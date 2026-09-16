"""Greedy Canary-Qwen transcription replayed from CUDA graphs.

Hugging Face ``generate`` runs the Qwen LLM eagerly, launching hundreds of small kernels from Python per
decoding step; humming's quantized matmuls (~190 us of Python each) make that CPU cost dominate. A CUDA
graph records the kernels once and replays them. Graphs need fixed shapes, so:

* **Audio** is padded to the next multiple of ``bucket_seconds`` and the batch to its full size; features
  are computed eagerly by the model's preprocessor, and an **encoder graph** per (batch size, bucket)
  runs the Conformer encoder and the projection into the LLM.
* A **prompt graph** per (batch size, bucket) builds the prompt exactly as SALM's ``generate`` does --
  ``<|im_start|>user\\nTranscribe the following: `` + the utterance's audio embeddings +
  ``<|im_end|>\\n<|im_start|>assistant\\n``, left-padded to the bucket's length -- with a GPU gather
  driven by the encoded lengths, then runs the LLM over it into a static KV cache and picks the first
  token. Positions count only the unpadded tokens, as generate derives them from the attention mask.
* A **step graph** per batch size embeds the last token, marks its cache slot in the attention mask,
  runs the LLM for one position and picks the next token.

Every prompt and step graph of one batch size shares one static KV cache and one attention mask, sized
for the longest prompt seen so far; a longer one rebuilds the cache and recaptures those graphs. Evaluation
orders utterances longest first, so that normally never happens.
Token selection is the argmax, as ``generate`` without sampling does; an utterance that has emitted
<|im_end|> keeps emitting it, and Python only replays the step graph until every utterance is done or
``max_new_tokens`` are generated. The static cache attends over all its slots through the mask, so logits
agree with ``generate`` to floating-point rounding rather than bit for bit.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple
from weakref import WeakKeyDictionary

import numpy as np
import torch
from nemo.collections.common.prompts.formatter import PromptFormatter
from transformers.cache_utils import StaticCache

BUCKET_SECONDS = 5.0
MAX_NEW_TOKENS = 128


def _compute_dtype(model) -> torch.dtype:
    """The model's floating-point dtype. Not its first parameter's: humming layers hold packed integer weights."""
    return model.embed_tokens.weight.dtype


def _capture(fn, pool, warmup: int = 3, before=None) -> torch.cuda.CUDAGraph:
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(warmup):
            if before is not None:
                before()
            fn()
    torch.cuda.current_stream().wait_stream(side)
    if before is not None:
        before()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        fn()
    return graph


class _BatchGraphs:
    """The static tensors, KV cache and graphs of one model at one batch size."""

    def __init__(self, model, batch_size: int, max_prompt: int, max_new_tokens: int, pool):
        self.batch_size = batch_size
        self.max_prompt = max_prompt
        self.max_new_tokens = max_new_tokens
        self.pool = pool
        device = next(model.parameters()).device
        llm_config = model.llm.config
        self.cache = StaticCache(config=llm_config, max_cache_len=max_prompt + max_new_tokens)
        self.mask = torch.zeros(batch_size, max_prompt + max_new_tokens, dtype=torch.bool, device=device)
        self.ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.positions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.write = torch.zeros(1, dtype=torch.long, device=device)
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.tokens = torch.full((batch_size, max_new_tokens), model.text_eos_id, dtype=torch.long, device=device)
        self.step_index = torch.zeros(1, dtype=torch.long, device=device)
        self.eos = model.text_eos_id
        self.buckets: Dict[int, torch.cuda.CUDAGraph] = {}
        # A graph reads the tensors it was captured on but does not keep them alive; the prompt graphs'
        # constant inputs are held here for as long as their graphs.
        self.bucket_inputs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.step_graph: Optional[torch.cuda.CUDAGraph] = None

    def reset_cache(self) -> None:
        for layer in self.cache.layers:
            if getattr(layer, "is_initialized", False):
                layer.cumulative_length.zero_()

    def choose(self, model, hidden: torch.Tensor) -> None:
        logits = model.llm.lm_head(hidden[:, -1]).float()
        chosen = logits.argmax(-1)
        chosen = torch.where(self.done, torch.full_like(chosen, self.eos), chosen)
        self.done.logical_or_(chosen == self.eos)
        self.ids.copy_(chosen.unsqueeze(1))
        self.tokens.index_copy_(1, self.step_index, chosen.unsqueeze(1))
        self.step_index.add_(1)

    def llm_hidden(self, model, embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return model.llm.model(
            inputs_embeds=embeds, attention_mask=self.mask, position_ids=positions,
            past_key_values=self.cache, use_cache=True,
        ).last_hidden_state


class CanaryQwenGraphs:
    """CUDA-graph transcription for one Canary-Qwen model; graphs are captured on first use.

    The model is passed to every call rather than held, so the cache does not keep it alive; the graphs
    are valid only while the model lives unchanged.
    """

    def __init__(self, bucket_seconds: float = BUCKET_SECONDS, max_new_tokens: int = MAX_NEW_TOKENS):
        self.bucket_seconds = bucket_seconds
        self.max_new_tokens = max_new_tokens
        self.encoders: Dict[Tuple[int, int, torch.dtype], Tuple[torch.cuda.CUDAGraph, dict]] = {}
        self.batches: Dict[Tuple[int, torch.dtype], _BatchGraphs] = {}
        self.pool = None

    @staticmethod
    def prompt_pieces(model) -> Tuple[List[int], List[int]]:
        """The prompt's token ids before and after the audio placeholder."""
        formatter = PromptFormatter.resolve(model.cfg.prompt_format)(model.tokenizer)
        user = {"role": "user", "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}}
        ids = formatter.encode_dialog(turns=[user])["input_ids"].tolist()
        split = ids.index(model.audio_locator_tag_id)
        return ids[:split], ids[split + 1:]

    def _pool(self):
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        return self.pool

    def _encoder(self, model, batch_size: int, features: torch.Tensor):
        """The encoder graph and its static tensors for this batch size and feature length."""
        dtype = _compute_dtype(model)
        key = (batch_size, features.shape[-1], dtype)
        if key in self.encoders:
            return self.encoders[key]
        static = {
            "features": features.to(dtype).clone(),
            "feature_lengths": torch.full((batch_size,), features.shape[-1], dtype=torch.long, device=features.device),
        }

        def encode():
            encoded, encoded_lengths = model.perception(
                processed_signal=static["features"], processed_signal_length=static["feature_lengths"])
            if "audio" not in static:
                static["audio"], static["audio_lengths"] = encoded, encoded_lengths
            else:
                static["audio"].copy_(encoded)
                static["audio_lengths"].copy_(encoded_lengths)

        graph = _capture(encode, self._pool())
        self.encoders[key] = (graph, static)
        return self.encoders[key]

    def _batch(self, model, batch_size: int, prompt_length: int) -> _BatchGraphs:
        """The batch size's KV cache and graphs, rebuilt (graphs recaptured) for a longer prompt."""
        key = (batch_size, _compute_dtype(model))
        current = self.batches.get(key)
        if current is None or current.max_prompt < prompt_length:
            if current is not None:
                del self.batches[key], current
                torch.cuda.empty_cache()
            self.batches[key] = _BatchGraphs(model, batch_size, prompt_length, self.max_new_tokens, self._pool())
        return self.batches[key]

    def _prompt_graph(self, model, graphs: _BatchGraphs, encoder_static: dict) -> torch.cuda.CUDAGraph:
        """The prompt graph for one bucket, and the batch size's step graph on first use."""
        audio_frames = encoder_static["audio"].shape[1]
        if audio_frames in graphs.buckets:
            return graphs.buckets[audio_frames]
        device = encoder_static["audio"].device
        prefix_ids, suffix_ids = self.prompt_pieces(model)
        n_prefix, n_suffix = len(prefix_ids), len(suffix_ids)
        text_ids = torch.tensor(prefix_ids + suffix_ids, device=device)
        prompt_length = n_prefix + audio_frames + n_suffix
        columns = torch.arange(prompt_length, device=device)
        graphs.bucket_inputs[audio_frames] = (text_ids, columns)

        def prompt():
            graphs.reset_cache()
            batch = graphs.batch_size
            audio, lengths = encoder_static["audio"], encoder_static["audio_lengths"].unsqueeze(1)
            relative = columns.unsqueeze(0) - (audio_frames - lengths)
            valid = relative >= 0
            source = torch.where(relative < n_prefix + lengths, relative, relative - lengths + audio_frames)
            source = source.clamp(0, prompt_length - 1)
            text = model.embed_tokens(text_ids).to(audio.dtype)
            pieces = torch.cat([
                text[:n_prefix].unsqueeze(0).expand(batch, -1, -1), audio,
                text[n_prefix:].unsqueeze(0).expand(batch, -1, -1),
            ], dim=1)
            embeds = torch.gather(pieces, 1, source.unsqueeze(-1).expand(-1, -1, audio.shape[-1]))
            embeds = embeds * valid.unsqueeze(-1).to(embeds.dtype)
            graphs.mask.zero_()
            graphs.mask[:, :prompt_length].copy_(valid)
            positions = relative.clamp(min=0)
            graphs.done.zero_()
            graphs.step_index.zero_()
            graphs.tokens.fill_(graphs.eos)
            graphs.choose(model, graphs.llm_hidden(model, embeds, positions))
            graphs.positions.copy_(positions[:, -1:] + 1)
            graphs.write.fill_(prompt_length)

        graphs.buckets[audio_frames] = _capture(prompt, graphs.pool)
        if graphs.step_graph is None:
            def step():
                graphs.mask.index_fill_(1, graphs.write, True)
                graphs.choose(model, graphs.llm_hidden(model, model.embed_tokens(graphs.ids), graphs.positions))
                graphs.positions.add_(1)
                graphs.write.add_(1)

            graphs.step_graph = _capture(step, graphs.pool, before=prompt)
        return graphs.buckets[audio_frames]

    @torch.inference_mode()
    def transcribe(self, model, audios: Sequence[np.ndarray], batch_size: int) -> List[str]:
        """Greedy transcripts of 16 kHz waveforms, ``batch_size`` utterances per replay."""
        model.eval()
        device = next(model.parameters()).device
        bucket = max(1, int(round(self.bucket_seconds * model.sampling_rate)))
        prefix_ids, suffix_ids = self.prompt_pieces(model)
        texts: List[str] = []
        for start in range(0, len(audios), batch_size):
            batch = [np.asarray(a, dtype=np.float32) for a in audios[start:start + batch_size]]
            count = len(batch)
            samples = bucket * math.ceil(max(len(a) for a in batch) / bucket)
            signal = torch.zeros(batch_size, samples, dtype=torch.float32)
            lengths = torch.full((batch_size,), samples, dtype=torch.long)
            for i, audio in enumerate(batch):
                signal[i, :len(audio)] = torch.from_numpy(audio)
                lengths[i] = len(audio)
            features, feature_lengths = model.perception.preprocessor(
                input_signal=signal.to(device), length=lengths.to(device))

            encoder_graph, encoder_static = self._encoder(model, batch_size, features)
            encoder_static["features"].copy_(features)
            encoder_static["feature_lengths"].copy_(feature_lengths)
            encoder_graph.replay()
            prompt_length = len(prefix_ids) + encoder_static["audio"].shape[1] + len(suffix_ids)
            graphs = self._batch(model, batch_size, prompt_length)
            prompt_graph = self._prompt_graph(model, graphs, encoder_static)
            prompt_graph.replay()
            generated = 1
            while generated < self.max_new_tokens and not bool(graphs.done[:count].all()):
                graphs.step_graph.replay()
                generated += 1
            for ids in graphs.tokens[:count, :generated].cpu():
                ends = (ids == model.text_eos_id).nonzero()
                ids = ids[: int(ends[0])] if len(ends) else ids
                texts.append(model.tokenizer.ids_to_text(ids).strip())
        return texts


_GRAPHS: "WeakKeyDictionary[torch.nn.Module, CanaryQwenGraphs]" = WeakKeyDictionary()


def canary_qwen_graphs(model) -> CanaryQwenGraphs:
    """The model's graph cache, created on first use and dropped with the model."""
    if model not in _GRAPHS:
        _GRAPHS[model] = CanaryQwenGraphs()
    return _GRAPHS[model]
