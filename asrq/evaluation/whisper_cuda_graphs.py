"""Greedy Whisper transcription replayed from CUDA graphs.

Hugging Face ``generate`` runs every decoding step eagerly. Each step then launches hundreds of small
kernels from Python, so at the batch sizes used for evaluation the CPU sets the pace and the GPU idles:
an fp16 whisper-large-v3 decoder step took 22 ms eagerly against 5.5 ms replayed from a graph. Low-bit
kernels make it worse, since humming spends ~190 us of Python per quantized matmul. A CUDA graph
records the kernels once and replays them without Python, so the GPU work alone decides the speed.

Per batch size three graphs are captured on first use and replayed for every later batch of that size:

* encoder: the 30 s log-mel window to the encoder output;
* prompt: the forced prompt (``<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`` for a
  multilingual model) over an empty static KV cache, filling the cross-attention cache from the encoder
  output and choosing the first token;
* step: one token per utterance, reading and extending the static cache and choosing the next token.

Graphs need fixed shapes, so the self-attention cache is a ``StaticCache`` of ``max_target_positions``
slots whose write position lives on the GPU, and a batch smaller than the captured size is padded
with silence and its extra rows dropped. Token selection runs inside the graphs and reproduces what
``generate`` does for a short-form transcription without timestamps: ``suppress_tokens`` are masked
at every step, ``begin_suppress_tokens`` at the first, the argmax is taken, and an utterance that has
emitted end-of-text keeps emitting it. Python only replays the step graph until every utterance is done.

The static cache attends over all its slots with a mask, where ``generate`` attends over the tokens so
far, so logits agree to floating-point rounding rather than bit for bit.
"""

from typing import Dict, List, Sequence, Tuple
from weakref import WeakKeyDictionary

import torch
from transformers.cache_utils import EncoderDecoderCache, StaticCache


def whisper_prompt_ids(model) -> List[int]:
    """The forced decoder prompt ``generate`` uses for English transcription without timestamps."""
    generation = model.generation_config
    prompt = [generation.decoder_start_token_id]
    if getattr(generation, "is_multilingual", False):
        prompt += [generation.lang_to_id["<|en|>"], generation.task_to_id["transcribe"]]
    prompt.append(generation.no_timestamps_token_id)
    return prompt


def _token_mask(tokens: Sequence[int], vocab_size: int, device, dtype) -> torch.Tensor:
    mask = torch.zeros(vocab_size, device=device, dtype=dtype)
    if tokens:
        mask[torch.tensor(list(tokens), device=device)] = float("-inf")
    return mask


class WhisperGraphs:
    """The captured encoder, prompt and step graphs of one model for one batch size.

    The model is used only while capturing; the graphs replay its kernels on its weights afterwards,
    so they are valid only while the model lives unchanged.

    Args:
        model: A WhisperForConditionalGeneration on CUDA, in the dtype it will run in.
        batch_size: Utterances per replay.
        suppress_tokens: Token ids masked at every step.
        begin_suppress_tokens: Token ids masked at the first generated token.
        warmup: Eager runs on a side stream before each capture, which also allocate the caches.
    """

    def __init__(self, model, batch_size: int, suppress_tokens: Sequence[int],
                 begin_suppress_tokens: Sequence[int], warmup: int = 3):
        self.model = model
        self.batch_size = batch_size
        config = model.config
        device, dtype = model.device, model.dtype
        self.prompt = whisper_prompt_ids(model)
        self.max_new_tokens = config.max_target_positions - len(self.prompt)
        self.eos = model.generation_config.eos_token_id

        decoder_config = config.__class__.from_dict(config.to_dict())
        decoder_config.num_hidden_layers = config.decoder_layers
        self.cache = EncoderDecoderCache(
            StaticCache(config=decoder_config, max_cache_len=config.max_target_positions),
            StaticCache(config=decoder_config, max_cache_len=config.max_source_positions),
        )
        vocab = config.vocab_size
        self.features = torch.zeros(batch_size, config.num_mel_bins, 2 * config.max_source_positions,
                                    device=device, dtype=dtype)
        self.prompt_ids = torch.tensor([self.prompt] * batch_size, device=device)
        self.suppress = _token_mask(suppress_tokens, vocab, device, torch.float32)
        self.begin_suppress = self.suppress + _token_mask(begin_suppress_tokens, vocab, device, torch.float32)
        self.ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.tokens = torch.full((batch_size, self.max_new_tokens), self.eos, dtype=torch.long, device=device)
        self.position = torch.zeros(1, dtype=torch.long, device=device)

        self.encoder_output = None
        self.encoder_graph = self._capture(self._encode, warmup)
        self.pool = self.encoder_graph.pool()
        self.prompt_graph = self._capture(self._prompt, warmup, pool=self.pool, before=self._reset_cache)
        self.step_graph = self._capture(self._step, warmup, pool=self.pool,
                                        before=lambda: (self._reset_cache(), self._prompt()))
        self.model = None

    def _capture(self, fn, warmup, pool=None, before=None) -> torch.cuda.CUDAGraph:
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

    def _zero_cache_positions(self) -> None:
        for layer in self.cache.cross_attention_cache.layers + self.cache.self_attention_cache.layers:
            if getattr(layer, "is_initialized", False):
                layer.cumulative_length.zero_()

    def _reset_cache(self) -> None:
        """Empty the cache and mark cross-attention as not yet computed, so the prompt fills it."""
        self._zero_cache_positions()
        for layer_idx in self.cache.is_updated:
            self.cache.is_updated[layer_idx] = False

    def _encode(self) -> None:
        output = self.model.model.encoder(self.features).last_hidden_state
        if self.encoder_output is None:
            self.encoder_output = output
        else:
            self.encoder_output.copy_(output)

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.model.decoder(
            input_ids=input_ids, encoder_hidden_states=self.encoder_output,
            past_key_values=self.cache, use_cache=True,
        ).last_hidden_state
        return self.model.proj_out(hidden[:, -1]).float()

    def _record(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        chosen = (logits + mask).argmax(-1)
        chosen = torch.where(self.done, torch.full_like(chosen, self.eos), chosen)
        self.done.logical_or_(chosen == self.eos)
        self.ids.copy_(chosen.unsqueeze(1))
        self.tokens.index_copy_(1, self.position, chosen.unsqueeze(1))
        self.position.add_(1)

    def _prompt(self) -> None:
        self._zero_cache_positions()
        self.done.zero_()
        self.position.zero_()
        self.tokens.fill_(self.eos)
        self._record(self._logits(self.prompt_ids), self.begin_suppress)

    def _step(self) -> None:
        self._record(self._logits(self.ids), self.suppress)

    @torch.inference_mode()
    def generate(self, features: torch.Tensor, max_new_tokens: int = None) -> torch.Tensor:
        """Greedy tokens for up to ``batch_size`` utterances, end-of-text padded, prompt excluded."""
        count = features.shape[0]
        if count > self.batch_size:
            raise ValueError(f"got {count} utterances for graphs captured at batch size {self.batch_size}")
        limit = min(max_new_tokens or self.max_new_tokens, self.max_new_tokens)
        self.features[:count].copy_(features)
        self.features[count:].zero_()
        self.encoder_graph.replay()
        self.prompt_graph.replay()
        generated = 1
        while generated < limit and not bool(self.done[:count].all()):
            self.step_graph.replay()
            generated += 1
        return self.tokens[:count, :generated].clone()


_GRAPHS: "WeakKeyDictionary[torch.nn.Module, Dict[Tuple[int, torch.dtype], WhisperGraphs]]" = WeakKeyDictionary()


def whisper_graphs(model, batch_size: int) -> WhisperGraphs:
    """The model's graphs for ``batch_size``, captured on first use and kept while the model lives.

    Suppressed tokens are read from the model's generation config when the graphs are captured, so it
    must still hold them; ``generate`` clears them only on its own working copy.
    """
    per_model = _GRAPHS.setdefault(model, {})
    key = (batch_size, model.dtype)
    if key not in per_model:
        generation = model.generation_config
        with torch.inference_mode():
            per_model[key] = WhisperGraphs(model, batch_size, generation.suppress_tokens or [],
                                           generation.begin_suppress_tokens or [])
    return per_model[key]
