"""CTC transcription for NeMo Conformer models, with the acoustic model replayed from CUDA graphs.

NeMo's ``transcribe`` runs the encoder eagerly. At small batch sizes each forward then launches
thousands of small kernels from Python, so the CPU sets the pace: Parakeet-CTC-1.1B's acoustic model
took 61.5 ms eagerly against 13.8 ms replayed from a graph for one 15 s utterance in fp16, and humming
W4A4 layers (~190 us of Python each) made eager W4A4 2.5x slower than fp16 while the graph made it
1.29x faster. At batch 128 the encoder is GPU-bound and a graph changes nothing.

A graph needs fixed shapes, while ``transcribe`` pads every batch to its longest utterance. Here each
batch is padded instead to the next multiple of ``bucket_seconds`` and to the full batch size, so one
graph per (batch size, bucket) covers every batch:

* the model's own preprocessor computes log-mel features from the padded audio with the true lengths;
* the graph runs the encoder and the CTC head on those features and lengths;
* NeMo's CTC decoding turns the log-probabilities of the real utterances into text.

The encoder masks every frame past an utterance's length, so padding further than ``transcribe``
would changes nothing but floating-point rounding at the utterance's last frames. Graphs share one
memory pool; evaluation orders utterances longest first, so the largest graph is captured first.
"""

import math
from typing import Dict, List, Sequence, Tuple
from weakref import WeakKeyDictionary

import numpy as np
import torch

BUCKET_SECONDS = 5.0


class _CTCGraph:
    """The encoder and CTC head of one model, captured for one batch size and feature length."""

    def __init__(self, model, batch_size: int, frames: int, pool, warmup: int = 3):
        device = next(model.parameters()).device
        dtype = next(p.dtype for p in model.encoder.parameters() if p.is_floating_point())
        self.features = torch.zeros(batch_size, model.encoder._feat_in, frames, device=device, dtype=dtype)
        self.lengths = torch.full((batch_size,), frames, dtype=torch.long, device=device)
        encoder, decoder = model.encoder, model.decoder

        def forward():
            encoded, encoded_lengths = encoder(audio_signal=self.features, length=self.lengths)
            return decoder(encoder_output=encoded), encoded_lengths

        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            for _ in range(warmup):
                forward()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.log_probs, self.encoded_lengths = forward()


class ParakeetGraphs:
    """The captured graphs of one model, keyed by batch size, feature length and dtype.

    Graphs replay the model's kernels on its weights, so they are valid only while the model lives
    unchanged. The model is passed to every call rather than held, so the cache does not keep it alive.

    Args:
        bucket_seconds: Audio is padded to a multiple of this many seconds; one graph per bucket.
    """

    def __init__(self, bucket_seconds: float = BUCKET_SECONDS):
        self.bucket_seconds = bucket_seconds
        self.graphs: Dict[Tuple[int, int, torch.dtype], _CTCGraph] = {}
        self.pool = None

    @torch.inference_mode()
    def transcribe(self, model, audios: Sequence[np.ndarray], batch_size: int) -> List[str]:
        """Greedy CTC transcripts of ``audios``, ``batch_size`` utterances per replay.

        The model is put in eval mode first. NeMo's ``transcribe`` returns with ``model.encoder`` in
        training mode while ``model.training`` stays False, and a graph captured then records dropout
        and batch-norm batch statistics; the preprocessor would also add dither to the features.
        """
        model.eval()
        device = next(model.parameters()).device
        dtype = next(p.dtype for p in model.encoder.parameters() if p.is_floating_point())
        sample_rate = model.preprocessor._sample_rate
        bucket = max(1, int(round(self.bucket_seconds * sample_rate)))
        texts: List[str] = []
        for start in range(0, len(audios), batch_size):
            batch = [np.asarray(a, dtype=np.float32) for a in audios[start:start + batch_size]]
            count = len(batch)
            samples = bucket * math.ceil(max(len(a) for a in batch) / bucket)
            signal = torch.zeros(count, samples, dtype=torch.float32)
            for i, audio in enumerate(batch):
                signal[i, :len(audio)] = torch.from_numpy(audio)
            lengths = torch.tensor([len(a) for a in batch], dtype=torch.long)
            features, feature_lengths = model.preprocessor(
                input_signal=signal.to(device), length=lengths.to(device)
            )

            key = (batch_size, features.shape[-1], dtype)
            if key not in self.graphs:
                if self.pool is None:
                    self.pool = torch.cuda.graph_pool_handle()
                self.graphs[key] = _CTCGraph(model, batch_size, features.shape[-1], self.pool)
            graph = self.graphs[key]
            graph.features[:count].copy_(features)
            graph.features[count:].zero_()
            graph.lengths[:count].copy_(feature_lengths)
            graph.lengths[count:].fill_(features.shape[-1])
            graph.graph.replay()

            hypotheses = model.decoding.ctc_decoder_predictions_tensor(
                graph.log_probs[:count].clone(),
                decoder_lengths=graph.encoded_lengths[:count].clone(),
                return_hypotheses=True,
            )
            texts.extend(h.text for h in hypotheses)
        return texts


_GRAPHS: "WeakKeyDictionary[torch.nn.Module, ParakeetGraphs]" = WeakKeyDictionary()


def parakeet_graphs(model) -> ParakeetGraphs:
    """The model's graph cache, created on first use and dropped with the model."""
    if model not in _GRAPHS:
        _GRAPHS[model] = ParakeetGraphs()
    return _GRAPHS[model]
