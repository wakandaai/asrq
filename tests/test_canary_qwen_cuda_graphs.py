"""CUDA-graph Canary-Qwen transcription must reproduce generate_canaryqwen's transcripts.

Canary-Qwen-2.5B from the local checkpoint cache, merged and in float16, on real LibriSpeech utterances from
the local calibration set: a full batch, a batch smaller than the graphs' batch size (padded), and graph reuse
across calls and length buckets. The static KV cache attends through a mask where generate attends over the
tokens so far, so the two agree to rounding: transcripts are compared word by word and a word error rate
between them of at most MAX_DISAGREEMENT is accepted.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("nemo")

import asrq.evaluation.openasr as openasr  # noqa: E402
from asrq.calibration.data import load_calibration_samples  # noqa: E402
from asrq.evaluation.canary_qwen_cuda_graphs import CanaryQwenGraphs, canary_qwen_graphs  # noqa: E402

CALIBRATION = Path("outputs/calibration/librispeech_train_clean_360_2048")
CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--nvidia--canary-qwen-2.5b"
MAX_DISAGREEMENT = 0.02

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA"),
    pytest.mark.skipif(not CALIBRATION.exists(), reason="local calibration audio not built"),
    pytest.mark.skipif(not CACHE.exists(), reason="canary-qwen-2.5b is not in the Hugging Face cache"),
]


@pytest.fixture(scope="module")
def canary():
    from asrq.models.nemo.canary_qwen import CanaryQwenQ

    model = CanaryQwenQ.load_model()[0].cuda().to(torch.float16).eval()
    samples = load_calibration_samples(str(CALIBRATION), text_source="reference", num_samples=8)
    audio = sorted((a for a, _ in samples), key=len, reverse=True)
    return model, audio


def _reference(model, audio, batch_size):
    data = {"audio": audio, "audio_len": [len(a) for a in audio]}
    with torch.inference_mode():
        return openasr.generate_canaryqwen(model, None, data, batch_size)


def _disagreement(a, b):
    import jiwer

    return jiwer.wer(" ".join(b).lower(), " ".join(a).lower())


def test_matches_generate_for_a_full_and_a_padded_batch(canary):
    model, audio = canary
    reference = _reference(model, audio[:5], batch_size=3)
    assert all(text.strip() for text in reference)
    graphed = CanaryQwenGraphs().transcribe(model, audio[:5], batch_size=3)
    assert len(graphed) == 5
    assert _disagreement(graphed, reference) <= MAX_DISAGREEMENT


def test_graphs_are_reused_across_calls_and_buckets(canary):
    model, audio = canary
    graphs = canary_qwen_graphs(model)
    assert canary_qwen_graphs(model) is graphs
    first = graphs.transcribe(model, audio, batch_size=2)
    encoders, batches = dict(graphs.encoders), {k: dict(v.buckets) for k, v in graphs.batches.items()}
    second = graphs.transcribe(model, audio, batch_size=2)
    assert first == second
    assert graphs.encoders == encoders and {k: dict(v.buckets) for k, v in graphs.batches.items()} == batches
    assert _disagreement(first, _reference(model, audio, batch_size=2)) <= MAX_DISAGREEMENT
