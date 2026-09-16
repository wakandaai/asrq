"""CUDA-graph CTC transcription must reproduce NeMo transcribe's transcripts.

Parakeet-CTC-1.1B from the local checkpoint cache, in float16 as evaluation runs it, on real
LibriSpeech utterances from the local calibration set: a full batch, a batch smaller than the graph's
batch size, and graph reuse across calls and length buckets.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from asrq.calibration.data import load_calibration_samples
from asrq.evaluation.parakeet_cuda_graphs import ParakeetGraphs, parakeet_graphs

CALIBRATION = Path("outputs/calibration/librispeech_train_clean_360_2048")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA"),
    pytest.mark.skipif(not CALIBRATION.exists(), reason="local calibration audio not built"),
]


@pytest.fixture(scope="module")
def parakeet():
    models = pytest.importorskip("nemo.collections.asr.models")
    try:
        model = models.ASRModel.from_pretrained("nvidia/parakeet-ctc-1.1b")
    except Exception as error:
        pytest.skip(f"parakeet-ctc-1.1b not available: {error}")
    model.cfg.decoding.strategy = "greedy_batch"
    model.change_decoding_strategy(model.cfg.decoding)
    model = model.eval().cuda().to(torch.float16)
    samples = load_calibration_samples(str(CALIBRATION), text_source="reference", num_samples=8)
    audio = sorted((a for a, _ in samples), key=len, reverse=True)
    return model, audio


def _reference(model, audio, batch_size):
    with torch.inference_mode():
        hypotheses = model.transcribe([np.asarray(a, dtype=np.float32) for a in audio],
                                      batch_size=batch_size, verbose=False, num_workers=1)
    return [h.text for h in hypotheses]


def test_matches_transcribe_for_a_full_and_a_padded_batch(parakeet):
    """transcribe runs first here: it leaves the encoder in training mode, which the graphs must undo."""
    model, audio = parakeet
    graphs = ParakeetGraphs()
    reference = _reference(model, audio[:5], batch_size=3)
    assert all(text.strip() for text in reference)
    assert graphs.transcribe(model, audio[:5], batch_size=3) == reference


def test_graphs_are_captured_once_per_bucket_and_reused(parakeet):
    model, audio = parakeet
    graphs = parakeet_graphs(model)
    assert parakeet_graphs(model) is graphs
    first = graphs.transcribe(model, audio, batch_size=2)
    captured = dict(graphs.graphs)
    second = graphs.transcribe(model, audio, batch_size=2)
    assert first == second == _reference(model, audio, batch_size=2)
    assert graphs.graphs == captured
    bucket = graphs.bucket_seconds * 16000
    buckets = {int(np.ceil(max(len(a) for a in audio[i:i + 2]) / bucket)) for i in range(0, len(audio), 2)}
    assert len(captured) == len(buckets)
