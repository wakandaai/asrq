"""CUDA-graph greedy transcription must reproduce Hugging Face generate's transcripts.

whisper-tiny on real LibriSpeech utterances from the local calibration set: generate as
generate_whisper calls it (English, transcribe, greedy) against the replayed graphs, for a full batch
and for a batch smaller than the captured size, which the graphs pad.
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from asrq.calibration.data import load_calibration_samples
from asrq.evaluation.whisper_cuda_graphs import whisper_graphs, whisper_prompt_ids

MODEL = "openai/whisper-tiny"
CALIBRATION = Path("outputs/calibration/librispeech_train_clean_360_2048")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA"),
    pytest.mark.skipif(not CALIBRATION.exists(), reason="local calibration audio not built"),
]


@pytest.fixture(scope="module")
def whisper():
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    audio = [a for a, _ in load_calibration_samples(str(CALIBRATION), text_source="reference", num_samples=6)]
    return model, processor, audio


def _features(model, processor, audio):
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    return features.to(device="cuda", dtype=model.dtype)


def _reference(model, processor, audio):
    with torch.inference_mode():
        ids = model.generate(_features(model, processor, audio), language="en", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)


def _graphed(model, processor, audio, batch_size):
    graphs = whisper_graphs(model, batch_size)
    tokens = graphs.generate(_features(model, processor, audio))
    return processor.batch_decode(tokens.cpu(), skip_special_tokens=True)


def test_the_prompt_is_the_one_generate_forces(whisper):
    model, processor, _ = whisper
    tokenizer = processor.tokenizer
    expected = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    assert whisper_prompt_ids(model) == tokenizer.convert_tokens_to_ids(expected)


def test_matches_generate_for_a_full_batch(whisper):
    model, processor, audio = whisper
    reference = _reference(model, processor, audio[:3])
    assert all(text.strip() for text in reference)
    assert _graphed(model, processor, audio[:3], batch_size=3) == reference


def test_a_smaller_batch_is_padded_and_still_matches(whisper):
    model, processor, audio = whisper
    reference = _reference(model, processor, audio[3:5])
    assert _graphed(model, processor, audio[3:5], batch_size=3) == reference


def test_graphs_are_reused_across_calls_and_stay_correct(whisper):
    model, processor, audio = whisper
    graphs = whisper_graphs(model, 3)
    assert whisper_graphs(model, 3) is graphs
    first = _graphed(model, processor, audio[:3], batch_size=3)
    second = _graphed(model, processor, audio[3:6], batch_size=3)
    assert first == _reference(model, processor, audio[:3])
    assert second == _reference(model, processor, audio[3:6])


def test_max_new_tokens_limits_the_output(whisper):
    model, processor, audio = whisper
    tokens = whisper_graphs(model, 3).generate(_features(model, processor, audio[:3]), max_new_tokens=5)
    assert tokens.shape == (3, 5)
