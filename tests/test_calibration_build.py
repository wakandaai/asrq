"""Tests for asrq.calibration.build: the per-model transcribers that produce calibration labels.

The Whisper tests run whisper-tiny on real speech from the LibriSpeech dummy set. The
Parakeet adapter is tested against a stand-in exposing NeMo's transcribe() interface, since the
real checkpoint is several gigabytes.
"""

import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asrq.calibration import build

MODEL = "openai/whisper-tiny"


@pytest.fixture(scope="module")
def whisper_tiny():
    try:
        processor = WhisperProcessor.from_pretrained(MODEL)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL).eval()
    except OSError:
        pytest.skip(f"{MODEL} is not available offline")
    return model, processor


@pytest.fixture(scope="module")
def speech():
    """Four real utterances of different lengths."""
    try:
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    except Exception as error:
        pytest.skip(f"the LibriSpeech dummy set is not available: {error}")
    ds = ds.cast_column("audio", Audio(decode=False))
    return [sf.read(io.BytesIO(ds[i]["audio"]["bytes"]), dtype="float32")[0] for i in range(4)]


def test_whisper_transcriber_does_not_flag_finished_transcripts(whisper_tiny, speech):
    """generate() strips the final end-of-text, so the longest transcript in a batch has none.

    Detecting truncation by a missing end-of-text token flagged exactly that transcript in
    every batch, and every transcript at batch size 1.
    """
    model, processor = whisper_tiny
    transcribe = build.make_whisper_transcriber(model, processor)
    batched = transcribe(speech)
    single = [transcribe([waveform])[0] for waveform in speech]

    assert all(text for text, _ in batched)
    assert not any(truncated for _, truncated in batched)
    assert not any(truncated for _, truncated in single)


def test_whisper_transcriber_flags_a_transcript_that_used_the_whole_budget(whisper_tiny, speech):
    model, processor = whisper_tiny
    outputs = build.make_whisper_transcriber(model, processor, max_new_tokens=3)(speech)
    assert all(truncated for _, truncated in outputs)


def test_parakeet_transcriber_reads_hypothesis_text_and_never_truncates():
    calls = []

    class FakeCTCModel:
        def transcribe(self, audio, batch_size, verbose, num_workers):
            calls.append((len(audio), batch_size, [a.dtype for a in audio]))
            return [SimpleNamespace(text=f"  utterance {i} ") for i in range(len(audio))]

    transcribe = build.make_parakeet_ctc_transcriber(FakeCTCModel(), batch_size=8)
    outputs = transcribe([np.zeros(160, dtype=np.float64), np.zeros(320, dtype=np.float32)])

    assert outputs == [("utterance 0", False), ("utterance 1", False)]
    assert calls == [(2, 8, [np.float32, np.float32])]


def test_an_unsupported_model_names_the_supported_ones():
    with pytest.raises(NotImplementedError, match="parakeet-ctc-1.1b.*canary-qwen-2.5b"):
        build.load_transcriber("nvidia/canary-1b", batch_size=4)
