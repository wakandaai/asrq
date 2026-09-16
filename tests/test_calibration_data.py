"""Tests for asrq.calibration.data: the on-disk calibration set shared by rotation and GPTQ.

Everything runs on synthetic FLAC files, so no dataset or model is downloaded; the
transcribers are tested in test_calibration_build.py. The module is loaded by path for the same
reason as the rotation tests: asrq/__init__.py does not import cleanly yet.
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_PATH = Path(__file__).resolve().parents[1] / "asrq" / "calibration" / "data.py"
_spec = importlib.util.spec_from_file_location("calibration_data", _PATH)
data = importlib.util.module_from_spec(_spec)
sys.modules["calibration_data"] = data
_spec.loader.exec_module(data)

MODEL = "openai/whisper-tiny"


def _flac_bytes(seconds=1.0, sample_rate=16000, seed=0):
    rng = np.random.default_rng(seed)
    audio = (0.1 * rng.standard_normal(int(seconds * sample_rate))).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="FLAC", subtype="PCM_16")
    return buffer.getvalue(), audio


def _make_set(root, count=5):
    """A calibration directory with count utterances and a manifest, as build_calibration_audio
    would leave it."""
    manifest = []
    for i in range(count):
        flac, _ = _flac_bytes(seconds=0.5 + 0.1 * i, seed=i)
        entry = data.save_audio_sample(root, f"utt-{i}", flac)
        entry.update(reference_text=f"REFERENCE {i}", speaker_id=i, chapter_id=0)
        manifest.append(entry)
    data._write_jsonl(root / data.MANIFEST_FILE, manifest)
    return manifest


def test_audio_is_stored_verbatim_and_decodes_to_the_original(tmp_path):
    flac, audio = _flac_bytes(seconds=0.75)
    entry = data.save_audio_sample(tmp_path, "utt", flac)

    assert (tmp_path / entry["audio_filepath"]).read_bytes() == flac
    assert entry["duration"] == pytest.approx(0.75)
    decoded = data.load_audio(tmp_path, entry)
    assert decoded.dtype == np.float32
    assert np.abs(decoded - audio).max() < 1 / 32767 + 1e-6


def test_audio_at_the_wrong_sample_rate_is_rejected(tmp_path):
    flac, _ = _flac_bytes(sample_rate=8000)
    with pytest.raises(ValueError, match="8000 Hz"):
        data.save_audio_sample(tmp_path, "utt", flac)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("and then and then and then and then and then and then and then", True),
        ("the quick brown fox jumps over the lazy dog while the cat watches it", False),
        ("yes yes yes", False),
    ],
)
def test_is_repetitive(text, expected):
    assert data.is_repetitive(text) is expected


def test_flagged_transcripts_are_dropped_and_order_is_kept(tmp_path):
    manifest = _make_set(tmp_path)
    loop = "so so so so so so so so so so so so so so so"
    outputs = {
        "utt-0": ("First.", False),
        "utt-1": (loop, False),
        "utt-2": ("Cut off mid", True),
        "utt-3": ("", False),
        "utt-4": ("Fifth one.", False),
    }

    in_order = iter(outputs.values())

    def transcribe(waveforms):
        assert all(w.dtype == np.float32 for w in waveforms)
        return [next(in_order) for _ in waveforms]

    path = data.transcribe_calibration_audio(tmp_path, MODEL, transcribe, batch_size=2)
    rows = {row["id"]: row for row in data._read_jsonl(path)}
    assert rows["utt-1"]["repetitive"] and rows["utt-2"]["truncated"] and rows["utt-3"]["empty"]

    samples = data.load_calibration_samples(tmp_path, MODEL)
    assert [text for _, text in samples] == ["First.", "Fifth one."]
    first = data.load_audio(tmp_path, manifest[0])
    assert np.array_equal(samples[0][0], first)


def test_num_samples_caps_the_usable_utterances(tmp_path):
    _make_set(tmp_path)
    data.transcribe_calibration_audio(
        tmp_path, MODEL, lambda waves: [("fine", False)] * len(waves), batch_size=3
    )
    assert len(data.load_calibration_samples(tmp_path, MODEL, num_samples=3)) == 3


def test_reference_text_keeps_every_utterance(tmp_path):
    _make_set(tmp_path)
    samples = data.load_calibration_samples(tmp_path, text_source="reference")
    assert [text for _, text in samples] == [f"REFERENCE {i}" for i in range(5)]


def test_missing_transcripts_say_how_to_create_them(tmp_path):
    _make_set(tmp_path)
    with pytest.raises(FileNotFoundError, match="python -m asrq.calibration.build --model"):
        data.load_calibration_samples(tmp_path, MODEL)


def test_transcripts_of_different_models_do_not_collide(tmp_path):
    _make_set(tmp_path)
    data.transcribe_calibration_audio(tmp_path, "a/model", lambda w: [("from a", False)] * len(w))
    data.transcribe_calibration_audio(tmp_path, "b/model", lambda w: [("from b", False)] * len(w))
    assert data.load_calibration_samples(tmp_path, "a/model")[0][1] == "from a"
    assert data.load_calibration_samples(tmp_path, "b/model")[0][1] == "from b"


def test_an_existing_set_is_reused_without_streaming(tmp_path, monkeypatch):
    _make_set(tmp_path, count=5)
    info = {
        "source": "openslr/librispeech_asr", "config": "all", "split": "train.clean.360",
        "seed": 42, "num_samples": 5, "max_duration": 30.0, "shuffle_buffer": 10_000,
    }
    (tmp_path / data.INFO_FILE).write_text(json.dumps(info))

    def no_download(*args, **kwargs):
        raise AssertionError("an existing calibration set must not be rebuilt")

    monkeypatch.setattr(data, "load_dataset", no_download)
    assert data.build_calibration_audio(tmp_path, num_samples=5) == tmp_path


def test_a_set_built_with_other_settings_is_not_overwritten(tmp_path, monkeypatch):
    _make_set(tmp_path, count=5)
    (tmp_path / data.INFO_FILE).write_text(json.dumps({"split": "train.other.500"}))
    monkeypatch.setattr(data, "load_dataset", lambda *a, **k: pytest.fail("must not stream"))
    with pytest.raises(ValueError, match="refusing to overwrite"):
        data.build_calibration_audio(tmp_path, num_samples=5)
