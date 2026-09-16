"""Calibration data shared by rotation learning and GPTQ: audio on disk, transcribed by the model.

A calibration set is a directory:

    <calibration_dir>/
        info.json                     how the audio was selected
        manifest.jsonl                one line per utterance: id, audio path, duration, reference
        audio/<id>.flac               the original LibriSpeech FLAC bytes, unmodified
        transcripts/<model>.jsonl     one line per utterance: id, the model's transcript, flags

The audio is selected once and shared by every model. Each model then transcribes it itself,
and those transcripts -- not the reference text -- are what calibration teacher-forces the
decoder with. The goal of calibration is to keep the quantized model behaving like the
full-precision one, so the decoder should be fed the sequences the full-precision model
actually produces. Reference text differs from them in casing, punctuation and normalization,
which puts every teacher-forced position in a context the model never reaches at inference,
and gives a label-based objective a formatting difference to fit instead of quantization error.

Transcription errors are kept deliberately: the target is the full-precision model's behavior,
not the truth. Only degenerate outputs are dropped -- a transcript that ran into the token
limit or loops on a phrase -- since those are failure modes rather than behavior to preserve.

This module only stores and loads calibration sets, so it imports no model code. Building a
set, which needs the models to transcribe with, lives in asrq.calibration.build.
"""

import io
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

SAMPLE_RATE = 16000
INFO_FILE = "info.json"
MANIFEST_FILE = "manifest.jsonl"
AUDIO_DIR = "audio"
TRANSCRIPTS_DIR = "transcripts"

Transcriber = Callable[[Sequence[np.ndarray]], List[Tuple[str, bool]]]


def model_slug(model_name: str) -> str:
    """File-system name for a model id, ``openai/whisper-large-v3`` -> ``openai--whisper-large-v3``.
    """
    return model_name.replace("/", "--")


def _read_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_audio_sample(calibration_dir: Path, sample_id: str, flac_bytes: bytes) -> Dict:
    """Write one utterance's FLAC bytes verbatim and return its manifest fields.

    The bytes are written as they came rather than decoded and re-encoded, so the stored audio
    is bit-identical to the source.
    """
    info = sf.info(io.BytesIO(flac_bytes))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{sample_id} is sampled at {info.samplerate} Hz, not {SAMPLE_RATE}")
    relative = Path(AUDIO_DIR) / f"{sample_id}.flac"
    path = calibration_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(flac_bytes)
    return {"id": sample_id, "audio_filepath": str(relative), "duration": info.duration}


def build_calibration_audio(
    calibration_dir: str,
    num_samples: int = 2048,
    split: str = "train.clean.360",
    seed: int = 42,
    max_duration: float = 30.0,
    shuffle_buffer: int = 10_000,
) -> Path:
    """Select num_samples LibriSpeech utterances and store them with a manifest.

    The dataset is streamed and shuffled through a buffer, so nothing beyond the utterances
    read into that buffer is downloaded. Utterances longer than max_duration are skipped:
    Whisper's encoder sees a fixed 30-second window, so a longer clip would be transcribed and
    calibrated on a truncated signal that ends mid-word.

    Use a training split. The evaluation sets are LibriSpeech test-clean and test-other among
    others, and calibrating on audio that is later evaluated on would leak.

    Does nothing if the directory already holds a set built with the same settings; raises if
    it holds one built with different settings, rather than silently mixing the two.

    Args:
        calibration_dir: Directory to write the set to.
        num_samples: Number of utterances to keep.
        split: LibriSpeech split, from the ``all`` configuration of openslr/librispeech_asr.
        seed: Shuffle seed.
        max_duration: Longest utterance to keep, in seconds.
        shuffle_buffer: Streaming shuffle buffer size. Larger mixes more speakers and chapters
            into the selection at the cost of reading more before the first sample.

    Returns:
        The calibration directory.
    """
    root = Path(calibration_dir)
    info = {
        "source": "openslr/librispeech_asr",
        "config": "all",
        "split": split,
        "seed": seed,
        "num_samples": num_samples,
        "max_duration": max_duration,
        "shuffle_buffer": shuffle_buffer,
    }
    info_path = root / INFO_FILE
    if info_path.exists():
        existing = json.loads(info_path.read_text())
        if existing != info:
            raise ValueError(
                f"{root} already holds a calibration set built with {existing}; refusing to "
                f"overwrite it with {info}. Use a different directory."
            )
        if len(_read_jsonl(root / MANIFEST_FILE)) == num_samples:
            return root

    stream = load_dataset("openslr/librispeech_asr", "all", split=split, streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)

    manifest = []
    progress = tqdm(total=num_samples, desc="Selecting calibration audio")
    for row in stream:
        duration = sf.info(io.BytesIO(row["audio"]["bytes"])).duration
        if duration > max_duration:
            continue
        entry = save_audio_sample(root, row["id"], row["audio"]["bytes"])
        entry.update(
            reference_text=row["text"],
            speaker_id=row["speaker_id"],
            chapter_id=row["chapter_id"],
        )
        manifest.append(entry)
        progress.update(1)
        if len(manifest) == num_samples:
            break
    progress.close()
    if len(manifest) < num_samples:
        raise ValueError(f"{split} ran out after {len(manifest)} utterances under {max_duration}s")

    _write_jsonl(root / MANIFEST_FILE, manifest)
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return root


def load_audio(calibration_dir: Path, entry: Dict) -> np.ndarray:
    """Decode one manifest entry's audio to a float32 waveform in [-1, 1]."""
    audio, sample_rate = sf.read(calibration_dir / entry["audio_filepath"], dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{entry['id']} is sampled at {sample_rate} Hz, not {SAMPLE_RATE}")
    return audio


def is_repetitive(
    text: str, n: int = 3, min_words: int = 12, max_distinct_ratio: float = 0.5
) -> bool:
    """Heuristic for a decoding loop: too few distinct word n-grams for the transcript's length.

    A looping decoder repeats the same phrase, so most of its n-grams are copies. Natural speech
    of this length almost never reuses half of its trigrams. Short transcripts are exempt, since
    a handful of words has too few n-grams to judge.
    """
    words = text.lower().split()
    if len(words) < min_words:
        return False
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / len(ngrams) < max_distinct_ratio


def transcribe_calibration_audio(
    calibration_dir: str,
    model_name: str,
    transcribe: Transcriber,
    batch_size: int = 16,
) -> Path:
    """Transcribe every utterance of a calibration set with one model and save the transcripts.

    Each transcript is stored with the flags load_calibration_samples filters on: ``truncated``
    when decoding stopped at the token limit instead of an end-of-text token, ``repetitive``
    when it looks like a decoding loop, and ``empty``.

    Transcribe with the unmodified full-precision model, before any transform or quantization
    touches it: the transcripts are the behavior calibration is meant to preserve.

    Args:
        calibration_dir: A directory written by build_calibration_audio.
        model_name: Model id, which names the transcripts file.
        transcribe: ``fn(waveforms) -> [(text, truncated), ...]`` for a batch of waveforms.
        batch_size: Utterances per call to transcribe.

    Returns:
        Path of the transcripts file.
    """
    root = Path(calibration_dir)
    manifest = _read_jsonl(root / MANIFEST_FILE)
    rows = []
    for start in tqdm(range(0, len(manifest), batch_size), desc=f"Transcribing with {model_name}"):
        entries = manifest[start : start + batch_size]
        outputs = transcribe([load_audio(root, entry) for entry in entries])
        for entry, (text, truncated) in zip(entries, outputs):
            rows.append(
                {
                    "id": entry["id"],
                    "text": text,
                    "truncated": bool(truncated),
                    "repetitive": is_repetitive(text),
                    "empty": not text.strip(),
                }
            )
    path = root / TRANSCRIPTS_DIR / f"{model_slug(model_name)}.jsonl"
    _write_jsonl(path, rows)
    return path


def load_calibration_samples(
    calibration_dir: str,
    model_name: Optional[str] = None,
    num_samples: Optional[int] = None,
    text_source: str = "transcript",
) -> List[Tuple[np.ndarray, str]]:
    """Load ``(waveform, text)`` pairs from a calibration set.

    With ``text_source="transcript"`` the text is model_name's own transcript, and utterances
    whose transcript is flagged truncated, repetitive or empty are dropped. With
    ``text_source="reference"`` it is the LibriSpeech reference text, and nothing is dropped;
    that suits a model whose calibration never reads the text, such as a CTC encoder.

    Args:
        calibration_dir: A directory written by build_calibration_audio.
        model_name: Model whose transcripts to use. Required for ``"transcript"``.
        num_samples: Keep at most this many, in manifest order. None keeps all.
        text_source: ``"transcript"`` or ``"reference"``.

    Returns:
        The pairs, in manifest order.
    """
    root = Path(calibration_dir)
    manifest = _read_jsonl(root / MANIFEST_FILE)

    if text_source == "reference":
        texts = {entry["id"]: entry["reference_text"] for entry in manifest}
    elif text_source == "transcript":
        if model_name is None:
            raise ValueError("text_source='transcript' needs the model whose transcripts to use")
        path = root / TRANSCRIPTS_DIR / f"{model_slug(model_name)}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"no transcripts for {model_name} in {root}. Create them with:\n"
                f"  python -m asrq.calibration.build --model {model_name} --calibration-dir {root}"
            )
        rows = _read_jsonl(path)
        kept = [r for r in rows if not (r["truncated"] or r["repetitive"] or r["empty"])]
        if len(kept) < len(rows):
            print(
                f"calibration: dropped {len(rows) - len(kept)} of {len(rows)} transcripts "
                f"flagged truncated, repetitive or empty"
            )
        texts = {r["id"]: r["text"] for r in kept}
    else:
        raise ValueError(f"text_source must be 'transcript' or 'reference', got {text_source!r}")

    entries = [entry for entry in manifest if entry["id"] in texts]
    if num_samples is not None:
        if len(entries) < num_samples:
            print(f"calibration: only {len(entries)} usable utterances, fewer than {num_samples}")
        entries = entries[:num_samples]
    return [(load_audio(root, entry), texts[entry["id"]]) for entry in entries]
