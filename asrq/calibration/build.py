"""Build a calibration set: select LibriSpeech audio, then transcribe it with a model.

Run once per model against the same directory; the audio is selected on the first run and
reused after that, and each model adds its own transcripts file:

    python -m asrq.calibration.build --model openai/whisper-large-v3 \\
        --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
    python -m asrq.calibration.build --model nvidia/parakeet-ctc-1.1b \\
        --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
    python -m asrq.calibration.build --model nvidia/canary-qwen-2.5b \\
        --calibration-dir outputs/calibration/librispeech_train_clean_360_2048

Transcription uses the unmodified full-precision checkpoint, loaded here directly rather than
through a ModelQ, so nothing a transform or quantizer does can reach the transcripts. The
layout of the set and why calibration uses transcripts are described in asrq.calibration.data.
"""

import argparse
from typing import List, Sequence, Tuple

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.models.salm import SALM
from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

from asrq.calibration.data import (
    SAMPLE_RATE,
    Transcriber,
    _read_jsonl,
    build_calibration_audio,
    transcribe_calibration_audio,
)

WHISPER_MODELS = ("openai/whisper-large-v3", "openai/whisper-tiny")
PARAKEET_CTC_MODELS = ("nvidia/parakeet-ctc-1.1b",)
CANARY_QWEN_MODELS = ("nvidia/canary-qwen-2.5b",)


def make_whisper_transcriber(model, processor, max_new_tokens: int = 440) -> Transcriber:
    """Greedy English transcription with the same forced prefix calibration teacher-forces.

    The prefix is ``<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>``, the one
    build_whisper_decoder_targets and the GPTQ calibration forward both start the decoder from,
    so a transcript is exactly the continuation the decoder produces after it. Greedy rather
    than beam search, because calibration should reproduce what the model does at inference.

    A transcript is reported truncated when it used the whole max_new_tokens budget. That
    cannot be read off an end-of-text token: Whisper's generate() strips the forced prefix and
    the final end-of-text from what it returns, so in a batch the longest transcript carries
    none even when it finished, and the shorter ones carry it only as padding. Counting the
    non-padding tokens is unambiguous instead; a transcript that ended by itself is always
    shorter than the budget. Whisper's decoder holds 448 positions, four taken by the prefix.
    """
    eos = processor.tokenizer.eos_token_id

    @torch.no_grad()
    def transcribe(waveforms: Sequence[np.ndarray]) -> List[Tuple[str, bool]]:
        features = processor.feature_extractor(
            list(waveforms), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device=model.device, dtype=model.dtype)
        generated = model.generate(
            input_features=features,
            language="en",
            task="transcribe",
            return_timestamps=False,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        lengths = (generated != eos).sum(dim=-1).tolist()
        return [(text.strip(), length >= max_new_tokens) for text, length in zip(texts, lengths)]

    return transcribe


def make_parakeet_ctc_transcriber(model, batch_size: int) -> Transcriber:
    """Greedy CTC transcription through NeMo's own transcribe(), as evaluation runs it.

    CTC emits one label per encoder frame and has no decoding budget to run out of, so no
    transcript is ever truncated. Repetitive and empty transcripts are still flagged by
    transcribe_calibration_audio.
    """

    @torch.no_grad()
    def transcribe(waveforms: Sequence[np.ndarray]) -> List[Tuple[str, bool]]:
        hypotheses = model.transcribe(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            batch_size=batch_size,
            verbose=False,
            num_workers=0,
        )
        return [(hypothesis.text.strip(), False) for hypothesis in hypotheses]

    return transcribe


def make_canary_qwen_transcriber(model, max_new_tokens: int = 256) -> Transcriber:
    """Greedy transcription with the prompt evaluation generates from, "Transcribe the following: <audio>".

    The budget is twice evaluation's 128 tokens, so the longest calibration utterances finish; one that
    still uses the whole budget without emitting <|im_end|> is reported truncated. The LLM returns only
    the generated tokens, right-padded with the pad token.
    """
    prompt = [{"role": "user", "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}}]

    @torch.no_grad()
    def transcribe(waveforms: Sequence[np.ndarray]) -> List[Tuple[str, bool]]:
        lengths = [len(w) for w in waveforms]
        audios = torch.zeros(len(waveforms), max(lengths))
        for i, waveform in enumerate(waveforms):
            audios[i, : len(waveform)] = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        generated = model.generate(
            prompts=[prompt] * len(waveforms),
            audios=audios.to(device=model.device, dtype=model.dtype),
            audio_lens=torch.tensor(lengths, device=model.device),
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                bos_token_id=model.text_bos_id, eos_token_id=model.text_eos_id, pad_token_id=model.text_pad_id,
            ),
        ).cpu()
        results = []
        for ids in generated:
            ends = (ids == model.text_eos_id).nonzero()
            truncated = len(ends) == 0
            ids = ids[: int(ends[0])] if not truncated else ids[ids != model.text_pad_id]
            results.append((model.tokenizer.ids_to_text(ids).strip(), truncated))
        return results

    return transcribe


def load_transcriber(model_name: str, batch_size: int) -> Transcriber:
    """Load model_name in full precision on the GPU and return its transcriber."""
    if model_name in WHISPER_MODELS:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=torch.float16)
        return make_whisper_transcriber(model.to("cuda").eval(), processor)
    if model_name in PARAKEET_CTC_MODELS:
        model = ASRModel.from_pretrained(model_name).to("cuda").eval()
        model.cfg.decoding.strategy = "greedy_batch"
        model.change_decoding_strategy(model.cfg.decoding)
        return make_parakeet_ctc_transcriber(model, batch_size)
    if model_name in CANARY_QWEN_MODELS:
        model = SALM.from_pretrained(model_name).to(device="cuda", dtype=torch.bfloat16).eval()
        return make_canary_qwen_transcriber(model)
    raise NotImplementedError(
        f"no transcriber for {model_name}; supported: {WHISPER_MODELS + PARAKEET_CTC_MODELS + CANARY_QWEN_MODELS}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="model id to transcribe with")
    parser.add_argument("--calibration-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=2048)
    parser.add_argument("--split", default="train.clean.360")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    build_calibration_audio(
        args.calibration_dir, args.num_samples, args.split, args.seed,
        args.max_duration, args.shuffle_buffer,
    )
    path = transcribe_calibration_audio(
        args.calibration_dir, args.model, load_transcriber(args.model, args.batch_size),
        args.batch_size,
    )
    rows = _read_jsonl(path)
    flagged = sum(r["truncated"] or r["repetitive"] or r["empty"] for r in rows)
    print(f"wrote {len(rows)} transcripts to {path}; {flagged} flagged and dropped at load time")


if __name__ == "__main__":
    main()
