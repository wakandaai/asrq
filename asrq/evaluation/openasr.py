# pyright: reportMissingImports=false

import os
import json
from typing import Any, Dict
import torch
from datasets import load_dataset, Audio
from asrq.evaluation.english_text_normalizer import normalizer
import numpy as np
import io
import soundfile as sf
from tqdm import tqdm
import lhotse
import evaluate
import time

# Nemo SALM model
from nemo.collections.speechlm2.models.salm import SALM

# transformers
from transformers import GenerationConfig

# metric - WER
wer_metric = evaluate.load("wer")



def write_manifest(
    references: list,
    transcriptions: list,
    model_id: str,
    dataset_path: str,
    dataset_name: str,
    split: str,
    audio_length: list = None, # type: ignore
    transcription_time: list = None, # type: ignore
):
    """
    Writes a manifest file (jsonl format) and returns the path to the file.

    Args:
        references: Ground truth reference texts.
        transcriptions: Model predicted transcriptions.
        model_id: String identifier for the model.
        dataset_path: Path to the dataset.
        dataset_name: Name of the dataset.
        split: Dataset split name.
        audio_length: Length of each audio sample in seconds.
        transcription_time: Transcription time of each sample in seconds.

    Returns:
        Path to the manifest file.
    """
    model_id = model_id.replace("/", "-")
    dataset_path = dataset_path.replace("/", "-")
    dataset_name = dataset_name.replace("/", "-")

    if len(references) != len(transcriptions):
        raise ValueError(
            f"The number of samples in `references` ({len(references)}) "
            f"must match `transcriptions` ({len(transcriptions)})."
        )

    if audio_length is not None and len(audio_length) != len(references):
        raise ValueError(
            f"The number of samples in `audio_length` ({len(audio_length)}) "
            f"must match `references` ({len(references)})."
        )
    if transcription_time is not None and len(transcription_time) != len(references):
        raise ValueError(
            f"The number of samples in `transcription_time` ({len(transcription_time)}) "
            f"must match `references` ({len(references)})."
        )

    audio_length = (
        audio_length if audio_length is not None else len(references) * [None]
    )
    transcription_time = (
        transcription_time
        if transcription_time is not None
        else len(references) * [None]
    )

    basedir = "./results/"
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    manifest_path = os.path.join(
        basedir, f"MODEL_{model_id}_DATASET_{dataset_name}_{split}.jsonl"
    )

    with open(manifest_path, "w", encoding="utf-8") as f:
        for idx, (text, transcript, audio_length, transcription_time) in enumerate(
            zip(references, transcriptions, audio_length, transcription_time)
        ):
            datum = {
                "audio_filepath": f"sample_{idx}",  # dummy value for Speech Data Processor
                "duration": audio_length,
                "time": transcription_time,
                "text": text,
                "pred_text": transcript,
            }
            f.write(f"{json.dumps(datum, ensure_ascii=False)}\n")
    return manifest_path


def get_text(sample:Dict[str, Any]) -> str:
    """Extracts the text from a dataset sample.
    
    Args:
        sample: A dictionary containing the dataset sample.
    Returns:
        The text from the sample.
    """
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def normalize(batch:Dict[str, str]) -> Dict[str, str]:
    """Normalizes the text in a dataset sample.
    
    Args:
        batch: A dictionary containing the dataset sample with an 'audio' key and a text key.
    Returns:
        The input batch with an additional 'norm_text' key containing the normalized text.
    """
    batch["original_text"] = get_text(batch)
    batch["norm_text"] = normalizer(batch["original_text"])
    return batch


def is_target_text_in_range(ref:str) -> bool:
    """Filters out samples with empty or ignore time segment in scoring transcriptions.
    
    Args:
        ref: Reference transcription.
    Returns:
        True if the reference transcription is valid, False otherwise.
    """
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: lhotse.CutSet) -> Dict[str, Any]:
        """Loads and returns the audio samples and their lengths from a batch of cuts.
        
        Args:
            cuts: A batch of lhotse.CutSet objects.
        Returns:
            A dictionary containing the cuts, audio samples, and their lengths.
        """
        cuts = lhotse.CutSet([c.to_mono(mono_downmix=True) if isinstance(c, lhotse.MultiCut) else c for c in cuts]) # type: ignore
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


def setup_dloader(audio_files, batch_size, num_workers):
    """ Sets up a DataLoader for the given audio files.

    Args:
        audio_files: List of paths to audio files.
        batch_size: Number of samples per batch.
        num_workers: Number of worker threads for loading data.
    Returns:
        A DataLoader that yields batches of audio samples.
    """
    cuts = lhotse.CutSet([lhotse.Recording.from_file(p).to_cut() for p in audio_files])
    cuts = cuts.resample(16000)
    return torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(cuts, max_cuts=batch_size),
        num_workers=num_workers,
        batch_size=None,
    )


def parse_hyp(answer: torch.Tensor, eos_tokens: torch.Tensor) -> torch.Tensor:
    """Parses the model's output to extract the transcription up to the first end-of-sequence token.
    
    Args:
        answer: A tensor containing the model's output token IDs.
        eos_tokens: A tensor containing the end-of-sequence token IDs.
    Returns:
        A tensor containing the token IDs up to the first end-of-sequence token.
    """
    end = (answer == torch.isin(answer, eos_tokens)).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


def transcribe(model: SALM, dloader: torch.utils.data.DataLoader) -> list[str]:
    """ Transcribes audio samples using the given model and DataLoader.

    Args:
        model: The SALM model for transcription.
        dloader: DataLoader that yields batches of audio samples.
    Returns:
        A list of transcribed texts.
    """
    hyps = []
    eos_tokens = torch.tensor([model.text_eos_id])
    for batch_idx, batch in enumerate(dloader):
        answer_ids = model.generate(
            prompts=[
                [
                    {"role": "user", "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}}
                ]
            ] * len(batch["cuts"]),
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            generation_config=GenerationConfig(
                max_new_tokens=128,
                bos_token_id=model.text_bos_id,
                eos_token_id=eos_tokens,
                pad_token_id=model.text_pad_id,
            ),
        )
        answer_ids = [parse_hyp(ans, eos_tokens) for ans in answer_ids.cpu()]
        hyps.extend(model.tokenizer.ids_to_text(ans).strip() for ans in answer_ids)
    return hyps



# batch_size = 192
# device = 0
# dataset_path = "hf-audio/esb-datasets-test-only-sorted"
# dataset = "ami"
# split = "test"
# cache_dir = ""

def generate_canaryqwen(model, processor, all_data, batch_size, max_new_tokens=None):
    predictions = []
    
    eos_tokens = torch.tensor([model.text_eos_id])
    for batch_start in range(0, len(all_data["audio"]), batch_size):
        batch_end = min(batch_start + batch_size, len(all_data["audio"]))
        batch_audios = all_data["audio"][batch_start:batch_end]
        batch_audio_lens = all_data["audio_len"][batch_start:batch_end]
        max_len = max(batch_audio_lens)
        temp = np.zeros((len(batch_audios), max_len), dtype=np.float32)
        for i, a in enumerate(batch_audios):
            temp[i, :len(a)] = a
        batch_audios = torch.tensor(temp).to(model.dtype)
        batch_audio_lens = torch.tensor(batch_audio_lens)
        eos_tokens = torch.tensor([model.text_eos_id])
    
        answer_ids = model.generate(
                prompts=[
                    [
                        {"role": "user", "slots": {"message": f"Transcribe the following: {model.audio_locator_tag}"}}
                    ]
                ] * len(batch_audios),
                audios=batch_audios.to(model.device, non_blocking=True),
                audio_lens=batch_audio_lens.to(model.device, non_blocking=True),
                generation_config=GenerationConfig(
                    max_new_tokens=128,
                    bos_token_id=model.text_bos_id,
                    eos_token_id=eos_tokens,
                    pad_token_id=model.text_pad_id,
                ),
        )
        answer_ids = [parse_hyp(ans, eos_tokens) for ans in answer_ids.cpu()]
        preds = [model.tokenizer.ids_to_text(ans).strip() for ans in answer_ids]
        predictions.extend(preds)
    return predictions


# for whisper
from torch.nn.attention import sdpa_kernel, SDPBackend
def generate_whisper(model, processor, all_data, batch_size, max_new_tokens=None):
    predictions = []
    for batch_start in range(0, len(all_data["audio"]), batch_size):
        batch_end = min(batch_start + batch_size, len(all_data["audio"]))
        batch_audios = all_data["audio"][batch_start:batch_end]
        inputs = processor(batch_audios, sampling_rate=16000, return_tensors="pt").to(model.device)
        inputs = {"input_features": inputs.input_features.to(model.dtype)}
        gkwargs = {
            "max_new_tokens": max_new_tokens,
        }
        if getattr(model.generation_config, "is_multilingual"):
            gkwargs["language"] = "en"
            gkwargs["task"] = "transcribe"
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            pred_ids = model.generate(**inputs, **gkwargs)
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)
        predictions.extend(pred_text)
    return predictions


def generate_canary(model, processor, all_data, batch_size):
    transcriptions = [
        val.text for val in
        model.transcribe([f for f in all_data["audio_files"]], batch_size=batch_size, verbose=False, pnc="nopnc", num_workers=1)
    ]
    return transcriptions


def generate_parakeet(model, processor, all_data, batch_size):
    # Disable CUDA graphs to avoid compilation issues with quantized models
    # if hasattr(model, 'cfg') and hasattr(model.cfg, 'decoding'):
    #     model.cfg.decoding.use_cuda_graph_decoder = False
    #     if hasattr(model, 'change_decoding_strategy'):
    #         model.change_decoding_strategy(model.cfg.decoding)
    # model.decoding.decoding.decoding_computer.disable_cuda_graphs()
    transcriptions = [
        val.text for val in
        model.transcribe([f for f in all_data["audio_files"]], batch_size=batch_size, verbose=False, num_workers=1)
    ]

    return transcriptions

def generate_granite(model, processor, all_data, batch_size):
    chat = [
        {
            "role": "system",
            "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
        },
        {
            "role": "user",
            "content": "<|audio|>can you transcribe the speech into a written format?",
        }
    ]
    text = processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    model_inputs = processor(
        [text] * len(all_data["audio"]),
        all_data["audio"],
        device=model.device,
        return_tensors="pt",
    ).to(model.device)
    model_outputs = model.generate(
        **model_inputs,
        bos_token_id=processor.tokenizer.bos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        repetition_penalty=1.0,
        max_new_tokens=200,
        num_beams=1,
        min_new_tokens=None,
    )
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[:, num_input_tokens:]
    output_text = processor.tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True  
    )

    return output_text

def evaluate_model(
    model, batch_size=192, dataset_path="hf-audio/esb-datasets-test-only-sorted", dataset="ami",
     split="test", cache_dir="", eval_id="", save_results_manifest=False, save_results_metrics=True, processor=None, generate_fn=None,
     batches_to_eval=None, create_audio_files=False
):
    eval_start_time = time.time()
    cache_dir = "../"#cache_dir or os.getcwd()
    DATA_CACHE_DIR = os.path.join(cache_dir, "audio_cache")
    DATASET_NAME = dataset
    SPLIT_NAME = split
    sample_file_id = 0

    CACHE_DIR = os.path.join(DATA_CACHE_DIR, DATASET_NAME, SPLIT_NAME)
    os.makedirs(CACHE_DIR, exist_ok=True)

    torch.set_float32_matmul_precision("medium")

    ds = load_dataset(
        dataset_path,
        dataset,
        split=split,
        # streaming=True,
        # token=True
    )
    if batches_to_eval is not None:
        ds = ds.take(batches_to_eval * batch_size) # type: ignore
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.map(normalize)
    ds = ds.filter(is_target_text_in_range, input_columns=["norm_text"])

    # ===
    all_data = {
        "audio": [],
        "references": [],
        "durations": [],
        "audio_len": [],
        "audio_files": [],
    }
    
    for sample in ds:
        audio = sample["audio"]["array"] # type: ignore
        sample_rate = sample["audio"]["sampling_rate"] # type: ignore
        audio_len = len(audio)
        duration = audio_len / sample_rate

        all_data["audio"].append(audio)
        all_data["references"].append(sample["norm_text"]) # type: ignore
        all_data["durations"].append(duration)
        all_data["audio_len"].append(audio_len)

        audio_file_path = os.path.join(CACHE_DIR, f"{sample_file_id}.wav")
        all_data["audio_files"].append(audio_file_path)
        if create_audio_files:
            sf.write(audio_file_path, audio, sample_rate)

        sample_file_id += 1

    sorted_indices = sorted(range(len(all_data["durations"])), key=lambda k: all_data["durations"][k], reverse=True)
    all_data["audio"] = [all_data["audio"][i] for i in sorted_indices]
    all_data["references"] = [all_data["references"][i] for i in sorted_indices]
    all_data["durations"] = [all_data["durations"][i] for i in sorted_indices]
    all_data["audio_len"] = [all_data["audio_len"][i] for i in sorted_indices]
    all_data["audio_files"] = [all_data["audio_files"][i] for i in sorted_indices]

    total_time = 0
    for i in range(2): # warmup first, then evaluate
        if i == 0: # for warmup, only 4 batches will be used.
            data = {k: v[:batch_size] for k, v in all_data.items()}
        else:
            data = all_data
            
        with torch.inference_mode():
            start_time = time.time()
            transcriptions = generate_fn(model, processor, data, batch_size=batch_size) # type: ignore
            end_time = time.time()
        
        if i == 1:
            total_time += end_time - start_time
            all_data = data
    
    if isinstance(transcriptions, tuple) and len(transcriptions) == 2: # type: ignore
        transcriptions = transcriptions[0] # type: ignore
    predictions = [normalizer(pred) for pred in transcriptions] # type: ignore
    avg_time = total_time / len(all_data["audio"])

    # for i in range(min(4, len(predictions))):
    #     print("Ref:", all_data["references"][i])
    #     print("Pred:", predictions[i])
    #     print("-----")

    if save_results_manifest:
        manifest_path = write_manifest(
            all_data["references"],
            predictions,
            eval_id,
            dataset_path,
            dataset,
            split,
            audio_length=all_data["durations"],
            transcription_time=[avg_time]*len(all_data["audio"]),
        )

        print("Results saved at path:", os.path.abspath(manifest_path))

    refs, preds = [], []
    for ref, pred in zip(all_data["references"], predictions):
        # if len(pred) > len(ref)*3:
        #     pred = pred[:len(ref)]
        refs.append(ref)
        preds.append(pred)
    wer = wer_metric.compute(references=refs, predictions=preds)
    wer = round(100*wer, 2) # type: ignore

    audio_length = sum(all_data["durations"])
    rtfx = audio_length / total_time
    rtfx = round(rtfx, 2)

    print("RTFX:", rtfx)
    print("WER:", wer, "%")

    if save_results_metrics:
        metric_file = os.path.join(cache_dir, "metrics.txt")
        with open(metric_file, "a") as f:
            f.write(f"{dataset} - {split} Results - {time.asctime()}\n")
            f.write(f"RTFX: {rtfx}\nWER: {wer:.3f}%\n\n")
            
    print("Evaluation completed in {:.2f} seconds".format(time.time() - eval_start_time))
    
    return {
        "wer": wer,
        "rtfx": rtfx,
    }

if __name__=="__main__":
    model_id = "nvidia/canary-qwen-2.5b"
    device = torch.device(f"cuda:{0}")
    model = SALM.from_pretrained(model_id).eval().to(torch.bfloat16).to(device)

    evaluate_model(
        model,
    )