import torch
import datetime
import os
from asrq.evaluation.openasr import DATASET_PATH, evaluate_model
from asrq.quantizers.activation import modify_linears_with_activation_quantization


# Mirrors the DATASET_CONFIGS lists in the Open ASR Leaderboard run scripts, e.g.
# third_party/open_asr_leaderboard/transformers/run_whisper.sh. The leaderboard now
# scores the cleaned variants of ami/gigaspeech/voxpopuli, and tedlium was dropped.
DATASET_SPLITS = [
    # ("ami_cleaned", "test"),
    ("ami", "test"),
    # ("gigaspeech_cleaned", "test"),
    ("earnings22", "test"),
    ("gigaspeech", "test"),
    # ("voxpopuli_cleaned_aa", "test"),
    ("librispeech", "test.clean"),
    ("librispeech", "test.other"),
    ("spgispeech", "test"),
    ("voxpopuli", "test"),
]


def evaluate_openasr(modelQ, cfg, generate_fn, evaluation_results_file, create_audio_files=False):
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(evaluation_results_file):
        with open(evaluation_results_file, "w") as f:
            f.write("model,method,quantizer,transform,wbits,abits,dataset,split,wer\n")

    model = modelQ.model.to(torch.float16).eval() # type: ignore
    if model.device != "cuda":
        model = model.to("cuda")
    if cfg.activation_bits < 16:
        linears_to_quantize = modelQ.for_activation_quantization()
        modify_linears_with_activation_quantization(model, linears_to_quantize, bits=cfg.activation_bits)

    for dataset, split in DATASET_SPLITS:
        print(f"Evaluating dataset {dataset} split {split}...")
        result = evaluate_model(
            model, batch_size=cfg.model.eval_batch_size, dataset_path=DATASET_PATH, dataset=dataset,
            split=split, cache_dir="", eval_id="whisper", save_results_manifest=False, save_results_metrics=False, processor=modelQ.processor, generate_fn=generate_fn,
            batches_to_eval=10, create_audio_files=create_audio_files
        )
        with open(evaluation_results_file, "a") as f:
            f.write(f"{cfg.model.name},{cfg.method},{cfg.quantizer.name},{cfg.transform.name},{cfg.quantizer.bits},{cfg.activation_bits},{dataset},{split},{result['wer']}\n")
