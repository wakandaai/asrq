import torch
import datetime
import os
from asrq.evaluation.openasr import DATASET_PATH, evaluate_model
from asrq.quantizers.activation import (
    attach_activation_quantization,
    build_activation_quantizers,
)


# The default evaluation set when the config gives no eval_datasets. Mirrors the DATASET_CONFIGS
# lists in the Open ASR Leaderboard run scripts, e.g.
# third_party/open_asr_leaderboard/transformers/run_whisper.sh. The leaderboard now
# scores the cleaned variants of ami/gigaspeech/voxpopuli, and tedlium was dropped.
DATASET_SPLITS = [
    ("ami_cleaned", "test"),
    # ("ami", "test"),
    ("earnings22", "test"),
    # "earnings22_cleaned_aa_chunked test ArtificialAnalysis/Earnings22-Cleaned-AA-chunked",
    # ("gigaspeech", "test"),
    ("gigaspeech_cleaned", "test"),
    ("librispeech", "test.clean"),
    ("librispeech", "test.other"),
    ("spgispeech", "test"),
    # ("voxpopuli", "test"),
    ("voxpopuli_cleaned_aa", "test"),
]


def apply_activation_quantization(modelQ, model, cfg):
    """Fake-quantize activations as the top-level config says, for evaluation.

    Reads activation_bits, activation_group_size, activation_symmetric and
    activation_groupwise_roles -- the settings the rotation search is also given -- and attaches a
    quantizer to the input of every layer in modelQ.activation_quantization_roles(). Roles in
    activation_groupwise_roles get groups of activation_group_size; the rest are quantized per
    token. Does nothing at 16 bits or more.

    Returns:
        The hook handles, or an empty list.
    """
    if cfg.activation_bits >= 16:
        return []
    roles = modelQ.activation_quantization_roles()
    groupwise_roles = cfg.get("activation_groupwise_roles", None)
    quantizers = build_activation_quantizers(
        roles,
        cfg.activation_bits,
        cfg.get("activation_group_size", -1),
        cfg.get("activation_symmetric", True),
        None if groupwise_roles is None else list(groupwise_roles),
    )
    counts = {}
    for quantizer in quantizers.values():
        counts[repr(quantizer)] = counts.get(repr(quantizer), 0) + 1
    print("activation quantization: " + ", ".join(f"{n} layers {q}" for q, n in counts.items()))
    return attach_activation_quantization(model, quantizers)


def evaluation_splits(cfg):
    """The ``(dataset, split)`` pairs to evaluate, from the config's eval_datasets.

    Each entry names an Open ASR Leaderboard dataset configuration and one of its splits, e.g.
    ``{dataset: librispeech, split: test.clean}``. Without eval_datasets, every set in
    DATASET_SPLITS is evaluated.
    """
    entries = cfg.get("eval_datasets", None)
    if entries is None:
        return list(DATASET_SPLITS)
    splits = []
    for entry in entries:
        if "dataset" not in entry or "split" not in entry:
            raise ValueError(f"each eval_datasets entry needs a dataset and a split, got {entry}")
        splits.append((entry["dataset"], entry["split"]))
    if not splits:
        raise ValueError("eval_datasets is empty; list at least one dataset and split")
    return splits


def evaluate_openasr(modelQ, cfg, generate_fn, evaluation_results_file, create_audio_files=False):
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(evaluation_results_file):
        with open(evaluation_results_file, "w") as f:
            f.write("model,method,quantizer,transform,wbits,abits,dataset,split,wer\n")

    model = modelQ.model.to(torch.float16).eval() # type: ignore
    if model.device != "cuda":
        model = model.to("cuda")
    if cfg.get("inference", "fake") == "humming":
        print("inference: humming -- activations are quantized inside the ASRQLinear layers")
    else:
        apply_activation_quantization(modelQ, model, cfg)

    batches_to_eval = cfg.get("eval_batches", None)
    if batches_to_eval is not None:
        print(f"Evaluating only the first {batches_to_eval} batches "
              f"({batches_to_eval * cfg.model.eval_batch_size} utterances) of each dataset split")

    for dataset, split in evaluation_splits(cfg):
        print(f"Evaluating dataset {dataset} split {split}...")
        result = evaluate_model(
            model, batch_size=cfg.model.eval_batch_size, dataset_path=DATASET_PATH, dataset=dataset,
            split=split, cache_dir="", eval_id="whisper", save_results_manifest=False, save_results_metrics=False, processor=modelQ.processor, generate_fn=generate_fn,
            batches_to_eval=batches_to_eval, create_audio_files=create_audio_files,
            dtype=getattr(torch, cfg.get("eval_dtype", "bfloat16")),
        )
        with open(evaluation_results_file, "a") as f:
            f.write(f"{cfg.model.name},{cfg.method},{cfg.quantizer.name},{cfg.transform.name},{cfg.quantizer.bits},{cfg.activation_bits},{dataset},{split},{result['wer']}\n")
