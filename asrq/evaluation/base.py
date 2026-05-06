import torch
import datetime
import os
from asrq.evaluation.openasr import evaluate_model
from asrq.quantizers.activation import modify_linears_with_activation_quantization



def evaluate_openasr(modelQ, cfg, generate_fn, evaluation_results_file, create_audio_files=False):
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists(evaluation_results_file):
        with open(evaluation_results_file, "w") as f:
            f.write("model,method,quantizer,transform,wbits,abits,dataset,split,wer\n")

    dataset_split = [
        ("ami", "test"),
        ("earnings22", "test"),
        ("gigaspeech", "test"),
        ("librispeech", "test.clean"),
        ("librispeech", "test.other"),
        ("spgispeech", "test"),
        ("tedlium", "test"),
        ("voxpopuli", "test"),
    ]
    model = modelQ.model.to(torch.float16).eval() # type: ignore
    if model.device != "cuda":
        model = model.to("cuda")
    if cfg.activation_bits < 16:
        linears_to_quantize = modelQ.for_activation_quantization()
        modify_linears_with_activation_quantization(model, linears_to_quantize, bits=cfg.activation_bits)

    for dataset, split in dataset_split:
        print(f"Evaluating dataset {dataset} split {split}...")
        result = evaluate_model(
            model, batch_size=cfg.model.eval_batch_size, dataset_path="hf-audio/esb-datasets-test-only-sorted", dataset=dataset,
            split=split, cache_dir="", eval_id="whisper", save_results_manifest=False, save_results_metrics=False, processor=modelQ.processor, generate_fn=generate_fn,
            batches_to_eval=None, create_audio_files=create_audio_files
        )
        with open(evaluation_results_file, "a") as f:
            f.write(f"{cfg.model.name},{cfg.method},{cfg.quantizer.name},{cfg.transform.name},{cfg.quantizer.bits},{cfg.activation_bits},{dataset},{split},{result['wer']}\n")
