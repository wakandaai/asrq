# pyright: reportMissingImports=false

import random
import datetime
import os
import hydra
import numpy as np
import torch
import transformers
from asrq.calibration.base import CalibConfig
from asrq.core.model import ModelQ
from asrq.evaluation.base import evaluate_openasr
from asrq.quantizers.base import QuantConfig
from asrq.transforms.base import BaseTransform, TransformConfig
import asrq.evaluation.openasr as openasr
from omegaconf import DictConfig, OmegaConf, open_dict


def set_seed(seed=42):
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior for some ops
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)


def build_evaluation_output_paths(
    *,
    model_name: str,
    method: str,
    quantizer_name: str,
    transform_name: str,
    weight_bits: int,
    activation_bits: int,
    timestamp: str,
) -> tuple[str, str]:
    evaluation_dir = os.path.join("results", "evaluations")
    os.makedirs(evaluation_dir, exist_ok=True)
    stem = (
        f"{model_name.replace('/','-')}_{method}_{quantizer_name}_{transform_name}_"
        f"{weight_bits}_{activation_bits}-{timestamp}"
    )
    return (
        os.path.join(evaluation_dir, f"{stem}_results.csv"),
        os.path.join(evaluation_dir, f"{stem}_config.yaml"),
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    print(f"Running ASRQ experiment:")
    print(f"  Quantizer:      {cfg.quantizer.name}")
    print(f"  Model:          {cfg.model.name}")
    print(f"  Transform:      {cfg.transform.name}")
    print(f"  Seed:           {cfg.seed}")

    # config assignments
    with open_dict(cfg.quantizer):
        cfg.quantizer.exclude_modules = cfg.model.exclude_modules
    with open_dict(cfg.transform):
        cfg.transform.model_name = cfg.model.name
        cfg.transform.wbits = cfg.quantizer.bits
        cfg.transform.abits = cfg.activation_bits

    if cfg.transform.name == "rotation":
        with open_dict(cfg.transform):
            cfg.transform.learn_rotation = False


    quant_cfg = QuantConfig.from_dictconfig(cfg.quantizer)
    calib_cfg = CalibConfig.from_dictconfig(cfg.calibration)

    # modelQ
    modelQ = ModelQ.from_pretrained(cfg.model.name, quant_cfg, calib_cfg)

    # Transforms
    if cfg.transform.name == "none":
        print("No transform will be applied.")
        modelQ.model.to("cuda")
    else:
        transform_cfg = TransformConfig.from_dictconfig(cfg.transform)
        transform = BaseTransform.from_config(transform_cfg)
        modelQ.model.to("cuda")
        transform.obtain_transform(modelQ)
        if cfg.transform.use:
            transform.apply_transform(modelQ)

    # quantization
    modelQ.quantize()

    # evaluation
    if cfg.evaluate:
        evaluation_results_file, evaluation_results_config_file = build_evaluation_output_paths(
            model_name=cfg.model.name,
            method=cfg.method,
            quantizer_name=cfg.quantizer.name,
            transform_name=cfg.transform.name,
            weight_bits=cfg.quantizer.bits,
            activation_bits=cfg.activation_bits,
            timestamp=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        )
        with open(evaluation_results_config_file, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        evaluate_openasr(modelQ=modelQ, cfg=cfg, generate_fn=getattr(openasr, cfg.model.generate_fn), evaluation_results_file=evaluation_results_file, create_audio_files=cfg.create_audio_files)


if __name__ == "__main__":
    main() # type: ignore
