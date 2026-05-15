# pyright: reportMissingImports=false

import random
import datetime
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
        evaluation_results_file_name_stem = f"results/evaluations/{cfg.model.name.replace('/','-')}_{cfg.method}_{cfg.quantizer.name}_{cfg.transform.name}_{cfg.quantizer.bits}_{cfg.activation_bits}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        evaluation_results_file = f"{evaluation_results_file_name_stem}_results.csv"
        evaluation_results_config_file = f"{evaluation_results_file_name_stem}_config.yaml"
        with open(evaluation_results_config_file, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        evaluate_openasr(modelQ=modelQ, cfg=cfg, generate_fn=getattr(openasr, cfg.model.generate_fn), evaluation_results_file=evaluation_results_file, create_audio_files=cfg.create_audio_files)


if __name__ == "__main__":
    main() # type: ignore
