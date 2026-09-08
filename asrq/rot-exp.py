# pyright: reportMissingImports=false

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random

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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    transformers.set_seed(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    assert cfg.transform.name == "rotation", (
        f"rot-exp.py only supports the rotation transform, got '{cfg.transform.name}'"
    )
    set_seed(cfg.seed)

    with open_dict(cfg.quantizer):
        cfg.quantizer.exclude_modules = cfg.model.exclude_modules
    with open_dict(cfg.transform):
        cfg.transform.model_name = cfg.model.name
        # The rotation is learned against the STE quantizers, so it needs the same
        # quantization settings the model will be deployed with (as exp.py does).
        cfg.transform.wbits = cfg.quantizer.bits
        cfg.transform.abits = cfg.activation_bits
        cfg.transform.wgroup = cfg.quantizer.get("group_size", None)
        # Default to gradient-based learning unless a Hadamard search was explicitly
        # requested (transform.hadamard_search=true); base.py's RotationTransform lets
        # learn_rotation win over hadamard_search, so it must stay off for search runs.
        if not cfg.transform.hadamard_search:
            cfg.transform.learn_rotation = True
    print(OmegaConf.to_yaml(cfg))

    quant_cfg = QuantConfig.from_dictconfig(cfg.quantizer)
    transform_cfg = TransformConfig.from_dictconfig(cfg.transform)
    calib_cfg = CalibConfig.from_dictconfig(cfg.calibration)

    # modelQ
    modelQ = ModelQ.from_pretrained(cfg.model.name, quant_cfg, calib_cfg)

    # Transform
    transform = BaseTransform.from_config(transform_cfg)
    modelQ.model.to("cuda")
    set_seed(cfg.seed)  # re-seed after model loading to ensure deterministic Q init + DataLoader shuffle
    transform.obtain_transform(modelQ)

    print("Done with obtaining rotations")


if __name__ == "__main__":
    main()
