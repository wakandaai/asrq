import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from asrq.core.model import ModelQ
from asrq.quantizers.base import QuantConfig
from asrq.transforms.base import BaseTransform, TransformConfig
from asrq.calibration.base import CalibConfig
import asrq.evaluation.openasr as openasr
from asrq.evaluation.base import evaluate_openasr

import torch
import numpy as np
import random
import transformers

def set_seed(seed=42):
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior for some ops
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)


# Load configs from tests/configs using Hydra
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

for transform in ["shrinking"]:
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=["quantizer=gptq", "model=parakeet", f"transform={transform}"])

    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # config assignments
    with open_dict(cfg.quantizer):
        cfg.quantizer.exclude_modules = cfg.model.exclude_modules
    with open_dict(cfg.transform):
        cfg.transform.model_name = cfg.model.name

    quant_cfg = QuantConfig.from_dictconfig(cfg.quantizer)
    transform_cfg = TransformConfig.from_dictconfig(cfg.transform)
    calib_cfg = CalibConfig.from_dictconfig(cfg.calibration)

    # modelQ
    modelQ = ModelQ.from_pretrained(cfg.model.name, quant_cfg, calib_cfg)

    # Transforms 
    transform = BaseTransform.from_config(transform_cfg)
    modelQ.model.to("cuda")
    transform.obtain_transform(modelQ)
    transform.apply_transform(modelQ)

    # quantization
    modelQ.quantize()

    # evaluation
    if cfg.evaluate:
        evaluate_openasr(modelQ=modelQ, cfg=cfg, generate_fn=getattr(openasr, cfg.model.generate_fn), create_audio_files=False)