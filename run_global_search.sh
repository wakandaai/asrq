#!/bin/bash
#SBATCH --job-name=parakeet-fnmse_gl_wa8a
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/parakeet-fnmse_gl_wa8a-%j.out
#SBATCH --error=logs/parakeet-fnmse_gl_wa8a-%j.err

source ~/.bashrc
conda activate asrq
cd /home/blessedg/asrq
echo "whisper glob search"
python - <<'PY'
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from asrq.core.model import ModelQ
from asrq.quantizers.base import QuantConfig
from asrq.calibration.base import CalibConfig
from asrq.transforms.base import BaseTransform, TransformConfig
import torch, random, numpy as np, transformers

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)

CONFIG_DIR = os.path.join(os.getcwd(), "asrq", "configs")
with initialize_config_dir(version_base=None, config_dir=os.path.abspath(CONFIG_DIR)):
    cfg = compose(config_name="config", overrides=[
        "model=canary_qwen",   # change to whisper if needed
        "quantizer=gptq",
        "quantizer.bits=4",
        "activation_bits=4",
        "evaluate=false",
        "transform=rotation",
        "transform.type=search",
        "transform.search_mode=global_qe",
        "calibration.num_samples=768",
        "transform.num_samples=768",
        "transform.batch_size=64",
        "transform.population_size=6",
        "transform.elite_count=2",
        "transform.path=/home/blessedg/asrq/outputs/rotation/canary_qwen_final_nmse_w4a4.pt",
        "transform.generations=8",
        "+transform.global_score_metric=final_nmse",
        "transform.patience=4",
        "transform.qe_generations=8",
        "transform.qe_patience=4",
        "transform.q2_refine_generations=8",
        "transform.outer_rounds=8",
        "transform.outer_patience=2",
        "transform.qe_min_delta=1e-3",
    ])

set_seed(cfg.seed)
with open_dict(cfg.quantizer):
    cfg.quantizer.exclude_modules = cfg.model.exclude_modules
with open_dict(cfg.transform):
    cfg.transform.model_name = cfg.model.name
    cfg.transform.learn_rotation = True
    cfg.transform.wbits = cfg.quantizer.bits
    cfg.transform.abits = cfg.activation_bits

quant_cfg = QuantConfig.from_dictconfig(cfg.quantizer)
transform_cfg = TransformConfig.from_dictconfig(cfg.transform)
calib_cfg = CalibConfig.from_dictconfig(cfg.calibration)

modelQ = ModelQ.from_pretrained(cfg.model.name, quant_cfg, calib_cfg)
transform = BaseTransform.from_config(transform_cfg)
modelQ.model.to("cuda")
transform.obtain_transform(modelQ)

print("Saved rotation to:", transform.cfg.path)
PY

echo "whisper glob quantize"

python -m asrq.exp \
  model=parakeet \
  quantizer=gptq \
  quantizer.bits=4 \
  activation_bits=4 \
  transform=rotation \
  transform.type=search \
  transform.type=global_qe_qd \
  transform.path=/home/blessedg/asrq/outputs/rotation/canary_qwen_final_nmse_w4a4.pt \
  evaluate=true

# transform.path=/home/blessedg/asrq/outputs/rotation/parakeet_globalqe_w4a4.pt \
# transform.path=/home/blessedg/asrq/outputs/rotation/openai-whisper-large-v3_rotation_w4a4_global_qe_qd_search.pt \


# Global-Qe-only search variant
# python - <<'PY'
# import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# from hydra import compose, initialize_config_dir
# from omegaconf import open_dict
# from asrq.core.model import ModelQ
# from asrq.quantizers.base import QuantConfig
# from asrq.calibration.base import CalibConfig
# from asrq.transforms.base import BaseTransform, TransformConfig
# import torch, random, numpy as np, transformers

# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     transformers.set_seed(seed)

# CONFIG_DIR = os.path.join(os.getcwd(), "asrq", "configs")
# with initialize_config_dir(version_base=None, config_dir=os.path.abspath(CONFIG_DIR)):
#     cfg = compose(config_name="config", overrides=[
#         "model=whisper",
#         "quantizer=gptq",
#         "quantizer.bits=4",
#         "activation_bits=8",
#         "evaluate=false",
#         "transform=rotation",
#         "transform.type=search",
#         "transform.search_mode=global_qe",
#         "transform.path=/home/blessedg/asrq/outputs/rotation/openai-whisper-large-v3_rotation_w4a8_global_qe.pt",
#         "calibration.num_samples=512",
#         "transform.num_samples=512",
#         "transform.batch_size=64",
#         "transform.qe_generations=8",
#         "transform.qe_patience=4",
    
#     ])

# set_seed(cfg.seed)
# with open_dict(cfg.quantizer):
#     cfg.quantizer.exclude_modules = cfg.model.exclude_modules
# with open_dict(cfg.transform):
#     cfg.transform.model_name = cfg.model.name
#     cfg.transform.learn_rotation = True
#     cfg.transform.wbits = cfg.quantizer.bits
#     cfg.transform.abits = cfg.activation_bits

# quant_cfg = QuantConfig.from_dictconfig(cfg.quantizer)
# transform_cfg = TransformConfig.from_dictconfig(cfg.transform)
# calib_cfg = CalibConfig.from_dictconfig(cfg.calibration)

# modelQ = ModelQ.from_pretrained(cfg.model.name, quant_cfg, calib_cfg)
# transform = BaseTransform.from_config(transform_cfg)
# modelQ.model.to("cuda")
# transform.obtain_transform(modelQ)

# print("Saved global_qe rotation to:", transform.cfg.path)
# PY
