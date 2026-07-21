srun --jobid=106635 --overlap bash -lc '
source ~/.bashrc
conda activate asrq
cd /home/blessedg/asrq
python - <<'"'"'PY'"'"'
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
        "model=parakeet",
        "quantizer=gptq",
        "transform=rotation",
        "transform.type=search",
        "transform.num_samples=128",
        "transform.batch_size=1",
        "transform.population_size=8",
        "transform.elite_count=2",
        "transform.generations=16",
        "transform.patience=4",
        "transform.parent_pool_fraction=0.5",
        "activation_bits=8",
        "evaluate=False",
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
'