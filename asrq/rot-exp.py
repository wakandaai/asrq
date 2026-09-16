# pyright: reportMissingImports=false

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random

import hydra
import numpy as np
import torch
import transformers
from omegaconf import DictConfig

from asrq.experiment import learn_rotation_experiment


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
    set_seed(cfg.seed)

    learn_rotation_experiment(cfg, reseed=lambda: set_seed(cfg.seed))


if __name__ == "__main__":
    main()
