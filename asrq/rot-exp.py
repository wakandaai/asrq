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


def set_seed(seed=42, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only lets an operation without a deterministic kernel run anyway. A Cayley search on Parakeet needs
    # it: the CE objective is the CTC loss, whose backward has no deterministic CUDA implementation.
    torch.use_deterministic_algorithms(True, warn_only=not deterministic)
    transformers.set_seed(seed)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    deterministic = cfg.get("deterministic", True)
    set_seed(cfg.seed, deterministic)

    learn_rotation_experiment(cfg, reseed=lambda: set_seed(cfg.seed, deterministic))


if __name__ == "__main__":
    main()
