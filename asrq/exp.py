# pyright: reportMissingImports=false

import random

import hydra
import numpy as np
import torch
import transformers
from omegaconf import DictConfig, OmegaConf

from asrq.experiment import run_experiment


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

    run_experiment(cfg)


if __name__ == "__main__":
    main() # type: ignore
