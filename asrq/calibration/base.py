# pyright: reportMissingImports=false
from asrq.transforms.base import TransformConfig
from omegaconf import DictConfig



class CalibConfig:
    def __init__(self, cfg: DictConfig) -> None:
        self.num_samples = cfg.num_samples

    @staticmethod
    def from_dictconfig(cfg: DictConfig) -> "CalibConfig":
        return CalibConfig(cfg)
    