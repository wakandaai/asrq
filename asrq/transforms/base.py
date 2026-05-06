# pyright: reportMissingImports=false

from datasets import load_dataset
import torch.nn as nn
import os
from abc import ABC
from omegaconf import DictConfig
import numpy as np
import soundfile as sf
from asrq.core.registry import get_transform_config_cls, get_transform_cls
from asrq.core.types import Processor



class TransformConfig:
    def __init__(self, cfg: DictConfig) -> None:
        self.name = cfg.name
        self.model_name = cfg.model_name
        self.path = cfg.path
       
    @staticmethod
    def from_dictconfig(cfg: DictConfig) -> "TransformConfig":
        """Create a TransformConfig instance from a DictConfig."""
        config_cls = get_transform_config_cls(cfg.name)
        return config_cls(cfg)


class BaseTransform(ABC):
    def __init__(self, transform_cfg: TransformConfig) -> None:
        self.cfg = transform_cfg
        # dummy audio for testing computational invariance after rotation
        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean", split="validation", trust_remote_code=True, 
        )
        sample = ds[0] # type: ignore
        self.audio = np.array(sample["audio"]["array"], dtype=np.float32) # type: ignore
        self.sr = sample["audio"]["sampling_rate"] # type: ignore
        sf.write("outputs/rotation_test_audio.wav", self.audio, self.sr)

        if not os.path.exists(f"outputs/{self.cfg.name}"):
            os.makedirs(f"outputs/{self.cfg.name}")
        if not self.cfg.path: # type: ignore
            if hasattr(self.cfg, "wbits") and hasattr(self.cfg, "abits"):
                self.cfg.path = f"outputs/{self.cfg.name}/{self.cfg.model_name.replace('/','-')}_{self.cfg.name}_w{self.cfg.wbits}a{self.cfg.abits}_{self.cfg.type}.pt" # type: ignore
            else:
                self.cfg.path = f"outputs/{self.cfg.name}/{self.cfg.model_name.replace('/','-')}_{self.cfg.name}{f'_{self.cfg.type}' if hasattr(self.cfg, 'type') else ''}.pt" # type: ignore

    @staticmethod
    def from_config(transform_cfg: TransformConfig) -> "BaseTransform":
        """Create a BaseTransform instance from a TransformConfig."""
        transform_cls = get_transform_cls(transform_cfg.name)
        return transform_cls(transform_cfg)

    def obtain_transform(self, modelQ) -> None:
        """Obtain the transformation parameters (e.g., scales for scaling transform) using the given model and processor."""
        raise NotImplementedError("obtain_transform method not implemented")

    def apply_transform(self, modelQ) -> None:
        """Apply the transformation to the given model and return the transformed model."""
        raise NotImplementedError("apply_transform method not implemented")