# pyright: reportMissingImports=false

from omegaconf import DictConfig
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from asrq.core.types import Processor
from asrq.transforms.base import BaseTransform, TransformConfig
from asrq.core.registry import TransformNames, register_transform, register_transform_config
from asrq.transforms.rotation.hadamard_utils import random_hadamard_matrix
from asrq.core.registry import ModelNames
from asrq.transforms.rotation import (
    obtain_rotations_for_whisper,
    rotate_whisper_model, 
    obtain_rotations_for_canary_qwen, 
    rotate_canary_qwen, 
    obtain_rotations_for_parakeet, 
    rotate_parakeet
)
from datasets import load_dataset
import soundfile as sf



@register_transform_config(TransformNames.rotation)
class RotationTransformConfig(TransformConfig):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.num_samples = cfg.num_samples
        self.epochs = cfg.epochs
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.learn_rotation = cfg.learn_rotation


@register_transform(TransformNames.rotation)
class RotationTransform(BaseTransform):
    cfg: RotationTransformConfig
    def __init__(self, cfg: RotationTransformConfig) -> None:
        super().__init__(cfg)

    def obtain_transform(self, modelQ) -> None:
        model = modelQ.model
        processor = modelQ.processor
        if self.cfg.learn_rotation is False:
            return

        if not self.cfg.learn_rotation:
            return
        if self.cfg.model_name == ModelNames.OPENAI_WHISPER_LARGE_V3:
            assert isinstance(model, WhisperForConditionalGeneration)
            obtain_rotations_for_whisper(
                model, processor, self.audio, self.sr, self.cfg.num_samples, 
                self.cfg.epochs, self.cfg.learning_rate, self.cfg.batch_size,
                self.cfg.path
            )
        elif self.cfg.model_name == ModelNames.NVIDIA_PARAKEET_CTC_1_1B:
            obtain_rotations_for_parakeet(
                model, "outputs/rotation_test_audio.wav", self.cfg.num_samples,
                self.cfg.epochs, self.cfg.batch_size, self.cfg.learning_rate,
                self.cfg.path, device="cuda"
            )
        elif self.cfg.model_name == ModelNames.NVIDIA_CANARY_QWEN_2_5B:
            obtain_rotations_for_canary_qwen(
                model, "outputs/rotation_test_audio.wav", self.cfg.num_samples,
                self.cfg.epochs, self.cfg.batch_size, self.cfg.learning_rate,
                self.cfg.path,
            )
        else:
            raise ValueError(f"Rotation learning not implemented for model {self.cfg.model_name}")


    def apply_transform(self, modelQ) -> None:
        model = modelQ.model
        processor = modelQ.processor
        path = self.cfg.path
        if self.cfg.model_name == ModelNames.OPENAI_WHISPER_LARGE_V3:
            rotate_whisper_model(model, processor, self.audio, self.sr, path, device="cuda") # type: ignore
        elif self.cfg.model_name == ModelNames.NVIDIA_PARAKEET_CTC_1_1B:
            rotate_parakeet(model, "outputs/rotation_test_audio.wav", path, device="cuda")
        elif self.cfg.model_name == ModelNames.NVIDIA_CANARY_QWEN_2_5B:
            rotate_canary_qwen(model, "outputs/rotation_test_audio.wav", path, device="cuda")
        else:
            raise ValueError(f"Rotation application not implemented for model {self.cfg.model_name}")
        

        