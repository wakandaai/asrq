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
    rotate_parakeet,
    obtain_rotations_for_whisper_search,
    obtain_rotations_for_parakeet_search,
)
from asrq.transforms.rotation.search import RotationSearchParams
from datasets import load_dataset
import soundfile as sf



@register_transform_config(TransformNames.rotation)
class RotationTransformConfig(TransformConfig):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.type = cfg.type
        self.num_samples = cfg.num_samples
        self.epochs = cfg.epochs
        self.learning_rate = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.learn_rotation = cfg.learn_rotation
        self.population_size = cfg.population_size
        self.elite_count = cfg.elite_count
        self.parent_pool_fraction = cfg.parent_pool_fraction
        self.generations = cfg.generations
        self.patience = cfg.patience
        self.mutate_both_probability = cfg.mutate_both_probability
        self.large_mutation_probability = cfg.large_mutation_probability
        self.small_mutation_min = cfg.small_mutation_min
        self.small_mutation_max = cfg.small_mutation_max
        self.medium_mutation_min = cfg.medium_mutation_min
        self.medium_mutation_max = cfg.medium_mutation_max
        self.large_mutation_fraction = cfg.large_mutation_fraction
        self.wbits = getattr(cfg, "wbits", 4)
        self.abits = getattr(cfg, "abits", 16)

    def search_params(self, seed: int) -> RotationSearchParams:
        return RotationSearchParams(
            population_size=self.population_size,
            elite_count=self.elite_count,
            parent_pool_fraction=self.parent_pool_fraction,
            generations=self.generations,
            patience=self.patience,
            mutate_both_probability=self.mutate_both_probability,
            large_mutation_probability=self.large_mutation_probability,
            small_mutation_min=self.small_mutation_min,
            small_mutation_max=self.small_mutation_max,
            medium_mutation_min=self.medium_mutation_min,
            medium_mutation_max=self.medium_mutation_max,
            large_mutation_fraction=self.large_mutation_fraction,
            seed=seed,
        )


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
            if self.cfg.type == "search":
                obtain_rotations_for_whisper_search(
                    model,
                    processor,
                    self.audio,
                    self.sr,
                    self.cfg.num_samples,
                    self.cfg.batch_size,
                    self.cfg.search_params(int(torch.initial_seed() & 0xFFFFFFFF)),
                    self.cfg.path,
                    weight_bits=self.cfg.wbits,
                    activation_bits=self.cfg.abits,
                )
            else:
                obtain_rotations_for_whisper(
                    model, processor, self.audio, self.sr, self.cfg.num_samples, 
                    self.cfg.epochs, self.cfg.learning_rate, self.cfg.batch_size,
                    self.cfg.path
                )
        elif self.cfg.model_name == ModelNames.NVIDIA_PARAKEET_CTC_1_1B:
            if self.cfg.type == "search":
                obtain_rotations_for_parakeet_search(
                    model,
                    "outputs/rotation_test_audio.wav",
                    self.cfg.num_samples,
                    self.cfg.batch_size,
                    self.cfg.search_params(int(torch.initial_seed() & 0xFFFFFFFF)),
                    self.cfg.path,
                    device="cuda",
                    weight_bits=self.cfg.wbits,
                    activation_bits=self.cfg.abits,
                )
            else:
                obtain_rotations_for_parakeet(
                    model, "outputs/rotation_test_audio.wav", self.cfg.num_samples,
                    self.cfg.epochs, self.cfg.batch_size, self.cfg.learning_rate,
                    self.cfg.path, device="cuda"
                )
        elif self.cfg.model_name == ModelNames.NVIDIA_CANARY_QWEN_2_5B:
            if self.cfg.type == "search":
                raise ValueError("Rotation search is not implemented for Canary-Qwen yet.")
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
        

        
