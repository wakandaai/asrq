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
    obtain_rotations_for_canary_qwen_search,
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
        self.search_mode = getattr(cfg, "search_mode", "local_q2")
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
        self.outer_rounds = getattr(cfg, "outer_rounds", 1)
        self.outer_patience = getattr(cfg, "outer_patience", 1)
        self.qe_min_delta = getattr(cfg, "qe_min_delta", 0.0)
        self.q2_refine_generations = getattr(cfg, "q2_refine_generations", self.generations)
        self.qe_generations = getattr(cfg, "qe_generations", self.generations)
        self.qe_patience = getattr(cfg, "qe_patience", self.patience)
        self.global_score_metric = getattr(cfg, "global_score_metric", "task_loss")
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
            global_score_metric=self.global_score_metric,
            seed=seed,
        )

    def q2_refine_search_params(self, seed: int) -> RotationSearchParams:
        return RotationSearchParams(
            population_size=self.population_size,
            elite_count=self.elite_count,
            parent_pool_fraction=self.parent_pool_fraction,
            generations=self.q2_refine_generations,
            patience=self.patience,
            mutate_both_probability=self.mutate_both_probability,
            large_mutation_probability=self.large_mutation_probability,
            small_mutation_min=self.small_mutation_min,
            small_mutation_max=self.small_mutation_max,
            medium_mutation_min=self.medium_mutation_min,
            medium_mutation_max=self.medium_mutation_max,
            large_mutation_fraction=self.large_mutation_fraction,
            global_score_metric=self.global_score_metric,
            seed=seed,
        )

    def qe_search_params(self, seed: int) -> RotationSearchParams:
        return RotationSearchParams(
            population_size=self.population_size,
            elite_count=self.elite_count,
            parent_pool_fraction=self.parent_pool_fraction,
            generations=self.qe_generations,
            patience=self.qe_patience,
            mutate_both_probability=self.mutate_both_probability,
            large_mutation_probability=self.large_mutation_probability,
            small_mutation_min=self.small_mutation_min,
            small_mutation_max=self.small_mutation_max,
            medium_mutation_min=self.medium_mutation_min,
            medium_mutation_max=self.medium_mutation_max,
            large_mutation_fraction=self.large_mutation_fraction,
            global_score_metric=self.global_score_metric,
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
                seed = int(torch.initial_seed() & 0xFFFFFFFF)
                if self.cfg.search_mode in {"alternating", "global_qe", "global_qe_qd"}:
                    obtain_rotations_for_whisper_search(
                        model,
                        processor,
                        self.audio,
                        self.sr,
                        self.cfg.num_samples,
                        self.cfg.batch_size,
                        self.cfg.search_params(seed),
                        self.cfg.path,
                        weight_bits=self.cfg.wbits,
                        activation_bits=self.cfg.abits,
                        search_mode=self.cfg.search_mode,
                        qe_search_params=self.cfg.qe_search_params(seed + 1),
                        q2_refine_search_params=(
                            self.cfg.q2_refine_search_params(seed + 2)
                            if self.cfg.search_mode == "alternating"
                            else None
                        ),
                        outer_rounds=self.cfg.outer_rounds,
                        outer_patience=self.cfg.outer_patience,
                        qe_min_delta=self.cfg.qe_min_delta,
                    )
                else:
                    obtain_rotations_for_whisper_search(
                        model,
                        processor,
                        self.audio,
                        self.sr,
                        self.cfg.num_samples,
                        self.cfg.batch_size,
                        self.cfg.search_params(seed),
                        self.cfg.path,
                        weight_bits=self.cfg.wbits,
                        activation_bits=self.cfg.abits,
                        search_mode=self.cfg.search_mode,
                    )
            else:
                obtain_rotations_for_whisper(
                    model, processor, self.audio, self.sr, self.cfg.num_samples, 
                    self.cfg.epochs, self.cfg.learning_rate, self.cfg.batch_size,
                    self.cfg.path
                )
        elif self.cfg.model_name == ModelNames.NVIDIA_PARAKEET_CTC_1_1B:
            if self.cfg.type == "search":
                seed = int(torch.initial_seed() & 0xFFFFFFFF)
                if self.cfg.search_mode in {"alternating", "global_qe", "global_qe_qd"}:
                    obtain_rotations_for_parakeet_search(
                        model,
                        "outputs/rotation_test_audio.wav",
                        self.cfg.num_samples,
                        self.cfg.batch_size,
                        self.cfg.search_params(seed),
                        self.cfg.path,
                        device="cuda",
                        weight_bits=self.cfg.wbits,
                        activation_bits=self.cfg.abits,
                        search_mode=self.cfg.search_mode,
                        qe_search_params=self.cfg.qe_search_params(seed + 1),
                        q2_refine_search_params=(
                            self.cfg.q2_refine_search_params(seed + 2)
                            if self.cfg.search_mode == "alternating"
                            else None
                        ),
                        outer_rounds=self.cfg.outer_rounds,
                        outer_patience=self.cfg.outer_patience,
                        qe_min_delta=self.cfg.qe_min_delta,
                    )
                else:
                    obtain_rotations_for_parakeet_search(
                        model,
                        "outputs/rotation_test_audio.wav",
                        self.cfg.num_samples,
                        self.cfg.batch_size,
                        self.cfg.search_params(seed),
                        self.cfg.path,
                        device="cuda",
                        weight_bits=self.cfg.wbits,
                        activation_bits=self.cfg.abits,
                        search_mode=self.cfg.search_mode,
                    )
            else:
                obtain_rotations_for_parakeet(
                    model, "outputs/rotation_test_audio.wav", self.cfg.num_samples,
                    self.cfg.epochs, self.cfg.batch_size, self.cfg.learning_rate,
                    self.cfg.path, device="cuda"
                )
        elif self.cfg.model_name == ModelNames.NVIDIA_CANARY_QWEN_2_5B:
            if self.cfg.type == "search":
                seed = int(torch.initial_seed() & 0xFFFFFFFF)
                if self.cfg.search_mode in {"alternating", "global_qe", "global_qe_qd"}:
                    obtain_rotations_for_canary_qwen_search(
                        model,
                        "outputs/rotation_test_audio.wav",
                        self.cfg.num_samples,
                        self.cfg.batch_size,
                        self.cfg.search_params(seed),
                        self.cfg.path,
                        device="cuda",
                        weight_bits=self.cfg.wbits,
                        activation_bits=self.cfg.abits,
                        search_mode=self.cfg.search_mode,
                        qe_search_params=self.cfg.qe_search_params(seed + 1),
                        q2_refine_search_params=(
                            self.cfg.q2_refine_search_params(seed + 2)
                            if self.cfg.search_mode == "alternating"
                            else None
                        ),
                        outer_rounds=self.cfg.outer_rounds,
                        outer_patience=self.cfg.outer_patience,
                        qe_min_delta=self.cfg.qe_min_delta,
                    )
                else:
                    obtain_rotations_for_canary_qwen_search(
                        model,
                        "outputs/rotation_test_audio.wav",
                        self.cfg.num_samples,
                        self.cfg.batch_size,
                        self.cfg.search_params(seed),
                        self.cfg.path,
                        device="cuda",
                        weight_bits=self.cfg.wbits,
                        activation_bits=self.cfg.abits,
                        search_mode=self.cfg.search_mode,
                    )
                return
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
        

        
