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
    search_rotations_for_whisper,
    search_rotations_for_canary_qwen,
    search_rotations_for_parakeet,
)
from asrq.transforms.rotation.hadamard_search import HadamardSearchConfig
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
        self.hadamard_search = cfg.hadamard_search
        # Evolutionary sign search hyperparameters; defaults match HadamardSearchConfig.
        _defaults = HadamardSearchConfig()
        self.search = HadamardSearchConfig(
            population=cfg.get("search_population", _defaults.population),
            generations=cfg.get("search_generations", _defaults.generations),
            survivors=cfg.get("search_survivors", _defaults.survivors),
            mutation_prob=cfg.get("search_mutation_prob", _defaults.mutation_prob),
            batches_per_eval=cfg.get("search_batches_per_eval", _defaults.batches_per_eval),
            crossover=cfg.get("search_crossover", _defaults.crossover),
            seed=cfg.get("search_seed", _defaults.seed),
        )
        # Set from cfg.activation_bits in exp.py. The rotation is learned with the STE
        # activation quantizer at this width, so it targets the deployment setting.
        self.activation_bits = cfg.abits
        self.weight_bits = cfg.wbits
        self.weight_group_size = cfg.get("wgroup", None)


@register_transform(TransformNames.rotation)
class RotationTransform(BaseTransform):
    cfg: RotationTransformConfig
    def __init__(self, cfg: RotationTransformConfig) -> None:
        super().__init__(cfg)

    # Rotation training only makes sense against quantized activations: the STE quantizer
    # and the online Hadamard on the down-projections are what the rotation is learned to
    # compensate for, and both are no-ops at 16 bits.
    SUPPORTED_ACTIVATION_BITS = (4, 8)

    def obtain_transform(self, modelQ) -> None:
        model = modelQ.model
        processor = modelQ.processor
        if not self.cfg.learn_rotation and not self.cfg.hadamard_search:
            return
        # learn_rotation wins over hadamard_search below, so decide which path is actually
        # going to run before exempting anything. Weight-only (activation_bits=16) is
        # searchable - the STE weight quantizer gives the search something to minimize
        # even with full-precision activations - but gradient-based training is not, since
        # at 16 bits there is nothing in its forward for the rotation to compensate for.
        weight_only = (
            self.cfg.hadamard_search
            and not self.cfg.learn_rotation
            and self.cfg.activation_bits >= 16
        )
        if not weight_only and self.cfg.activation_bits not in self.SUPPORTED_ACTIVATION_BITS:
            raise ValueError(
                f"Rotation training expects activation_bits in {self.SUPPORTED_ACTIVATION_BITS}, "
                f"got {self.cfg.activation_bits}. Set activation_bits=4 or 8, or turn off "
                f"learn_rotation/hadamard_search to reuse a saved rotation."
            )
        if self.cfg.learn_rotation:
            if self.cfg.model_name == ModelNames.OPENAI_WHISPER_LARGE_V3:
                assert isinstance(model, WhisperForConditionalGeneration)
                obtain_rotations_for_whisper(
                    model, processor, self.audio, self.sr, self.cfg.num_samples,
                    self.cfg.epochs, self.cfg.learning_rate, self.cfg.batch_size,
                    self.cfg.path, activation_bits=self.cfg.activation_bits
                )
            elif self.cfg.model_name == ModelNames.NVIDIA_PARAKEET_CTC_1_1B:
                obtain_rotations_for_parakeet(
                    model, "outputs/rotation_test_audio.wav", self.cfg.num_samples,
                    self.cfg.epochs, self.cfg.batch_size, self.cfg.learning_rate,
                    self.cfg.path, device="cuda", activation_bits=self.cfg.activation_bits
                )
            elif self.cfg.model_name == ModelNames.NVIDIA_CANARY_QWEN_2_5B:
                obtain_rotations_for_canary_qwen(
                    model, "outputs/rotation_test_audio.wav", self.cfg.num_samples,
                    self.cfg.epochs, self.cfg.batch_size, self.cfg.learning_rate,
                    self.cfg.path, activation_bits=self.cfg.activation_bits
                )
            else:
                raise ValueError(f"Rotation learning not implemented for model {self.cfg.model_name}")
        elif self.cfg.hadamard_search:
            # Derivative-free alternative to learn_rotation: keep the Hadamard structure
            # and search only the sign vectors, scoring each candidate by the quantized
            # model's task loss. Writes the same file apply_transform reads.
            #
            # Weight-only: quantize the weights in the forward instead of the activations,
            # and make the residual-stream rotation block diagonal at the weight group
            # size, so it mixes channels only inside a quantization group. Q2 stays a full
            # head_dim rotation either way.
            search_kwargs = dict(
                activation_bits=self.cfg.activation_bits,
                search_cfg=self.cfg.search,
                quantize_weights=weight_only,
                weight_bits=self.cfg.weight_bits,
                weight_group_size=self.cfg.weight_group_size if weight_only else None,
                rotation_block_size=self.cfg.weight_group_size if weight_only else None,
            )
            if self.cfg.model_name == ModelNames.OPENAI_WHISPER_LARGE_V3:
                assert isinstance(model, WhisperForConditionalGeneration)
                search_rotations_for_whisper(
                    model, processor, self.cfg.num_samples, self.cfg.batch_size,
                    self.cfg.path, **search_kwargs,
                )
            elif self.cfg.model_name == ModelNames.NVIDIA_PARAKEET_CTC_1_1B:
                search_rotations_for_parakeet(
                    model, self.cfg.num_samples, self.cfg.batch_size, self.cfg.path, **search_kwargs,
                )
            elif self.cfg.model_name == ModelNames.NVIDIA_CANARY_QWEN_2_5B:
                search_rotations_for_canary_qwen(
                    model, self.cfg.num_samples, self.cfg.batch_size, self.cfg.path, **search_kwargs,
                )
            else:
                raise ValueError(f"Hadamard search not implemented for model {self.cfg.model_name}")


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
        

        