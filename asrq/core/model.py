# pyright: reportMissingImports=false

import torch
import torch.nn as nn
import numpy as np
import json

from typing import Dict, List, Tuple, Any
from asrq.core.registry import ModelQ_Registry, QuantizerNames
from asrq.quantizers.base import QuantConfig
from asrq.transforms.base import TransformConfig
from asrq.calibration.base import CalibConfig
from asrq.core.linear import ASRQLinear
from abc import ABC, abstractmethod
from datasets import load_dataset
from itertools import islice
from tqdm import tqdm
from asrq.evaluation.english_text_normalizer import normalizer
from asrq.core.types import Processor
from asrq.core.registry import get_quant_cls




# ========== Quantizer Base Classes ==========
class ModelQ(ABC):
    model_id = ""
    modules_to_quantize = (nn.Linear,)
    is_linear_only = True
    calibration_samples: List[Tuple[np.ndarray, str]]
    qparams: Dict[str, Any]

    @classmethod
    def load_model(cls) -> Tuple[nn.Module, Processor]:
        """Load the model given the huggingface model ID, and return the model and 
        processor.
        
        Args:
            model_id (str): The huggingface model ID.

        Returns:
            Tuple[nn.Module, Processor]: The loaded model and processor.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    
    @classmethod
    def from_pretrained(
        cls, model_id:str, 
        quant_cfg: QuantConfig,
        calib_cfg: CalibConfig,
    ) -> 'ModelQ':
        """Load a huggingface pretrained model and create a ModelQ instance.
        
        A ModelQ instance expects a model, processor, quantization configuration, transform configuration, 
        and a calibration configuration. These parameters enable a ModelQ instance to perform 
        quantization using the specified method and calibration samples.

        This method calls the corresponding 'from_pretrained' method of the ModelQ 
        subclass registered in the ModelQ_Registry for the specified model_id.

        Args:
            model_id (str): The huggingface model ID.
            quant_cfg (QuantConfig): Quantization configuration.
            calib_cfg (CalibConfig): Calibration configuration.

        Returns:
            ModelQ: An instance of the ModelQ subclass corresponding to the model_id.
        """
        if model_id not in ModelQ_Registry:
            raise ValueError(f"Model {model_id} not found in registry.")

        modelQ =  ModelQ_Registry[model_id].from_pretrained(
            quant_cfg=quant_cfg,
            calib_cfg=calib_cfg
        )
        modelQ.model_id = model_id
        return modelQ

    @classmethod
    def get_supported_models(cls)->List[str]:
        """Get a list of supported model IDs in the ModelQ registry."""
        return list(ModelQ_Registry.keys())

    def __init__(
        self, 
        model: nn.Module, 
        processor: Processor,
        quant_cfg: QuantConfig,
        calib_cfg: CalibConfig,
    ) -> None:
        self.model = model
        self.processor = processor
        self.quant_cfg = quant_cfg
        self.calib_cfg = calib_cfg
        self.qparams = {}
        self._init()

    def _init(self)->None:
        """Initialize any additional attributes needed for quantization."""
        # Calibration Samples
        self.calibration_samples = []
        num_samples = self.calib_cfg.num_samples
        if num_samples is not None:
            ds = load_dataset("openslr/librispeech_asr", "all", split="train.clean.360")
            ds = ds.shuffle(seed=42)
            subset = list(islice(ds, num_samples))
            for sample in tqdm(subset, desc="Loading calibration samples"):
                audio = sample["audio"]["array"] # type: ignore
                text = normalizer(sample["text"]) # type: ignore
                self.calibration_samples.append((audio, text))

        # Quantization method
        self.quant_cls = get_quant_cls(self.quant_cfg.name)

    def set_calibration_samples(self, samples):
        self.calibration_samples = samples

    def set_quantization_method(self, quant_config: QuantConfig):
        self.quant_cfg = quant_config
        self.quant_cls = get_quant_cls(self.quant_cfg.name)

    def quantize(self):
        """Quantize the model to the desired bitwidth using the specified method."""
        # Implement the quantization logic here. Weights should be saved to disk.
        with torch.inference_mode():
            # First quantize the speech encoder
            print("Quantizing speech encoder...")
            self.quantize_speech_encoder()

            # Then quantize the text decoder
            print("Quantizing text decoder...")
            self.quantize_text_decoder()
        print("Quantization complete.")

    def should_quantize_module(self, name, module):
        """Check if a module should be quantized based on its name and type."""
        if name in self.quant_cfg.exclude_modules:
            return False
        return isinstance(module, nn.Linear)

    def quantize_speech_encoder(self):
        """Quantize the speech encoder."""
        raise NotImplementedError("Speech encoder quantization not implemented.")

    def quantize_text_decoder(self):
        """Quantize the text decoder."""
        raise NotImplementedError("Text decoder quantization not implemented.")
    
    def get_asrq_linear_targets(self) -> Dict[str, Tuple[int, int]]:
        """Map fully-qualified nn.Linear module names (dotted path from
        self.model) to the (wbits, abits) they should be quantized to by
        to_asrq_linear(). Subclasses (e.g. WhisperQ) override this with
        their own model-specific layer names -- see
        WhisperQ.for_activation_quantization() for the equivalent
        name-building pattern (encoder/decoder blocks, variable layer
        count from config).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_asrq_linear_targets() to use to_asrq_linear()"
        )

    def to_asrq_linear(self, group_size: int = -1, symmetric: bool = True) -> None:
        """Replace every nn.Linear named by get_asrq_linear_targets() with
        an ASRQLinear quantized to that layer's (wbits, abits), in place.

        Unlike quantize_speech_encoder()/quantize_text_decoder() (which run
        a calibration-driven Quantizer -- e.g. RTNQuantizer -- and need
        module.quantize_() called separately for ASRQLinear targets to
        become usable, see WhisperQ's block-quantization methods), this is
        a direct, calibration-free RTN quantization path: each target
        layer's existing float weight (and bias, if any) is copied via
        ASRQLinear.from_linear(), which quantizes and packs it immediately
        -- the returned module is ready for forward() with no further
        steps. group_size/symmetric apply uniformly to every replaced
        layer; per-layer wbits/abits come from get_asrq_linear_targets().
        """
        targets = self.get_asrq_linear_targets()
        named_modules = dict(self.model.named_modules())
        for name, (wbits, abits) in targets.items():
            if name not in named_modules:
                raise ValueError(f"Module '{name}' not found in model")
            module = named_modules[name]
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Module '{name}' is not an nn.Linear (got {type(module).__name__})")

            new_module = ASRQLinear.from_linear(
                module, wbits=wbits, abits=abits, group_size=group_size, symmetric=symmetric,
            )
            parent_name, _, attr = name.rpartition(".")
            parent = named_modules[parent_name] if parent_name else self.model
            setattr(parent, attr, new_module)

    def learn_rotation(self, rotation_path:str)->None:
        """Learn a rotation matrix for the model and save it to disk."""
        raise NotImplementedError("Rotation learning not implemented.")
    
    def rotate_model(self, rotation_path:str)->None:
        """Rotate the model using a learned rotation matrix."""
        raise NotImplementedError("Model rotation not implemented.")

