# pyright: reportMissingImports=false

import gc
import os
import torch
import torch.nn as nn
import numpy as np
import json

from typing import Dict, List, Optional, Tuple, Any
from asrq.core.registry import ModelQ_Registry, QuantizerNames
from asrq.quantizers.base import QuantConfig
from asrq.transforms.base import TransformConfig
from asrq.calibration.base import CalibConfig
from asrq.calibration.data import load_calibration_samples
from asrq.core.linear import ASRQLinear, replace_with_asrq_linear
from abc import ABC, abstractmethod
from datasets import load_dataset
from itertools import islice
from tqdm import tqdm
from asrq.evaluation.english_text_normalizer import normalizer
from asrq.core.types import Processor
from asrq.core.registry import get_quant_cls
from asrq.core.utils import cuda_empty_cache
from asrq.quantizers.norm_tweak import apply_norm_tweaks, capture_norm_tweaks
from asrq.quantizers.output_refit import OutputRefit, insert_block_output_linear




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
        if self.calib_cfg.path:
            self.calibration_samples = load_calibration_samples(
                self.calib_cfg.path,
                model_name=self._registered_name(),
                num_samples=num_samples,
                text_source=self.calib_cfg.text_source,
            )
        elif num_samples is not None:
            ds = load_dataset("openslr/librispeech_asr", "all", split="train.clean.360")
            ds = ds.shuffle(seed=42)
            subset = list(islice(ds, num_samples))
            for sample in tqdm(subset, desc="Loading calibration samples"):
                audio = sample["audio"]["array"] # type: ignore
                text = normalizer(sample["text"]) # type: ignore
                self.calibration_samples.append((audio, text))

        # Quantization method
        self.quant_cls = get_quant_cls(self.quant_cfg.name)

    def activation_quantization_roles(self) -> Dict[str, Optional[str]]:
        """``{layer_name: role}`` for every layer whose input is activation-quantized.

        Roles decide which layers are quantized group-wise; see asrq.quantizers.activation. Evaluation,
        the rotation search, the scaling transform's layer lists and to_asrq_linear all use this one
        mapping, so each model defines it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement activation_quantization_roles()")

    def _registered_name(self) -> str:
        """The model id this class is registered under.

        model_id is only assigned once the constructor returns, which is after calibration
        samples are loaded, so the name is looked up from the registry instead.
        """
        for name, cls in ModelQ_Registry.items():
            if cls is type(self):
                return name
        raise ValueError(f"{type(self).__name__} is not registered")

    def set_calibration_samples(self, samples):
        self.calibration_samples = samples

    def set_quantization_method(self, quant_config: QuantConfig):
        self.quant_cfg = quant_config
        self.quant_cls = get_quant_cls(self.quant_cfg.name)

    QUANTIZED_FORMAT = 2

    def save_quantized(self, path: str, fingerprint: str) -> None:
        """Save the quantized model so load_quantized can restore it without quantizing again.

        The state dict is stored at each tensor's own dtype, together with what a freshly loaded and transformed model
        lacks: the output Linears block_output_refit_insert added (``{block name: width}``) and the norm weights norm tweaking gave scale-free norms (module names whose
        ``weight`` is a buffer). The rotation is not stored; load_quantized expects it applied, as exp.py does
        before quantizing. Written to a temporary file and renamed, so an interrupted save leaves no file behind.

        Tensors are not narrowed to float16: evaluation casts the model to ``eval_dtype`` (bfloat16 by default), and
        rounding to float16 first changes some bfloat16 values, which changed the WER of a loaded model.
        """
        output_linears = {
            name: module.output_linear.in_features
            for name, module in self.model.named_modules()
            if isinstance(getattr(module, "output_linear", None), nn.Linear)
        }
        weight_buffers = [name for name, module in self.model.named_modules() if "weight" in module._buffers]
        state = {
            key: (value.detach().cpu())
            for key, value in self.model.state_dict().items()
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f"{path}.tmp"
        torch.save({
            "format": self.QUANTIZED_FORMAT, "fingerprint": fingerprint, "state_dict": state,
            "output_linears": output_linears, "weight_buffers": weight_buffers,
        }, temporary)
        os.replace(temporary, path)

    def load_quantized(self, path: str, fingerprint: Optional[str] = None) -> None:
        """Restore a model saved by save_quantized into this (transformed, unquantized) model, in place.

        The inserted output Linears and norm weight buffers are recreated first, then every tensor is loaded; a
        missing or unexpected key is an error, as is a fingerprint other than the given one.
        """
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("format") != self.QUANTIZED_FORMAT:
            raise ValueError(f"{path} has quantized-model format {saved.get('format')}, expected {self.QUANTIZED_FORMAT}")
        if fingerprint is not None and saved["fingerprint"] != fingerprint:
            raise ValueError(
                f"{path} was saved with different settings (fingerprint {saved['fingerprint']}, these give "
                f"{fingerprint}); delete it or point quantized_path elsewhere"
            )
        state = saved["state_dict"]
        for block_name, width in saved["output_linears"].items():
            insert_block_output_linear(self.model.get_submodule(block_name), width)
        for name in saved["weight_buffers"]:
            module = self.model.get_submodule(name)
            if "weight" not in module._buffers:
                reference = next(iter(module.buffers()), None)
                device = reference.device if reference is not None else next(self.model.parameters()).device
                module.register_buffer("weight", state[f"{name}.weight"].to(device))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"{path} does not match the model: missing {list(missing)[:5]}, unexpected {list(unexpected)[:5]}"
            )

    def quantize(self):
        """Quantize the model to the desired bitwidth using the specified method."""
        if getattr(self.quant_cfg, "block_output_refit", False) and getattr(self.quant_cfg, "block_output_refit_insert", False):
            for name, width in self.transformer_block_widths().items():
                insert_block_output_linear(self.model.get_submodule(name), width)
        # Implement the quantization logic here. Weights should be saved to disk.
        with torch.inference_mode():
            # First quantize the speech encoder
            print("Quantizing speech encoder...")
            self.quantize_speech_encoder()

            # Then quantize the text decoder
            print("Quantizing text decoder...")
            self.quantize_text_decoder()
        # The block-wise quantizers hold one captured input per calibration sample, kept alive by the
        # reference cycles of their hooks until the cycle collector runs: 33 GB for whisper-large-v3 on
        # 2048 samples, which leaves evaluation without memory.
        gc.collect()
        cuda_empty_cache()
        print("Quantization complete.")

    def should_quantize_module(self, name, module):
        """Check if a module should be quantized based on its name and type."""
        if name in self.quant_cfg.exclude_modules or name.endswith(".output_linear"):
            return False
        return isinstance(module, nn.Linear)

    def quantize_speech_encoder(self):
        """Quantize the speech encoder."""
        raise NotImplementedError("Speech encoder quantization not implemented.")

    def quantize_text_decoder(self):
        """Quantize the text decoder."""
        raise NotImplementedError("Text decoder quantization not implemented.")
    
    def norm_tweak_targets(self) -> Dict[str, List[str]]:
        """``{norm_name: [layer names]}``: each norm whose output feeds only those layers, in one block.

        Norm tweaking scales these norms' outputs after the layers are quantized; models without it return
        nothing.
        """
        return {}

    def capture_norm_tweaks(self, quantizers: Dict[str, Any]) -> list:
        """Before a block's layers are quantized, keep what norm tweaking needs; empty when it is off."""
        if not getattr(self.quant_cfg, "norm_tweak", False):
            return []
        return capture_norm_tweaks(self.norm_tweak_targets(), quantizers, dict(self.model.named_modules()))

    def apply_norm_tweaks(self, captured: list) -> None:
        """After a block's layers are quantized, scale its norms; see asrq.quantizers.norm_tweak."""
        if not captured:
            return
        results = apply_norm_tweaks(captured, dict(self.model.named_modules()), self.quant_cfg.norm_tweak_ridge)
        for name, (s, ratio) in results.items():
            tqdm.write(f"Tweaked {name}: s in [{float(s.min()):.3f}, {float(s.max()):.3f}], "
                       f"output error x{ratio:.3f}")

    def transformer_block_widths(self) -> Dict[str, int]:
        """``{block_name: residual width}`` of the blocks block_output_refit_insert gives an output Linear: those
        that end in a residual sum rather than a linear layer. None by default."""
        return {}

    def block_output_layer(self, block_name: str) -> Optional[str]:
        """The linear layer a block's output comes from, when it ends in one: the identity Linear a rotation
        inserts after a Conformer block's output norm (``<block>.norm_out.1``), or the ``output_linear`` of
        insert_block_output_linear. None for other blocks."""
        for suffix in ("norm_out.1", "output_linear"):
            try:
                layer = self.model.get_submodule(f"{block_name}.{suffix}")
            except AttributeError:
                continue
            if isinstance(layer, nn.Linear):
                return f"{block_name}.{suffix}"
        return None

    def capture_refit_targets(self, block_name: str) -> Optional[list]:
        """A list to collect the full-precision block's outputs in before its layers are quantized, when the block's
        output layer is to be refit; None otherwise."""
        if not getattr(self.quant_cfg, "block_output_refit", False) or self.block_output_layer(block_name) is None:
            return None
        return []

    def refit_block_output(self, block_name: str, quantizers: Dict[str, Any], run, targets: Optional[list]) -> None:
        """After a block's layers are quantized, refit its output layer to the full-precision outputs in targets;
        see asrq.quantizers.output_refit. ``run(i)`` runs the block on calibration sample i. An output layer that is
        quantized itself is left alone, since the refit would replace its quantized weights."""
        if not targets:
            return
        name = self.block_output_layer(block_name)
        if name in quantizers:
            tqdm.write(f"Not refitting {name}: it is quantized")
            return
        layer = self.model.get_submodule(name)
        refit = OutputRefit(layer)
        current = {}
        handle = layer.register_forward_hook(
            lambda _m, inputs, _output: refit.add(inputs[0], current["target"].to(inputs[0].device))
        )
        try:
            with torch.no_grad():
                for index, target in enumerate(targets):
                    current["target"] = target
                    run(index)
        finally:
            handle.remove()
        before, after = refit.solve(self.quant_cfg.block_output_refit_ridge)
        tqdm.write(f"Refit {name}: block output error {before:.3e} -> {after:.3e} (relative to its energy)")

    def online_hadamard_layers(self) -> Dict[str, str]:
        """``{layer_name: activation_name}`` for layers fed by an online Hadamard module.

        Models whose rotation puts an online Hadamard in front of a layer override this, so
        to_asrq_linear can fuse the Hadamard into that layer's humming call.
        """
        return {}

    def to_asrq_linear(self, cfg) -> Dict[str, ASRQLinear]:
        """Replace every activation-quantized layer with a real low-bit ASRQLinear, in place.

        For measuring speed: evaluation otherwise fakes the quantization. Each layer named by
        activation_quantization_roles() gets the quantizer's weight bits and group size, and the
        config's activation settings -- activation_bits, and activation_group_size for the roles
        in activation_groupwise_roles, one scale per token for the rest. Run it after quantize():
        the fake-quantized weights are loaded onto humming's matching grid. A layer fed by an
        online Hadamard takes it into its own humming call. Layers without a role keep their
        fake-quantized fp16 weights.

        Evaluation must not also fake the activation quantization on these layers; set
        ``inference: humming`` in the config, which exp.py and evaluation both read.
        """
        if not self.quant_cfg.symmetric:
            raise ValueError("ASRQLinear supports symmetric weight quantization only")
        activation_bits = cfg.activation_bits if cfg.activation_bits < 16 else 16
        groupwise = set(cfg.activation_groupwise_roles)
        layers = {
            name: (
                self.quant_cfg.bits,
                activation_bits,
                cfg.activation_group_size if role in groupwise else 0,
            )
            for name, role in self.activation_quantization_roles().items()
        }
        return replace_with_asrq_linear(
            self.model, layers, self.quant_cfg.group_size, self.online_hadamard_layers()
        )

    def learn_rotation(self, rotation_path:str)->None:
        """Learn a rotation matrix for the model and save it to disk."""
        raise NotImplementedError("Rotation learning not implemented.")
    
    def rotate_model(self, rotation_path:str)->None:
        """Rotate the model using a learned rotation matrix."""
        raise NotImplementedError("Model rotation not implemented.")

