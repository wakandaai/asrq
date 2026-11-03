# pyright: reportMissingImports=false

import copy
import gc
import os
import torch
import torch.nn as nn
import numpy as np
import json

from typing import Dict, Iterator, List, Optional, Tuple, Any
from asrq.core.registry import ModelQ_Registry, QuantizerNames
from asrq.quantizers.base import QuantConfig
from asrq.transforms.base import TransformConfig
from asrq.calibration.base import CalibConfig
from asrq.calibration.data import load_calibration_samples
from asrq.core.linear import ASRQLinear, replace_with_asrq_linear
from abc import ABC, abstractmethod
from datasets import load_dataset
from functools import partial
from itertools import islice
from tqdm import tqdm
from asrq.evaluation.english_text_normalizer import normalizer
from asrq.core.types import Processor
from asrq.core.registry import get_quant_cls
from asrq.core.utils import cuda_empty_cache
from asrq import tracking
from asrq.quantizers.scale_recovery import ScaleTarget, apply_scale_recovery, capture_scale_recovery, quantize_scale_group
from asrq.quantizers.weight_recovery import WeightRecovery




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

    QUANTIZED_FORMAT = 4

    def save_quantized(self, path: str, fingerprint: str) -> None:
        """Save the quantized model so load_quantized can restore it without quantizing again.

        The state dict is stored at each tensor's own dtype, together with what a freshly loaded and transformed
        model lacks: the norm weights scale recovery gave scale-free norms (module names whose ``weight`` is a
        buffer). The rotation is not stored; load_quantized expects it applied, as exp.py does before quantizing. Written to a temporary file and renamed, so an interrupted save leaves no file behind.

        Tensors are not narrowed to float16: evaluation casts the model to ``eval_dtype`` (bfloat16 by default), and
        rounding to float16 first changes some bfloat16 values, which changed the WER of a loaded model.
        """
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
            "weight_buffers": weight_buffers,
        }, temporary)
        os.replace(temporary, path)

    def load_quantized(self, path: str, fingerprint: Optional[str] = None) -> None:
        """Restore a model saved by save_quantized into this (transformed, unquantized) model, in place.

        The norm weight buffers are recreated first, then every tensor is loaded; a missing or unexpected key is
        an error, as is a fingerprint other than the given one.
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

    def layer_quant_cfg(self, name: str):
        """The quantizer config for one layer: the experiment's, or a copy with block_output_linear_bits for the
        Linear a rotation inserts after a Conformer block's output norm."""
        bits = getattr(self.quant_cfg, "block_output_linear_bits", None)
        if bits is None or not name.endswith(".norm_out.1"):
            return self.quant_cfg
        config = copy.copy(self.quant_cfg)
        config.bits = bits
        return config

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
    
    def scale_recovery_targets(self) -> List[ScaleTarget]:
        """The norms of this model's transformer blocks and the quantized layers each one feeds, which share its
        scale; see asrq.quantizers.scale_recovery. Empty by default.

        Conformer blocks are left out: they end in a linear layer, which weight recovery corrects instead
        (see block_output_layer).
        """
        return []

    def capture_scale_recovery(self, quantizers: Dict[str, Any]) -> list:
        """Before a block's layers are quantized, the groups to recover; empty when scale recovery is off."""
        if not getattr(self.quant_cfg, "scale_recovery", False):
            return []
        return capture_scale_recovery(self.scale_recovery_targets(), quantizers)

    def quantize_layers(
        self, quantizers: Dict[str, Any], groups: list, skip: Optional[set] = None, recover=None,
    ) -> Iterator[str]:
        """Quantize a block's layers, recording each one's quantization parameters in qparams, and yield each
        name once its layer is quantized.

        The layers a norm feeds are quantized together by quantize_scale_group, when the first of them comes up,
        which also fits their scale; it is folded into the norm straight away, so the rest of the block runs on
        a model that computes what it will compute at inference. Every other layer is quantized by its own
        quantizer. Names in ``skip`` are left for the caller, as apply_weight_recovery quantizes a block's output
        layer itself, after refitting it.

        ``recover(name)``, from layer_recovery, refits a layer just before it is quantized; the layers a norm
        feeds are all recovered before the group is quantized, since they share one Hessian and one scale.
        """
        leaders = {group.target.layers[0]: group for group in groups}
        shared = {name for group in groups for name in group.target.layers}
        skip = skip or set()
        recover = recover or (lambda name: None)
        for name, quantizer in quantizers.items():
            if name in skip:
                continue
            if name in leaders:
                group = leaders[name]
                for layer in group.target.layers:
                    recover(layer)
                group.scale, group.ratio, qparams = quantize_scale_group(
                    [quantizers[layer] for layer in group.target.layers], self.quant_cfg.scale_recovery_ridge,
                    self.quant_cfg.scale_recovery_method,
                )
                self.apply_scale_recovery([group])
                for layer, params in zip(group.target.layers, qparams):
                    self.qparams[layer] = params
                    yield layer
            elif name not in shared:
                recover(name)
                self.qparams[name] = quantizer()
                yield name

    def apply_scale_recovery(self, captured: list) -> None:
        """Fold the scales of groups whose layers are quantized into their norms; see
        asrq.quantizers.scale_recovery. quantize_layers calls it per group, as each one is quantized."""
        if not captured:
            return
        results = apply_scale_recovery(captured, dict(self.model.named_modules()))
        ratios = []
        for name, (s, ratio) in results.items():
            tqdm.write(f"Scaled {name}: s in [{float(s.min()):.3f}, {float(s.max()):.3f}], "
                       f"output error x{ratio:.3f}")
            ratios.append(ratio)
        if ratios:
            tracking.step_metric("scale_recovery", "scale_recovery/group")
            self._scale_recovery_groups = getattr(self, "_scale_recovery_groups", 0)
            scales = torch.cat([results[name][0].flatten().float().cpu() for name in results])
            self._scale_recovery_groups += len(ratios)
            tracking.log({
                "scale_recovery/group": self._scale_recovery_groups,
                "scale_recovery/error_ratio": sum(ratios) / len(ratios),
                "scale_recovery/error_ratio_max": max(ratios),
                "scale_recovery/scale_min": float(scales.min()),
                "scale_recovery/scale_max": float(scales.max()),
                "scale_recovery/scales": tracking.histogram(scales),
            })

    def block_output_layer(self, block_name: str) -> Optional[str]:
        """The layer a block's output comes from, when it ends in one: the identity Linear a rotation inserts after
        a Conformer block's output norm (``<block>.norm_out.1``). None for other blocks."""
        try:
            layer = self.model.get_submodule(f"{block_name}.norm_out.1")
        except AttributeError:
            return None
        return f"{block_name}.norm_out.1" if isinstance(layer, nn.Linear) else None

    def weight_recovery_quantizes(self, block_name: str, quantizers: Dict[str, Any]) -> set:
        """``{output layer name}`` when apply_weight_recovery will quantize that layer itself, for quantize_layers
        to skip; empty otherwise."""
        name = self.block_output_layer(block_name)
        if name is None or name not in quantizers or self.capture_weight_recovery(block_name) is None:
            return set()
        return {name}

    def capture_weight_recovery(self, block_name: str) -> Optional[list]:
        """A list to collect the full-precision block's outputs in before its layers are quantized, when the block's
        output layer is to be refit; None otherwise."""
        if not getattr(self.quant_cfg, "weight_recovery", False) or self.block_output_layer(block_name) is None:
            return None
        return []

    def apply_weight_recovery(self, block_name: str, quantizers: Dict[str, Any], run, targets: Optional[list]) -> None:
        """After a block's other layers are quantized, refit its output layer to the full-precision outputs in
        targets; see asrq.quantizers.weight_recovery. ``run(i)`` runs the block on calibration sample i.

        An output layer that is quantized itself (``quantize_block_output_linear``) is refit first and quantized
        afterwards, so it keeps the correction and still ends up on the grid. Its Hessian is collected in the
        same pass, from the quantized block's inputs, which is what it will see; quantize_layers leaves it to
        this method (see weight_recovery_quantizes).
        """
        if not targets:
            return
        name = self.block_output_layer(block_name)
        layer = self.model.get_submodule(name)
        quantizer = quantizers.get(name)
        if quantizer is not None and getattr(quantizer, "H", None) is not None:
            quantizer.H.zero_()
            quantizer.nsamples = 0
        refit = WeightRecovery(layer)
        current = {}

        def hook(_module, inputs, _output):
            refit.add(inputs[0], current["target"].to(inputs[0].device))
            if quantizer is not None:
                quantizer.add_batch((inputs[0], None))

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                for index, target in enumerate(targets):
                    current["target"] = target
                    run(index)
        finally:
            handle.remove()
        before, after = refit.solve(self.quant_cfg.weight_recovery_ridge)
        tqdm.write(f"Recovered {name}: block output error {before:.3e} -> {after:.3e} (relative to its energy)")
        tracking.step_metric("weight_recovery", "weight_recovery/block")
        self._weight_recovery_blocks = getattr(self, "_weight_recovery_blocks", 0) + 1
        tracking.log({
            "weight_recovery/block": self._weight_recovery_blocks, "weight_recovery/error_before": before, "weight_recovery/error_after": after,
            "weight_recovery/error_ratio": after / before if before > 0 else 1.0,
        })
        if quantizer is not None:
            self.qparams[name] = quantizer()
            tqdm.write(f"Quantized {name} after recovering it")

    def in_execution_order(self, quantizers: Dict[str, Any], run) -> Dict[str, Any]:
        """The block's quantizers reordered by when their layers run, or unchanged when per-layer weight
        recovery is off.

        A block loop collects its quantizers from ``block.named_modules()``, which is the order the
        submodules were assigned in ``__init__``, not the order the forward pass calls them: a Conformer
        block defines its convolution module before its attention module and runs it after. Plain GPTQ does
        not care, because every Hessian is collected before any layer is quantized and quantizing one layer
        leaves the others untouched. layer_recovery does care: a layer can only be refit against the drift
        that is already at its input, so a layer quantized before the layers that feed it never sees theirs.

        The order is read from the model rather than declared per model: forward hooks fire in the order the
        layers run, so one pass over one calibration sample gives it exactly, including layers a config
        skips. A quantized layer the pass never calls keeps its place at the end.

        Args:
            quantizers: ``{layer name: quantizer}`` for the block, from named_modules.
            run: ``run(index)`` runs the block on calibration sample ``index``.
        """
        if not getattr(self.quant_cfg, "layer_weight_recovery", False):
            return quantizers
        order: List[str] = []
        handles = []

        def hook(_module, _inputs, _output, name=""):
            if name not in order:
                order.append(name)

        for name, quantizer in quantizers.items():
            handles.append(quantizer.module.register_forward_hook(partial(hook, name=name)))
        try:
            with torch.no_grad():
                run(0)
        finally:
            for handle in handles:
                handle.remove()
        order += [name for name in quantizers if name not in order]
        return {name: quantizers[name] for name in order}

    def layer_recovery(self, block: nn.Module, quantizers: Dict[str, Any], run, num_samples: int):
        """``recover(name)`` for quantize_layers, which refits a layer onto its full-precision output just
        before it is quantized; None when per-layer weight recovery is off.

        Where apply_weight_recovery corrects a Conformer block's whole error at the linear layer it ends in,
        this corrects one layer at a time, so it needs no such layer and reaches transformer blocks too. A
        copy of the block, taken before anything in it is quantized, provides each layer's full-precision
        output Y; the layer's input Z comes from the block as it stands, with the layers quantized so far in
        place; the layer is refit to map Z onto Y and is quantized afterwards, so GPTQ puts the refit weight
        on the grid. See asrq.quantizers.weight_recovery.

        Both runs read the same block input, so the fit corrects the error this block's own quantized layers
        introduce, as the block output refit does; error inherited from earlier blocks sits in Y as much as in
        Z. Reaching for that inherited error instead was measured worse: a single linear map undoes only the
        part of it that its own input spans.

        A layer with no quantized layer upstream of it therefore sees Z equal to Y's input, where the fit
        returns the weights the layer already has, and is skipped. One calibration sample decides that: the
        two runs are the same computation except where a quantized layer lies between the block input and the
        layer. Which layers those are depends on the order the block is quantized in; see in_execution_order.

        The layer's Hessian is collected again in the same pass, from the inputs it is about to be quantized
        on. The one from the calibration pass is stale twice over: refitting a layer and quantizing it both
        move the inputs of everything downstream of it.

        Args:
            block: The block being quantized, with none of its layers quantized yet.
            quantizers: ``{layer name: quantizer}`` for the block, as quantize_layers takes them.
            run: ``run(block, index)`` runs a given block on calibration sample ``index``.
            num_samples: The number of calibration samples ``run`` accepts.

        Returns:
            ``recover(name)``, or None when quantizer.layer_weight_recovery is off.
        """
        if not getattr(self.quant_cfg, "layer_weight_recovery", False):
            return None
        teacher = copy.deepcopy(block).eval()
        sub_names = {id(module): name for name, module in block.named_modules()}
        teacher_modules = dict(teacher.named_modules())
        teacher_layers = {
            name: teacher_modules[sub_names[id(quantizer.module)]] for name, quantizer in quantizers.items()
        }

        def first_input(layer, source):
            """The layer's input on the first calibration sample, running ``source`` in place of the block."""
            captured = {}

            def hook(_module, inputs, _output):
                captured.setdefault("x", inputs[0].detach())

            handle = layer.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    run(source, 0)
            finally:
                handle.remove()
            return captured.get("x")

        def recover(name):
            quantizer = quantizers[name]
            layer, teacher_layer = quantizer.module, teacher_layers[name]
            reference, current = first_input(teacher_layer, teacher), first_input(layer, block)
            if reference is None or current is None or torch.equal(reference, current):
                return
            refit = WeightRecovery(layer)
            if getattr(quantizer, "H", None) is not None:
                quantizer.H.zero_()
                quantizer.nsamples = 0
            target = {}

            def teacher_hook(_module, _inputs, output):
                target["y"] = output

            def layer_hook(_module, inputs, output):
                refit.add(inputs[0], target["y"].to(inputs[0].device))
                quantizer.add_batch((inputs[0], output))

            handles = [
                teacher_layer.register_forward_hook(teacher_hook),
                layer.register_forward_hook(layer_hook),
            ]
            try:
                with torch.no_grad():
                    for index in range(num_samples):
                        run(teacher, index)
                        run(block, index)
            finally:
                for handle in handles:
                    handle.remove()
            before, after = refit.solve(self.quant_cfg.layer_weight_recovery_ridge)
            tqdm.write(f"Recovered {name}: output error {before:.3e} -> {after:.3e} (relative to its energy)")
            tracking.step_metric("layer_weight_recovery", "layer_weight_recovery/layer")
            self._layer_recovery_layers = getattr(self, "_layer_recovery_layers", 0) + 1
            tracking.log({
                "layer_weight_recovery/layer": self._layer_recovery_layers,
                "layer_weight_recovery/error_before": before,
                "layer_weight_recovery/error_after": after,
                "layer_weight_recovery/error_ratio": after / before if before > 0 else 1.0,
            })

        return recover

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
                self.layer_quant_cfg(name).bits,
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

