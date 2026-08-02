# pyright: reportMissingImports=false
from omegaconf import DictConfig
import torch
import math
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from asrq.core.registry import get_quantizer_config_cls



class QuantConfig(ABC):
    """Base quantization configuration class."""
    @staticmethod
    def from_dictconfig(cfg:DictConfig):
        """Create a QuantConfig instance from a DictConfig."""
        quant_config_cls = get_quantizer_config_cls(cfg.name)
        return quant_config_cls(cfg)

    def __init__(self, cfg:DictConfig)->None:
        self.name = cfg.name
        self.bits = cfg.bits
        self.exclude_modules = cfg.exclude_modules


class LinearQuantConfig(QuantConfig):
    """Quantization configuration for linear uniform quantization."""
    def __init__(self, cfg: DictConfig)->None:
        super().__init__(cfg)
        self.group_size = cfg.group_size
        self.symmetric = cfg.symmetric

def is_pointwise_conv1d(module: torch.nn.Module) -> bool:
    """A kernel-size-1 Conv1d, i.e. a linear layer in convolution clothing.

    The Conformer's ``conv.pointwise_conv1`` / ``conv.pointwise_conv2`` are these: their
    weight is ``(out, in, 1)`` and their activations are ``(B, C, T)`` rather than the
    ``(out, in)`` / ``(B, T, C)`` a linear layer uses. Reshaping the weight and
    transposing the activations makes every quantizer here apply unchanged.
    """
    return isinstance(module, torch.nn.Conv1d) and tuple(module.kernel_size) == (1,)


class Quantizer(ABC):
    """Base quantizer class."""
    def __init__(self, module:torch.nn.Module, name:str, quant_config:QuantConfig)->None:
        self.module = module
        self.name = name
        self.quant_config = quant_config
        self.is_pointwise_conv = is_pointwise_conv1d(module)
        if isinstance(module, torch.nn.Conv1d) and not self.is_pointwise_conv:
            raise ValueError(
                f"'{name}' is a Conv1d with kernel_size={tuple(module.kernel_size)}; only "
                f"pointwise (kernel_size=1) convolutions can be quantized as linear layers"
            )

    def weight_2d(self) -> torch.Tensor:
        """The weight as ``(out_features, in_features)``, whatever the module type."""
        w = self.module.weight.data # type: ignore
        return w.reshape(w.shape[0], -1) if self.is_pointwise_conv else w

    def set_weight_2d(self, w2d: torch.Tensor) -> None:
        """Write a ``(out_features, in_features)`` weight back in the module's own shape."""
        target = self.module.weight.data # type: ignore
        target.copy_(w2d.reshape(target.shape).to(target.dtype))

    def activations_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten an input batch to ``(tokens, in_features)``.

        Conv1d activations are ``(B, C, T)``, so the channel axis is moved last first -
        otherwise the statistics would be gathered over time instead of over channels.
        """
        if self.is_pointwise_conv:
            x = x.transpose(1, 2)
        return x.reshape(-1, x.shape[-1])

    def find_quant_params(self, w):
        """Find quantization parameters (scales and zeros) for the given weights."""
        raise NotImplementedError("find_quant_params not implemented")
        
    def add_batch(self, batch: Tuple[torch.Tensor, Any])->None:
        """Add a batch of data for calibration."""
        raise NotImplementedError("add_batch not implemented")
    
    def __call__(self) -> Tuple[Any, Any]:
        """Quantize the module."""
        raise NotImplementedError("quantize not implemented")

        
class LinearQuantizer(Quantizer):
    """Base quantizer class for linear uniform quantization."""
    quant_config: LinearQuantConfig
    def __init__(self, module:torch.nn.Linear, name:str, quant_config:LinearQuantConfig)->None:
        super().__init__(module, name, quant_config)
        self.maxq = 2 ** (self.quant_config.bits - 1 ) - 1 if self.quant_config.symmetric else 2 ** self.quant_config.bits - 1
        # Two's complement is asymmetric: b bits span [-2^(b-1), 2^(b-1)-1], so the
        # negative side has one more code than the positive one. find_quant_params picks a
        # scale that uses it (see there), so the clamp has to allow it.
        self.minq = -(self.maxq + 1) if self.quant_config.symmetric else 0

    def find_quant_params(self, w):
        """Find quantization parameters (scales and zeros) for the given weights.

        The symmetric branch follows Humming's rule (``quant_weight.cuh``) rather than the
        textbook ``abs_max / maxq``, which wastes the extra negative code. For b bits the
        codes run ``[-(maxq+1), maxq]``, so:

            scale = +/- max(max_abs / (maxq + 1),  min_abs / maxq)

        where ``max_abs``/``min_abs`` are the larger/smaller of the group's two extremes.
        Taking the max of the two candidates is the smallest scale that clips neither side:
        the dominant extreme needs ``scale >= max_abs/(maxq+1)`` to fit the -(maxq+1) code,
        the weaker one needs ``scale >= min_abs/maxq`` to fit +maxq. The scale is negated
        when the positive extreme dominates, which is what puts *it* on the -(maxq+1) code
        (``-(maxq+1) * -|s| = +(maxq+1)|s|``).

        Negative scales stay inside the quantizer: this path fake-quantizes in place
        (``round(w/s).clamp(...) * s``), so nothing downstream sees the sign.
        """
        assert w.ndim == 2, "Only 2D weight matrices are supported for GPTQ quantization."
        groupsize = self.quant_config.group_size
        if groupsize == -1:
            groupsize = w.shape[1]
        assert w.shape[1] % groupsize == 0, f"Weight matrix columns ({w.shape[1]}) must be divisible by group size ({groupsize})."
        w = w.reshape(w.shape[0], w.shape[1] // groupsize, groupsize)
        if self.quant_config.symmetric:
            maxq = 2 ** (self.quant_config.bits - 1) - 1
            w_max = torch.amax(w, dim=2, keepdim=True)
            w_min = torch.amin(w, dim=2, keepdim=True)
            max_abs = torch.maximum(w_max, w_min.abs())
            min_abs = torch.minimum(w_max, w_min.abs())
            if maxq > 0:
                scales = torch.maximum(max_abs / (maxq + 1), min_abs / maxq)
            else:  # 1-bit has no positive code; fall back to plain abs-max
                scales = max_abs
            scales = torch.where(w_max > w_min.abs(), -scales, scales)
            # An all-zero group has no scale to speak of; 1 keeps the division finite and
            # quantizes it to all zeros, matching Humming.
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            zeros = torch.zeros_like(scales)
        else:
            maxq = 2 ** self.quant_config.bits - 1
            scales = (torch.max(w, dim=2, keepdim=True).values - torch.min(w, dim=2, keepdim=True).values) / maxq
            zeros = torch.min(w, dim=2, keepdim=True).values
        return scales, zeros
    

class HessianAddBatchMixin:
    """Mixin class for adding batches to compute the Hessian."""
    def __init__(self, module):
        self.nsamples = 0
        in_features = module.weight.shape[1] # type: ignore  # (out, in) or (out, in, 1)
        self.H = torch.zeros(in_features, in_features).to(module.weight.device) # type: ignore

    def add_batch(self, batch: Tuple[torch.Tensor, Any])->None:
        """Add a batch of data for calibration."""
        input, _ = batch
        assert input.ndim == 3, "Input must be 3D (batch_size, seq_len, input_dim)"
        # activations_2d handles the (B, C, T) layout of a pointwise conv.
        X = self.activations_2d(input) # type: ignore[attr-defined]
        zero_mask = (X.abs().sum(dim=1) != 0)
        X = X[zero_mask]
        n_new_samples = X.shape[0]
        self.H *= (self.nsamples/(self.nsamples + n_new_samples))
        self.nsamples += n_new_samples
        X = math.sqrt(2/self.nsamples) * X.float()
        self.H += X.T @ X