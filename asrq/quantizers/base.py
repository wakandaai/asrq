# pyright: reportMissingImports=false
from omegaconf import DictConfig
import torch
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

from asrq.core.registry import get_quantizer_config_cls
from asrq.quantizers.gptq_solver import add_to_hessian
from asrq.quantizers.weight_rounding import round_weight, weight_quant_params  # noqa: F401  round_weight re-exported



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
        # Conformer models: also quantize the Linear a rotation inserts after each block's output
        # norm. Set from the model config, like exclude_modules; see ParakeetCTCQ.
        self.quantize_block_output_linear = bool(cfg.get("quantize_block_output_linear", False))
        # Bits for the Linear a rotation inserts after a Conformer block's output norm, when it is quantized;
        # None gives it the same bits as every other layer. Its error lands straight in the residual stream, so
        # it can be worth keeping it wider than the rest.
        bits = cfg.get("block_output_linear_bits", None)
        self.block_output_linear_bits = int(bits) if bits else None
        self.scale_recovery = bool(cfg.get("scale_recovery", False))
        self.scale_recovery_ridge = float(cfg.get("scale_recovery_ridge", 0.01))
        self.scale_recovery_method = str(cfg.get("scale_recovery_method", "lockstep"))
        self.block_output_refit = bool(cfg.get("block_output_refit", False))
        self.block_output_refit_ridge = float(cfg.get("block_output_refit_ridge", 0.01))


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
        """Scales and zeros for ``w`` with this quantizer's bits, group size and symmetry; see weight_quant_params."""
        return weight_quant_params(w, self.quant_config.bits, self.quant_config.group_size, self.quant_config.symmetric)
    

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
        self.H, self.nsamples = add_to_hessian(self.H, self.nsamples, self.activations_2d(input)) # type: ignore[attr-defined]
