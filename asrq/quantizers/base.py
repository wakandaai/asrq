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

class Quantizer(ABC):
    """Base quantizer class."""
    def __init__(self, module:torch.nn.Module, name:str, quant_config:QuantConfig)->None:
        self.module = module
        self.name = name
        self.quant_config = quant_config

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
        self.minq = -self.maxq if self.quant_config.symmetric else 0

    def find_quant_params(self, w):
        """Find quantization parameters (scales and zeros) for the given weights."""
        assert w.ndim == 2, "Only 2D weight matrices are supported for GPTQ quantization."
        groupsize = self.quant_config.group_size
        if groupsize == -1:
            groupsize = w.shape[1]
        assert w.shape[1] % groupsize == 0, f"Weight matrix columns ({w.shape[1]}) must be divisible by group size ({groupsize})."
        w = w.reshape(w.shape[0], w.shape[1] // groupsize, groupsize)
        if self.quant_config.symmetric:
            maxq = 2 ** (self.quant_config.bits - 1) - 1
            scales = torch.max(torch.abs(w), dim=2, keepdim=True).values / maxq
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
        self.H = torch.zeros(module.weight.shape[1], module.weight.shape[1]).to(module.weight.device) # type: ignore

    def add_batch(self, batch: Tuple[torch.Tensor, Any])->None:
        """Add a batch of data for calibration."""
        input, _ = batch
        assert input.ndim == 3, "Input must be 3D (batch_size, seq_len, input_dim)"
        X = input.reshape(-1, input.shape[-1])
        zero_mask = (X.abs().sum(dim=1) != 0)
        X = X[zero_mask]
        n_new_samples = X.shape[0]
        self.H *= (self.nsamples/(self.nsamples + n_new_samples))
        self.nsamples += n_new_samples
        X = math.sqrt(2/self.nsamples) * X.float()
        self.H += X.T @ X