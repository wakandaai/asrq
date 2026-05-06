# pyright: reportMissingImports=false
# pyright: reportIncompatibleVariableOverride=false

import torch
import torch.nn as nn
from omegaconf import DictConfig
from asrq.core.registry import QuantizerNames, register_quantizer, register_quantizer_config
from asrq.quantizers.base import LinearQuantConfig, LinearQuantizer
from typing import Tuple, Any



@register_quantizer_config(QuantizerNames.RTN)
class RTNConfig(LinearQuantConfig):
    def __init__(self, cfg:DictConfig)->None:
        super().__init__(cfg)
        self.group_size = cfg.group_size
        self.symmetric = cfg.symmetric


@register_quantizer(QuantizerNames.RTN)
class RTNQuantizer(LinearQuantizer):
    """RTN quantizer for linear layers using a simple rounding-based quantization approach."""
    quant_config: RTNConfig
    def __init__(self, module: nn.Linear, name: str, cfg: RTNConfig):
        super().__init__(module, name, cfg)

    def add_batch(self, batch: Tuple[torch.Tensor, Any])->None:
        """RTN does not require calibration samples, so this method is a no-op."""
        pass

    def __call__(self)-> Tuple[Any, Any]:
        """Quantize the module using RTN."""
        W = self.module.weight.data.clone() # type: ignore
        scales, zeros = self.find_quant_params(W)
        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = W.shape[1]
        W = W.reshape(W.shape[0], W.shape[1] // group_size, group_size)
        if self.quant_config.symmetric:
            q = torch.round(W / scales).clamp(self.minq, self.maxq) * scales
        else:
            q = torch.round((W - zeros) / scales).clamp(self.minq, self.maxq) * scales + zeros
        q = q.reshape(self.module.weight.data.shape) # type: ignore
        self.module.weight.data.copy_(q) # type: ignore
        return scales, zeros