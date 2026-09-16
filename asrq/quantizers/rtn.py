# pyright: reportMissingImports=false
# pyright: reportIncompatibleVariableOverride=false

import torch
import torch.nn as nn
from omegaconf import DictConfig
from asrq.core.registry import QuantizerNames, register_quantizer, register_quantizer_config
from asrq.quantizers.base import LinearQuantConfig, LinearQuantizer
from asrq.quantizers.weight_rounding import round_weight
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
        # weight_2d/set_weight_2d keep this identical for nn.Linear and pointwise Conv1d.
        q, scales, zeros = round_weight(
            self.weight_2d().clone(), self.quant_config.bits, self.quant_config.group_size, self.quant_config.symmetric
        )
        self.set_weight_2d(q)
        return scales, zeros