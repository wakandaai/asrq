# pyright: reportMissingImports=false
# pyright: reportIncompatibleVariableOverride=false

from typing import Any, Tuple
import torch.nn as nn
from omegaconf import DictConfig
from asrq.quantizers.base import (
    HessianAddBatchMixin,
    LinearQuantizer,
    LinearQuantConfig,
)
from asrq.core.registry import (
    QuantizerNames,
    register_quantizer,
    register_quantizer_config
)
from asrq.core.utils import cuda_empty_cache, cuda_synchronize
from asrq.quantizers.gptq_solver import gptq_factors, gptq_quantize



@register_quantizer_config(QuantizerNames.GPTQ)
class GPTQConfig(LinearQuantConfig):
    """Configuration for GPTQ quantization.
    
    Attributes:
        percdamp: Percentage of Hessian diagonal to use for damping.
        block_size: Size of blocks for processing columns.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """Initialize GPTQ configuration.
        
        Args:
            cfg: Configuration dictionary containing percdamp, act_order, and block_size.
        """
        super().__init__(cfg)
        self.percdamp = cfg.percdamp
        self.block_size = cfg.block_size


@register_quantizer(QuantizerNames.GPTQ)
class GPTQQuantizer(HessianAddBatchMixin, LinearQuantizer):
    """GPTQ quantizer for linear layers using Hessian-aware quantization."""
    
    quant_config: GPTQConfig
    
    def __init__(self, module: nn.Linear, name: str, cfg: GPTQConfig) -> None:
        """Initialize GPTQ quantizer.
        
        Args:
            module: Linear layer to quantize.
            name: Name of the module.
            cfg: GPTQ configuration.
        """
        HessianAddBatchMixin.__init__(self, module)
        LinearQuantizer.__init__(self, module, name, cfg)

    def __call__(self) -> Tuple[Any, Any]:
        """Perform GPTQ quantization on the module's weights.
        
        Quantizes weights using a block-wise Hessian-aware approach with
        optimal rounding based on second-order information.
        """
        # weight_2d/set_weight_2d keep this identical for nn.Linear and pointwise Conv1d.
        W = self.weight_2d().clone()
        config = self.quant_config
        factors = gptq_factors(self.H, config.percdamp)
        del self.H
        self.set_weight_2d(gptq_quantize(W, factors, config.bits, config.group_size, config.symmetric, config.block_size))
        cuda_synchronize()
        cuda_empty_cache()
        return tuple(t.squeeze(-1) for t in self.find_quant_params(W))