# pyright: reportMissingImports=false
# pyright: reportIncompatibleVariableOverride=false

from typing import Any, Tuple
import torch
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
        W = self.module.weight.data.clone() # type: ignore
        columns = W.shape[1]
        # initial scales and zeros
        scales, zeros = self.find_quant_params(W)
        scales.squeeze_(-1); zeros.squeeze_(-1)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        # act order and static groups
        perm = torch.argsort(torch.diag(H), descending=True)
        inv_perm = torch.argsort(perm)
        W = W[:, perm]
        H = H[perm][:, perm]
        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        damp = self.quant_config.percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=W.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        group_size = self.quant_config.group_size
        if group_size == -1:
            group_size = columns
        for i1 in range(0, columns, self.quant_config.block_size):
            i2 = min(i1 + self.quant_config.block_size, columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                idx = i1 + i
                idx = perm[idx]
                s, z = scales[:, idx// group_size], zeros[:, idx//group_size]
                q = torch.round((w - z) / s).clamp(self.minq, self.maxq) * s + z
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w-q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        cuda_synchronize()
        Q = Q[:, inv_perm]
        self.module.weight.data =  Q.reshape(self.module.weight.shape).to(self.module.weight.data.dtype) # type: ignore

        cuda_empty_cache()
        return (scales, zeros)