import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import types



# Include activation quantization
def activation_quantization_forward_patch(linear, bits, layer_name):
    max_q =  (2 ** bits - 1)
    def forward(self, x):
        w = linear.weight
        bias = linear.bias
        
        # compute scales and zeros for the activations
        if bits < 16:

            # zero_point = x.min(dim=dim, keepdim=True).values
            # scale = (x.max(dim=dim, keepdim=True).values - zero_point) / max_q
            # x = torch.round((x - zero_point) / scale) * scale + zero_point

            # symmetric quantization
            # x is (B, T, dim)
            # obtain scales
            scales = x.abs().max(dim=-1, keepdim=True).values / (2 ** (bits-1) - 1)
            scales = scales.where(scales != 0, 1e-3) # avoid division by 0
            x = torch.round(x / scales) * scales
            name = layer_name
        
        return F.linear(x, w.to(x.dtype), bias)

    linear.forward = types.MethodType(forward, linear)



def modify_linears_with_activation_quantization(model: nn.Module, linears_to_quantize, bits: int = 4) -> None:
    named_modules = dict(model.named_modules())
    for layer_name in linears_to_quantize:
        linear = named_modules.get(layer_name) # type: ignore
        if linear is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        activation_quantization_forward_patch(linear, bits, layer_name)


def activation_fn_scaler(act_fn, scale):
    class ScaledActivation(nn.Module):
        def __init__(self, act_fn, scale):
            super(ScaledActivation, self).__init__()
            self.act_fn = act_fn
            self.scale = scale

        def forward(self, x):
            return self.act_fn(x) * self.scale

    return ScaledActivation(act_fn, scale)
