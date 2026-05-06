# pyright: reportMissingImports=false

import torch
import torch.nn as nn



class LinearQ(nn.Module):
    """A linear layer containing quantized weights using ASRQ"""
    def __init__(self, in_features, out_features, bits, group_size=128,bias=True):
        super(LinearQ, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        weight = torch.zeros((out_features, in_features//(8//bits)), dtype=torch.uint8)
        self.register_buffer('weight', weight)
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None
        scales = torch.zeros((out_features, (in_features+group_size-1)//group_size), dtype=torch.float16)
        zeros = torch.zeros((out_features, (in_features+group_size-1)//group_size), dtype=torch.float16)
        self.bits = bits
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)

    def load_quantized_params(self, scales, zeros, quantized_weight):
        """Load the quantized parameters into the layer"""
        self.scales = scales.to(self.scales.device)
        self.zeros = zeros.to(self.zeros.device)
        self.weight = quantized_weight.to(self.weight.device)

    def dequantize_weight(self, device="cpu", dtype=torch.float32):
        """Dequantize the quantized weights"""
        group_size = self.group_size
        W_dequant = torch.zeros((self.out_features, self.in_features), device=device, dtype=dtype)

        scale = self.scales.repeat_interleave(group_size, dim=1).to(device).to(dtype)
        zero = self.zeros.repeat_interleave(group_size, dim=1).to(device).to(dtype)
        if self.bits == 8 or self.bits ==3:
            W_dequant = self.weight.to(dtype) * scale[:,:self.in_features] + zero[:,:self.in_features]

        elif self.bits == 4:
            W_dequant[:, 0::2] = (self.weight & 0x0F).to(dtype)
            W_dequant[:, 1::2] = (self.weight >> 4).to(dtype)
            W_dequant = W_dequant * scale[:,:self.in_features] + zero[:,:self.in_features]

        elif self.bits == 2:
            W_dequant[:, 0::4] = (self.weight & 0x03).to(dtype)
            W_dequant[:, 1::4] = ((self.weight >> 2) & 0x03).to(dtype)
            W_dequant[:, 2::4] = ((self.weight >> 4) & 0x03).to(dtype)
            W_dequant[:, 3::4] = ((self.weight >> 6) & 0x03).to(dtype)
            W_dequant = W_dequant * scale[:,:self.in_features] + zero[:,:self.in_features]
        return W_dequant

    def forward(self, input):
        # Dequantize weights
        x = input#.to(torch.float32)
        W_dequant = self.dequantize_weight(x.device, x.dtype)

        # perform matmul
        output = torch.matmul(x, W_dequant.t())
        if self.bias is not None:
            output += self.bias

        return output


class LinearQQ(LinearQ):
    """A linear layer containing quantized weights using ASRQ, with quantized activations"""
    def __init__(self, in_features, out_features, wbits, abits, group_size=128,bias=True):
        super(LinearQQ, self).__init__(in_features, out_features, wbits, group_size, bias)
        self.register_buffer("act_scales", torch.ones(in_features))
        self.amax = 2** (abits - 1) - 1
        self.abits = abits

    def quantize_activation(self, input):
        # obtain scales
        # static quantization for activations
        scales = input.abs().max(dim=1, keepdim=True).values / self.amax
        qinput = (input / scales).round().clamp(-self.amax, self.amax)
        return qinput, scales
    
    def forward(self, input):
        qinput, act_scales = self.quantize_activation(input)



from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

        