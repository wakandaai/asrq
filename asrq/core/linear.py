from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

_SCHEMES: dict[Tuple[int, int], bool] = {
    (4, 4): True,
    (4, 8): True,
    (4, 16): False,
    (2, 16): False,
}

_GROUP_SIZE = 128

_INPUT_TRANSFORMS = {None, "scale", "hadamard"}


def _is_pointwise_conv(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv1d) and tuple(module.kernel_size) == (1,) and module.groups == 1


class ASRQLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, wbits=4, abits=16,
                 group_size=-1, input_transform=None, input_scale=None,
                 hadamard_block_size=None, conv1d_layout=False):
        super().__init__(in_features, out_features, bias=bias)
        self.wbits = wbits
        self.abits = abits
        self.group_size = group_size
        self.input_transform = input_transform
        self.hadamard_block_size = hadamard_block_size
        self.conv1d_layout = conv1d_layout
        self._validate()
        self.register_buffer(
            "input_scale",
            None if input_scale is None else input_scale.detach().clone().flatten(),
        )
        self.quantized = False
        self._humming_layer = None

    @property
    def symmetric(self) -> bool:
        return _SCHEMES[(self.wbits, self.abits)]

    def _validate(self) -> None:
        if (self.wbits, self.abits) not in _SCHEMES:
            raise ValueError(
                f"Unsupported (wbits={self.wbits}, abits={self.abits}); "
                f"supported schemes are {sorted(_SCHEMES)}"
            )
        if self.group_size not in (-1, _GROUP_SIZE):
            raise ValueError(f"group_size must be -1 (per-channel) or {_GROUP_SIZE}; got {self.group_size}")
        if self.group_size == _GROUP_SIZE and self.in_features % _GROUP_SIZE != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be a multiple of group_size ({_GROUP_SIZE})"
            )
        if self.input_transform not in _INPUT_TRANSFORMS:
            raise ValueError(
                f"Unsupported input_transform={self.input_transform!r}; supported values are "
                f"{sorted(t for t in _INPUT_TRANSFORMS if t)} or None"
            )
        if self.hadamard_block_size is not None:
            if self.input_transform != "hadamard":
                raise ValueError("hadamard_block_size only applies to input_transform='hadamard'")
            if self.in_features % self.hadamard_block_size != 0:
                raise ValueError(
                    f"in_features ({self.in_features}) must be a multiple of "
                    f"hadamard_block_size ({self.hadamard_block_size})"
                )


    @classmethod
    def from_linear(cls, module: Union[nn.Linear, nn.Conv1d], wbits=4, abits=16,
                    group_size=-1, input_transform=None, input_scale=None,
                    hadamard_block_size=None) -> "ASRQLinear":
        is_conv = isinstance(module, nn.Conv1d)
        if is_conv:
            if not _is_pointwise_conv(module):
                raise ValueError(
                    f"Only pointwise, ungrouped Conv1d can be quantized as a linear layer; got "
                    f"kernel_size={tuple(module.kernel_size)}, groups={module.groups}"
                )
            in_features, out_features = module.in_channels, module.out_channels
            weight = module.weight.data.reshape(out_features, in_features)
        else:
            in_features, out_features = module.in_features, module.out_features
            weight = module.weight.data

        if input_transform is None:
            input_transform, input_scale, hadamard_block_size = cls._detect_input_transform(module)

        layer = cls(
            in_features, out_features, bias=module.bias is not None,
            wbits=wbits, abits=abits, group_size=group_size,
            input_transform=input_transform, input_scale=input_scale,
            hadamard_block_size=hadamard_block_size, conv1d_layout=is_conv,
        )
        layer.weight.data.copy_(weight)
        if module.bias is not None:
            layer.bias.data.copy_(module.bias.data)
        layer.quantize_()
        return layer

    @staticmethod
    def _detect_input_transform(module):
        from asrq.transforms.scaling.base import ScaledInputConv1d, ScaledInputLinear

        if isinstance(module, (ScaledInputLinear, ScaledInputConv1d)):
            return "scale", module.scale, None
        if getattr(module, "_hadamard_fused", False):
            return "hadamard", None, getattr(module, "_hadamard_block_size", None)
        return None, None, None

    @torch.no_grad()
    def quantize_(self) -> None:
        from humming.layer import HummingLayer

        weight = self.weight.data.cuda().half()
        bias = self.bias.data.cuda().half() if self.bias is not None else None

        weight_config = {"dtype": f"int{self.wbits}", "has_zero_point": not self.symmetric}
        if self.group_size == _GROUP_SIZE:
            weight_config["group_size"] = _GROUP_SIZE
        elif not self.symmetric:
            weight_config["is_fp_zero_point"] = True
        input_config = None if self.abits == 16 else {"dtype": f"int{self.abits}", "group_size": 0}

        layer = HummingLayer(
            shape_n=self.out_features, shape_k=self.in_features,
            weight_config=weight_config, input_config=input_config,
            has_bias=bias is not None, torch_dtype=torch.float16,
        ).cuda()
        layer.load_from_unquantized(weight)
        if bias is not None:
            layer.bias.data.copy_(bias)
        layer.transform()

        self._humming_layer = layer
        if bias is not None:
            self.bias.data = bias
        self.register_parameter("weight", None)
        self.quantized = True


    def set_input_transform(self, input_transform, input_scale=None, hadamard_block_size=None) -> None:
        self.input_transform = input_transform
        self.hadamard_block_size = hadamard_block_size
        self._validate()
        if input_transform == "scale" and input_scale is None:
            raise ValueError("input_transform='scale' needs an input_scale")
        device = self.bias.device if self.bias is not None else "cuda"
        self.input_scale = None if input_scale is None else input_scale.detach().clone().flatten().to(device)

    def _transform_input(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.input_transform is None:
            return x2d
        if self.input_transform == "scale":
            if self.input_scale is None:
                raise RuntimeError("input_transform='scale' requires input_scale to be set")
            return (x2d / self.input_scale.to(dtype=x2d.dtype, device=x2d.device)).contiguous()

        from asrq.transforms.rotation.hadamard_utils import matmul_hadU_auto

        if self.hadamard_block_size is None:
            return matmul_hadU_auto(x2d).contiguous()
        blocks = x2d.reshape(-1, self.in_features // self.hadamard_block_size, self.hadamard_block_size)
        return matmul_hadU_auto(blocks).reshape(x2d.shape).contiguous()

    def _matmul(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.abits == 16:
            return self._humming_layer(x2d)
        from humming import ops

        xq, x_scale = ops.quant_input(x2d, f"int{self.abits}", group_size=0, scale_dtype="float32")
        return self._humming_layer(inputs=xq, input_scale=x_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            raise RuntimeError(
                "ASRQLinear must be quantized (call .quantize_() or build via .from_linear()) before use"
            )
        if self.conv1d_layout:
            x = x.transpose(1, 2)
        batch_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).half().contiguous()
        out = self._matmul(self._transform_input(x2d)).reshape(*batch_shape, self.out_features)
        return out.transpose(1, 2) if self.conv1d_layout else out

    def extra_repr(self) -> str:
        group = "per-channel" if self.group_size == -1 else f"group={self.group_size}"
        mode = "sym" if self.symmetric else "asym"
        extra = ", layout=conv1d" if self.conv1d_layout else ""
        if self.input_transform is not None:
            extra += f", input_transform={self.input_transform}"
            if self.hadamard_block_size is not None:
                extra += f"(block={self.hadamard_block_size})"
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, w{self.wbits}a{self.abits} ({mode}, {group}){extra}")


class LinearQ(nn.Module):
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
        self.scales = scales.to(self.scales.device)
        self.zeros = zeros.to(self.zeros.device)
        self.weight = quantized_weight.to(self.weight.device)

    def dequantize_weight(self, device="cpu", dtype=torch.float32):
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
        x = input
        W_dequant = self.dequantize_weight(x.device, x.dtype)

        output = torch.matmul(x, W_dequant.t())
        if self.bias is not None:
            output += self.bias

        return output
