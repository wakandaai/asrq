"""Real low-bit linear layers running humming's quantized matmul, for measuring speedups.

Evaluation measures WER with fake quantization: weights are rounded to their low-bit grid and stored
back in fp16, and activations are rounded by hooks before an ordinary fp16 matmul. That reproduces
what a low-bit model computes but not how fast it computes it. ASRQLinear replaces such a layer with
a humming layer that stores packed low-bit weights and multiplies quantized activations with them
directly, so the same model can be timed as it would actually run.
"""

import json
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from humming.layer import HummingLayer

# from typing import Optional, Tuple, Union

# import torch
# import torch.nn as nn

# _SCHEMES: dict[Tuple[int, int], bool] = {
#     (4, 4): True,
#     (4, 8): True,
#     (4, 16): False,
#     (2, 16): False,
# }

# _GROUP_SIZE = 128

# _INPUT_TRANSFORMS = {None, "scale", "hadamard"}


# def _is_pointwise_conv(module: nn.Module) -> bool:
#     return isinstance(module, nn.Conv1d) and tuple(module.kernel_size) == (1,) and module.groups == 1


# class ASRQLinear(nn.Linear):

#     def __init__(self, in_features, out_features, bias=True, wbits=4, abits=16,
#                  group_size=-1, input_transform=None, input_scale=None,
#                  hadamard_block_size=None, conv1d_layout=False):
#         super().__init__(in_features, out_features, bias=bias)
#         self.wbits = wbits
#         self.abits = abits
#         self.group_size = group_size
#         self.input_transform = input_transform
#         self.hadamard_block_size = hadamard_block_size
#         self.conv1d_layout = conv1d_layout
#         self._validate()
#         self.register_buffer(
#             "input_scale",
#             None if input_scale is None else input_scale.detach().clone().flatten(),
#         )
#         self.quantized = False
#         self._humming_layer = None

#     @property
#     def symmetric(self) -> bool:
#         return _SCHEMES[(self.wbits, self.abits)]

#     def _validate(self) -> None:
#         if (self.wbits, self.abits) not in _SCHEMES:
#             raise ValueError(
#                 f"Unsupported (wbits={self.wbits}, abits={self.abits}); "
#                 f"supported schemes are {sorted(_SCHEMES)}"
#             )
#         if self.group_size not in (-1, _GROUP_SIZE):
#             raise ValueError(f"group_size must be -1 (per-channel) or {_GROUP_SIZE}; got {self.group_size}")
#         if self.group_size == _GROUP_SIZE and self.in_features % _GROUP_SIZE != 0:
#             raise ValueError(
#                 f"in_features ({self.in_features}) must be a multiple of group_size ({_GROUP_SIZE})"
#             )
#         if self.input_transform not in _INPUT_TRANSFORMS:
#             raise ValueError(
#                 f"Unsupported input_transform={self.input_transform!r}; supported values are "
#                 f"{sorted(t for t in _INPUT_TRANSFORMS if t)} or None"
#             )
#         if self.hadamard_block_size is not None:
#             if self.input_transform != "hadamard":
#                 raise ValueError("hadamard_block_size only applies to input_transform='hadamard'")
#             if self.in_features % self.hadamard_block_size != 0:
#                 raise ValueError(
#                     f"in_features ({self.in_features}) must be a multiple of "
#                     f"hadamard_block_size ({self.hadamard_block_size})"
#                 )


#     @classmethod
#     def from_linear(cls, module: Union[nn.Linear, nn.Conv1d], wbits=4, abits=16,
#                     group_size=-1, input_transform=None, input_scale=None,
#                     hadamard_block_size=None) -> "ASRQLinear":
#         is_conv = isinstance(module, nn.Conv1d)
#         if is_conv:
#             if not _is_pointwise_conv(module):
#                 raise ValueError(
#                     f"Only pointwise, ungrouped Conv1d can be quantized as a linear layer; got "
#                     f"kernel_size={tuple(module.kernel_size)}, groups={module.groups}"
#                 )
#             in_features, out_features = module.in_channels, module.out_channels
#             weight = module.weight.data.reshape(out_features, in_features)
#         else:
#             in_features, out_features = module.in_features, module.out_features
#             weight = module.weight.data

#         if input_transform is None:
#             input_transform, input_scale, hadamard_block_size = cls._detect_input_transform(module)

#         layer = cls(
#             in_features, out_features, bias=module.bias is not None,
#             wbits=wbits, abits=abits, group_size=group_size,
#             input_transform=input_transform, input_scale=input_scale,
#             hadamard_block_size=hadamard_block_size, conv1d_layout=is_conv,
#         )
#         layer.weight.data.copy_(weight)
#         if module.bias is not None:
#             layer.bias.data.copy_(module.bias.data)
#         layer.quantize_()
#         return layer

#     @staticmethod
#     def _detect_input_transform(module):
#         from asrq.transforms.scaling.base import ScaledInputConv1d, ScaledInputLinear

#         if isinstance(module, (ScaledInputLinear, ScaledInputConv1d)):
#             return "scale", module.scale, None
#         if getattr(module, "_hadamard_fused", False):
#             return "hadamard", None, getattr(module, "_hadamard_block_size", None)
#         return None, None, None

#     @torch.no_grad()
#     def quantize_(self) -> None:
#         from humming.layer import HummingLayer

#         weight = self.weight.data.cuda().half()
#         bias = self.bias.data.cuda().half() if self.bias is not None else None

#         weight_config = {"dtype": f"int{self.wbits}", "has_zero_point": not self.symmetric}
#         if self.group_size == _GROUP_SIZE:
#             weight_config["group_size"] = _GROUP_SIZE
#         elif not self.symmetric:
#             weight_config["is_fp_zero_point"] = True
#         input_config = None if self.abits == 16 else {"dtype": f"int{self.abits}", "group_size": 0}

#         layer = HummingLayer(
#             shape_n=self.out_features, shape_k=self.in_features,
#             weight_config=weight_config, input_config=input_config,
#             has_bias=bias is not None, torch_dtype=torch.float16,
#         ).cuda()
#         layer.load_from_unquantized(weight)
#         if bias is not None:
#             layer.bias.data.copy_(bias)
#         layer.transform()

#         self._humming_layer = layer
#         if bias is not None:
#             self.bias.data = bias
#         self.register_parameter("weight", None)
#         self.quantized = True


#     def set_input_transform(self, input_transform, input_scale=None, hadamard_block_size=None) -> None:
#         self.input_transform = input_transform
#         self.hadamard_block_size = hadamard_block_size
#         self._validate()
#         if input_transform == "scale" and input_scale is None:
#             raise ValueError("input_transform='scale' needs an input_scale")
#         device = self.bias.device if self.bias is not None else "cuda"
#         self.input_scale = None if input_scale is None else input_scale.detach().clone().flatten().to(device)

#     def _transform_input(self, x2d: torch.Tensor) -> torch.Tensor:
#         if self.input_transform is None:
#             return x2d
#         if self.input_transform == "scale":
#             if self.input_scale is None:
#                 raise RuntimeError("input_transform='scale' requires input_scale to be set")
#             return (x2d / self.input_scale.to(dtype=x2d.dtype, device=x2d.device)).contiguous()

#         from asrq.transforms.rotation.hadamard_utils import matmul_hadU_auto

#         if self.hadamard_block_size is None:
#             return matmul_hadU_auto(x2d).contiguous()
#         blocks = x2d.reshape(-1, self.in_features // self.hadamard_block_size, self.hadamard_block_size)
#         return matmul_hadU_auto(blocks).reshape(x2d.shape).contiguous()

#     def _matmul(self, x2d: torch.Tensor) -> torch.Tensor:
#         if self.abits == 16:
#             return self._humming_layer(x2d)
#         from humming import ops

#         xq, x_scale = ops.quant_input(x2d, f"int{self.abits}", group_size=0, scale_dtype="float32")
#         return self._humming_layer(inputs=xq, input_scale=x_scale)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if not self.quantized:
#             raise RuntimeError(
#                 "ASRQLinear must be quantized (call .quantize_() or build via .from_linear()) before use"
#             )
#         if self.conv1d_layout:
#             x = x.transpose(1, 2)
#         batch_shape = x.shape[:-1]
#         x2d = x.reshape(-1, self.in_features).half().contiguous()
#         out = self._matmul(self._transform_input(x2d)).reshape(*batch_shape, self.out_features)
#         return out.transpose(1, 2) if self.conv1d_layout else out

#     def extra_repr(self) -> str:
#         group = "per-channel" if self.group_size == -1 else f"group={self.group_size}"
#         mode = "sym" if self.symmetric else "asym"
#         extra = ", layout=conv1d" if self.conv1d_layout else ""
#         if self.input_transform is not None:
#             extra += f", input_transform={self.input_transform}"
#             if self.hadamard_block_size is not None:
#                 extra += f"(block={self.hadamard_block_size})"
#         return (f"in_features={self.in_features}, out_features={self.out_features}, "
#                 f"bias={self.bias is not None}, w{self.wbits}a{self.abits} ({mode}, {group}){extra}")


# class LinearQ(nn.Module):
#     def __init__(self, in_features, out_features, bits, group_size=128,bias=True):
#         super(LinearQ, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.group_size = group_size
#         weight = torch.zeros((out_features, in_features//(8//bits)), dtype=torch.uint8)
#         self.register_buffer('weight', weight)
#         if bias:
#             self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
#         else:
#             self.bias = None
#         scales = torch.zeros((out_features, (in_features+group_size-1)//group_size), dtype=torch.float16)
#         zeros = torch.zeros((out_features, (in_features+group_size-1)//group_size), dtype=torch.float16)
#         self.bits = bits
#         self.register_buffer("scales", scales)
#         self.register_buffer("zeros", zeros)

#     def load_quantized_params(self, scales, zeros, quantized_weight):
#         self.scales = scales.to(self.scales.device)
#         self.zeros = zeros.to(self.zeros.device)
#         self.weight = quantized_weight.to(self.weight.device)

#     def dequantize_weight(self, device="cpu", dtype=torch.float32):
#         group_size = self.group_size
#         W_dequant = torch.zeros((self.out_features, self.in_features), device=device, dtype=dtype)

#         scale = self.scales.repeat_interleave(group_size, dim=1).to(device).to(dtype)
#         zero = self.zeros.repeat_interleave(group_size, dim=1).to(device).to(dtype)
#         if self.bits == 8 or self.bits ==3:
#             W_dequant = self.weight.to(dtype) * scale[:,:self.in_features] + zero[:,:self.in_features]

#         elif self.bits == 4:
#             W_dequant[:, 0::2] = (self.weight & 0x0F).to(dtype)
#             W_dequant[:, 1::2] = (self.weight >> 4).to(dtype)
#             W_dequant = W_dequant * scale[:,:self.in_features] + zero[:,:self.in_features]

#         elif self.bits == 2:
#             W_dequant[:, 0::4] = (self.weight & 0x03).to(dtype)
#             W_dequant[:, 1::4] = ((self.weight >> 2) & 0x03).to(dtype)
#             W_dequant[:, 2::4] = ((self.weight >> 4) & 0x03).to(dtype)
#             W_dequant[:, 3::4] = ((self.weight >> 6) & 0x03).to(dtype)
#             W_dequant = W_dequant * scale[:,:self.in_features] + zero[:,:self.in_features]
#         return W_dequant

#     def forward(self, input):
#         x = input
#         W_dequant = self.dequantize_weight(x.device, x.dtype)

#         output = torch.matmul(x, W_dequant.t())
#         if self.bias is not None:
#             output += self.bias

#         return output


class ASRQLinear(nn.Module):
    """A linear or pointwise-convolution layer computed by humming's low-bit kernels.

    Weights are symmetric ``weight_bits`` integers with one fp16 scale per ``weight_group_size``
    inputs. Activations are either left in fp16 (``activation_bits`` 16) or quantized on the fly
    to ``activation_bits`` integers, with one scale per token (``activation_group_size`` 0) or one
    per group of that many features -- the same granularities evaluation fakes, chosen per layer
    from its activation role.

    The layer an online Hadamard feeds (fc2 after a rotation) takes it here rather than as a
    separate module: humming applies the Hadamard and the input quantization in one call, which
    measured +1-3% over no Hadamard against +25-40% as a module in front. The random signs of a
    randomized Hadamard come from a seed that humming's input kernel hashes per channel
    (``hadamard_sign_seed``), so they cost nothing; a stored sign vector from an older checkpoint is
    multiplied just before the call instead. The Hadamard, signs included, must already be absorbed
    into the weight passed in, as a folded rotation leaves it.

    humming runs in fp16 only. Evaluation casts a whole model to bfloat16, so the layer keeps its
    own tensors in fp16 through such casts, converts its input to fp16 and returns the output in
    the input's dtype.

    Layers with quantized activations run humming's batch-invariant mode. Its default schedules for
    some shapes use Stream-K, which accumulates the output concurrently across CUDA streams; the
    completion order varies, so fp16 sums round differently between identical calls (one to two ulps
    per element), and through 32 layers that moved whisper-large-v3 W4A4's encoder output 11% between
    runs and WER by up to 0.07. Batch-invariant schedules disable Stream-K: outputs are bit-identical
    across runs and W4A4 measured 1-6% faster end to end on an RTX A6000 (batch 1-64). Layers with
    fp16 activations keep the default schedules: batch-invariant ones made the W4A16 and W2A16
    decoding step at batch size 1 slower than fp16 (0.78x), the default ones faster (1.10x, 1.14x).
    Their default schedules are nondeterministic for some shapes (a 1280x1280 layer at 64 tokens).

    Build it with from_linear rather than the constructor, which leaves the weights uninitialised.

    Args:
        in_features, out_features: Layer shape.
        bias: Whether the layer has a bias.
        weight_bits: Weight bit width.
        weight_group_size: Inputs per weight scale; 0 or -1 for one scale per output channel.
        activation_bits: Activation bit width; 16 or more keeps activations in fp16.
        activation_group_size: Features per activation scale; 0 or -1 for one scale per token.
        hadamard_block_size: Block size of an online Hadamard on the input, or None.
        hadamard_signs: Optional ``(in_features,)`` +-1 vector applied before that Hadamard.
        conv1d_layout: True for a pointwise Conv1d, whose input is (batch, channels, time).
        hadamard_sign_seed: Seed of the Hadamard's signs, generated inside humming's kernel; 0 for none.
            Takes the place of hadamard_signs, which must then be None.
    """

    BATCH_INVARIANT = json.dumps({"use_batch_invariant": True})

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 4,
        weight_group_size: int = 128,
        activation_bits: int = 16,
        activation_group_size: int = 0,
        hadamard_block_size: Optional[int] = None,
        hadamard_signs: Optional[torch.Tensor] = None,
        conv1d_layout: bool = False,
        hadamard_sign_seed: int = 0,
    ):
        super().__init__()
        if hadamard_sign_seed and hadamard_signs is not None:
            raise ValueError("pass either hadamard_signs or hadamard_sign_seed, not both")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.weight_group_size = max(weight_group_size, 0)
        self.activation_bits = min(activation_bits, 16)
        self.activation_group_size = max(activation_group_size, 0)
        self.hadamard_block_size = hadamard_block_size
        self.hadamard_sign_seed = int(hadamard_sign_seed)
        self.conv1d_layout = conv1d_layout
        self._validate()

        input_config = {}
        if self.activation_bits < 16:
            input_config = {"dtype": f"int{self.activation_bits}"}
            if self.activation_group_size:
                input_config.update(quant_mode="dynamic_group", group_size=activation_group_size)
            else:
                input_config.update(quant_mode="dynamic_token")
        self.compute_config = self.BATCH_INVARIANT if self.activation_bits < 16 else None
        self.humming = HummingLayer(
            shape_n=out_features,
            shape_k=in_features,
            weight_config={"dtype": f"int{weight_bits}", "group_size": self.weight_group_size},
            input_config=input_config,
            has_bias=bias,
            torch_dtype=torch.float16,
        )
        signs = None if hadamard_signs is None else hadamard_signs.detach().to(torch.float16)
        self.register_buffer("hadamard_signs", signs)

    def _validate(self) -> None:
        if self.weight_group_size and self.in_features % self.weight_group_size:
            raise ValueError(f"weight_group_size {self.weight_group_size} does not divide "
                             f"in_features {self.in_features}")
        if self.activation_bits < 16:
            if self.activation_bits not in (4, 8):
                raise ValueError(f"activation_bits must be 4, 8 or 16, got {self.activation_bits}")
            size = self.activation_group_size
            if size and (self.in_features % size or size < 256 // self.activation_bits):
                raise ValueError(
                    f"activation_group_size {size} must divide in_features {self.in_features} and "
                    f"be at least {256 // self.activation_bits} for {self.activation_bits}-bit "
                    f"activations"
                )
        if self.hadamard_block_size is not None and self.in_features % self.hadamard_block_size:
            raise ValueError(f"hadamard_block_size {self.hadamard_block_size} does not divide "
                             f"in_features {self.in_features}")

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        weight_bits: int = 4,
        weight_group_size: int = 128,
        activation_bits: int = 16,
        activation_group_size: int = 0,
        hadamard_block_size: Optional[int] = None,
        hadamard_signs: Optional[torch.Tensor] = None,
        hadamard_sign_seed: int = 0,
    ) -> "ASRQLinear":
        """Quantize an nn.Linear or pointwise nn.Conv1d into an ASRQLinear on the GPU.

        The weight is quantized by humming with symmetric group scales. For a layer that was
        already fake-quantized to the same grid, such as by GPTQ, that reproduces its values.
        """
        is_conv = isinstance(layer, nn.Conv1d)
        if is_conv and (layer.kernel_size != (1,) or layer.groups != 1):
            raise ValueError("only pointwise Conv1d layers (kernel_size 1, groups 1) are supported")
        if not is_conv and not isinstance(layer, nn.Linear):
            kind = type(layer).__name__
            raise TypeError(f"expected nn.Linear or pointwise nn.Conv1d, got {kind}")
        weight = layer.weight.detach()
        weight = weight.squeeze(-1) if is_conv else weight
        module = cls(
            weight.shape[1], weight.shape[0], bias=layer.bias is not None,
            weight_bits=weight_bits, weight_group_size=weight_group_size,
            activation_bits=activation_bits, activation_group_size=activation_group_size,
            hadamard_block_size=hadamard_block_size, hadamard_signs=hadamard_signs,
            conv1d_layout=is_conv, hadamard_sign_seed=hadamard_sign_seed,
        ).cuda()
        with torch.no_grad():
            module.humming.load_from_unquantized(weight.to(device="cuda", dtype=torch.float16))
            if layer.bias is not None:
                bias = layer.bias.detach().to(device="cuda", dtype=torch.float16)
                module.humming.bias.copy_(bias)
        module.humming.transform()
        return module

    def _apply(self, fn, recurse=True):
        # Keep the fp16 originals: casting to bfloat16 and back would round every scale to
        # bfloat16's 7 fraction bits, which moved the output by 5%. Only the device may change.
        originals = [
            (store, name, tensor.data)
            for module in self.modules()
            for store in (module._parameters, module._buffers)
            for name, tensor in store.items()
            if tensor is not None and tensor.is_floating_point()
        ]
        super()._apply(fn, recurse)
        for store, name, original in originals:
            store[name].data = original.to(device=store[name].device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.conv1d_layout:
            x = x.transpose(1, 2)
        leading = x.shape[:-1]
        x = x.reshape(-1, self.in_features).to(torch.float16)
        if self.hadamard_signs is not None:
            x = x * self.hadamard_signs
        out = self.humming(x.contiguous(), hadamard_block_size=self.hadamard_block_size,
                           hadamard_sign_seed=self.hadamard_sign_seed, compute_config=self.compute_config)
        out = out.reshape(*leading, self.out_features).to(dtype)
        return out.transpose(1, 2) if self.conv1d_layout else out

    def extra_repr(self) -> str:
        weight_groups = f"g{self.weight_group_size}" if self.weight_group_size else "per-channel"
        if self.activation_bits >= 16:
            activations = "A16"
        elif self.activation_group_size:
            activations = f"A{self.activation_bits} g{self.activation_group_size}"
        else:
            activations = f"A{self.activation_bits} per-token"
        hadamard = ""
        if self.hadamard_block_size:
            hadamard = f", hadamard={self.hadamard_block_size}"
            hadamard += "+signs" if self.hadamard_signs is not None else ""
            hadamard += f"+signs(seed {self.hadamard_sign_seed})" if self.hadamard_sign_seed else ""
        return (f"in={self.in_features}, out={self.out_features}, W{self.weight_bits} "
                f"{weight_groups}, {activations}{hadamard}, conv1d={self.conv1d_layout}")


def _parent_and_attribute(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parent = model
    *path, attribute = name.split(".")
    for part in path:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, attribute


def replace_with_asrq_linear(
    model: nn.Module,
    layers: Mapping[str, Tuple[int, int, int]],
    weight_group_size: int,
    online_hadamards: Optional[Mapping[str, str]] = None,
) -> Dict[str, ASRQLinear]:
    """Swap named layers for ASRQLinear, in place.

    A layer fed by an online Hadamard module -- an activation wrapped as
    ``nn.Sequential(activation, hadamard)`` by a folded rotation -- takes that Hadamard into its
    ASRQLinear, where humming fuses it with the input quantization, and the activation is
    unwrapped back to the bare activation. A layer carrying the Hadamard at its own input, as an
    ``input_hadamard`` child run by a pre-hook (a SwiGLU down projection), takes it the same way;
    the replaced layer's hook goes with it. The Hadamard module is recognised by its
    ``block_size`` and ``signs`` attributes, so this module does not depend on the rotation code; its
    ``sign_seed``, when set, is passed on so humming generates the signs itself.

    Args:
        model: The model to modify.
        layers: ``{layer_name: (weight_bits, activation_bits, activation_group_size)}``.
        weight_group_size: Inputs per weight scale.
        online_hadamards: ``{layer_name: activation_name}`` for layers behind an online Hadamard;
            an activation_name of None means the Hadamard sits at the layer's own input.

    Returns:
        ``{layer_name: the new ASRQLinear}``.
    """
    online_hadamards = online_hadamards or {}
    replaced = {}
    for name, (weight_bits, activation_bits, activation_group_size) in layers.items():
        block_size = signs = None
        seed = 0
        if online_hadamards.get(name) is not None:
            act_parent, act_attribute = _parent_and_attribute(model, online_hadamards[name])
            activation = getattr(act_parent, act_attribute)
            if isinstance(activation, nn.Sequential) and hasattr(activation[-1], "block_size"):
                block_size, signs = activation[-1].block_size, activation[-1].signs
                seed = getattr(activation[-1], "sign_seed", 0)
                unwrapped = activation[0] if len(activation) == 2 else activation[:-1]
                setattr(act_parent, act_attribute, unwrapped)
        parent, attribute = _parent_and_attribute(model, name)
        layer = parent[int(attribute)] if attribute.isdigit() else getattr(parent, attribute)
        input_hadamard = getattr(layer, "input_hadamard", None)
        if hasattr(input_hadamard, "block_size"):
            block_size, signs = input_hadamard.block_size, input_hadamard.signs
            seed = getattr(input_hadamard, "sign_seed", 0)
        new = ASRQLinear.from_linear(
            layer, weight_bits, weight_group_size, activation_bits, activation_group_size,
            hadamard_block_size=block_size, hadamard_signs=None if seed else signs, hadamard_sign_seed=seed,
        )
        if attribute.isdigit():
            parent[int(attribute)] = new
        else:
            setattr(parent, attribute, new)
        replaced[name] = new
    return replaced
