"""Activation fake quantization, shared by the rotation search and evaluation.

Both use the same quantizer on the same layers, configured from the same settings, so a rotation is
searched against exactly the activation quantization it will be evaluated with.

Which layers are quantized, and how, is decided by each layer's role:

    q, k, v    attention query, key and value projections
    attn_out   the attention output projection
    fc1        the first layer of a feed-forward or pointwise-convolution module
    fc2        the second one, which reads a nonlinearity's output
    block_out  the Linear a rotation inserts after a Conformer block's output norm, when quantized

A model supplies the ``{layer_name: role}`` mapping; the configuration names which roles are
quantized group-wise, and every other role is quantized per token. A role says which basis a
layer's input is in when rotations are applied, which is what decides the granularity it needs:
q, k, v, fc1 and block_out read the R1-rotated residual stream, attn_out reads the head-wise R2
basis, and fc2 reads the output of the online Hadamard.
"""

from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Callable, Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn

ACTIVATION_ROLES = ("q", "k", "v", "attn_out", "fc1", "fc2", "block_out")


def fake_quantize_activations(
    x: torch.Tensor, bits: int, group_size: int, symmetric: bool
) -> torch.Tensor:
    """Quantize and dequantize x group-wise along its last axis, with a straight-through gradient.

    Each token is split into groups of group_size consecutive features, and every group gets
    its own scale (and zero point, if asymmetric) from its own range. Groups never span tokens,
    so padding cannot change another token's scale.

    Symmetric, with q_max = 2**(bits - 1) - 1:

        s = max|x_g| / q_max,  x_g <- clamp(round(x_g / s), -q_max - 1, q_max) * s

    Asymmetric, with q_max = 2**bits - 1:

        s = (max x_g - min x_g) / q_max,  z = round(-min x_g / s)
        x_g <- (clamp(round(x_g / s) + z, 0, q_max) - z) * s

    The gradient is the straight-through estimator, x + (q(x) - x).detach(): rounding has zero
    gradient almost everywhere, so without it nothing would reach the rotations upstream. The
    forward value is exactly q(x).

    The arithmetic runs in at least float32 and is cast back to x's dtype; half-precision scales
    of small groups would otherwise round to zero.

    Args:
        x: Activations with features on the last axis.
        bits: Bit width.
        group_size: Features per group. Must divide the feature width. 0 or -1 quantizes each
            token as a single group.
        symmetric: Symmetric around zero if True, otherwise with a per-group zero point.

    Returns:
        The fake-quantized activations, in x's shape and dtype.
    """
    width = x.shape[-1]
    group = width if group_size in (0, -1) else group_size
    if width % group:
        raise ValueError(f"feature width {width} is not divisible by group_size {group}")

    dtype = torch.promote_types(x.dtype, torch.float32)
    groups = x.to(dtype).reshape(*x.shape[:-1], width // group, group)
    if symmetric:
        q_max = 2 ** (bits - 1) - 1
        scale = (groups.abs().amax(dim=-1, keepdim=True) / q_max).clamp_min(1e-8)
        dequantized = torch.clamp(torch.round(groups / scale), -q_max - 1, q_max) * scale
    else:
        q_max = 2**bits - 1
        low = groups.amin(dim=-1, keepdim=True)
        high = groups.amax(dim=-1, keepdim=True)
        scale = ((high - low) / q_max).clamp_min(1e-8)
        zero = torch.round(-low / scale)
        dequantized = (torch.clamp(torch.round(groups / scale) + zero, 0, q_max) - zero) * scale

    quantized = dequantized.reshape(x.shape).to(x.dtype)
    return x + (quantized - x).detach()


def make_activation_quantizer(
    bits: int, group_size: int, symmetric: bool
) -> Callable[[torch.Tensor], torch.Tensor]:
    """fake_quantize_activations with its settings bound."""
    return partial(
        fake_quantize_activations, bits=bits, group_size=group_size, symmetric=symmetric
    )


class ActivationQuantizer:
    """A switchable fake quantizer.

    Everything holding a reference to one instance -- patched forwards during a rotation search,
    input hooks during evaluation -- is switched by flipping ``enabled``, without patching
    again. That is what lets one model act as both sides of a distillation objective: the same
    weights and rotations, once with quantized activations and once without.

    Args:
        bits: Bit width.
        group_size: Features per group; 0 or -1 for one group per token.
        symmetric: Symmetric around zero if True, otherwise with a per-group zero point.
    """

    def __init__(self, bits: int, group_size: int, symmetric: bool):
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.enabled = True

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return fake_quantize_activations(x, self.bits, self.group_size, self.symmetric)

    def __repr__(self) -> str:
        granularity = "per-token" if self.group_size in (0, -1) else f"group={self.group_size}"
        mode = "sym" if self.symmetric else "asym"
        return f"ActivationQuantizer(A{self.bits}, {granularity}, {mode})"

    @contextmanager
    def disabled(self):
        """Run the enclosed forward passes in full precision."""
        previous, self.enabled = self.enabled, False
        try:
            yield
        finally:
            self.enabled = previous


def build_activation_quantizers(
    layer_roles: Mapping[str, Optional[str]],
    bits: int,
    group_size: int,
    symmetric: bool,
    groupwise_roles: Optional[Iterable[str]] = None,
) -> Dict[str, ActivationQuantizer]:
    """One activation quantizer per layer: group-wise for the requested roles, per token otherwise.

    A layer whose role is None is quantized per token. That is the fallback for a model that
    lists the layers to quantize without saying what they are.

    Layers with the same granularity share one quantizer instance, so switching one switches
    all of them.

    Args:
        layer_roles: ``{layer_name: role}``; roles from ACTIVATION_ROLES, or None.
        bits: Bit width.
        group_size: Features per group for the group-wise roles.
        symmetric: Symmetric around zero if True, otherwise with a per-group zero point.
        groupwise_roles: Roles quantized group-wise. None makes every role group-wise; an empty
            list makes every layer per token.

    Returns:
        ``{layer_name: quantizer}`` for every layer in layer_roles.
    """
    groupwise = set(ACTIVATION_ROLES if groupwise_roles is None else groupwise_roles)
    unknown = (groupwise | {r for r in layer_roles.values() if r is not None}) - set(
        ACTIVATION_ROLES
    )
    if unknown:
        raise ValueError(f"unknown activation roles {sorted(unknown)}; expected {ACTIVATION_ROLES}")

    by_group_size: Dict[int, ActivationQuantizer] = {}
    quantizers = {}
    for name, role in layer_roles.items():
        size = group_size if role in groupwise else -1
        if size not in by_group_size:
            by_group_size[size] = ActivationQuantizer(bits, size, symmetric)
        quantizers[name] = by_group_size[size]
    return quantizers


@contextmanager
def quantizers_disabled(quantizers: Mapping[object, ActivationQuantizer]):
    """Switch off every quantizer in the mapping for the enclosed forward passes."""
    with ExitStack() as stack:
        for quantizer in {id(q): q for q in quantizers.values()}.values():
            stack.enter_context(quantizer.disabled())
        yield


def _module_at(model: nn.Module, name: str) -> nn.Module:
    module = model
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def attach_activation_quantization(
    model: nn.Module, quantizers: Mapping[str, Callable[[torch.Tensor], torch.Tensor]]
) -> List[torch.utils.hooks.RemovableHandle]:
    """Fake-quantize each named layer's input with a forward pre-hook.

    A pre-hook rather than a replaced forward, so the layer keeps its own forward: whatever it is
    -- a Linear, a pointwise Conv1d, a layer patched by a transform -- the quantizer sees exactly
    the tensor that layer receives. For fc2 behind an online Hadamard that is the rotated input,
    which is the basis its weight was folded for.

    A Conv1d reads (batch, channels, time), so its input is quantized along the channel axis.

    Args:
        model: The model to modify in place.
        quantizers: ``{layer_name: fn(x) -> x}``, as build_activation_quantizers returns.

    Returns:
        The hook handles.
    """
    handles = []
    for name, quantize in quantizers.items():
        module = _module_at(model, name)
        channels_first = isinstance(module, nn.Conv1d)

        def hook(_module, args, quantize=quantize, channels_first=channels_first):
            x = args[0]
            if channels_first:
                x = quantize(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = quantize(x)
            return (x, *args[1:])

        handles.append(module.register_forward_pre_hook(hook))
    return handles


def activation_fn_scaler(act_fn, scale):
    class ScaledActivation(nn.Module):
        def __init__(self, act_fn, scale):
            super(ScaledActivation, self).__init__()
            self.act_fn = act_fn
            self.scale = scale

        def forward(self, x):
            return self.act_fn(x) * self.scale

    return ScaledActivation(act_fn, scale)
