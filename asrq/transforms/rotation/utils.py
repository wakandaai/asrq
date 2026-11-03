import types
from contextlib import contextmanager
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from asrq import tracking
import torch.nn as nn
import torch.nn.functional as F
from humming import ops

from asrq.quantizers.activation import (  # noqa: F401  re-exported for the rotation search
    ActivationQuantizer,
    build_activation_quantizers,
    fake_quantize_activations,
    make_activation_quantizer,
    quantizers_disabled,
)
from asrq.transforms.rotation.cayley_sgd import SGDG
from asrq.transforms.rotation.hadamard_utils import random_hadamard_matrix



def _weight_2d(layer: Union[nn.Linear, nn.Conv1d]) -> torch.Tensor:
    """Return an (out_features, in_features) view of a Linear or pointwise Conv1d weight.

    A pointwise Conv1d stores its weight as (out_channels, in_channels, 1) and applies it
    along the channel axis, so dropping the kernel axis yields the equivalent matrix. The
    result is a view, so in-place writes to it update the layer's weight.
    """
    if isinstance(layer, nn.Conv1d):
        if layer.kernel_size != (1,) or layer.groups != 1:
            raise ValueError(
                f"only pointwise Conv1d (kernel_size=1, groups=1) can be folded, got "
                f"kernel_size={layer.kernel_size}, groups={layer.groups}"
            )
        return layer.weight.data.squeeze(-1)
    return layer.weight.data


def _rotate_heads(t: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Right-multiply t by blockdiag(R2, ..., R2) along its last axis.

    R2 is a head-dimension rotation: it is head_dim x head_dim and acts on each attention head
    separately, so on a d_model-wide tensor it acts as a block-diagonal matrix with one R2 per
    head. Reshaping the last axis into (num_heads, head_dim) applies it without materialising
    that block-diagonal matrix. The number of heads is inferred from the sizes.

    With R2 as wide as the axis itself there is a single block and this reduces to ``t @ R2``.
    """
    head_dim = R2.shape[0]
    if t.shape[-1] % head_dim:
        raise ValueError(f"axis of size {t.shape[-1]} is not divisible by head_dim {head_dim}")
    blocks = t.reshape(*t.shape[:-1], t.shape[-1] // head_dim, head_dim)
    return (blocks @ R2).reshape(t.shape)


class _RMSNormWithScale(nn.Module):
    """A LayerNorm rewritten as an RMSNorm, with centering and shift folded into its neighbours.

    For a row vector x of width n, let M = I - (1/n) * ones be the mean subtraction (centering)
    matrix. M is symmetric and idempotent, and X @ M == X - mean(X, dim=-1, keepdim=True) for a
    batch of row vectors X of shape (..., n). Because mean((x - mean(x))**2) == mean((x @ M)**2),
    a LayerNorm decomposes exactly as

        LayerNorm(x) = RMSNorm(x @ M) * gamma + beta

    so it becomes an RMSNorm once M and beta are pushed into the surrounding layers. Only the
    scale gamma is kept here.

    nn.Linear computes X @ W.T + b on batched row vectors, so centering linear1's output gives

        (X @ W1.T + b1) @ M = X @ (W1.T @ M) + b1 @ M = X @ (M @ W1).T + M @ b1

    using M.T == M, i.e. M folds into linear1 as W1 <- M @ W1 and b1 <- M @ b1. The shift then
    folds into linear2, since

        (RMSNorm(Xc) * gamma + beta) @ W2.T + b2
            = (RMSNorm(Xc) * gamma) @ W2.T + (beta @ W2.T + b2)

    i.e. b2 <- b2 + beta @ W2.T, creating the bias if linear2 has none.

    This must be applied before _ScaleFreeRMSNorm on the same next layers: the shift lands in their bias
    using their original weight, which _ScaleFreeRMSNorm's scale fold then leaves untouched. Reversed,
    the shift would be computed from the already-scaled weight and be wrong.

    Either list may hold several layers, and either may be empty:

    - Several previous layers: the norm's input is their sum, and M is linear, so centering
      the sum is the same as centering each contribution. M folds into every one of them.
    - Several next layers: each reads the norm's output independently, so the shift folds into
      each of their biases separately.
    - No previous layers: the centering is not folded anywhere, so the conversion is not
      output-equivalent on its own. This covers a norm reading the residual stream, or one
      whose input passes through a nonlinearity a centering cannot be folded back through.
      Centering that input is then the caller's responsibility.
    - No next layers: the shift has nowhere to go, so it is kept here and applied in forward.

    Args:
        layer_norm: The LayerNorm instance to convert.
        previous_layers: Layers whose outputs are summed to form this LayerNorm's input.
        next_layers: Layers that consume this LayerNorm's output.
    """

    def __init__(
        self,
        layer_norm: nn.LayerNorm,
        previous_layers: Sequence[Union[nn.Linear, nn.Conv1d]],
        next_layers: Sequence[Union[nn.Linear, nn.Conv1d]],
    ):
        super().__init__()
        self.eps = layer_norm.eps
        self.normalized_shape = layer_norm.normalized_shape
        self.register_buffer("scale", layer_norm.weight.data.clone())

        dim = self.normalized_shape[0]
        device, dtype = layer_norm.weight.device, layer_norm.weight.dtype
        M = torch.eye(dim, device=device, dtype=dtype) - torch.full(
            (dim, dim), 1.0 / dim, device=device, dtype=dtype
        )

        for previous in previous_layers:
            M1 = M.to(dtype=previous.weight.dtype, device=previous.weight.device)
            w = _weight_2d(previous)
            w.copy_(M1 @ w)
            if previous.bias is not None:
                previous.bias.data.copy_(M1 @ previous.bias.data)

        for nxt in next_layers:
            beta = layer_norm.bias.data.to(dtype=nxt.weight.dtype, device=nxt.weight.device)
            shift = beta @ _weight_2d(nxt).T
            if nxt.bias is not None:
                nxt.bias.data.copy_(nxt.bias.data + shift)
            else:
                nxt.bias = nn.Parameter(shift)

        self.register_buffer(
            "shift", None if next_layers else layer_norm.bias.data.clone()
        )

    def forward(self, x):
        normed = F.rms_norm(x, self.normalized_shape, self.scale, self.eps)
        return normed if self.shift is None else normed + self.shift



def _is_hf_rmsnorm(module: nn.Module) -> bool:
    """A Hugging Face-style RMSNorm (e.g. Qwen3RMSNorm): a ``weight`` scale and ``variance_epsilon``."""
    return hasattr(module, "variance_epsilon") and isinstance(getattr(module, "weight", None), torch.Tensor)


class _ScaleFreeRMSNorm(nn.Module):
    """An RMSNorm reduced to pure normalization, with its scale folded into the next layer.

    nn.Linear computes Y @ W.T + b on batched row vectors and the RMSNorm output row is
    (normed * gamma), so

        (normed * gamma) @ W.T + b = normed @ (W @ diag(gamma)).T + b

    i.e. gamma folds into the next layer as W <- W * gamma, broadcast over W's in_features
    axis, leaving the bias unchanged. The scale is read from .scale for an _RMSNormWithScale or
    .weight for an nn.RMSNorm or a Hugging Face-style RMSNorm.

    A Hugging Face-style RMSNorm (Qwen3RMSNorm) normalizes in float32 whatever its input dtype,
    because squaring a half-precision residual stream with large activations overflows; the
    converted norm keeps doing that.

    On the same next layers this must be applied after _RMSNormWithScale, whose shift fold depends
    on the unscaled weight. Each next layer reads the norm's output independently, so the
    scale folds into every one of them.

    A norm with no next layers cannot be converted: the scale has nowhere to go and must stay
    in the norm, so leave it as an _RMSNormWithScale rather than calling this.

    Scale recovery (asrq.quantizers.scale_recovery) can give it a per-channel ``weight`` buffer again, after the
    next layers are quantized; the output is then multiplied by it.

    Args:
        rms_norm: The RMSNorm instance to convert.
        next_layers: Layers that consume this RMSNorm's output. Must not be empty.
    """

    def __init__(
        self,
        rms_norm: Union[nn.RMSNorm, _RMSNormWithScale, nn.Module],
        next_layers: Sequence[Union[nn.Linear, nn.Conv1d]],
    ):
        super().__init__()
        if not next_layers:
            raise ValueError(
                "a norm with no next layers has nowhere to fold its scale; keep it as an "
                "_RMSNormWithScale instead of converting it"
            )
        self.upcast = _is_hf_rmsnorm(rms_norm)
        if self.upcast:
            self.eps = rms_norm.variance_epsilon
            self.normalized_shape = tuple(rms_norm.weight.shape)
        else:
            self.eps = rms_norm.eps
            self.normalized_shape = rms_norm.normalized_shape

        scale = rms_norm.scale if isinstance(rms_norm, _RMSNormWithScale) else rms_norm.weight
        for nxt in next_layers:
            gamma = scale.data.to(dtype=nxt.weight.dtype, device=nxt.weight.device)
            _weight_2d(nxt).mul_(gamma)

    def forward(self, x):
        weight = getattr(self, "weight", None)
        if getattr(self, "upcast", False):
            weight = None if weight is None else weight.float()
            return F.rms_norm(x.float(), self.normalized_shape, weight, self.eps).to(x.dtype)
        return F.rms_norm(x, self.normalized_shape, None if weight is None else weight.to(x.dtype), self.eps)


ROT_INPUT_R1 = 1
ROT_INPUT_R1_OUTPUT_R2 = 2
ROT_INPUT_R2_OUTPUT_R1 = 3
ROT_OUTPUT_R1 = 4
ROT_INPUT_R1_OUTPUT_R1 = 5


def _rotate_weight(weight, bias, kind, R1, R2=None, H=None):
    """Return ``(weight, bias)`` rewritten so a layer applies the rotations its kind names.

    nn.Linear computes X @ W.T + b on batched row vectors, so rotating a layer's input or its
    output is exactly a rewrite of its weight and bias. The five kinds, with B the head-wise
    rotation blockdiag(R2, ..., R2):

        ROT_INPUT_R1            Linear(X R1.T)    = X @ (W @ R1).T + b
        ROT_INPUT_R1_OUTPUT_R2  Linear(X R1.T) B  = X @ (B.T @ W @ R1).T + b @ B
        ROT_INPUT_R2_OUTPUT_R1  Linear(X B.T) R1  = X @ (R1.T @ W @ B).T + b @ R1
        ROT_OUTPUT_R1           Linear(X) R1      = X @ (R1.T @ W).T + b @ R1
        ROT_INPUT_R1_OUTPUT_R1  Linear(X R1.T) R1 = X @ (R1.T @ W @ R1).T + b @ R1

    R1 is the residual-stream rotation, applied wherever a layer reads the stream (Q, K, V, FC1)
    or writes into it (the attention output projection, FC2). R2 is head_dim x head_dim and
    rotates each attention head separately, so on a d_model-wide axis it acts as B, which is
    never materialised; see _rotate_heads. ROT_INPUT_R1_OUTPUT_R1 is ROT_INPUT_R2_OUTPUT_R1 with
    B = R1, for a layer that both reads and writes the stream -- the Linear inserted after a
    Conformer block's output norm.

    ROT_OUTPUT_R1 is for a layer whose input cannot be rotated backwards, because a nonlinearity
    precedes it. H, the online Hadamard such a layer's input is rotated by instead, is folded in
    as W <- W @ H; with H @ H.T = I the layer then behaves as the unrotated one followed by R1,
    while its input is seen, and quantized, in the Hadamard basis.

    The arithmetic runs in the dtype promoted over the weight and every rotation given, and the
    result is returned in that dtype: a float64 rotation over a lower-precision weight keeps full
    precision through the matmul and in its gradient, which is what Cayley SGD needs to stay on
    the Stiefel manifold. Callers round once on the way back into the layer, so folding a
    rotation reproduces the patched model rather than drifting from it.

    Args:
        weight: The layer's ``(out_features, in_features)`` weight.
        bias: The layer's bias, or None.
        kind: One of the ROT_* rotation kinds.
        R1: The residual-stream rotation. May be an nn.Parameter.
        R2: The head-dimension rotation, for the two kinds that name it. May be an nn.Parameter.
        H: An orthogonal matrix applied to the layer's input as X @ H, for ROT_OUTPUT_R1.

    Returns:
        ``(weight, bias)`` in the promoted dtype; the bias is None when the layer has none.
    """
    if kind == ROT_INPUT_R1_OUTPUT_R1:
        R2 = R1
    if kind in (ROT_INPUT_R1_OUTPUT_R2, ROT_INPUT_R2_OUTPUT_R1) and R2 is None:
        raise ValueError(f"rotation type {kind} needs R2")
    dtype = weight.dtype
    for rotation in (R1, R2, H):
        if rotation is not None:
            dtype = torch.promote_types(dtype, rotation.dtype)
    device = weight.device
    R1 = R1.to(dtype=dtype, device=device)
    R2 = None if R2 is None else R2.to(dtype=dtype, device=device)
    w = weight.to(dtype)
    b = None if bias is None else bias.to(dtype)
    if kind == ROT_INPUT_R1:
        w = w @ R1
    elif kind == ROT_INPUT_R1_OUTPUT_R2:
        w = _rotate_heads(w.T, R2).T @ R1
        b = None if b is None else _rotate_heads(b, R2)
    elif kind in (ROT_INPUT_R2_OUTPUT_R1, ROT_INPUT_R1_OUTPUT_R1):
        w = R1.T @ _rotate_heads(w, R2)
        b = None if b is None else b @ R1
    elif kind == ROT_OUTPUT_R1:
        w = R1.T @ w
        b = None if b is None else b @ R1
    else:
        raise ValueError(f"Unsupported rotation type: {kind}")
    if H is not None:
        w = w @ H.to(dtype=dtype, device=device)
    return w, b


def fold_rotation(layer, kind, R1, R2=None):
    """Fold a rotation into a layer's weight and bias, in place.

    The inference counterpart of patch_rotation, and numerically equivalent to it: both build the
    rotated weight with _rotate_weight and round it once into the layer's dtype. An FC2's online
    Hadamard is absorbed separately, by absorb_hadamard_rotation_into_fc2, which applies humming's
    own kernel instead of a dense matrix.

    Args:
        layer: The Linear or pointwise Conv1d to fold into.
        kind: One of the ROT_* rotation kinds.
        R1: The residual-stream rotation.
        R2: The head-dimension rotation, for the kinds that name it.

    Returns:
        None. The layer's weight and bias are modified in place.
    """
    w = _weight_2d(layer)
    bias = None if layer.bias is None else layer.bias.data
    rotated, rotated_bias = _rotate_weight(w, bias, kind, R1, R2)
    w.copy_(rotated)
    if bias is not None:
        bias.copy_(rotated_bias)


def _transform_input(x, H, quantize_input, is_conv):
    """Apply the online Hadamard, then the input quantizer, on x's feature axis.

    A pointwise Conv1d reads (batch, channels, time), so its features are not on the last axis.
    Both transforms act on the features, so the axis is moved once for the pair and moved back.
    """
    if H is None and quantize_input is None:
        return x
    if is_conv:
        x = x.transpose(1, 2)
    if H is not None:
        x = x @ H.to(dtype=x.dtype, device=x.device)
    if quantize_input is not None:
        x = quantize_input(x)
    return x.transpose(1, 2) if is_conv else x


def patch_rotation(layer, kind, R1, R2=None, H=None, quantize_input=None):
    """Patch a layer's forward pass to apply its rotation, leaving its weight untouched.

    Equivalent to fold_rotation, but the rotated weight is rebuilt every forward, so R1 and R2
    stay parameters an optimiser can move. The layer's own weight is used as a Parameter rather
    than through _weight_2d, so gradients reach it as well; freeze the weights separately if only
    the rotations are being searched. _weight_2d is still called once here, to reject a
    non-pointwise Conv1d.

    R1 and R2 are captured by reference and are never copied or registered on the layer, so one
    shared nn.Parameter can be patched into every layer that reads the residual stream and
    optimised as a single rotation, as Cayley SGD over the Stiefel manifold needs. Registering
    them per layer would instead give each layer an independent copy that diverges during the
    search.

    The replacement is bound with types.MethodType rather than assigned as a bare closure over
    the layer. Both shadow the class method on this instance only, but a bound method is rebound
    to the copy by copy.deepcopy, whereas a closure would keep pointing at the original layer and
    silently ignore the copy's weights. R1 and R2 stay shared across copies, which is what the
    search wants.

    Patching a layer again replaces this forward rather than stacking on it, so calling it a
    second time with a different quantizer replaces the quantizer.

    Args:
        layer: The Linear or pointwise Conv1d to patch.
        kind: One of the ROT_* rotation kinds.
        R1: The residual-stream rotation. May be an nn.Parameter.
        R2: The head-dimension rotation, for the kinds that name it. May be an nn.Parameter.
        H: The online Hadamard applied to the layer's input as X @ H, for ROT_OUTPUT_R1. A dense
            matrix rather than humming's kernel, which is not differentiable; see
            online_hadamard_matrix. Fixed, not searched.
        quantize_input: Optional ``fn(x) -> x`` fake-quantizing the layer's input, features last,
            so the search sees the quantization error the deployed layer will. That input is
            already in the basis the layer's input quantizer will see: R1's for a layer reading
            the residual stream, R2's for the attention output projection -- which is what makes
            R2 matter to the search at all -- and H's for an FC2 behind an online Hadamard. See
            fake_quantize_activations.

    Returns:
        None. The layer is modified in place.
    """
    _weight_2d(layer)
    is_conv = isinstance(layer, nn.Conv1d)

    def forward_with_rotation(self, x):
        weight = self.weight.squeeze(-1) if is_conv else self.weight
        out_dtype = weight.dtype
        w, bias = _rotate_weight(weight, self.bias, kind, R1, R2, H)
        w = w.to(out_dtype)
        bias = None if bias is None else bias.to(out_dtype)
        x = _transform_input(x, H, quantize_input, is_conv)
        if is_conv:
            return F.conv1d(x, w.unsqueeze(-1), bias)
        return F.linear(x, w, bias)

    layer.forward = types.MethodType(forward_with_rotation, layer)


def absorb_hadamard_rotation_into_fc2(
    linear: Union[nn.Linear, nn.Conv1d],
    block_size: int,
    signs: Optional[torch.Tensor] = None,
):
    """Absorb humming's block-diagonal Hadamard into the weight of an FC2 layer.

    Linear(XH.T) = X @ H.T @ W.T + b
                 = X @ (W @ H).T + b

    H is never materialised. The same humming kernel that will run online is applied directly
    to the weight, whose last axis is in_features, so it computes W @ H by construction. That
    is what makes the pair exact: the absorbed matrix cannot drift from the one applied online,
    whatever orientation or normalisation the kernel uses.

    The product runs in float32, the widest humming accepts, and is rounded once on the way
    back into the weight, so the layer's dtype is unchanged.

    With signs, the rotation is the randomized Hadamard D @ H, D = diag(signs), and the weight
    becomes W @ D @ H; OnlineHadamard applies the matching X @ D @ H. See random_hadamard_signs
    for why FC2 needs it.

    Args:
        linear: The FC2 layer to absorb the Hadamard into.
        block_size: Hadamard block size, matching the value used online. humming requires a
            power of two, at most 512, that divides in_features. Setting it equal to the
            activation quantization group size gives the best quantization SNR.
        signs: Optional ``(in_features,)`` vector of +-1, the diagonal of D.

    Returns:
        None. The linear layer's weight is modified in place.
    """
    w = _weight_2d(linear)
    weight = w.to(torch.float32)
    if signs is not None:
        weight = weight * signs.to(device=weight.device, dtype=weight.dtype)
    rotated, _, _ = ops.process_input(weight.contiguous(), hadamard_block_size=block_size)
    w.copy_(rotated)


def block_random_hadamard_matrix(size: int, block_size: int, device) -> torch.Tensor:
    """A block-diagonal orthogonal matrix of independent random Hadamard blocks.

    Used as a residual-stream rotation that only mixes channels within consecutive blocks. With
    the block size equal to the activation quantization group size, each block lines up with
    one quantization group: an outlier channel is spread across the channels of its own group,
    lowering that group's scale, without leaking into the other groups the way a full-width
    rotation does. Any orthogonal R1 keeps the rewrite exact, so nothing else changes.

    Args:
        size: Width of the matrix.
        block_size: Width of each block; must divide size.
        device: Device to build the matrix on.

    Returns:
        A ``(size, size)`` float64 matrix.
    """
    if size % block_size:
        raise ValueError(f"block_size {block_size} does not divide size {size}")
    blocks = [random_hadamard_matrix(block_size, device) for _ in range(size // block_size)]
    return torch.block_diag(*blocks)


def block_diagonal_mask(size: int, block_size: int, device) -> torch.Tensor:
    """Boolean ``(size, size)`` mask that is True inside the diagonal blocks."""
    block_ids = torch.arange(size, device=device) // block_size
    return block_ids.unsqueeze(0) == block_ids.unsqueeze(1)


def get_parent_module(model, module_name):
    """Get the parent module of a given module name in a model.

    Args:
        model: The model containing the module.
        module_name: The name of the module whose parent is to be found.

    Returns:
        The parent module of the specified module.
    """
    parent_module = model
    for name in module_name.split(".")[:-1]:
        if name.isdigit():
            parent_module = parent_module[int(name)]
        else:
            parent_module = getattr(parent_module, name)
    return parent_module

def online_hadamard_matrix(in_features: int, block_size: int, device, dtype=torch.float32):
    """Materialise, as a dense matrix, the block-diagonal Hadamard humming applies online.

    humming's kernel is not differentiable, so a rotation search cannot call it. H is only a
    constant in the forward pass though, so it can be built once here and applied as a plain
    matmul while R is being learned, then dropped at inference in favour of the fused kernel.
    Building it from the kernel rather than independently is what guarantees the searched
    rotation is optimised against the transform that will actually run.

    Only the block_size x block_size tile is taken from the kernel; the full matrix is exactly
    that tile repeated down the diagonal.

    Args:
        in_features: Width of the layer's input.
        block_size: Hadamard block size. A power of two, at most 512, dividing in_features.
        device: Device to build the matrix on.
        dtype: Dtype of the returned matrix.

    Returns:
        An (in_features, in_features) orthogonal block-diagonal Hadamard matrix.
    """
    if in_features % block_size:
        raise ValueError(
            f"in_features={in_features} is not divisible by block_size={block_size}"
        )
    tile, _, _ = ops.process_input(
        torch.eye(block_size, device=device, dtype=torch.float32),
        hadamard_block_size=block_size,
    )
    return torch.block_diag(*([tile.to(dtype)] * (in_features // block_size)))



def _transform_stream(x: torch.Tensor, R1: torch.Tensor, inverse: bool, center: bool) -> torch.Tensor:
    """Rotate the residual stream into the R1 basis, or back out of it, in x's dtype.

    The matmul runs in the dtype promoted over x and R1, so a float32 rotation is applied to a
    half-precision stream at full precision and only the result is rounded. Centering is written
    as x - mean(x) rather than x @ M, which is identical and linear in the feature width instead
    of quadratic.
    """
    out_dtype = x.dtype
    dtype = torch.promote_types(out_dtype, R1.dtype)
    rotation = R1.to(dtype=dtype, device=x.device)
    y = x.to(dtype)
    if inverse:
        return (y @ rotation.T).to(out_dtype)
    if center:
        y = y - y.mean(dim=-1, keepdim=True)
    return (y @ rotation).to(out_dtype)


def center_and_rotate_residual_stream(x: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Mean-center x along its feature axis, then rotate it into the R1 basis.

    This is the other half of the norm conversion. _RMSNormWithScale folds the centering into the
    layers feeding each norm, but a stack's first norm reads the residual stream, which is not the
    output of any single layer, so there is nothing to fold into. Centering the stream once on
    entry covers it: every later contribution is centered by the fold into the layer that wrote
    it, and a sum of centered vectors is centered, so every downstream RMSNorm sees the same
    input its LayerNorm would have centered for itself.

    Call unrotate_residual_stream on the stack's output to return to the original basis.

    Args:
        x: Residual stream, with features on the last axis.
        R1: The residual-stream rotation. May be an nn.Parameter.

    Returns:
        The centered, rotated stream, in x's dtype.
    """
    return _transform_stream(x, R1, inverse=False, center=True)


def rotate_residual_stream(x: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Rotate x into the R1 basis without centering it.

    The entry transform for a stack whose norms are RMSNorms to begin with, such as a Qwen LLM:
    RMSNorm does not center its input, so centering the stream would change the function.

    Args:
        x: Residual stream, with features on the last axis.
        R1: The residual-stream rotation. May be an nn.Parameter.

    Returns:
        The rotated stream, in x's dtype.
    """
    return _transform_stream(x, R1, inverse=False, center=False)


def unrotate_residual_stream(x: torch.Tensor, R1: torch.Tensor) -> torch.Tensor:
    """Rotate x out of the R1 basis, undoing center_and_rotate_residual_stream's rotation.

    The centering is not undone, and does not need to be: the norms it feeds were LayerNorms,
    which centered their input anyway, so the converted stack computes the same function.

    Args:
        x: Stack output, in the rotated basis, with features on the last axis.
        R1: The residual-stream rotation that was applied on entry.

    Returns:
        The stream in the original basis, in x's dtype.
    """
    return _transform_stream(x, R1, inverse=True, center=False)


def _apply_to_stream(transform, args, kwargs, keyword):
    """Replace the residual stream in a module's arguments with transform of it.

    The stream is read from the first positional argument, or from a keyword when the stack passes
    it by name: HuggingFace calls ``layer(hidden_states, ...)`` while NeMo calls
    ``layer(x=audio_signal, ...)``.
    """
    if keyword is not None and keyword in kwargs:
        return args, {**kwargs, keyword: transform(kwargs[keyword])}
    return (transform(args[0]), *args[1:]), kwargs


def _stream_pre_hook(transform, keyword):
    """A forward pre-hook applying transform to the residual stream on its way into a module."""
    def hook(_module, args, kwargs):
        return _apply_to_stream(transform, args, kwargs, keyword)

    return hook


class ResidualStreamRotation(nn.Module):
    """Rotates, or rotates back, the residual stream on its way into another module.

    The rotated model has to transform the stream at points where it does not pass through any
    module boundary of its own: Whisper builds a stack's input from convolutions, a functional GELU
    and positional embeddings inside the stack's forward, and hands it straight to layers[0]. A
    module can only act on another module's input or output, and wrapping layers[0] would rename
    every parameter under it. So the transform is this module, registered as a child of the layer
    it acts on, and that layer's forward pre-hook is this module's own pre_hook method.

    Being a module rather than a closure is what makes the rotation part of the model:

    - The rotation is a buffer, so it is in state_dict and is saved and loaded with the weights.
    - The hook is a bound method of a module, which pickles, so torch.save(model) keeps the hook
      and torch.load restores a working rotated model; a closure hook makes the model unpicklable.
    - It survives copy.deepcopy and follows .to(device).

    The buffer stays float32 when the model is cast to half precision: rounding R1 to fp16 would
    make it measurably non-orthogonal, and the transform promotes the stream's dtype to float32 for
    the matmul anyway.

    Args:
        R1: The residual-stream rotation.
        inverse: False centers and rotates into the R1 basis (a stack's entry); True rotates back
            out of it (in front of a norm that keeps its scale).
        keyword: Name of the keyword argument carrying the stream, or None if positional.
        center: On entry, center the stream before rotating it. False for a stack of RMSNorms,
            which do not center; see rotate_residual_stream.
    """

    def __init__(self, R1: torch.Tensor, inverse: bool = False, keyword: Optional[str] = None,
                 center: bool = True):
        super().__init__()
        self.inverse = inverse
        self.keyword = keyword
        self.center = center
        self.register_buffer("rotation", R1.detach().to(torch.float32).clone())

    def _apply(self, fn, recurse=True):
        rotation = self._buffers["rotation"]
        super()._apply(fn, recurse)
        self._buffers["rotation"] = rotation.to(device=fn(rotation).device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _transform_stream(x, self.rotation, self.inverse, getattr(self, "center", True))

    def pre_hook(self, _module, args, kwargs):
        return _apply_to_stream(self, args, kwargs, self.keyword)

    def extra_repr(self) -> str:
        direction = "unrotate" if self.inverse else ("center+rotate" if getattr(self, "center", True) else "rotate")
        return f"{direction}, width={self.rotation.shape[0]}, keyword={self.keyword}"


class _RotationHandle:
    """Removes a persistent residual-stream rotation: its hook and its child module."""

    def __init__(self, module: nn.Module, child_name: str, hook_handle):
        self.module = module
        self.child_name = child_name
        self.hook_handle = hook_handle

    def remove(self) -> None:
        self.hook_handle.remove()
        if hasattr(self.module, self.child_name):
            delattr(self.module, self.child_name)


def _attach_rotation(module, R1, inverse, keyword, child_name, center=True):
    """Attach a saved ResidualStreamRotation as a child of module, run by its forward pre-hook."""
    if hasattr(module, child_name):
        raise ValueError(f"{type(module).__name__} already has a {child_name}")
    rotation = ResidualStreamRotation(R1, inverse=inverse, keyword=keyword, center=center)
    module.add_module(child_name, rotation)
    hook = module.register_forward_pre_hook(rotation.pre_hook, with_kwargs=True)
    return _RotationHandle(module, child_name, hook)


def add_residual_stream_entry_hook(
    module: nn.Module, R1: torch.Tensor, keyword: str = None, persistent: bool = False,
    center: bool = True,
):
    """Center and rotate the residual stream on its way into a module, via a forward pre-hook.

    Attach this to the first layer of a stack. A hook is used rather than a rewritten forward so
    the surrounding code stays whatever the model library provides, and nothing has to be kept in
    step with it across versions.

    Two forms, for the two uses:

    - ``persistent=False``: a closure over R1 itself. The rotation search needs this, since R1 is
      an nn.Parameter being optimised and the hook must see its current value.
    - ``persistent=True``: a ResidualStreamRotation child named ``input_rotation`` holding a copy
      of R1, for a rotation that is final. It is saved with the model; see
      ResidualStreamRotation.

    Args:
        module: The module whose input is the residual stream, typically ``layers[0]``.
        R1: The residual-stream rotation.
        keyword: Name of the keyword argument carrying the stream, or None if positional.
        persistent: Store the rotation as a module that is saved with the model.
        center: Center the stream before rotating it. Leave True for a stack of converted
            LayerNorms; set False for a stack of RMSNorms, which never centered their input.

    Returns:
        A handle whose ``.remove()`` undoes the patch.
    """
    if persistent:
        return _attach_rotation(module, R1, False, keyword, "input_rotation", center=center)
    transform = center_and_rotate_residual_stream if center else rotate_residual_stream
    return module.register_forward_pre_hook(
        _stream_pre_hook(lambda x: transform(x, R1), keyword), with_kwargs=True
    )


def add_unrotate_input_hook(
    module: nn.Module, R1: torch.Tensor, keyword: str = None, persistent: bool = False
):
    """Rotate the residual stream out of the R1 basis on its way into a module.

    Use this in front of a norm that has to keep its scale. RMSNorm is only equivariant to a
    rotation while it has no scale and no shift: scaling elementwise is multiplying by
    diag(gamma), which does not commute with R1. So a norm with no next layer to absorb its
    scale cannot sit inside the rotated region, and the stream has to leave the rotated basis
    before it rather than after.

    Such a norm still folds its centering into its previous layers as usual; only the scale and
    shift stay behind.

    ``persistent`` chooses between a closure and a saved ResidualStreamRotation child named
    ``input_unrotation``, as in add_residual_stream_entry_hook.

    Args:
        module: The module whose input should leave the rotated basis, typically the last norm
            of a stack.
        R1: The residual-stream rotation applied on entry.
        keyword: Name of the keyword argument carrying the stream, or None if positional.
        persistent: Store the rotation as a module that is saved with the model.

    Returns:
        A handle whose ``.remove()`` undoes the patch.
    """
    if persistent:
        return _attach_rotation(module, R1, True, keyword, "input_unrotation")
    return module.register_forward_pre_hook(
        _stream_pre_hook(lambda x: unrotate_residual_stream(x, R1), keyword), with_kwargs=True
    )


def replace_module(model, module_name: str, new_module: nn.Module) -> None:
    """Replace the module at a dotted name, in place, keeping its position.

    The last path segment is a container index rather than an attribute when it is a digit,
    which happens once insert_linear_after_norm has wrapped a norm in an nn.Sequential: the
    norm moves from ``p.norm_out`` to ``p.norm_out.0``. Indexing the container is equivalent
    to setattr with the string key here, but says what is meant.

    Args:
        model: The model containing the module.
        module_name: Dotted name of the module to replace.
        new_module: The module to put in its place.
    """
    parent = get_parent_module(model, module_name)
    attribute = module_name.split(".")[-1]
    if attribute.isdigit():
        parent[int(attribute)] = new_module
    else:
        setattr(parent, attribute, new_module)


def insert_linear_after_norm(model, norm_name: str) -> Tuple[str, str]:
    """Insert an identity Linear immediately after a norm, and return both their new names.

    _RMSNormWithScale folds a centering matrix into the layers feeding a norm, which requires those to be
    Linear or pointwise Conv1d layers. At the end of a Conformer block the layer feeding the next
    block's first norm is itself a norm, so there is nothing to fold into. Putting an identity
    Linear after it gives the next norm a real weight to absorb the centering, at the cost of one
    d_model x d_model matmul per block.

    The Linear starts as the identity, so the model's output is unchanged until something is
    folded into it. Once the surrounding norms are converted it holds the centering matrix, and
    the block ends with a Linear rather than a norm.

    The norm is replaced by an nn.Sequential of the two, so a norm previously at ``p.norm_out``
    becomes ``p.norm_out.0`` with the Linear at ``p.norm_out.1``. Use the returned names when
    building the norm mapping.

    Args:
        model: The model containing the norm.
        norm_name: Dotted name of the norm to insert after.

    Returns:
        ``(norm_name, linear_name)`` after the insertion.
    """
    parent = get_parent_module(model, norm_name)
    attribute = norm_name.split(".")[-1]
    norm = getattr(parent, attribute)

    dim = norm.normalized_shape[0]
    weight = norm.weight
    linear = nn.Linear(dim, dim, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(dim, device=weight.device, dtype=weight.dtype))
        linear.bias.zero_()

    setattr(parent, attribute, nn.Sequential(norm, linear))
    return f"{norm_name}.0", f"{norm_name}.1"


def get_module(model, module_name: str) -> nn.Module:
    """Return the module at a dotted name by walking attributes, not named_modules().

    named_modules() lists a module instance once however many parents hold it, so a shared
    instance is missing under every name but the first. NeMo's Conformer feed-forwards all hold
    one Swish, created once as a default argument, which is exactly that case.
    """
    parent = get_parent_module(model, module_name)
    attribute = module_name.split(".")[-1]
    return parent[int(attribute)] if attribute.isdigit() else getattr(parent, attribute)


def random_hadamard_signs(width: int, device, generator: Optional[torch.Generator] = None):
    """A random +-1 vector, the diagonal D of a randomized online Hadamard X @ D @ H.

    A deterministic Hadamard maps a constant vector onto a single coordinate: the first row of H
    is flat, so a block's mean ends up in that block's first entry. FC2 reads a GELU output, which
    is mostly non-negative -- on whisper-large-v3 its per-token mean is 0.56-0.58 of its RMS in
    the middle and late encoder layers -- and the plain Hadamard put 36% of each block's energy
    into that one coordinate. The group-128 crest factor rose from 2.5 to 6.5 and the per-token
    one to 8.3, so FC2 quantized worse with its rotation than without. Flipping random signs first
    scatters the mean across the block (crest 2.7 and 3.7). A SwiGLU down-projection input is
    roughly zero-mean, which is why LLM rotation methods do not hit this.

    Args:
        width: Width of the FC2 input.
        device: Device of the returned vector.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        A float32 ``(width,)`` tensor of +-1.
    """
    bits = torch.randint(0, 2, (width,), generator=generator)
    return (bits.to(torch.float32) * 2 - 1).to(device)


def random_sign_seed(generator: Optional[torch.Generator] = None) -> int:
    """A seed for seeded_hadamard_signs, drawn from torch's generator; never 0, which means no signs."""
    return int(torch.randint(1, 2**31, (1,), generator=generator))


def seeded_hadamard_signs(seed: int, width: int, device) -> torch.Tensor:
    """The +-1 vector humming applies before its Hadamard when called with ``hadamard_sign_seed=seed``.

    Channel j gets -1 when bit 0 of lowbias32(j + seed * 0x9E3779B9 mod 2**32) is set, the hash the input
    kernel computes (third_party/humming, branch hadamard-sign-seed). The signs are what
    random_hadamard_signs draws -- see there for why an FC2's Hadamard needs them -- but a single integer
    reproduces them for every width, so the online transform generates them itself instead of reading and
    multiplying a stored vector.

    Args:
        seed: A positive seed, as random_sign_seed returns.
        width: Number of channels.
        device: Device of the returned vector.

    Returns:
        A float32 ``(width,)`` tensor of +-1.
    """
    if seed <= 0:
        raise ValueError(f"the sign seed must be positive, got {seed}")
    x = (np.arange(width, dtype=np.uint64) + np.uint64(seed * 0x9E3779B9 & 0xFFFFFFFF)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7FEB352D)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846CA68B)) & np.uint64(0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    return torch.from_numpy(np.where(x & np.uint64(1), -1.0, 1.0).astype(np.float32)).to(device)


class OnlineHadamard(nn.Module):
    """Apply humming's block-diagonal Hadamard to the feature axis: X <- X @ H.

    The counterpart of absorb_hadamard_rotation_into_fc2, which folds H.T into the next layer's
    weight. Together they reconstruct the unrotated layer,

        (X @ H) @ (W @ H).T + b = X @ H @ H.T @ W.T + b = X @ W.T + b

    while that layer's input is seen, and quantized, in the Hadamard basis.

    It is a module in front of the layer rather than a patch of the layer's forward so that the
    layer's input really is X @ H. Anything observing that input through a forward hook -- the
    Hessian a GPTQ-style quantizer collects, activation statistics -- then sees the tensor the
    weight actually multiplies. A patched forward would hand such a hook X instead, which is the
    wrong basis for W @ H.

    Inference only: the humming kernel is not differentiable. During a rotation search use
    patch_rotation with a dense H, which applies the same matrix as a plain matmul.

    Args:
        block_size: Hadamard block size. A power of two, at most 512, dividing the feature
            width. Must match the value absorbed into the next layer.
        signs: Optional ``(width,)`` +-1 vector; if given the transform is X @ D @ H with
            D = diag(signs), the randomized Hadamard. See random_hadamard_signs.
        channels_first: True for (batch, channels, time) inputs, as a pointwise Conv1d reads;
            the Hadamard is then applied to the channel axis.
        sign_seed: The seed signs came from (seeded_hadamard_signs), or 0. With a seed humming generates
            the signs inside its Hadamard kernel instead of this module multiplying by them; signs is
            then kept for inspection and must equal ``seeded_hadamard_signs(sign_seed, width)``.
    """

    def __init__(
        self, block_size: int, channels_first: bool = False, signs: Optional[torch.Tensor] = None,
        sign_seed: int = 0,
    ):
        super().__init__()
        self.block_size = block_size
        self.channels_first = channels_first
        self.sign_seed = int(sign_seed)
        self.register_buffer("signs", None if signs is None else signs.detach().clone())

    def pre_hook(self, _module, args):
        return (self(args[0]), *args[1:])

    def forward(self, x):
        if self.channels_first:
            x = x.transpose(1, 2)
        if self.signs is not None and not self.sign_seed:
            x = x * self.signs.to(dtype=x.dtype)
        rotated, _, _ = ops.process_input(
            x.contiguous(), hadamard_block_size=self.block_size, hadamard_sign_seed=self.sign_seed
        )
        return rotated.transpose(1, 2) if self.channels_first else rotated

    def extra_repr(self) -> str:
        signs = f"sign_seed={self.sign_seed}" if self.sign_seed else f"random_signs={self.signs is not None}"
        return f"block_size={self.block_size}, channels_first={self.channels_first}, {signs}"


def _online_hadamard_for(model, fc2_name, block_size, signs, sign_seed):
    """An FC2 layer and the OnlineHadamard that applies X @ H to its input.

    humming's kernel mixes blocks of block_size, so the width it is applied to has to be a
    multiple of it; the layout follows the layer, since a pointwise Conv1d reads
    (batch, channels, time).
    """
    fc2 = get_module(model, fc2_name)
    in_features = _weight_2d(fc2).shape[1]
    if in_features % block_size:
        raise ValueError(
            f"{fc2_name} has in_features={in_features}, not divisible by block_size={block_size}"
        )
    hadamard = OnlineHadamard(
        block_size, channels_first=isinstance(fc2, nn.Conv1d), signs=signs, sign_seed=sign_seed
    )
    return fc2, hadamard


def insert_online_hadamard_after_activation(
    model, activation_name: str, fc2_name: str, block_size: int,
    signs: Optional[torch.Tensor] = None, sign_seed: int = 0,
) -> str:
    """Wrap an activation as nn.Sequential(activation, OnlineHadamard), in place.

    This puts X @ H between the nonlinearity and the FC2 layer it feeds, leaving the FC2 layer
    itself untouched and under its original name. Only elementwise dropout sits between the two
    in the supported models, which is the identity in eval mode, so moving H ahead of it changes
    nothing at inference.

    The activation is wrapped separately in its own parent rather than modified, because it may
    be one instance shared with other parents; wrapping one parent's reference leaves the others
    alone. For the same reason both names are resolved by attribute path; see get_module.

    Idempotent: an activation already ending in an OnlineHadamard is left as it is.

    Args:
        model: The model to modify in place.
        activation_name: Dotted name of the activation feeding the FC2 layer.
        fc2_name: Dotted name of that FC2 layer, whose H.T must already be absorbed or be about
            to be. Its type decides the layout: a Conv1d reads (batch, channels, time).
        block_size: Hadamard block size, matching the value absorbed into the FC2 layer.
        signs: The random signs absorbed into the FC2 layer, if any.
        sign_seed: The seed they came from, if any; see OnlineHadamard.

    Returns:
        The dotted name of the inserted OnlineHadamard.
    """
    fc2, hadamard = _online_hadamard_for(model, fc2_name, block_size, signs, sign_seed)
    parent = get_parent_module(model, activation_name)
    attribute = activation_name.split(".")[-1]
    activation = getattr(parent, attribute)
    if isinstance(activation, nn.Sequential) and isinstance(activation[-1], OnlineHadamard):
        return f"{activation_name}.{len(activation) - 1}"

    setattr(parent, attribute, nn.Sequential(activation, hadamard))
    return f"{activation_name}.1"


def insert_online_hadamard_before_layer(
    model, fc2_name: str, block_size: int, signs: Optional[torch.Tensor] = None, sign_seed: int = 0,
) -> str:
    """Apply X @ H to an FC2 layer's own input, as an OnlineHadamard child run by a forward pre-hook.

    For an FC2 whose input is not the output of a single activation module, such as a SwiGLU
    down projection reading ``act(gate(x)) * up(x)``: wrapping the activation there would rotate only
    one factor of the product. The Hadamard is a child named ``input_hadamard`` and its pre_hook
    method is the hook, so, like ResidualStreamRotation, it is saved and loaded with the model and
    the layer keeps its name. A forward hook on the layer, such as the one a GPTQ-style quantizer
    collects its Hessian with, sees the input after the pre-hook, which is X @ H.

    Idempotent: a layer that already has an ``input_hadamard`` is left as it is.

    Args:
        model: The model to modify in place.
        fc2_name: Dotted name of the FC2 layer, whose H.T must already be absorbed or be about to be.
        block_size: Hadamard block size, matching the value absorbed into the layer.
        signs: The random signs absorbed into the layer, if any.
        sign_seed: The seed they came from, if any; see OnlineHadamard.

    Returns:
        The dotted name of the inserted OnlineHadamard.
    """
    fc2, hadamard = _online_hadamard_for(model, fc2_name, block_size, signs, sign_seed)
    if not hasattr(fc2, "input_hadamard"):
        fc2.add_module("input_hadamard", hadamard)
        fc2.register_forward_pre_hook(hadamard.pre_hook)
    return f"{fc2_name}.input_hadamard"


def check_hadamard_matches_groups(
    hadamard_block_size, activation_group_size, groupwise_roles, fc2_online_hadamard=True
) -> None:
    """Raise if fc2 is quantized group-wise behind an online Hadamard of a different block size.

    The online Hadamard mixes channels within blocks of hadamard_block_size. When those blocks line
    up with the activation quantization groups, an outlier is spread only across the group it is
    already in. A block wider than a group leaks it into neighbouring groups and raises their
    scales too; a narrower one leaves each group's halves unmixed. Nothing else links the two
    settings, so a mismatch would silently cost accuracy.

    Args:
        hadamard_block_size: Block size of fc2's online Hadamard.
        activation_group_size: Features per activation quantization group.
        groupwise_roles: Roles quantized group-wise; None means every role.
        fc2_online_hadamard: Whether fc2 has an online Hadamard at all.
    """
    fc2_groupwise = groupwise_roles is None or "fc2" in groupwise_roles
    if fc2_online_hadamard and fc2_groupwise and hadamard_block_size != activation_group_size:
        raise ValueError(
            f"fc2's input is quantized in groups of {activation_group_size} but its online "
            f"Hadamard mixes blocks of {hadamard_block_size}; set hadamard_block_size equal to "
            f"activation_group_size so each block lines up with one quantization group"
        )


def _group_rotation(R1, group):
    """The R1 a layers_to_rotate entry uses: R1 itself, or R1[stream] for a model with several streams.

    A model with more than one residual stream -- Canary-Qwen's Conformer encoder and Qwen LLM -- has
    one R1 per stream, passed as ``{stream: R1}``, and each entry names its stream as a fourth
    element: ``(attention_name, head_dim, layers, stream)``.
    """
    if not isinstance(R1, Mapping):
        return R1
    if len(group) < 4:
        raise ValueError(
            f"{group[0]}: with one R1 per residual stream every layers_to_rotate entry needs its "
            f"stream name as a fourth element"
        )
    return R1[group[3]]


def _fc2_names(layers_to_rotate):
    """The layers an online Hadamard applies to: every one listed as ROT_OUTPUT_R1."""
    return [name for group in layers_to_rotate for name, kind in group[2] if kind == ROT_OUTPUT_R1]


def _check_online_hadamard_layers(layers_to_rotate, online_hadamard_layers) -> None:
    """Every FC2 layer that absorbs H must have exactly one activation applying it online.

    A ROT_OUTPUT_R1 layer missing from online_hadamard_layers would hold W @ H with nothing
    applying H to its input, and a listed layer of another kind would get H applied online with
    nothing absorbing it. Neither raises anywhere else; both just return a wrong model.
    """
    fc2_names = set(_fc2_names(layers_to_rotate))
    listed = [fc2_name for _, fc2_name in online_hadamard_layers]
    duplicated = {name for name in listed if listed.count(name) > 1}
    missing, extra = fc2_names - set(listed), set(listed) - fc2_names
    if missing or extra or duplicated:
        raise ValueError(
            "online_hadamard_layers must pair every ROT_OUTPUT_R1 layer with exactly one "
            f"activation (or None for its own input); missing={sorted(missing)}, wrong kind={sorted(extra)}, "
            f"listed twice={sorted(duplicated)}"
        )


def _fc2_input_widths(model, layers_to_rotate):
    """The distinct input widths of the FC2 layers an online Hadamard applies to."""
    moduledict = dict(model.named_modules())
    return {_weight_2d(moduledict[name]).shape[1] for name in _fc2_names(layers_to_rotate)}


def _signs_for(hadamard_signs, layer):
    """The random signs of a layer's online Hadamard, which are kept per input width."""
    return None if hadamard_signs is None else hadamard_signs[_weight_2d(layer).shape[1]]


def _effective_kind(kind, use_r2):
    """The rotation a layer actually gets when there is no R2.

    Without an R2 the V projection keeps only R1 on its input and the attention output projection
    only R1 on its output, so the attention output reaches the output projection unrotated.
    """
    if use_r2:
        return kind
    return {
        ROT_INPUT_R1_OUTPUT_R2: ROT_INPUT_R1,
        ROT_INPUT_R2_OUTPUT_R1: ROT_OUTPUT_R1,
    }.get(kind, kind)


def _rotation_targets(layers_to_rotate, R1, R2s, use_r2, fc2_online_hadamard):
    """Yield ``(layer_name, kind, r1, r2, online_hadamard)`` for every layer to rotate.

    kind is what the layer actually gets, after _effective_kind; online_hadamard says whether its
    input is rotated by an online Hadamard, which only an FC2 -- a ROT_OUTPUT_R1 layer as listed,
    before any substitution -- ever is. r1 is the rotation of the residual stream the layer's
    group names, and r2 its attention module's head-dimension rotation.
    """
    for group in layers_to_rotate:
        attention_name, _head_dim, layer_names = group[:3]
        r1 = _group_rotation(R1, group)
        r2 = R2s[attention_name] if use_r2 else None
        for layer_name, kind in layer_names:
            online_hadamard = fc2_online_hadamard and kind == ROT_OUTPUT_R1
            yield layer_name, _effective_kind(kind, use_r2), r1, r2, online_hadamard


def fold_norms(model, norm_layers) -> None:
    """Convert every norm named in norm_layers to a plain normalization, in place.

    A LayerNorm becomes an RMSNorm with no scale and no shift, which is what makes the residual
    stream rotation-equivariant: RMSNorm(x @ R1) == RMSNorm(x) @ R1 holds only without them.
    The centering goes backwards into the layers feeding the norm, the shift forwards into the
    layers reading it, and the scale forwards as well. An nn.RMSNorm or a Hugging Face-style
    RMSNorm (Qwen3RMSNorm) has no centering or shift to begin with, so only its scale is folded.

    _RMSNormWithScale runs before _ScaleFreeRMSNorm on the same next layers because the shift
    fold reads their unscaled weight; reversed, the shift would be computed from the already-
    scaled weight.

    A norm with an empty next list keeps its scale and shift, since there is nowhere to put
    them. Such a norm cannot sit inside the rotated region -- see add_unrotate_input_hook.

    Args:
        model: The model whose norms are converted, modified in place.
        norm_layers: ``(norm_name, previous_names, next_names)`` entries. Either neighbour
            list may be empty; see _RMSNormWithScale for what that means.
    """
    moduledict = dict(model.named_modules())
    for norm_name, previous_names, next_names in norm_layers:
        norm = moduledict[norm_name]
        previous_layers = [moduledict[name] for name in previous_names]
        next_layers = [moduledict[name] for name in next_names]
        if isinstance(norm, nn.LayerNorm):
            converted = _RMSNormWithScale(norm, previous_layers, next_layers)
        elif isinstance(norm, nn.RMSNorm) or _is_hf_rmsnorm(norm):
            converted = norm
        else:
            raise ValueError(f"Unsupported norm layer type: {type(norm)}")
        if next_layers:
            converted = _ScaleFreeRMSNorm(converted, next_layers)
        replace_module(model, norm_name, converted)


def masked_kl_divergence(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean KL(teacher || student) over the positions selected by mask.

    Computed in float32 whatever the logits' dtype: a half-precision log-softmax over a vocabulary
    of tens of thousands underflows for the unlikely tokens and turns their contribution into
    noise.

    Args:
        student_logits: ``(..., vocab)`` logits of the model being optimised.
        teacher_logits: Logits of the same shape from the full-precision model.
        mask: Boolean tensor over the leading axes; True where the position counts.

    Returns:
        A scalar: the KL divergence per selected position.
    """
    student = F.log_softmax(student_logits[mask].float(), dim=-1)
    teacher = F.log_softmax(teacher_logits[mask].float(), dim=-1)
    return F.kl_div(student, teacher, log_target=True, reduction="batchmean")


@contextmanager
def exact_float32_arithmetic():
    """Disable TF32 for cuBLAS matmuls and cuDNN convolutions, restoring both on exit.

    TF32 computes float32 matmuls and convolutions with 19-bit mantissas. PyTorch enables it for
    cuDNN convolutions by default, and a Conformer runs many: subsampling, depthwise and
    pointwise. Rewriting the weights changes where that rounding lands, and on parakeet-ctc-1.1b
    it moved the logits by 1.8e-3 -- as much as a genuinely wrong fold does. With TF32 off the
    same rewrite is exact to 1.5e-6. The equivalence checks therefore run under this, and the
    search itself keeps whatever the caller configured.
    """
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def check_logits_unchanged(
    model, compute_logits, batch, reference_logits, stage, tolerance=1e-2
) -> float:
    """Raise if a rewrite that is meant to be output-equivalent changed the model's logits.

    Every step of the rewrite -- the norm fold, the centering hooks, the rotation patches -- is
    an exact reparameterisation, so the logits on a fixed batch must come back unchanged up to
    floating-point rounding. Checking after each step separates a wrong fold from a wrong
    mapping.

    The comparison is on the logits rather than a loss so that it cannot be vacuous: a
    distillation loss against the model itself is zero whatever the model computes, and a
    scalar loss can in principle hide a compensating error. The error is the largest absolute
    difference over the counted positions, relative to the largest reference logit.

    The tolerance has to allow for more than fp64 rounding: the folds are computed in promoted
    precision but rounded back into the model's dtype, so a half-precision model accumulates
    real error that the unmodified model never had. TF32 has to be off for both the reference
    and this pass, or its rounding alone can exceed the tolerance; see exact_float32_arithmetic.

    Args:
        model: The model, already rewritten up to this stage.
        compute_logits: ``fn(model, batch) -> (logits, mask)``.
        batch: A fixed batch, the same one the reference logits were computed on.
        reference_logits: The original model's logits on that batch.
        stage: Name of the stage just applied, used in the message.
        tolerance: Largest relative logit error that is accepted.

    Returns:
        The relative logit error after this stage.
    """
    with torch.no_grad():
        logits, mask = compute_logits(model, batch)
    reference = reference_logits[mask].float()
    error = float(
        (logits[mask].float() - reference).abs().max() / reference.abs().max().clamp_min(1e-12)
    )
    print(f"  after {stage}: relative logit error {error:.2e}")
    tracking.summary({f"rotation/logit_error_{stage.replace(' ', '_')}": error})
    if error > tolerance:
        raise RuntimeError(
            f"{stage} changed the model's output: the logits moved by {error:.2e} relative to "
            f"the original, above the {tolerance:.0e} tolerance. This rewrite is meant to be "
            f"exactly output-equivalent, so the cause is a bug in one of the folds, in the norm "
            f"mapping, or in the residual-stream hooks -- not a precision effect."
        )
    return error


def _cached_online_hadamard(hadamards, width, block_size, signs, device):
    """The dense D @ H a search applies to an FC2 input of this width, built once and kept.

    One matrix per width serves every FC2 of that width, and the signs are the ones absorbed into
    those layers' weights; see random_hadamard_signs.
    """
    if width not in hadamards:
        H = online_hadamard_matrix(width, block_size, device)
        if signs is not None:
            H = signs[width].to(device=device).unsqueeze(1) * H
        hadamards[width] = H
    return hadamards[width]


def patch_rotations(
    model,
    layers_to_rotate,
    R1,
    R2s,
    hadamards,
    hadamard_block_size,
    quantize_inputs: Optional[Mapping[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
    hadamard_signs: Optional[Mapping[int, torch.Tensor]] = None,
    use_r2: bool = True,
    fc2_online_hadamard: bool = True,
) -> None:
    """Patch the forward pass of every layer in layers_to_rotate to apply its rotation.

    Used while the rotation is being searched: the weights are left untouched and the rotation
    is recomputed each forward, so gradients reach R1 and the R2s. fold_rotations is the
    counterpart that writes the finished rotation into the weights.

    Patching a layer again replaces its forward rather than stacking on it, so calling this a
    second time with different quantizers replaces them.

    quantize_inputs maps a layer name to the quantizer applied to that layer's input, as
    asrq.quantizers.activation.build_activation_quantizers returns for the model's activation
    roles -- the same quantizers evaluation attaches. A layer missing from the mapping is left
    unquantized, which includes the identity Linear inserted after a Conformer block's output
    norm: it carries only the folded centering and has no activation role.

    Args:
        model: The model to patch.
        layers_to_rotate: ``(attention_name, head_dim, [(layer_name, kind), ...])`` entries,
            with a fourth element naming the residual stream when R1 is a mapping.
        R1: The residual-stream rotation, or ``{stream: R1}`` for a model with several streams.
        R2s: One head-dimension rotation per attention module, keyed by attention_name.
        hadamards: Cache of online Hadamard matrices, keyed by input width. Filled as needed.
        hadamard_block_size: Block size of the online Hadamard applied to each FC2 input.
        quantize_inputs: Optional ``{layer_name: fn(x) -> x}``. None leaves every input in full
            precision.
        hadamard_signs: Optional ``{fc2_input_width: signs}``; each FC2's online Hadamard is then
            the randomized D @ H. See random_hadamard_signs.
        use_r2: If False there is no R2; see _effective_kind. R2s is then ignored and may be None.
        fc2_online_hadamard: If False FC2 has no online Hadamard: it only rotates its output by
            R1, and its input stays in the nonlinearity's own basis.
    """
    quantize_inputs = quantize_inputs or {}
    moduledict = dict(model.named_modules())
    device = next(model.parameters()).device
    for layer_name, kind, r1, r2, online_hadamard in _rotation_targets(
        layers_to_rotate, R1, R2s, use_r2, fc2_online_hadamard
    ):
        layer = moduledict[layer_name]
        H = None
        if online_hadamard:
            H = _cached_online_hadamard(
                hadamards, _weight_2d(layer).shape[1], hadamard_block_size, hadamard_signs, device
            )
        patch_rotation(layer, kind, r1, r2, H, quantize_inputs.get(layer_name))


def fold_rotations(
    model, layers_to_rotate, R1, R2s, hadamard_block_size, online_hadamard_layers,
    hadamard_signs=None, use_r2=True, fc2_online_hadamard=True, hadamard_sign_seed=0,
) -> None:
    """Fold a finished rotation into the weights of every layer in layers_to_rotate.

    The inference counterpart of patch_rotations, and numerically equivalent to it: both build
    the rotated weight with _rotate_weight and round once into the layer's dtype.

    Each FC2 gets its online Hadamard's H absorbed as well, after its output rotation, which
    reproduces the ``R1.T @ W @ H`` the search computed. The matching X @ H is then inserted
    after the activation feeding that FC2, so the FC2 layer is left an ordinary Linear or Conv1d;
    see OnlineHadamard. An FC2 paired with no activation gets it at its own input instead; see
    insert_online_hadamard_before_layer.

    Args:
        model: The model to fold into, modified in place.
        layers_to_rotate: ``(attention_name, head_dim, [(layer_name, kind), ...])`` entries,
            with a fourth element naming the residual stream when R1 is a mapping.
        R1: The residual-stream rotation, or ``{stream: R1}`` for a model with several streams.
        R2s: One head-dimension rotation per attention module, keyed by attention_name.
        hadamard_block_size: Block size of the online Hadamard applied to each FC2 input. Must
            match the value the rotation was searched with.
        online_hadamard_layers: ``(activation_name, fc2_name)`` pairs, one for every ROT_OUTPUT_R1
            layer; an activation_name of None puts the Hadamard at the FC2's own input.
            Unused when fc2_online_hadamard is False.
        hadamard_signs: Optional ``{fc2_input_width: signs}`` for the randomized Hadamard.
        use_r2: Fold R2 into the V and attention output projections; see _effective_kind.
        fc2_online_hadamard: Absorb and insert FC2's online Hadamard; see patch_rotations.
        hadamard_sign_seed: The seed hadamard_signs came from, if any; the inserted OnlineHadamards
            then let humming generate the signs; see seeded_hadamard_signs.
    """
    if fc2_online_hadamard:
        _check_online_hadamard_layers(layers_to_rotate, online_hadamard_layers)
    moduledict = dict(model.named_modules())
    for layer_name, kind, r1, r2, online_hadamard in _rotation_targets(
        layers_to_rotate, R1, R2s, use_r2, fc2_online_hadamard
    ):
        layer = moduledict[layer_name]
        fold_rotation(layer, kind, r1, r2)
        if online_hadamard:
            absorb_hadamard_rotation_into_fc2(
                layer, hadamard_block_size, _signs_for(hadamard_signs, layer)
            )

    for activation_name, fc2_name in online_hadamard_layers if fc2_online_hadamard else ():
        signs = _signs_for(hadamard_signs, get_module(model, fc2_name))
        seed = hadamard_sign_seed if signs is not None else 0
        if activation_name is None:
            insert_online_hadamard_before_layer(model, fc2_name, hadamard_block_size, signs, seed)
        else:
            insert_online_hadamard_after_activation(
                model, activation_name, fc2_name, hadamard_block_size, signs, seed
            )


def _save_rotations(path, R1, R2s, multi_stream, settings) -> None:
    """Save a searched rotation in the checkpoint format apply_rotations reads, with its settings."""
    torch.save(
        {
            "R1": (
                {stream: p.data.detach().cpu() for stream, p in R1.items()} if multi_stream
                else R1.data.detach().cpu()
            ),
            "R2s": {name: R2.data.detach().cpu() for name, R2 in R2s.items()},
            **settings,
        },
        path,
    )


def _init_rotations(widths, multi_stream, r1_block_size, layers_to_rotate, learn_r2, device):
    """Draw the rotations a search starts from, with the masks that keep R1 block-diagonal.

    One R1 per residual stream, from a random Hadamard matrix, and one head-wise R2 per attention
    module. All of them are float32 nn.Parameters for SGDG to move over the Stiefel manifold.

    Args:
        widths: ``{stream: width}``; the key is None for a model with a single residual stream.
        multi_stream: Whether to return R1 keyed by stream or as the one rotation.
        r1_block_size: If set, each R1 is block-diagonal with blocks of this width and a mask is
            returned to keep it so; see learn_rotations.
        layers_to_rotate: The entries whose attention modules and head dimensions give the R2s.
        learn_r2: Draw the R2s at all.
        device: Device to draw on.

    Returns:
        ``(R1, r1_parameters, r1_masks, R2s)``, where r1_parameters lists the same parameters
        flatly for the optimiser and r1_masks is empty unless r1_block_size is set.
    """
    def draw(width):
        matrix = (
            random_hadamard_matrix(width, device) if r1_block_size is None
            else block_random_hadamard_matrix(width, r1_block_size, device)
        )
        return nn.Parameter(matrix.to(torch.float32))

    per_stream = {stream: draw(width) for stream, width in widths.items()}
    r1_parameters = list(per_stream.values())
    r1_masks = [] if r1_block_size is None else [
        block_diagonal_mask(p.shape[0], r1_block_size, device) for p in r1_parameters
    ]
    R2s = {}
    if learn_r2:
        R2s = {
            attention_name: nn.Parameter(random_hadamard_matrix(head_dim, device).to(torch.float32))
            for attention_name, head_dim, *_ in layers_to_rotate
        }
    return (per_stream if multi_stream else per_stream[None]), r1_parameters, r1_masks, R2s


def _fc2_hadamard_signs(model, layers_to_rotate, device):
    """One seed for every FC2's randomized online Hadamard, and the signs it generates per width.

    See random_hadamard_signs for why an FC2's Hadamard needs random signs, and
    seeded_hadamard_signs for why the seed is what gets stored.
    """
    seed = random_sign_seed()
    widths = sorted(_fc2_input_widths(model, layers_to_rotate))
    return seed, {width: seeded_hadamard_signs(seed, width, device) for width in widths}


def _cayley_sgd(
    parameters, masks, loss_fn, train_loader, epochs, learning_rate, objective_name, log_every,
    device,
):
    """Optimise orthogonal parameters over the Stiefel manifold, and return the last epoch's mean loss.

    SGDG keeps every parameter orthogonal, which plain SGD would not: a rotation that drifts off
    the manifold stops being orthogonal and the reparameterisation stops being exact. The learning
    rate decays linearly to zero over the whole run.

    Args:
        parameters: The rotations to optimise.
        masks: Gradient masks, aligned with the leading parameters, one each; a block-diagonal R1
            stays block-diagonal because SGDG builds its update only from products of a parameter
            and its gradient, so a masked gradient gives a masked step. May be empty.
        loss_fn: ``fn(batch, grad) -> scalar tensor``, the objective being minimised.
        train_loader: Calibration dataloader yielding batches for loss_fn.
        epochs: Number of passes over train_loader.
        learning_rate: Initial learning rate.
        objective_name: What loss_fn measures, for the printed lines.
        log_every: Summarise every log_every steps with the window's mean, min and max loss, the
            learning rate and the parameters' distance from orthogonality. An epoch average alone
            hides a search that diverges and recovers. 0 disables it.
        device: Device the parameters are on.

    Returns:
        The mean loss of the last epoch.
    """
    optimizer = SGDG(parameters, lr=learning_rate, stiefel=True)
    num_steps = max(len(train_loader) * epochs, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, (num_steps - step) / num_steps)
    )
    step, avg_loss = 0, 0.0
    for epoch in range(epochs):
        total_loss, num_batches, window = 0.0, 0, []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(batch, grad=True)
            loss.backward()
            for parameter, mask in zip(parameters, masks):
                parameter.grad.masked_fill_(~mask, 0.0)
            lr = scheduler.get_last_lr()[0]
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            step += 1
            window.append(loss.item())
            if log_every and step % log_every == 0:
                with torch.no_grad():
                    drift = max(
                        float((p.T @ p - torch.eye(p.shape[0], device=device)).abs().max())
                        for p in parameters
                    )
                print(
                    f"  step {step}/{num_steps}: {objective_name} {sum(window) / len(window):.6f} "
                    f"(min {min(window):.6f}, max {max(window):.6f}), lr {lr:.4g}, "
                    f"|R^T R - I| {drift:.1e}",
                    flush=True,
                )
                tracking.step_metric("search", "search/step")
                tracking.log({
                    "search/step": step, "search/loss": sum(window) / len(window), "search/loss_min": min(window),
                    "search/loss_max": max(window), "search/learning_rate": lr, "search/orthogonality_drift": drift,
                    "search/epoch": epoch + 1,
                })
                window = []
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{epochs}, average {objective_name}: {avg_loss:.6f}", flush=True)
        tracking.summary({f"search/epoch_{epoch + 1}_loss": avg_loss, "search/final_loss": avg_loss,
                          "search/name": "cayley", "search/steps": step})
    return avg_loss


def _report_initial_objective(loss_fn, objective_name, batch, activation_bits) -> float:
    """Print and record the objective at the random-Hadamard initialisation.

    The baseline the search has to beat: a random Hadamard is already a good rotation, so a run
    whose loss ends near this one learned nothing.
    """
    initial = float(loss_fn(batch, grad=False))
    print(
        f"{objective_name} with {activation_bits}-bit activations at the random Hadamard "
        f"initialisation: {initial:.6f}"
    )
    tracking.summary({"rotation/initial_objective": initial, "rotation/objective": objective_name,
                      "rotation/activation_bits": activation_bits})
    return initial


def _search_objective(model, objective, compute_logits, compute_loss, quantizers):
    """The loss a rotation search minimises, and what to call it in the log.

    Every rewrite the rotation makes is exact, so without quantization every rotation computes the
    same function and there is nothing to learn. Quantizing the inputs of the rotated layers makes
    the output depend on how well a rotation spreads outliers, and both objectives measure that
    dependence:

    - ``"kl"``: the KL divergence from the full-precision model to the same model with
      fake-quantized activations. It can only be lowered by recovering quantization error, and
      it uses the teacher's whole distribution at every position. The teacher is the model
      itself with the activation quantizer switched off, which by the exactness of the rewrite
      is the original model; see ActivationQuantizer. Its logits are recomputed each step
      without gradients rather than cached, since a vocabulary-sized tensor per token per
      sample would not fit.
    - ``"ce"``: the model's own training loss on the calibration labels, through compute_loss --
      cross-entropy for Whisper, CTC for Parakeet. It can also be lowered by fitting whatever
      the labels differ from the model in. With reference text that includes casing,
      punctuation and normalization, and on Whisper it measurably moved the quantized model
      further from the full-precision one while the loss fell. With labels taken from the
      full-precision model's own transcripts that gap closes, but the loss still only sees the
      top token and still rewards being more confident than the full-precision model.

    Args:
        model: The patched model, whose rotations the loss is differentiated against.
        objective: ``"kl"`` or ``"ce"``.
        compute_logits: ``fn(model, batch) -> (logits, mask)``.
        compute_loss: ``fn(model, batch) -> scalar tensor``; required for ``"ce"``.
        quantizers: The activation quantizers the search runs with, which ``full_precision``
            switches off.

    Returns:
        ``(loss_fn, name, full_precision)``: ``loss_fn(batch, grad)`` is the objective, name is
        what it measures, and full_precision is a context manager disabling the quantizers.
    """
    def full_precision():
        return quantizers_disabled(quantizers)

    def loss_fn(batch, grad):
        if objective == "ce":
            with torch.set_grad_enabled(grad):
                return compute_loss(model, batch)
        with torch.no_grad(), full_precision():
            teacher_logits, mask = compute_logits(model, batch)
        with torch.set_grad_enabled(grad):
            student_logits, _ = compute_logits(model, batch)
            return masked_kl_divergence(student_logits, teacher_logits, mask)

    name = "KL to full precision" if objective == "kl" else "training loss"
    return loss_fn, name, full_precision


def learn_rotations(
    model,
    path,
    layers_to_rotate,
    norm_layers,
    hidden_size,
    learning_rate,
    compute_logits,
    train_loader,
    epochs,
    hadamard_block_size,
    online_hadamard_layers,
    attach_rotation_hooks,
    tolerance=1e-2,
    activation_bits=None,
    activation_group_size=-1,
    activation_symmetric=True,
    activation_roles=None,
    groupwise_roles=None,
    objective="kl",
    compute_loss=None,
    log_every=25,
    r1_block_size=None,
    hadamard_random_signs=True,
    learn_r2=True,
    fc2_online_hadamard=True,
    model_name=None,
):
    """Search the rotations R1 and R2 for a model and save them to disk.

    The model's norms are folded in place before the rotations are patched in, so the saved
    rotations only apply to a model that has had the same folding applied; apply_rotations does
    both in that order.

    The rotations are patched into the forward passes rather than folded into the weights, so
    one shared R1 and one R2 per attention module stay as parameters the optimiser can move.
    SGDG keeps them on the Stiefel manifold, which plain SGD would not: a rotation that drifts
    off it stops being orthogonal and the reparameterisation stops being exact.

    Every rewrite is exact, so the search only has something to minimise because the rotated
    layers' inputs are fake-quantized; see _search_objective for the two objectives.

    R1 and the R2s start from random Hadamard matrices and are optimised with Cayley SGD over the
    Stiefel manifold, so every iterate stays orthogonal and the reparameterisation stays exact.

    The rewrites are checked against the original model's logits on a fixed batch before the
    search starts, with quantization off; see check_logits_unchanged.

    Args:
        model: The model to learn rotations for.
        path: Where to save the learned rotations.
        layers_to_rotate: ``(attention_name, head_dim, [(layer_name, kind), ...])``
            entries. One R2 is learned per attention module, since R2 is a head-dimension
            rotation applied to that module's V and undone by its output projection.
            Rotation types 1, 4 and 5 use only R1 and ignore R2, so layers of those types
            may be listed under whichever attention module they share a layer with. For a model
            with several residual streams each entry carries its stream name as a fourth element.
        norm_layers: ``(norm_name, previous_names, next_names)`` entries. Either neighbour
            list may be empty; see _RMSNormWithScale for what that means.
        hidden_size: Width of the residual stream, and so the size of R1. For a model with several
            residual streams, ``{stream: width}``: one R1 is learned per stream, and R1 is passed
            to attach_rotation_hooks and saved as ``{stream: R1}``.
        learning_rate: Initial learning rate, decayed linearly to zero over the run.
        compute_logits: ``fn(model, batch) -> (logits, mask)``, where mask selects the
            positions the objective counts.
        train_loader: Calibration dataloader yielding batches for compute_logits.
        epochs: Number of passes over train_loader.
        hadamard_block_size: Block size of the online Hadamard applied to each FC2 input.
            Must match the value used at inference, and is best set to the activation
            quantization group size.
        online_hadamard_layers: ``(activation_name, fc2_name)`` pairs, one for every
            ROT_OUTPUT_R1 layer. The search applies H as a dense matmul inside each FC2's patched
            forward, so these are not used for the search itself; they are checked here so that
            a mapping apply_rotations would reject fails before the search rather than after.
        attach_rotation_hooks: ``fn(model, R1, persistent=False) -> handles`` called once R1
            exists, for the model-specific hooks that center and rotate each stack's residual
            stream and rotate it back out. Needed because norms whose input is the residual stream
            have no previous layer to fold the centering into. Required, so that leaving the hooks
            out is a decision: pass None only for a model whose residual stream needs no
            transform of its own, such as one entered through an embedding R1 can be folded into.
        tolerance: Largest relative logit error accepted at each verification step.
        activation_bits: Bit width of the activation fake quantization applied during the
            search. None disables it, which leaves the search with nothing to learn.
        activation_group_size: Features per activation quantization group for the roles in
            groupwise_roles. Set it to the group size used at evaluation.
        activation_symmetric: Symmetric activation quantization if True, else asymmetric.
        activation_roles: ``{layer_name: role}`` for the layers whose inputs are quantized, the
            same mapping evaluation uses. Required when activation_bits is set.
        groupwise_roles: Roles quantized group-wise; the rest per token. None makes every role
            group-wise. See asrq.quantizers.activation.build_activation_quantizers.
        objective: ``"kl"`` or ``"ce"``; see _search_objective.
        compute_loss: ``fn(model, batch) -> scalar tensor``, the model's training loss on the
            batch's labels. Required for ``objective="ce"``, unused for ``"kl"``.
        hadamard_random_signs: Use the randomized online Hadamard X @ D @ H for every FC2, with the
            signs generated from one random seed (seeded_hadamard_signs), saved with the rotations as
            ``hadamard_sign_seed``. See random_hadamard_signs for why a plain Hadamard hurts FC2.
        learn_r2: Learn one head-wise R2 per attention module alongside R1. If False there is no
            R2 at all: it is neither trained nor applied, and the attention output projection's
            input stays unrotated.
        fc2_online_hadamard: Put an online Hadamard on every FC2 input. If False FC2 only
            rotates its output by R1 and its input stays unrotated. With both False only R1 is
            searched, and the inputs left unrotated are best quantized group-wise.
        r1_block_size: If set, R1 is block-diagonal with blocks of this width, initialised from
            block_random_hadamard_matrix and kept block-diagonal throughout the search by
            masking its gradient to the blocks. SGDG builds its update only from products of
            R1 and its gradient, so a block-diagonal gradient gives a block-diagonal step. Set
            it to the activation quantization group size. None keeps R1 full-width.
        log_every: Print the mean loss of the last log_every steps, with the learning rate and
            R1's distance from orthogonality, every log_every steps. An epoch average alone
            hides a search that diverges and recovers. 0 disables it.
        model_name: The model being rotated, saved in the checkpoint so applying it to another model is refused.

    Returns:
        ``(R1, R2s)``, the learned rotations, also saved to path; R1 is ``{stream: R1}`` when
        hidden_size is a mapping.
    """
    if objective not in ("kl", "ce"):
        raise ValueError(f"objective must be 'kl' or 'ce', got {objective!r}")
    if objective == "ce" and compute_loss is None:
        raise ValueError("objective='ce' needs compute_loss")
    if activation_bits is not None and not activation_roles:
        raise ValueError("activation quantization needs activation_roles")
    if activation_bits is not None:
        check_hadamard_matches_groups(
            hadamard_block_size, activation_group_size, groupwise_roles, fc2_online_hadamard
        )
    multi_stream = isinstance(hidden_size, Mapping)
    widths = dict(hidden_size) if multi_stream else {None: hidden_size}
    for width in widths.values():
        if r1_block_size is not None and width % r1_block_size:
            raise ValueError(f"r1_block_size {r1_block_size} does not divide hidden_size {width}")
    if fc2_online_hadamard:
        _check_online_hadamard_layers(layers_to_rotate, online_hadamard_layers)

    device = next(model.parameters()).device
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    reference_batch = next(iter(train_loader))
    with torch.no_grad(), exact_float32_arithmetic():
        reference_logits, _ = compute_logits(model, reference_batch)
    print("verifying the rewrite against the original model's logits")

    def verify(stage):
        with exact_float32_arithmetic():
            check_logits_unchanged(
                model, compute_logits, reference_batch, reference_logits, stage, tolerance
            )

    fold_norms(model, norm_layers)
    handles = []
    if attach_rotation_hooks is not None:
        identity = {stream: torch.eye(width, device=device) for stream, width in widths.items()}
        handles = attach_rotation_hooks(model, identity if multi_stream else identity[None])
    verify("norm folding")
    for handle in handles:
        handle.remove()

    R1, r1_parameters, r1_masks, R2s = _init_rotations(
        widths, multi_stream, r1_block_size, layers_to_rotate, learn_r2, device
    )
    if attach_rotation_hooks is not None:
        attach_rotation_hooks(model, R1)

    quantizers = {}
    if activation_bits is None:
        print(
            "WARNING: no activation quantization; the output does not depend on the rotation, "
            "so the search will not change it"
        )
    else:
        quantizers = build_activation_quantizers(
            activation_roles, activation_bits, activation_group_size, activation_symmetric,
            groupwise_roles,
        )
    hadamard_sign_seed, hadamard_signs = 0, None
    if fc2_online_hadamard and hadamard_random_signs:
        hadamard_sign_seed, hadamard_signs = _fc2_hadamard_signs(model, layers_to_rotate, device)
    patch_rotations(
        model, layers_to_rotate, R1, R2s, {}, hadamard_block_size, quantizers, hadamard_signs,
        use_r2=learn_r2, fc2_online_hadamard=fc2_online_hadamard,
    )

    objective_loss, objective_name, full_precision = _search_objective(
        model, objective, compute_logits, compute_loss, quantizers
    )
    with full_precision():
        verify("rotation patching")

    if quantizers:
        if objective == "ce":
            with torch.no_grad(), full_precision():
                unquantized = float(compute_loss(model, reference_batch))
            print(f"training loss in full precision: {unquantized:.6f}")
        _report_initial_objective(objective_loss, objective_name, reference_batch, activation_bits)

    _cayley_sgd(
        r1_parameters + list(R2s.values()), r1_masks, objective_loss, train_loader, epochs,
        learning_rate, objective_name, log_every, device,
    )

    _save_rotations(path, R1, R2s, multi_stream, {
        "model": model_name,
        "hadamard_block_size": hadamard_block_size,
        "learn_r2": learn_r2,
        "fc2_online_hadamard": fc2_online_hadamard,
        "hadamard_sign_seed": hadamard_sign_seed or None,
        "r1_block_size": r1_block_size,
        "objective": objective,
        "activation_quantization": {
            "bits": activation_bits,
            "group_size": activation_group_size,
            "symmetric": activation_symmetric,
            "groupwise_roles": None if groupwise_roles is None else sorted(groupwise_roles),
        },
        "search": {"name": "cayley"},
    })
    return R1, R2s


def check_rotation_settings(
    path: str, activation: Mapping, weight: Optional[Mapping] = None, checkpoint: Optional[Mapping] = None,
    model_name: Optional[str] = None,
) -> None:
    """Raise when a saved rotation was searched under different quantization settings than the run applying it.

    A rotation is searched against a quantizer, so applying it under another one is unsound: the search's
    fitness is the KL of exactly the quantization being used. The checkpoint records what it assumed in
    ``activation_quantization`` and, for the weight-only searches that used to be supported, in
    ``search["weight_quantization"]``, which is still read so those checkpoints stay checked.

    Args:
        path: The checkpoint's path, for the message.
        activation: The run's ``{"bits", "group_size", "symmetric", "groupwise_roles"}``; bits None or >= 16
            means activations are not quantized.
        weight: The run's ``{"bits", "group_size", "symmetric"}``, checked against a weight-only search.
        checkpoint: The loaded checkpoint; read from path when not given.
        model_name: The model being quantized, checked against the one the rotation was searched for.
    """
    checkpoint = torch.load(path, map_location="cpu") if checkpoint is None else checkpoint
    searched_model = checkpoint.get("model")
    if model_name is not None and searched_model is not None and searched_model != model_name:
        raise ValueError(
            f"the rotation in {path} was searched for {searched_model}, but this run quantizes {model_name}. "
            f"A rotation belongs to the model it was searched on; search one for this model."
        )
    saved = dict(checkpoint.get("activation_quantization") or {})
    search = checkpoint.get("search") or {}
    mismatches = []
    bits = activation.get("bits")
    bits = None if bits is None or bits >= 16 else bits
    saved_bits = saved.get("bits")
    if saved_bits != bits:
        mismatches.append(f"activation bits {saved_bits} searched, {bits} requested")
    elif bits is not None:
        for key in ("group_size", "symmetric"):
            if saved.get(key) != activation.get(key):
                mismatches.append(f"activation {key} {saved.get(key)!r} searched, {activation.get(key)!r} requested")
        roles = activation.get("groupwise_roles")
        roles = None if roles is None else sorted(roles)
        if saved.get("groupwise_roles") != roles:
            mismatches.append(f"group-wise roles {saved.get('groupwise_roles')} searched, {roles} requested")
    searched_weights = search.get("weight_quantization") if search.get("quantization") == "weights" else None
    if searched_weights is not None and weight is not None:
        for key in ("bits", "group_size", "symmetric"):
            if searched_weights.get(key) != weight.get(key):
                mismatches.append(
                    f"weight {key} {searched_weights.get(key)!r} searched, {weight.get(key)!r} requested"
                )
    if mismatches:
        raise ValueError(
            f"the rotation in {path} was searched under different quantization settings: "
            + "; ".join(mismatches)
            + ". Search a rotation for these settings, or set transform.check_settings=false to apply it anyway."
        )


def _load_rotations(checkpoint, device):
    """The ``(R1, R2s)`` a checkpoint holds, on device; R1 is ``{stream: R1}`` for several streams."""
    saved_r1 = checkpoint["R1"]
    R1 = (
        {stream: rotation.to(device) for stream, rotation in saved_r1.items()}
        if isinstance(saved_r1, Mapping) else saved_r1.to(device)
    )
    return R1, {name: R2.to(device) for name, R2 in checkpoint["R2s"].items()}


def _check_rotations_fit_model(path, checkpoint, model, layers_to_rotate, R1, R2s, use_r2) -> None:
    """Raise if a saved rotation does not match the model it is about to be folded into.

    A rotation searched for another model can still line up in some dimensions, so both halves are
    checked: one R2 per attention module this model has, and an R1 as wide as the residual stream
    each group reads. Without this the folds raise an opaque shape error somewhere in the middle of
    a half-rotated model, or none at all.
    """
    searched_for = f" It records {checkpoint['model']}." if checkpoint.get("model") else ""
    if use_r2:
        missing = [group[0] for group in layers_to_rotate if group[0] not in R2s]
        if missing:
            raise ValueError(
                f"the rotation in {path} has no R2 for {len(missing)} of this model's attention modules, the "
                f"first being {missing[0]!r}; it was searched for a different model." + searched_for
            )
    moduledict = dict(model.named_modules())
    for group in layers_to_rotate:
        first_layer = group[2][0][0]
        width = _weight_2d(moduledict[first_layer]).shape[1]
        rotation = _group_rotation(R1, group)
        if rotation.shape[0] != width:
            raise ValueError(
                f"the rotation in {path} has a {tuple(rotation.shape)} R1, but {first_layer} of this model reads "
                f"{width} channels; it was searched for a different model." + searched_for
            )


def apply_rotations(
    model,
    path,
    layers_to_rotate,
    norm_layers,
    hadamard_block_size,
    online_hadamard_layers,
    attach_rotation_hooks,
):
    """Fold saved rotations into a model's weights, for inference.

    Mirrors learn_rotations without the search: the norms are folded first, then the rotations
    read from path are folded into the weights instead of patched into the forward passes. The
    norm fold has to come first because the rotation folds assume the norms carry no scale --
    only then does the residual stream commute with R1.

    Which rotations exist is read from the checkpoint: a rotation searched without R2 or without
    FC2's online Hadamard is folded without them, and a checkpoint that predates those options is
    folded with both.

    The online Hadamard becomes an OnlineHadamard module after each FC2's activation, running
    humming's kernel, rather than the dense matmul used during the search. Both apply the same
    matrix, so the searched rotation stays valid; see online_hadamard_matrix for why it is built
    from the kernel in the first place.

    Args:
        model: The model to rotate, modified in place.
        path: Checkpoint written by learn_rotations.
        layers_to_rotate: ``(attention_name, head_dim, [(layer_name, kind), ...])`` entries,
            with a fourth element naming the residual stream for a checkpoint with one R1 per stream.
        norm_layers: ``(norm_name, previous_names, next_names)`` entries.
        hadamard_block_size: Block size of the online Hadamard. Must match the value the
            rotation was searched with, which is stored in the checkpoint and checked here.
        online_hadamard_layers: ``(activation_name, fc2_name)`` pairs, one for every
            ROT_OUTPUT_R1 layer.
        attach_rotation_hooks: ``fn(model, R1, persistent) -> handles``, called with
            ``persistent=True``; see learn_rotations. Required: without it a Whisper or Parakeet
            model is rotated but its residual stream is not, and its output is wrong.

    The residual-stream transforms are attached as ResidualStreamRotation modules, so the rotated
    model is self-contained: torch.save(model) and torch.load restore it without calling this
    again, and its state_dict holds R1. Loading that state_dict into a fresh model still needs the
    same structure, which calling this with the same checkpoint first provides.

    Returns:
        The handles of the residual-stream rotations; ``.remove()`` on one detaches it.
    """
    checkpoint = torch.load(path, map_location="cpu")
    use_r2 = checkpoint.get("learn_r2", True)
    fc2_online_hadamard = checkpoint.get("fc2_online_hadamard", True)
    saved_block_size = checkpoint.get("hadamard_block_size")
    if fc2_online_hadamard and saved_block_size not in (None, hadamard_block_size):
        raise ValueError(
            f"the rotation in {path} was searched with hadamard_block_size="
            f"{saved_block_size}, but {hadamard_block_size} was requested. The absorbed and "
            f"the online Hadamard have to be the same matrix, so these cannot differ."
        )

    device = next(model.parameters()).device
    R1, R2s = _load_rotations(checkpoint, device)
    _check_rotations_fit_model(path, checkpoint, model, layers_to_rotate, R1, R2s, use_r2)
    hadamard_sign_seed = checkpoint.get("hadamard_sign_seed") or 0
    saved_signs = checkpoint.get("hadamard_signs")
    hadamard_signs = None
    if hadamard_sign_seed:
        hadamard_signs = {
            width: seeded_hadamard_signs(hadamard_sign_seed, width, device)
            for width in _fc2_input_widths(model, layers_to_rotate)
        }
    elif saved_signs is not None:
        hadamard_signs = {int(w): v.to(device) for w, v in saved_signs.items()}

    fold_norms(model, norm_layers)
    handles = []
    if attach_rotation_hooks is not None:
        handles = attach_rotation_hooks(model, R1, persistent=True)
    fold_rotations(
        model, layers_to_rotate, R1, R2s, hadamard_block_size, online_hadamard_layers,
        hadamard_signs, use_r2=use_r2, fc2_online_hadamard=fc2_online_hadamard,
        hadamard_sign_seed=hadamard_sign_seed,
    )
    return handles
