# pyright: reportMissingImports=false

import torch
from typing import Optional, Union, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import types
from asrq.transforms.rotation.hadamard_utils import matmul_hadU, matmul_hadU_auto, random_hadamard_matrix
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm



class ConvertedLayerNorm(nn.Module):
    """Converts a LayerNorm layer to one with an RMSNorm such that the scales and shift 
    can be fused into adjacent linear layers.

    LayerNorm(X) = RMSNorm(XM)diag(α)√D + 1N β⊤  
    LayerNorm(X) = RMSNorm(XM)diag(weight) + bias
    M = I - (1/D) 1N 1N⊤ 

    """
    def __init__(self, layernorm: nn.LayerNorm) -> None:
        super().__init__()
        self.eps = layernorm.eps
        self.weight = layernorm.weight
        self.normalized_shape = layernorm.normalized_shape
        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            self.bias = layernorm.bias
        else:
            self.bias = None
        self.M = torch.eye(layernorm.normalized_shape[0], device=layernorm.weight.device) - \
            torch.ones((layernorm.normalized_shape[0], layernorm.normalized_shape[0]), device=layernorm.weight.device) * (1.0 / layernorm.normalized_shape[0])
        if self.M.device.type != "cuda":
            raise Exception("ConvertedLayerNorm M matrix must be on CUDA device")

    def rmsnorm(self, x: torch.Tensor) -> torch.Tensor:
        # return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return F.rms_norm(x, normalized_shape=self.normalized_shape, eps=self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x @ self.M) * self.weight + (self.bias if self.bias is not None else 0.0)


class RMSNormFused(nn.Module):
    """A plain RMSNorm layer with no bias, no weight, used after fusing the LayerNorm parameters into adjacent linears."""
    def __init__(self, norm: ConvertedLayerNorm) -> None:
        super().__init__()
        self.eps = norm.eps
        self.normalized_shape = norm.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, normalized_shape=self.normalized_shape, eps=self.eps)


class RMSNormFusedM(nn.Module):
    """RMSNorm when mean subtraction has been fused into the model but the weight and bias have not yet been fused.
    
    This is specifically used for the last layer norm, just after the decoder 
    """
    def __init__(self, norm: ConvertedLayerNorm) -> None:
        super().__init__()
        self.eps = norm.eps
        self.normalized_shape = norm.normalized_shape
        self.weight = norm.weight
        self.bias = norm.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.weight.shape) == 1:
            return F.rms_norm(x, normalized_shape=self.normalized_shape, eps=self.eps) * self.weight + (self.bias if self.bias is not None else 0.0)
        else:
            return F.rms_norm(x, normalized_shape=self.normalized_shape, eps=self.eps) @ self.weight + (self.bias if self.bias is not None else 0.0)


class STEQuantize(torch.autograd.Function):
    """Symmetric fake-quantization with a straight-through gradient.

    Matches the quantizer the model is deployed with
    (``asrq.quantizers.activation.activation_quantization_forward_patch``): abs-max scale,
    no zero point, so the rotation is trained against the scheme it will be evaluated on.
    """
    @staticmethod
    def forward(ctx, x, bit, row_wise=True):
        dim = 1 if row_wise else 0
        scale = x.abs().max(dim=dim, keepdim=True).values / (2 ** (bit - 1) - 1)
        scale = scale.where(scale != 0, 1e-3)  # avoid division by 0
        q = torch.round(x / scale) * scale
        return q
        
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None
    

def ste_quantize_weight(w: torch.Tensor, bit: int, group_size: Optional[int] = None) -> torch.Tensor:
    """Symmetric fake-quantization of a weight, per output channel or per group.

    ``group_size=None`` quantizes each output channel as a unit (STEQuantize's own dim-1
    reduction on an ``(out, in)`` weight). A group size instead quantizes each contiguous
    run of that many input channels separately, matching the deployed quantizers'
    groupwise scales - and matching a block-diagonal rotation of the same width, whose
    whole point is to keep the mixing inside one group.

    Done in float32: at high bit widths ``(x - zero_point) / scale`` overflows fp16.
    """
    if bit >= 16:
        return w
    org_shape = w.shape
    flat = w.float() if group_size is None else w.float().reshape(-1, group_size)
    q = STEQuantize.apply(flat, bit, True)  # type: ignore
    return q.reshape(org_shape).to(w.dtype)


def random_orthogonal_matrix(
    n: int,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    Draws a random Gaussian matrix and orthogonalises it, adjusting signs so
    the result is uniformly distributed over O(n) (Haar measure).

    Args:
        n: Matrix dimension.
        device: Target device.
        seed: Optional RNG seed.

    Returns:
        Orthogonal ``(n, n)`` tensor in ``float64``.
    """
    if seed is not None:
        _gen = torch.Generator()
        _gen.manual_seed(seed)
        random_matrix = torch.randn(n, n, dtype=torch.float64, generator=_gen).to(device)
    else:
        random_matrix = torch.randn(n, n, dtype=torch.float64, device=device)
    q, r = torch.linalg.qr(random_matrix)
    # Fix the sign ambiguity so we sample from Haar measure
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(
    n: int,
    mode: str = "hadamard",
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Factory: return an orthogonal matrix of size *n* using the given *mode*.

    Args:
        n: Matrix dimension.
        mode: ``"hadamard"`` or ``"random"``.
        device: Target device.
        seed: Optional RNG seed.

    Returns:
        Orthogonal ``(n, n)`` tensor in ``float64``.
    """
    if mode == "hadamard":
        return random_hadamard_matrix(n, device, seed=seed)
    elif mode == "random":
        return random_orthogonal_matrix(n, device=device, seed=seed)
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'hadamard' or 'random'.")


def convert_layernorm_to_rmsnorm(layernorm: nn.LayerNorm) -> ConvertedLayerNorm:
    result = ConvertedLayerNorm(layernorm)
    return result


def convert_model_layernorms_to_rmsnorms(model: nn.Module) -> None:
    """Convert all LayerNorm modules in the model to ConvertedLayerNorm (RMSNorm with fusion-friendly parameters)."""
    for name, child in model.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(model, name, convert_layernorm_to_rmsnorm(child))
        else:
            # Always recurse into children, including ModuleList/Sequential members
            convert_model_layernorms_to_rmsnorms(child)


@torch.no_grad()
def fuse_normalization_weights_and_bias_into_adjacent_linears(
        model: nn.Module,
        norm_n_adj_linears: List[Tuple[str, List[str], List[str]]],
        ) -> None:
    """For each norm layer specified in norm_n_adj_linears, fuse its weight and bias into the adjacent linear layers."""
    named_modules = dict(model.named_modules())
    print("Starting fusion of normalization weights and biases into adjacent linear layers...")
    print(f"Found {len(norm_n_adj_linears)} norm layers to process.")
    for norm_name, pre_linear_names, post_linear_names in norm_n_adj_linears:
        # print(f"Processing norm '{norm_name}' with preceding linears {pre_linear_names} and succeeding linears {post_linear_names}...")
        # print(f"Norm: {norm_name}\nPre linears: {pre_linear_names}\nPost linears: {post_linear_names}\n")
        norm_module = named_modules.get(norm_name)
        if norm_module is None:
            raise ValueError(f"Normalization module '{norm_name}' not found in model.")
        
        if isinstance(norm_module, ConvertedLayerNorm):
            M = norm_module.M.double()
            # pre linear
            for pre_linear_name in pre_linear_names:
                if pre_linear_name.startswith("first_layer"):
                    # special case for the first layer norm. Here we ensure that the input into the block is already mean-subtracted
                    # So we have a mean-subtracted residual stream
                    # So we introduce a mean subtraction just before the block
                    # This should be done elsewhere in the code. 
                    # It would involve modifying the forward pass implementation
                    continue
                if pre_linear_name == "last":
                    # special case for the last layer norm, just after the encoder or decoder blocks
                    # The input here is coming from the residual stream and it is alread mean-subtracted.
                    continue

                pre_linear = named_modules.get(pre_linear_name)
                if pre_linear is None:
                    raise ValueError(f"Preceding linear module '{pre_linear_name}' not found in model.")

                # For the fist normalization layer of an intermidiate conformer block, the preceding layer can be
                # a layer normalization
                if isinstance(pre_linear, ConvertedLayerNorm): #.norm_out
                    # Fuse the mean subtraction into the weight and bias of the LayerNorm
                    W = torch.diag(pre_linear.weight.data.double()) @ M # type: ignore
                    pre_linear.weight.data = W.to(dtype=pre_linear.weight.data.dtype, device=pre_linear.weight.data.device) # type: ignore
                    if pre_linear.bias is not None:
                        beta = pre_linear.bias.data.double().unsqueeze(0) # type: ignore
                        pre_linear.bias.data = (beta @ M).squeeze(0).to(dtype=pre_linear.bias.data.dtype, device=pre_linear.bias.data.device) # type: ignore
                else:
                # fuse the Mean Subtraction Matrix into the preceding linear layer
                    W = M @ pre_linear.weight.data.double().flatten(1) # type: ignore
                    pre_linear.weight.data = W.to(dtype=pre_linear.weight.data.dtype, device=pre_linear.weight.data.device).reshape(pre_linear.weight.data.shape) # type: ignore
                    if pre_linear.bias is not None:
                        b = pre_linear.bias.data.double().unsqueeze(0)  # (1, hidden_size) # type: ignore
                        pre_linear.bias.data = (b @ M).squeeze(0).to(dtype=pre_linear.bias.data.dtype, device=pre_linear.bias.data.device) # type: ignore
            # fuse the weight and bias into the succeeding linear layer
            for post_linear_name in post_linear_names:
                post_linear = named_modules.get(post_linear_name)
                if post_linear is None:
                    raise ValueError(f"Succeeding linear module '{post_linear_name}' not found in model.")
                W_old = post_linear.weight.data.double().flatten(1)  # save before gamma multiplication # type: ignore
                W_new = W_old * norm_module.weight.double().unsqueeze(0)  # (out, hidden) * (1, hidden) # type: ignore
                post_linear.weight.data = W_new.to(dtype=post_linear.weight.data.dtype, device=post_linear.weight.data.device).reshape(post_linear.weight.shape) # type: ignore
                if norm_module.bias is not None:
                    beta = norm_module.bias.data.double()
                    if post_linear.bias is None:
                        out_features = post_linear.weight.data.shape[0]
                        post_linear.bias = nn.Parameter(torch.zeros(out_features, dtype=post_linear.weight.data.dtype, device=post_linear.weight.data.device))
                    # bias_new = bias_old + beta @ W_old^T  (must use W_old, not W_new)
                    post_linear.bias.data = (post_linear.bias.data.double() + beta.unsqueeze(0) @ W_old.t()).squeeze(0).to(dtype=post_linear.bias.data.dtype, device=post_linear.bias.data.device) # type: ignore
                    
            # Reset norm weight/bias to identity after fusion
            if post_linear_names:
                norm_module.weight.data = torch.ones_like(norm_module.weight.data) # type: ignore
                if norm_module.bias is not None:
                    norm_module.bias.data = torch.zeros_like(norm_module.bias.data) # type: ignore

            # Mean subtraction matrix becomes identity after fusion, so we can remove it to get a plain RMSNorm
            if pre_linear_names:
                norm_module.M = torch.eye(norm_module.M.size(0), dtype=norm_module.M.dtype, device=norm_module.M.device) # type: ignore

            # now modify the norm_module entirely
            if pre_linear_names and post_linear_names:
                # This is the standard case where we have both pre and post linears. We can replace with a plain RMSNorm with no bias or weight
                new_module = RMSNormFused(norm_module)
                # now replace the norm module
                parent_module = model
                names = norm_name.split(".")
                for n in names[:-1]:
                    if n.isdigit():
                        parent_module = parent_module[int(n)] # type: ignore
                    else:
                        parent_module = getattr(parent_module, n)
                setattr(parent_module, names[-1], new_module)
            elif pre_linear_names and not post_linear_names:
                # This is the special case for the last layer norm in the encoder or decoder, where we only have pre linears and no post linears. In this case we replace with an RMSNorm that still has the weight and bias, but with the mean subtraction removed.
                new_module = RMSNormFusedM(norm_module)
                # now replace the norm module
                parent_module = model
                names = norm_name.split(".")
                for n in names[:-1]:
                    if n.isdigit():
                        parent_module = parent_module[int(n)] # type: ignore
                    else:
                        parent_module = getattr(parent_module, n)
                setattr(parent_module, names[-1], new_module)

        elif isinstance(norm_module, nn.LayerNorm):
            # Fuse LayerNorm gamma/beta directly into succeeding linears.
            # The LayerNorm itself remains (mean-subtraction + variance norm),
            # but with weight=1, bias=0 it becomes a plain normalisation op.
            breakpoint()
            gamma = norm_module.weight.data.double()
            beta = norm_module.bias.data.double() if norm_module.bias is not None else None
            for post_linear_name in post_linear_names:
                post_linear = named_modules.get(post_linear_name)
                if post_linear is None:
                    raise ValueError(f"Succeeding linear module '{post_linear_name}' not found in model.")
                post_dtype = post_linear.weight.data.dtype
                post_dev = post_linear.weight.data.device
                W_old = post_linear.weight.data.double().flatten(1)  # save before gamma multiplication # type: ignore
                # W_new = W_old * diag(gamma)  (broadcast over out_features)
                post_linear.weight.data = (W_old * gamma.unsqueeze(0)).to(dtype=post_dtype, device=post_dev).reshape( # type: ignore
                    post_linear.weight.data.shape # type: ignore
                )
               
                if beta is not None:
                    # bias_new = bias_old + beta @ W_old^T  (must use W_old, not W_new)
                    if post_linear.bias is None:
                        out_features = post_linear.weight.data.shape[0]
                        post_linear.bias = nn.Parameter(torch.zeros(out_features, dtype=post_dtype, device=post_dev)) # type: ignore
                    try:
                        post_linear.bias.data = (
                            post_linear.bias.data.double() + beta.unsqueeze(0) @ W_old.t()
                        ).squeeze(0).to(dtype=post_dtype, device=post_dev) # type: ignore
                    except:
                        breakpoint()
            # Reset norm to identity after fusion
            if post_linear_names:
                norm_module.weight.data = torch.ones_like(norm_module.weight.data)
                if norm_module.bias is not None:
                    norm_module.bias.data = torch.zeros_like(norm_module.bias.data)

        elif isinstance(norm_module, (nn.RMSNorm, Qwen3RMSNorm)):
            # fuse the weight into the succeeding linear layer
            for post_linear_name in post_linear_names:
                post_linear = named_modules.get(post_linear_name)
                if post_linear is None:
                    raise ValueError(f"Succeeding linear module '{post_linear_name}' not found in model.")
                W = post_linear.weight.data.double() * norm_module.weight.double().unsqueeze(0)  # (1, hidden_size) # type: ignore
                post_linear.weight.data = W.to(dtype=post_linear.weight.data.dtype, device=post_linear.weight.data.device) # type: ignore
            # Reset norm weight to identity after fusion
            if post_linear_names:
                norm_module.weight.data = torch.ones_like(norm_module.weight.data)

        else:
            raise ValueError(f"Normalization module '{norm_name}' of type {type(norm_module).__name__} not supported for fusion. Only LayerNorm, RMSNorm, and Qwen3RMSNorm are supported.")
            if hasattr(norm_module, 'weight') and not hasattr(norm_module, 'bias'):
                for post_linear_name in post_linear_names:
                    post_linear = named_modules.get(post_linear_name)
                    if post_linear is None:
                        raise ValueError(f"Succeeding linear module '{post_linear_name}' not found in model.")
                    W = post_linear.weight.data.double() * norm_module.weight.double().unsqueeze(0)
                    post_linear.weight.data = W.to(dtype=post_linear.weight.data.dtype, device=post_linear.weight.data.device)
                if post_linear_names:
                    norm_module.weight.data = torch.ones_like(norm_module.weight.data)
            else:
                print(f"Warning: Norm module '{norm_name}' of type {type(norm_module).__name__} not handled by fusion.")

    print("Fused normalization weights and biases into adjacent linear layers.")


def modify_linear_with_rotation_param(
        linear: Union[nn.Linear, nn.Conv1d, RMSNormFusedM],
        Q: Optional[nn.Parameter],
        Q2: Optional[nn.Parameter] = None,
        for_rotated_input: bool = True,
        for_norm_out: bool = False,
        quantize_row_wise: bool = True,
        bit: int = 4,
        activation_bits: int = 16,
        quantize_weights: bool = False,
        weight_group_size: Optional[int] = None,
        online_hadamard: bool = False,
        hadamard_block_size: Optional[int] = None,
        rotation_dtype: Optional[torch.dtype] = None,
) -> None:
    """Rotate this layer's weight on the fly, inside its forward pass.

    This is the training-time form of the rotation: nothing is written to the weight, so
    the same layer can be re-evaluated under a different Q just by changing Q's contents.
    :func:`fuse_rotation_param_into_linear` is the inference-time counterpart, which bakes
    the rotation in once.

    Three weights layouts are handled, differing only in which axis is contracted:
    ``nn.Linear`` (F.linear transposes, so `in` is the last dim), pointwise ``nn.Conv1d``
    ((out, in, k) activations (B, C, T)), and ``RMSNormFusedM`` (``rms_norm(x) @ w``, no
    transpose, so `in` is dim 0).

    Args:
        Q: residual-stream rotation, or None to skip it (canary's LoRA B matrices).
        Q2: head-wise rotation applied within ``Q2.shape[0]``-sized blocks.
        for_rotated_input: the input already arrives rotated, so the weight is
            right-multiplied by Q. False means the output feeds the rotated residual, so
            the weight is left-multiplied by Q^T instead.
        for_norm_out: this is a norm whose weight becomes ``Q^T diag(w) Q``.
        activation_bits: width of the STE activation quantizer, applied every forward so
            the rotation is learned against quantized activations. >= 16 disables it.
        quantize_weights / bit / weight_group_size: STE weight quantizer, for the
            weight-only setting where activations stay in fp16.
        online_hadamard: additionally rotate x by a Hadamard H at run time and apply the
            same H to the weight's contracted axis. Because the layer transposes or
            contracts that axis, the pair cancels - ``(x H)(W H)^T == x W^T`` - so the
            output is unchanged while both operands quantize in the Hadamard basis.
        hadamard_block_size: rotate independent blocks of this width rather than the whole
            contracted axis, e.g. the head dim to match a head-wise Q2.
        rotation_dtype: precision for the rotation matmuls. None uses Q's dtype; whisper
            passes float64.
    """
    is_conv = isinstance(linear, nn.Conv1d)
    is_norm_fused = isinstance(linear, RMSNormFusedM)

    def rot_dtype(default: torch.Tensor) -> torch.dtype:
        return rotation_dtype if rotation_dtype is not None else default.dtype

    def hadamard_rotate_last(t: torch.Tensor) -> torch.Tensor:
        """Right-multiply the last dim of ``t`` by H (blockwise if requested)."""
        if hadamard_block_size is None:
            return matmul_hadU_auto(t)
        org_shape = t.shape
        t = t.reshape(*org_shape[:-1], org_shape[-1] // hadamard_block_size, hadamard_block_size)
        return matmul_hadU_auto(t).reshape(org_shape)

    def hadamard_rotate_input(x: torch.Tensor) -> torch.Tensor:
        if is_conv:
            # Conv1d activations are (B, C, T): the contraction is over C, not the last dim.
            return hadamard_rotate_last(x.transpose(1, 2).contiguous()).transpose(1, 2)
        return hadamard_rotate_last(x)

    def hadamard_rotate_weight(w: torch.Tensor) -> torch.Tensor:
        if w.dim() < 2:
            raise ValueError(
                "online_hadamard needs a 2-D weight to rotate; got a 1-D weight, which is "
                "an elementwise scale. Fuse the norm weight first, or pass online_hadamard=False."
            )
        if is_conv:
            return hadamard_rotate_last(w.transpose(1, 2).contiguous()).transpose(1, 2)
        if is_norm_fused:
            return hadamard_rotate_last(w.t().contiguous()).t()
        return hadamard_rotate_last(w)

    def rotate_head_blocks(t: torch.Tensor, hdim: int) -> torch.Tensor:
        """Apply Q2 within each ``hdim``-wide block of the last dim."""
        org_shape = t.shape
        blocks = t.reshape(-1, org_shape[-1] // hdim, hdim)
        return (blocks.to(rot_dtype(Q2)) @ Q2).reshape(org_shape)  # type: ignore[arg-type]

    def modified_forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- input side: rotate, then quantize, so x is quantized in the rotated basis
        if online_hadamard:
            if is_conv and self.groups != 1:
                raise ValueError(
                    f"online_hadamard mixes channels, which is invalid for a grouped conv "
                    f"(groups={self.groups}). Pass online_hadamard=False for this layer."
                )
            x = hadamard_rotate_input(x)
        if activation_bits < 16:
            # Flattened to 2-D first: STEQuantize reduces over dim 1, which is per-token
            # only for (tokens, channels) - on a raw (B, T, C) it would reduce over time.
            x = STEQuantize.apply(  # type: ignore[misc]
                x.reshape(-1, x.shape[-1]).float(), activation_bits, quantize_row_wise
            ).reshape(x.shape).to(x.dtype)

        # ---- weight side: apply Q (and Q2) for this layer's role
        rotated_weight = self.weight
        rotated_bias = self.bias
        orig_shape = self.weight.shape

        if for_norm_out:
            # W_rotated = Q^T diag(w) Q -- a full DxD matrix when the norm weight is 1-D.
            w = self.weight.double()
            if w.dim() == 1:
                w = torch.diag(w)
            rotated_weight = (Q.t().double() @ w) @ Q.double()  # type: ignore[union-attr]
            if rotated_bias is not None:
                rotated_bias = rotated_bias.unsqueeze(0).double() @ Q.double()  # type: ignore[union-attr]
        elif for_rotated_input:
            if Q is not None:
                rotated_weight = self.weight.to(rot_dtype(Q)).flatten(1) @ Q.to(rot_dtype(Q))
            if Q2 is not None:
                # Q2 rotates within heads of the OUTPUT dim here, hence the transpose.
                rotated_weight = rotate_head_blocks(rotated_weight.t(), Q2.shape[0]).t()
                if self.bias is not None:
                    rotated_bias = rotate_head_blocks(self.bias, Q2.shape[0]).to(self.bias.dtype)
        else:
            if Q is not None:
                rotated_weight = Q.to(rot_dtype(Q)).T @ self.weight.to(rot_dtype(Q)).flatten(1)
                if self.bias is not None:
                    rotated_bias = (self.bias.data.to(rot_dtype(Q)) @ Q.to(rot_dtype(Q))).to(x.dtype)
            if Q2 is not None:
                # No transpose: Q2 rotates within heads of the INPUT dim (the last one).
                rotated_weight = rotate_head_blocks(rotated_weight, Q2.shape[0])

        if for_norm_out and rotated_weight.shape != orig_shape:
            w = rotated_weight
            if rotated_bias is not None:
                rotated_bias = rotated_bias.squeeze(0).to(x.dtype)
        else:
            w = rotated_weight.reshape(orig_shape)
            if rotated_bias is not None:
                rotated_bias = rotated_bias.reshape(self.bias.shape).to(x.dtype)

        if quantize_weights:
            # Weight-only setting: with 16-bit activations this is the only thing in the
            # forward for the rotation to be scored against.
            w = ste_quantize_weight(w, bit, weight_group_size)
        if online_hadamard:
            w = hadamard_rotate_weight(w.to(x.dtype))

        # ---- the layer's own forward, with the rotated weight
        if is_norm_fused:
            out = F.rms_norm(x, normalized_shape=self.normalized_shape, eps=self.eps) @ w.to(x.dtype)
            return out + (rotated_bias if self.bias is not None else 0.0)
        if is_conv:
            return F.conv1d(x, w.to(x.dtype), rotated_bias,
                            self.stride, self.padding, self.dilation, self.groups)
        return F.linear(x, w.to(x.dtype), rotated_bias)

    linear.forward = types.MethodType(modified_forward, linear)


def apply_to_rotated_layers(model, layers_to_rotate, resolve, apply) -> None:
    """Walk a model's rotation plan and hand each layer to ``apply``.

    Every model's modify/fuse pass is the same loop - look the layer up, work out which
    rotations it needs, call the per-layer function - differing only in that middle step.
    That policy lives in the model's own ``resolve``; this driver owns the walk.

    Args:
        layers_to_rotate: ``(layer_name, for_rotated_input)`` pairs from the model's
            ``get_*_layers_to_rotate``.
        resolve: ``(layer_name) -> (Q, Q2, extra_kwargs)``. ``Q`` may be None (canary's
            LoRA B matrices rotate on the Q2 side only); ``extra_kwargs`` carries per-layer
            decisions such as ``for_norm_out`` and ``online_hadamard``.
        apply: ``modify_linear_with_rotation_param`` or
            ``fuse_rotation_param_into_linear``, plus any keyword arguments already bound.
    """
    named_modules = dict(model.named_modules())
    for layer_name, for_rotated_input in layers_to_rotate:
        layer = named_modules.get(layer_name)
        if layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model.")
        Q, Q2, extra = resolve(layer_name)
        apply(layer, Q, Q2=Q2, for_rotated_input=for_rotated_input, **extra)


def matches_layer_suffix(layer_name: str, suffixes: Tuple[str, ...]) -> bool:
    """Whether ``layer_name`` ends in one of ``suffixes`` on a module boundary.

    Used to pick out the down-projections, which are the only layers that get an online
    Hadamard: their input is the MLP intermediate rather than the Q-rotated residual
    stream, so nothing else has spread the outliers in that activation.
    """
    return any(layer_name == s or layer_name.endswith("." + s) for s in suffixes)


def fuse_hadamard_into_linear(
        linear: nn.Linear | RMSNormFusedM | nn.Conv1d,
        hadamard_block_size: Optional[int] = None,
):
    """Bake the Hadamard rotation H into ``linear``'s weight and rotate its input at run time.

    Where :func:`modify_linear_with_rotation_param` recomputes ``W H`` on every forward,
    this folds H into the weight once, and registers a forward pre-hook that applies H to
    the activations. Inference then costs one fast Hadamard transform on the input plus
    the ordinary linear/conv - no per-forward weight math at all.

    Call this *after* Q has been fused, since the two rotations compose on the same axis.
    The weight side is fused in float64 with the exact pure-torch butterfly (it happens
    once), while the run-time input transform goes through HadaCore.

    Returns the pre-hook handle, so the rotation can be removed with ``handle.remove()``
    (the weight stays fused - undoing that needs a second call with the same H).
    """
    if getattr(linear, "_hadamard_fused", False):
        raise RuntimeError(
            "The Hadamard rotation is already fused into this layer; fusing twice would "
            "rotate the weight a second time while the input is only rotated once."
        )

    is_conv = isinstance(linear, nn.Conv1d)
    is_norm_fused = isinstance(linear, RMSNormFusedM)

    def blockwise(t, transform):
        """Apply ``transform`` to the last dim of ``t``, in blocks if requested."""
        if hadamard_block_size is None:
            return transform(t)
        org_shape = t.shape
        t = t.reshape(*org_shape[:-1], org_shape[-1] // hadamard_block_size, hadamard_block_size)
        return transform(t).reshape(org_shape)

    w = linear.weight.data
    if w.dim() < 2:
        raise ValueError(
            "fuse_hadamard_into_linear needs a 2-D weight; got a 1-D weight, which is an "
            "elementwise scale. Fuse the norm weight into the adjacent linears first."
        )
    dtype, device = w.dtype, w.device

    if is_conv:
        # The Q fusion above flattens (out, in, k) to (out, in*k), which only lines up
        # with an `in`-sized rotation for pointwise convs; keep the same restriction.
        if linear.kernel_size[0] != 1:
            raise ValueError(f"Only pointwise Conv1d is supported, got kernel_size={linear.kernel_size}")
        if linear.groups != 1:
            raise ValueError(
                f"A Hadamard mixes channels, which is invalid for a grouped conv (groups={linear.groups})."
            )
        fused = blockwise(w.double().flatten(1), matmul_hadU).reshape(w.shape)
    elif is_norm_fused:
        # RMSNormFusedM does `rms_norm(x) @ w` with no transpose, so `in` is dim 0 and
        # the weight needs H^T on the left: (w^T H)^T.
        fused = blockwise(w.double().t().contiguous(), matmul_hadU).t()
    else:
        # F.linear transposes the weight, so `in` is already the last dim.
        fused = blockwise(w.double(), matmul_hadU)
    linear.weight.data = fused.to(dtype=dtype, device=device).contiguous()

    def hadamard_pre_hook(module, args):
        x = args[0]
        if is_conv:
            # Conv1d activations are (B, C, T); the contraction is over C.
            x = blockwise(x.transpose(1, 2).contiguous(), matmul_hadU_auto).transpose(1, 2)
        else:
            x = blockwise(x, matmul_hadU_auto)
        return (x,) + tuple(args[1:])

    # Recorded so a later swap can recover the transform - ASRQLinear.from_linear reads
    # these to rebuild the rotation as its own input transform, since the pre-hook below
    # does not survive being replaced by a different module.
    linear._hadamard_fused = True
    linear._hadamard_block_size = hadamard_block_size
    return linear.register_forward_pre_hook(hadamard_pre_hook)


def fuse_rotation_param_into_linear(
        linear: nn.Linear | RMSNormFusedM,
        Q: torch.Tensor,
        Q2: Optional[torch.Tensor] = None,
        for_rotated_input: bool = True,
        online_hadamard: bool = False,
        hadamard_block_size: Optional[int] = None,
) -> None:
    """Fuse the rotation parameter Q into the given linear layer's weights (and bias if for_rotated_input=False).

    With ``online_hadamard=True`` a Hadamard rotation is fused in on top of Q once Q is
    folded in, leaving only a fast Hadamard transform on the input at inference time - see
    :func:`fuse_hadamard_into_linear`.
    """
    dtype = linear.weight.data.dtype
    device = linear.weight.data.device
    if isinstance(linear, RMSNormFusedM):
        w = linear.weight.data.double()
        if w.dim() == 1:
            w = torch.diag(w)
        linear.weight.data = (Q.double().t() @ w @ Q.double()).to(linear.weight.dtype)
        if linear.bias is not None:
            linear.bias.data = (linear.bias.data.unsqueeze(0).double() @ Q.double()).to(linear.bias.dtype)
    elif for_rotated_input: 
        if Q is not None:
            Q_d = Q.double().to(device)
            linear.weight.data = (linear.weight.data.double().flatten(1) @ Q_d).to(dtype=dtype, device=device).reshape(linear.weight.shape)
        if Q2 is not None:
            hdim = Q2.shape[0]
            w_ = linear.weight.data.double().t()
            org_shape = w_.shape
            temp = w_.reshape(-1, org_shape[-1]//hdim, hdim)
            temp = (temp.double() @ Q2.double())
            linear.weight.data = temp.reshape(org_shape).t().to(dtype=dtype, device=device)
            if linear.bias is not None:
                org_shape = linear.bias.shape
                temp = linear.bias.data.double().reshape(-1, org_shape[-1]//hdim, hdim)
                temp = (temp.double() @ Q2.double())
                linear.bias.data = temp.reshape(org_shape).to(dtype=linear.bias.data.dtype, device=linear.bias.data.device)
    else:
        if Q is not None:
            Q_d = Q.double().to(device)
            linear.weight.data = (Q_d.T @ linear.weight.data.double().flatten(1)).to(dtype=dtype, device=device).reshape(linear.weight.shape)
            if linear.bias is not None:
                linear.bias.data = (linear.bias.data.double().unsqueeze(0) @ Q_d).to(dtype=dtype, device=device).reshape(linear.bias.shape)
        if Q2 is not None:
            hdim = Q2.shape[0]
            # No transpose here: Q2 rotates within heads of the INPUT dimension
            # (last dim of weight shape (out, in)), matching the on-the-fly version.
            w_ = linear.weight.data.double()
            org_shape = w_.shape
            temp = w_.reshape(-1, org_shape[-1]//hdim, hdim)
            temp = (temp.double() @ Q2.double())
            linear.weight.data = temp.reshape(org_shape).to(dtype=dtype, device=device)

    if online_hadamard:
        fuse_hadamard_into_linear(linear, hadamard_block_size=hadamard_block_size)


# class