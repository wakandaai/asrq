# pyright: reportMissingImports=false

import torch
from typing import Optional, Union, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
import types
from asrq.transforms.rotation.hadamard_utils import random_hadamard_matrix
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
    @staticmethod
    def forward(ctx, x, bit, row_wise=True):
        dim = 1 if row_wise else 0
        zero_point = x.min(dim=dim, keepdim=True).values
        scale = (x.max(dim=dim, keepdim=True).values - zero_point) / (2 ** bit - 1)
        q = torch.round((x - zero_point) / scale) * scale + zero_point
        return q
        
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None
    

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
        linear: nn.Linear,
        Q: nn.Parameter,
        Q2: Optional[nn.Parameter] = None,
        for_rotated_input: bool = True,
        quantize_row_wise: bool = True,
        bit: int = 4,
) -> None:
    """Modify the given linear layer to include the rotation parameter Q in its forward pass."""

    def modified_forward(self, x: torch.Tensor) -> torch.Tensor:
        # quantize the input activations with STE quantization
        if getattr(self, "_rotation_quantize_activation", False):
            x = STEQuantize.apply(x, getattr(self, "_rotation_activation_bits", 8), True)
        # Apply the rotation to the weight
        rotated_bias = self.bias
        dtype = self.weight.dtype
        double_type = torch.float64
        if for_rotated_input:
            rotated_weight = (self.weight.to(double_type) @ Q.to(double_type)).to(dtype)
            if Q2 is not None:
                hdim = Q2.shape[0]
                w_ = rotated_weight.t()
                org_shape = w_.shape
                temp = w_.reshape(-1, org_shape[-1]//hdim, hdim)
                temp = (temp.to(double_type) @ Q2.to(double_type)).to(dtype)
                rotated_weight = temp.reshape(org_shape).t()
                if self.bias is not None:
                    org_shape = self.bias.shape
                    temp = self.bias.reshape(-1, org_shape[-1]//hdim, hdim)
                    temp = (temp.to(double_type) @ Q2.to(double_type)).to(dtype)
                    rotated_bias = temp.reshape(org_shape).to(self.bias.dtype)

        else:
            rotated_weight = (Q.T.to(double_type) @ self.weight.data.to(double_type)).to(dtype)
            if self.bias is not None:
                rotated_bias = (self.bias.data.to(double_type) @ Q.to(double_type)).to(x.dtype)
            if Q2 is not None:
                hdim = Q2.shape[0]
                org_shape = rotated_weight.shape
                temp = rotated_weight.reshape(-1, org_shape[-1]//hdim, hdim)
                temp = (temp.to(double_type) @ Q2.to(double_type)).to(dtype)
                rotated_weight = temp.reshape(org_shape)

        
        # Perform RTN quantization of weights           
        if getattr(self, "_rotation_quantize_weight", False):
            w = STEQuantize.apply(
                rotated_weight,
                getattr(self, "_rotation_weight_bits", bit),
                quantize_row_wise,
            )
        else:
            w = rotated_weight
        # w = rotated_weight.to(dtype)
        # continue with the normal linear forward using the rotated weight
        return F.linear(x, w.to(x.dtype), rotated_bias)

    linear._rotation_search_ready = True
    linear._rotation_quantize_weight = False
    linear._rotation_quantize_activation = False
    linear._rotation_weight_bits = bit
    linear._rotation_activation_bits = 8
    linear.forward = types.MethodType(modified_forward, linear)


def set_rotation_fake_quant_state(
        model: nn.Module,
        *,
        enabled: bool,
        activation_bits: int = 8,
        weight_bits: int = 4,
) -> None:
    """Toggle fake-quantized replay for linears patched by rotation utilities."""
    for module in model.modules():
        if getattr(module, "_rotation_search_ready", False):
            module._rotation_quantize_weight = enabled
            module._rotation_quantize_activation = enabled and activation_bits < 16
            module._rotation_weight_bits = weight_bits
            module._rotation_activation_bits = activation_bits


def fuse_rotation_param_into_linear(
        linear: nn.Linear | RMSNormFusedM,
        Q: torch.Tensor,
        Q2: Optional[torch.Tensor] = None,
        for_rotated_input: bool = True,
) -> None:
    """Fuse the rotation parameter Q into the given linear layer's weights (and bias if for_rotated_input=False)."""
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
