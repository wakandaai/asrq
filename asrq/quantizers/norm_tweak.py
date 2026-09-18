"""Norm tweaking in closed form: a per-channel scale on each norm that feeds GPTQ-quantized layers.

A norm with output X (tokens x d) feeds layers l with full-precision weights W_l and, after GPTQ, quantized
weights Q_l. Scaling the norm's output by s gives the layers X diag(s), and s is chosen so the quantized layers
reproduce the full-precision outputs as closely as possible:

    L(s) = sum_l || X W_l.T - X diag(s) Q_l.T ||_F^2

With x_j, q_lj the j-th columns of X and Q_l, X diag(s) Q_l.T = sum_j s_j x_j q_lj.T, so L is quadratic in s
and dL/ds_j = 0 gives the d x d system

    G s = b,    G = H * sum_l Q_l.T Q_l    (elementwise),    b = diag(H sum_l W_l.T Q_l)

with H = X.T X, the Hessian GPTQ accumulates for those layers. A ridge term pulling s toward 1 (no change),
lambda = ridge * mean(diag(G)), keeps channels the calibration barely excites near 1:

    (G + lambda I) s = b + lambda

The bias of each layer is the same on both sides and cancels. The fit covers weight quantization only: with
quantized activations s also changes what the activation quantizer rounds, which is not linear in s.
"""

from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


def _weight_2d(module: nn.Module) -> torch.Tensor:
    weight = module.weight
    return weight.reshape(weight.shape[0], -1)


def norm_tweak_scale(
    H: torch.Tensor, full_precision: Sequence[torch.Tensor], quantized: Sequence[torch.Tensor], ridge: float,
) -> Tuple[torch.Tensor, float]:
    """Solve for the scale s of one norm; see the module docstring.

    Args:
        H: ``(d, d)`` Hessian of the layers' shared input.
        full_precision: ``(out_l, d)`` weights W_l before quantization.
        quantized: ``(out_l, d)`` weights Q_l after quantization.
        ridge: Strength of the pull toward s = 1, relative to the mean diagonal of G.

    Returns:
        ``(s, ratio)``: the float64 scale, and L(s) / L(1), the output error left relative to no tweak.
    """
    H = H.double()
    width = H.shape[0]
    QtQ = torch.zeros(width, width, dtype=torch.float64, device=H.device)
    WtQ = torch.zeros_like(QtQ)
    for W, Q in zip(full_precision, quantized):
        W, Q = W.double(), Q.double()
        QtQ += Q.T @ Q
        WtQ += W.T @ Q
    G = H * QtQ
    b = (H * WtQ.T).sum(dim=1)
    penalty = ridge * G.diagonal().mean()
    s = torch.linalg.solve(G + penalty * torch.eye(width, dtype=torch.float64, device=H.device), b + penalty)

    def error(scale):
        total = 0.0
        for W, Q in zip(full_precision, quantized):
            E = W.double() - Q.double() * scale
            total += float((E @ H * E).sum())
        return total

    unscaled = error(torch.ones_like(s))
    return s, (error(s) / unscaled if unscaled > 0 else 1.0)


def scale_norm_output(norm: nn.Module, s: torch.Tensor) -> None:
    """Multiply a norm's output by s per channel, in place.

    A norm with an affine weight (LayerNorm, RMSNorm) has it, and its bias, multiplied by s. A scale-free norm, as a
    rotation leaves them, gets s as a new ``weight`` buffer, which it multiplies its output by.
    """
    weight = getattr(norm, "weight", None)
    if weight is not None:
        weight.data.mul_(s.to(weight))
        bias = getattr(norm, "bias", None)
        if bias is not None:
            bias.data.mul_(s.to(bias))
        return
    reference = next((t for t in list(norm.parameters()) + list(norm.buffers()) if t.is_floating_point()), None)
    device = reference.device if reference is not None else s.device
    norm.register_buffer("weight", s.to(device=device, dtype=torch.float32))


def capture_norm_tweaks(
    norm_targets: Mapping[str, Sequence[str]], quantizers: Mapping[str, object], modules: Mapping[str, nn.Module],
) -> List[Tuple[str, List[str], torch.Tensor, List[torch.Tensor]]]:
    """Keep what the closed form needs before a block's layers are quantized.

    Only norms all of whose layers are among this block's GPTQ quantizers are kept: the Hessian of their shared
    input (from the first layer's quantizer, which discards it when it quantizes) and every layer's full-precision
    weight.

    Args:
        norm_targets: ``{norm_name: [layer names it feeds]}``.
        quantizers: ``{layer_name: quantizer}`` of the block being quantized; quantizers without a Hessian (RTN)
            are skipped.
        modules: ``{name: module}`` of the model.
    """
    captured = []
    for norm_name, layer_names in norm_targets.items():
        layer_names = list(layer_names)
        if not layer_names or not all(name in quantizers for name in layer_names):
            continue
        H = getattr(quantizers[layer_names[0]], "H", None)
        if H is None:
            continue
        weights = [_weight_2d(modules[name]).detach().clone() for name in layer_names]
        captured.append((norm_name, layer_names, H.detach().clone(), weights))
    return captured


def apply_norm_tweaks(
    captured: Sequence[Tuple[str, List[str], torch.Tensor, List[torch.Tensor]]], modules: Mapping[str, nn.Module],
    ridge: float,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Solve and apply the scale of every captured norm, once its layers are quantized.

    Returns:
        ``{norm_name: (s, error ratio)}``.
    """
    results = {}
    for norm_name, layer_names, H, weights in captured:
        quantized = [_weight_2d(modules[name]).detach() for name in layer_names]
        s, ratio = norm_tweak_scale(H, weights, [q.to(H.device) for q in quantized], ridge)
        scale_norm_output(modules[norm_name], s)
        results[norm_name] = (s, ratio)
    return results

