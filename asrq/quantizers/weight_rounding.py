"""The weight grid of the RTN and GPTQ quantizers, as functions of a ``(out, in)`` weight.

Kept free of the quantizer registry so the rotation search can round weights exactly as the quantizers will.
"""

import torch


def weight_quant_params(w: torch.Tensor, bits: int, group_size: int, symmetric: bool):
    """Scales and zeros of a ``(out, in)`` weight, one per row and group of ``group_size`` inputs (-1: per row).

    The symmetric branch follows Humming's rule (``quant_weight.cuh``) rather than the
    textbook ``abs_max / maxq``, which wastes the extra negative code. For b bits the
    codes run ``[-(maxq+1), maxq]``, so:

        scale = +/- max(max_abs / (maxq + 1),  min_abs / maxq)

    where ``max_abs``/``min_abs`` are the larger/smaller of the group's two extremes.
    Taking the max of the two candidates is the smallest scale that clips neither side:
    the dominant extreme needs ``scale >= max_abs/(maxq+1)`` to fit the -(maxq+1) code,
    the weaker one needs ``scale >= min_abs/maxq`` to fit +maxq. The scale is negated
    when the positive extreme dominates, which is what puts *it* on the -(maxq+1) code
    (``-(maxq+1) * -|s| = +(maxq+1)|s|``).

    Negative scales stay inside the quantizer: this path fake-quantizes in place
    (``round(w/s).clamp(...) * s``), so nothing downstream sees the sign.
    """
    assert w.ndim == 2, "Only 2D weight matrices are supported for GPTQ quantization."
    groupsize = group_size
    if groupsize == -1:
        groupsize = w.shape[1]
    assert w.shape[1] % groupsize == 0, f"Weight matrix columns ({w.shape[1]}) must be divisible by group size ({groupsize})."
    w = w.reshape(w.shape[0], w.shape[1] // groupsize, groupsize)
    if symmetric:
        maxq = 2 ** (bits - 1) - 1
        w_max = torch.amax(w, dim=2, keepdim=True)
        w_min = torch.amin(w, dim=2, keepdim=True)
        max_abs = torch.maximum(w_max, w_min.abs())
        min_abs = torch.minimum(w_max, w_min.abs())
        if maxq > 0:
            scales = torch.maximum(max_abs / (maxq + 1), min_abs / maxq)
        else:  # 1-bit has no positive code; fall back to plain abs-max
            scales = max_abs
        scales = torch.where(w_max > w_min.abs(), -scales, scales)
        # An all-zero group has no scale to speak of; 1 keeps the division finite and
        # quantizes it to all zeros, matching Humming.
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        zeros = torch.zeros_like(scales)
    else:
        maxq = 2 ** bits - 1
        scales = (torch.max(w, dim=2, keepdim=True).values - torch.min(w, dim=2, keepdim=True).values) / maxq
        zeros = torch.min(w, dim=2, keepdim=True).values
    return scales, zeros


def round_weight(w: torch.Tensor, bits: int, group_size: int, symmetric: bool):
    """Round a ``(out, in)`` weight to its grid (RTN), with the scales and zeros of weight_quant_params.

    Symmetric codes span ``[-2^(b-1), 2^(b-1) - 1]``; asymmetric codes ``[0, 2^b - 1]`` offset by the group
    minimum. Returns ``(rounded weight, scales, zeros)``.
    """
    maxq = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
    minq = -(maxq + 1) if symmetric else 0
    scales, zeros = weight_quant_params(w, bits, group_size, symmetric)
    size = w.shape[1] if group_size == -1 else group_size
    groups = w.reshape(w.shape[0], w.shape[1] // size, size)
    if symmetric:
        q = torch.round(groups / scales).clamp(minq, maxq) * scales
    else:
        q = torch.round((groups - zeros) / scales).clamp(minq, maxq) * scales + zeros
    return q.reshape(w.shape), scales, zeros
