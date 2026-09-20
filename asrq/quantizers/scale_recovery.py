"""Scale recovery: a per-input-channel scale on each GPTQ-quantized layer, absorbed by whatever feeds it.

A quantized layer computes X Q.T. Scaling its input channels, X diag(s), is the same as scaling column j of Q
by s_j, so the layer's grid gains one free degree of freedom per channel. The scale is fitted while GPTQ
quantizes, column by column (see gptq_quantize_columns): when column j is rounded, GPTQ has already pushed the
compensation of the earlier columns into it, so its updated value w~_j is what GPTQ wants the column to be, and
w~_j - q_j is column j's own rounding error. The scale that best stretches the rounded column onto it is plain
least squares,

    s_j = <w~_j, q_j> / <q_j, q_j>,

and the error the later columns compensate is then w~_j - s_j q_j. A ridge pulls s_j toward 1 (no change).

Nothing is added to the model: diag(s) is folded into the norm feeding the layer, which has its output scaled
(its affine weight and bias, or a new ``weight`` buffer when a rotation left it scale-free). Only layers a norm
feeds are scaled. Folding into a preceding layer's rows instead -- an attention output projection into the value
projection, a SwiGLU down projection into the up projection -- is exact and free but measured no better, and was
removed; see dev/docs/scale_recovery_and_output_refit.md.

Several layers can share one norm -- q/k/v share theirs, gate/up share theirs -- and then share one s.
Two ways to fit it (SCALE_RECOVERY_METHODS):

- ``lockstep``: their weights are stacked row-wise and quantized as one matrix. GPTQ treats rows independently,
  so each layer gets exactly its own GPTQ result, while s_j is fitted over all of their rows at once and every
  layer compensates the error it has with the shared scale.
- ``average``: each layer is quantized on its own with its own scales, and the norm gets their average
  weighted by each layer's column energy <q_lj, q_lj>. For one column this is the lockstep formula, but each
  layer compensated its own scale's error rather than the shared one's, so from the second column on the two
  differ.

The fit covers weight quantization only: with quantized activations s also changes what the activation quantizer
rounds.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from asrq.quantizers.gptq_solver import gptq_factors, gptq_quantize_columns

SCALE_RECOVERY_METHODS = ("lockstep", "average")


def error_ratio(
    H: torch.Tensor, full_precision: Sequence[torch.Tensor], quantized: Sequence[torch.Tensor], s: torch.Tensor,
) -> float:
    """L(s) / L(1), with L(s) = sum_l tr((W_l - Q_l diag(s)) H (W_l - Q_l diag(s)).T): the quantized layers' output
    error with the scale, relative to without it."""
    H = H.double()
    s = s.to(H)

    def error(scale):
        total = 0.0
        for W, Q in zip(full_precision, quantized):
            E = W.double() - Q.double() * scale
            total += float((E @ H * E).sum())
        return total

    unscaled = error(torch.ones_like(s))
    return error(s) / unscaled if unscaled > 0 else 1.0


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


@dataclass
class ScaleTarget:
    """One norm and the quantized layers it feeds, which share its scale.

    Args:
        norm: The norm whose output takes the scale.
        layers: The layers sharing this scale, all reading the norm's output.
    """

    norm: str
    layers: List[str]


@dataclass
class ScaleGroup:
    """A ScaleTarget of the block being quantized, with the scale fitted while its layers were quantized."""

    target: ScaleTarget
    scale: Optional[torch.Tensor] = None
    ratio: Optional[float] = None


def capture_scale_recovery(
    targets: Sequence[ScaleTarget], quantizers: Mapping[str, object]
) -> List[ScaleGroup]:
    """The targets of a block to recover: those all of whose layers are among its GPTQ quantizers.

    Args:
        targets: The model's ScaleTargets; see ModelQ.scale_recovery_targets.
        quantizers: ``{layer_name: quantizer}`` of the block being quantized; quantizers without a Hessian (RTN)
            are skipped.
    """
    captured = []
    for target in targets:
        if not target.layers or not all(name in quantizers for name in target.layers):
            continue
        if getattr(quantizers[target.layers[0]], "H", None) is None:
            continue
        captured.append(ScaleGroup(target))
    return captured


def quantize_scale_group(
    quantizers: Sequence[object], ridge: float, method: str = "lockstep",
) -> Tuple[torch.Tensor, float, list]:
    """GPTQ-quantize the layers one norm feeds, fitting their column scale as it goes; ``method`` is one of
    SCALE_RECOVERY_METHODS (see the module docstring).

    The quantizers are GPTQ quantizers whose layers read the same input, so they hold the same Hessian; the
    first one's is used. Each layer's weight is replaced by its grid weight, and every quantizer's Hessian is
    released, as quantizing it alone would.

    Returns:
        ``(s, ratio, qparams)``: the scale, L(s) / L(1) for the quantized layers (see error_ratio), and each
        layer's ``(scales, zeros)`` as its quantizer returns them.
    """
    config = quantizers[0].quant_config
    H = quantizers[0].H
    factors = gptq_factors(H, config.percdamp)
    originals = [quantizer.weight_2d().clone() for quantizer in quantizers]
    grid = (factors, config.bits, config.group_size, config.symmetric, config.block_size)
    if method == "lockstep":
        Q, s = gptq_quantize_columns(torch.cat(originals), *grid, scale_ridge=ridge)
        quantized = list(torch.split(Q, [W.shape[0] for W in originals]))
    elif method == "average":
        results = [gptq_quantize_columns(W, *grid, scale_ridge=ridge) for W in originals]
        quantized = [Q for Q, _ in results]
        energies = [(Q * Q).sum(dim=0) for Q in quantized]
        total = sum(energies)
        weighted = sum(energy * scale for energy, (_, scale) in zip(energies, results))
        s = torch.where(total > 0, weighted / total.clamp_min(torch.finfo(total.dtype).tiny), torch.ones_like(total))
    else:
        raise ValueError(f"scale_recovery_method must be one of {SCALE_RECOVERY_METHODS}, got {method!r}")
    for quantizer, weight in zip(quantizers, quantized):
        quantizer.set_weight_2d(weight)
    ratio = error_ratio(H, originals, quantized, s)
    for quantizer in quantizers:
        del quantizer.H
    qparams = [tuple(t.squeeze(-1) for t in quantizer.find_quant_params(W)) for quantizer, W in zip(quantizers, originals)]
    return s, ratio, qparams


def apply_scale_recovery(
    captured: Sequence[ScaleGroup], modules: Mapping[str, nn.Module]
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Fold every captured group's scale into its norm, once its layers are quantized.

    Returns:
        ``{norm_name: (s, error ratio)}``.
    """
    results = {}
    for group in captured:
        if group.scale is None:
            raise RuntimeError(f"{group.target.norm}'s layers were not quantized through quantize_scale_group")
        scale_norm_output(modules[group.target.norm], group.scale)
        results[group.target.norm] = (group.scale, group.ratio)
    return results
