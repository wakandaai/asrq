"""GPTQ on a weight and its Hessian, split into the Hessian's factorisation and the column pass.

The GPTQ quantizer collects a layer's Hessian from calibration inputs and runs both halves once. The rotation
search runs them for every candidate rotation, on Hessians collected once: gptq_factors depends only on the
Hessian, so a layer whose input does not change with the rotation reuses its factors. Kept free of the
quantizer registry so the search can import it directly.

The algorithm is act-order GPTQ with static groups: scales and zeros come from the unpermuted weight, columns
are quantized in order of decreasing Hessian diagonal, each column's rounding error is spread over the columns
not yet quantized through the upper Cholesky factor of the damped inverse Hessian, and the column loop runs in
blocks of block_size with the error of a whole block applied to the rest at once.
"""

import math
from dataclasses import dataclass

import torch

from asrq.quantizers.weight_rounding import weight_quant_params


def add_to_hessian(H: torch.Tensor, nsamples: int, X: torch.Tensor):
    """Fold a batch of ``(tokens, in)`` inputs into the running Hessian ``2 / n * sum(x.T @ x)``.

    All-zero rows (padding) are left out. Returns ``(H, nsamples)``; H is updated in place.
    """
    X = X[X.abs().sum(dim=1) != 0]
    new = X.shape[0]
    H *= nsamples / (nsamples + new)
    nsamples += new
    X = math.sqrt(2 / nsamples) * X.float()
    H += X.T @ X
    return H, nsamples


@dataclass
class GPTQFactors:
    """What GPTQ needs from a Hessian: its dead columns, the act-order permutation and the upper Cholesky
    factor of the damped inverse, permuted."""

    dead: torch.Tensor
    perm: torch.Tensor
    inv_perm: torch.Tensor
    hinv: torch.Tensor


def gptq_factors(H: torch.Tensor, percdamp: float) -> GPTQFactors:
    """Factor a ``(in, in)`` Hessian for gptq_quantize; H is not modified.

    A column with a zero diagonal saw no input: its diagonal is set to 1 and gptq_quantize zeroes its weight.
    Columns are ordered by decreasing diagonal, ``percdamp`` times the mean diagonal is added, and the
    permuted inverse is factored as ``Hinv = U.T @ U`` with U upper triangular.
    """
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    perm = torch.argsort(torch.diag(H), descending=True)
    H = H[perm][:, perm]
    columns = torch.arange(H.shape[0], device=H.device)
    H[columns, columns] += percdamp * torch.mean(torch.diag(H))
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    return GPTQFactors(dead, perm, torch.argsort(perm), hinv)


def gptq_quantize(W: torch.Tensor, factors: GPTQFactors, bits: int, group_size: int, symmetric: bool,
                  block_size: int) -> torch.Tensor:
    """GPTQ-quantize a ``(out, in)`` weight with its Hessian's factors; returns the dequantized weight.

    Returns the weight in its original column order and does not modify W.
    """
    maxq = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
    minq = -(maxq + 1) if symmetric else 0
    W = W.clone()
    columns = W.shape[1]
    scales, zeros = weight_quant_params(W, bits, group_size, symmetric)
    group_size = columns if group_size == -1 else group_size
    groups = factors.perm // group_size
    scales = scales.squeeze(-1)[:, groups]
    zeros = zeros.squeeze(-1)[:, groups]
    W[:, factors.dead] = 0
    W = W[:, factors.perm]
    Q = torch.zeros_like(W)
    hinv = factors.hinv
    for i1 in range(0, columns, block_size):
        i2 = min(i1 + block_size, columns)
        W1 = W[:, i1:i2].clone()
        errors = torch.zeros_like(W1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            s, z = scales[:, i1 + i], zeros[:, i1 + i]
            q = torch.round((w - z) / s).clamp(minq, maxq) * s + z
            Q[:, i1 + i] = q
            error = (w - q) / hinv1[i, i]
            W1[:, i:] -= error.unsqueeze(1) * hinv1[i, i:].unsqueeze(0)
            errors[:, i] = error
        W[:, i2:] -= errors @ hinv[i1:i2, i2:]
    return Q[:, factors.inv_perm]


def gptq_factors_batched(H: torch.Tensor, percdamp: float) -> GPTQFactors:
    """gptq_factors for a stack of ``(layers, in, in)`` Hessians, each factored independently; H is not
    modified."""
    H = H.clone()
    layers, width, _ = H.shape
    diagonal = torch.diagonal(H, dim1=1, dim2=2)
    dead = diagonal == 0
    diagonal[dead] = 1
    perm = torch.argsort(diagonal, dim=1, descending=True)
    H = torch.gather(torch.gather(H, 1, perm.unsqueeze(2).expand(-1, -1, width)), 2, perm.unsqueeze(1).expand(-1, width, -1))
    diagonal = torch.diagonal(H, dim1=1, dim2=2)
    diagonal += percdamp * diagonal.mean(dim=1, keepdim=True)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    return GPTQFactors(dead, perm, torch.argsort(perm, dim=1), hinv)


def gptq_quantize_batched(W: torch.Tensor, factors: GPTQFactors, bits: int, group_size: int, symmetric: bool,
                          block_size: int) -> torch.Tensor:
    """gptq_quantize for a stack of ``(layers, out, in)`` weights of one shape, with gptq_factors_batched of
    their Hessians.

    Every layer goes through the column loop together, so the loop runs once per column of the shape rather
    than once per column of every layer, which is what the per-layer loop spends its time on.
    """
    maxq = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1
    minq = -(maxq + 1) if symmetric else 0
    layers, rows, columns = W.shape
    scales, zeros = weight_quant_params(W.reshape(layers * rows, columns), bits, group_size, symmetric)
    scales, zeros = scales.reshape(layers, rows, -1), zeros.reshape(layers, rows, -1)
    group_size = columns if group_size == -1 else group_size
    groups = (factors.perm // group_size).unsqueeze(1).expand(-1, rows, -1)
    scales = torch.gather(scales, 2, groups)
    zeros = torch.gather(zeros, 2, groups)
    W = W.masked_fill(factors.dead.unsqueeze(1), 0)
    W = torch.gather(W, 2, factors.perm.unsqueeze(1).expand(-1, rows, -1))
    Q = torch.zeros_like(W)
    hinv = factors.hinv
    for i1 in range(0, columns, block_size):
        i2 = min(i1 + block_size, columns)
        W1 = W[:, :, i1:i2].clone()
        errors = torch.zeros_like(W1)
        hinv1 = hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, :, i]
            s, z = scales[:, :, i1 + i], zeros[:, :, i1 + i]
            q = torch.round((w - z) / s).clamp(minq, maxq) * s + z
            Q[:, :, i1 + i] = q
            error = (w - q) / hinv1[:, i, i].unsqueeze(1)
            W1[:, :, i:] -= error.unsqueeze(2) * hinv1[:, i, i:].unsqueeze(1)
            errors[:, :, i] = error
        W[:, :, i2:] -= errors @ hinv[:, i1:i2, i2:]
    return torch.gather(Q, 2, factors.inv_perm.unsqueeze(1).expand(-1, rows, -1))
