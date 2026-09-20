"""Refit a Conformer block's output linear layer so the quantized block reproduces the full-precision block's output.

A block ending in a linear layer (the identity Linear a rotation inserts after each Conformer block's output
norm) computes its output as Z W.T + b, where Z is that layer's input. Once the block's other layers are quantized,
Z differs from the full-precision block's, and so does the output. Refitting W and b to map the quantized block's
Z onto the full-precision output Y corrects whatever part of the block's error is linear in Z, including the error
of layers no norm feeds (the attention output projection, the second feed-forward and pointwise layers).

With Za = [Z, 1] and Wa = [W, b], the least-squares fit pulled toward the current weights Wa0 is

    Wa = argmin ||Za Wa.T - Y||_F^2 + lambda ||Wa - Wa0||_F^2
       = (Y.T Za + lambda Wa0) (Za.T Za + lambda I)^-1

with lambda = ridge * mean(diag(Z.T Z)). Only the statistics Za.T Za and Y.T Za are needed, accumulated over the
calibration tokens; the output error before and after is tracked from the same sums and sum(Y^2).

"""

from typing import Tuple

import torch
import torch.nn as nn


class OutputRefit:
    """Accumulate the statistics of one output layer's refit, then solve and apply it.

    Args:
        layer: The block's output nn.Linear.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        width_in = layer.in_features + 1
        device = next(layer.parameters()).device
        self.gram = torch.zeros(width_in, width_in, dtype=torch.float64, device=device)
        self.cross = torch.zeros(layer.out_features, width_in, dtype=torch.float64, device=device)
        self.target_energy = 0.0

    def add(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Add ``(tokens, in)`` inputs of the layer and the ``(tokens, out)`` full-precision block outputs."""
        z = inputs.reshape(-1, inputs.shape[-1]).double()
        y = targets.reshape(-1, targets.shape[-1]).to(z)
        za = torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)], dim=1)
        self.gram += za.T @ za
        self.cross += y.T @ za
        self.target_energy += float((y * y).sum())

    def _error(self, weights: torch.Tensor) -> float:
        return float((weights @ self.gram * weights).sum() - 2 * (weights * self.cross).sum()) + self.target_energy

    def _current(self) -> torch.Tensor:
        """The layer's map as ``(out, in + 1)``, weight beside bias."""
        weight, bias = self.layer.weight, self.layer.bias
        zeros = torch.zeros_like(weight[:, :1])
        return torch.cat([weight.detach().double(), (zeros if bias is None else bias.detach()[:, None]).double()], dim=1)

    def solve(self, ridge: float) -> Tuple[float, float]:
        """Replace the layer's weight and bias by the refit ones.

        Returns:
            ``(before, after)``: the output error relative to the full-precision output's energy.
        """
        current = self._current()
        penalty = ridge * self.gram.diagonal()[:-1].mean()
        eye = torch.eye(self.gram.shape[0], dtype=torch.float64, device=self.gram.device)
        refit = torch.linalg.solve(self.gram + penalty * eye, (self.cross + penalty * current).T).T
        weight, bias = self.layer.weight, self.layer.bias
        with torch.no_grad():
            weight.copy_(refit[:, :-1].to(weight.dtype))
            if bias is None:
                self.layer.bias = nn.Parameter(refit[:, -1].to(weight.dtype))
            else:
                bias.copy_(refit[:, -1].to(bias.dtype))
        energy = max(self.target_energy, 1e-30)
        return self._error(current) / energy, self._error(refit) / energy
