"""Weight recovery: refit a quantized layer in closed form so it reproduces the full-precision output.

One of the two error recovery techniques, beside scale recovery (asrq/quantizers/scale_recovery.py), which
recovers a transformer block by scaling the layers its norms feed. This one refits a layer's own weight and
bias, and is applied two ways; see ModelQ.apply_weight_recovery and ModelQ.layer_recovery.

- **A block's output layer.** A Conformer block ends in a linear layer (the identity Linear a rotation
  inserts after its output norm), so the block's whole error can be corrected there. That layer computes
  Z W.T + b, where Z is its input. Once the block's other layers are quantized, Z differs from the
  full-precision block's, and so does the output. Refitting W and b to map the quantized block's Z onto the
  full-precision output Y corrects whatever part of the block's error is linear in Z, including the error of
  layers no norm feeds (the attention output projection, the second feed-forward and pointwise layers).
- **Every quantized layer.** The same fit, one layer at a time: Y is what the layer produced before anything
  in its block was quantized, Z is its input as the block stands, with the layers quantized so far in place.
  This needs no layer at the end of the block, so it reaches transformer blocks too, and it runs before each
  layer is quantized, so GPTQ puts the refit weight on the grid.

With Za = [Z, 1] and Wa = [W, b], the least-squares fit pulled toward the current weights Wa0 is

    Wa = argmin ||Za Wa.T - Y||_F^2 + lambda ||Wa - Wa0||_F^2
       = (Y.T Za + lambda Wa0) (Za.T Za + lambda I)^-1

with lambda = ridge * mean(diag(Z.T Z)). Only the statistics Za.T Za and Y.T Za are needed, accumulated over
the calibration tokens; the output error before and after is tracked from the same sums and sum(Y^2).

Z and Y have to come from two different streams. Fitting a layer to its own full-precision output on its own
full-precision input returns the weights it already has: that fit is solved exactly by Wa0.

The statistics are accumulated in float64, so a layer of in_features F holds an (F + 1)^2 matrix: 210 MB for
Whisper's fc2, and more for a wide LLM down projection. They are freed with the WeightRecovery.
"""

from typing import Tuple

import torch
import torch.nn as nn

from asrq.quantizers.base import is_pointwise_conv1d


def _tokens(x: torch.Tensor, channels_first: bool) -> torch.Tensor:
    """Flatten activations to ``(tokens, channels)``; a pointwise conv's are ``(B, C, T)``."""
    if channels_first:
        x = x.transpose(1, 2)
    return x.reshape(-1, x.shape[-1])


class WeightRecovery:
    """Accumulate the statistics of one layer's refit, then solve and apply it.

    Args:
        layer: The layer to refit: an nn.Linear or a pointwise Conv1d, whose ``(out, in, 1)`` weight and
            ``(B, C, T)`` activations are read as a linear layer's.
    """

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.channels_first = is_pointwise_conv1d(layer)
        weight = self._weight_2d()
        width_in = weight.shape[1] + 1
        device = weight.device
        self.gram = torch.zeros(width_in, width_in, dtype=torch.float64, device=device)
        self.cross = torch.zeros(weight.shape[0], width_in, dtype=torch.float64, device=device)
        self.target_energy = 0.0

    def _weight_2d(self) -> torch.Tensor:
        """The layer's weight as ``(out, in)``, whatever the module type."""
        weight = self.layer.weight.data
        return weight.reshape(weight.shape[0], -1) if self.channels_first else weight

    def add(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch of the layer's inputs and the full-precision outputs they should map onto.

        Both are the layer's own activation layout, as a forward hook sees them.
        """
        z = _tokens(inputs, self.channels_first).double()
        y = _tokens(targets, self.channels_first).to(z)
        za = torch.cat([z, torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)], dim=1)
        self.gram += za.T @ za
        self.cross += y.T @ za
        self.target_energy += float((y * y).sum())

    def _error(self, weights: torch.Tensor) -> float:
        return float((weights @ self.gram * weights).sum() - 2 * (weights * self.cross).sum()) + self.target_energy

    def _current(self) -> torch.Tensor:
        """The layer's map as ``(out, in + 1)``, weight beside bias."""
        weight, bias = self._weight_2d(), self.layer.bias
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
            weight.copy_(refit[:, :-1].reshape(weight.shape).to(weight.dtype))
            if bias is None:
                self.layer.bias = nn.Parameter(refit[:, -1].to(weight.dtype))
            else:
                bias.copy_(refit[:, -1].to(bias.dtype))
        energy = max(self.target_energy, 1e-30)
        return self._error(current) / energy, self._error(refit) / energy
