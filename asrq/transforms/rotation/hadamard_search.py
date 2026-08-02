# pyright: reportMissingImports=false

"""Evolutionary search over the sign vectors of randomized Hadamard rotations.

A randomized Hadamard rotation is ``Q = diag(d) @ H`` for a sign vector ``d`` in
{-1, +1}^n - which is exactly what :func:`random_hadamard_matrix` builds, with ``d``
drawn at random. Every such ``Q`` is orthogonal, so the *unquantized* model is invariant
to the choice: the search is only meaningful when the forward pass quantizes something,
because then ``d`` decides how the outliers land relative to the quantization grid.

The search is derivative-free by construction (a sign vector has no useful gradient), so
this is a plain evolutionary loop: evaluate a population, keep the best, mutate them by
flipping signs, repeat. Only forward passes are needed.

Candidates are evaluated through the on-the-fly rotation path
(``modify_*_layers_with_rotation_params``), not by fusing rotations into the weights: the
patched forwards close over the rotation Parameters, so writing new values into them
swaps the rotation for the whole model at once, with no weight surgery to undo between
candidates.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from asrq.transforms.rotation.hadamard_utils import matmul_hadU


@dataclass
class HadamardSearchConfig:
    """Hyperparameters for :func:`evolutionary_sign_search`.

    Each generation costs ``population - survivors`` evaluations (survivors keep their
    score), and each evaluation is ``batches_per_eval`` forward passes over the whole
    model, so the total is roughly
    ``(population + generations * (population - survivors)) * batches_per_eval`` forwards.
    """

    population: int = 16
    generations: int = 8
    survivors: int = 4
    mutation_prob: float = 0.05     # per-sign flip probability when mutating
    batches_per_eval: int = 4       # calibration batches per candidate
    crossover: bool = True          # breed from two survivors instead of one
    seed: Optional[int] = None
    verbose: bool = True


def hadamard_from_signs(signs: torch.Tensor, device=None, dtype=torch.float32,
                        block_size: Optional[int] = None) -> torch.Tensor:
    """Build the randomized Hadamard rotation for ``signs``.

    With ``block_size=None`` this is ``Q = diag(signs) @ H``, the same construction as
    :func:`random_hadamard_matrix` with the sign vector supplied instead of drawn.

    With ``block_size=b`` it is block diagonal - ``diag(blocks)`` of ``b x b`` randomized
    Hadamards, one per contiguous run of ``b`` channels. Set ``b`` to the weight
    quantization group size and the rotation mixes channels only *within* a quantization
    group, so each group's scale still covers exactly the channels the rotation touched.
    A full-width rotation would spread each group's outliers over the whole hidden dim and
    across group boundaries.

    Built in float64 and cast down, as the rest of the rotation code does.
    """
    d = signs.to(torch.float64)
    if block_size is None:
        Q = matmul_hadU(torch.diag(d))
    else:
        n = d.numel()
        if n % block_size != 0:
            raise ValueError(f"rotation size ({n}) must be a multiple of block_size ({block_size})")
        H_block = matmul_hadU(torch.eye(block_size, dtype=torch.float64))
        Q = torch.block_diag(*[
            torch.diag(d[i:i + block_size]) @ H_block for i in range(0, n, block_size)
        ])
    return Q.to(device=device if device is not None else signs.device, dtype=dtype)


def random_signs(size: int, generator: torch.Generator, device="cpu") -> torch.Tensor:
    return (torch.randint(0, 2, (size,), generator=generator, device=device) * 2 - 1).to(torch.float64)


def write_rotations_(params: Dict[str, torch.Tensor], signs: Dict[str, torch.Tensor],
                     block_sizes: Optional[Dict[str, Optional[int]]] = None) -> None:
    """Write the rotations for ``signs`` into the live rotation Parameters, in place.

    In place is the point: the modified forwards and the monkey-patched residual stream
    close over these tensors, so this swaps the rotation everywhere at once.

    ``block_sizes`` optionally makes individual rotations block diagonal (see
    :func:`hadamard_from_signs`); names absent from it stay full width.
    """
    block_sizes = block_sizes or {}
    for name, param in params.items():
        Q = hadamard_from_signs(
            signs[name], device=param.device, dtype=param.dtype,
            block_size=block_sizes.get(name),
        )
        param.data.copy_(Q)


def _mutate(signs: torch.Tensor, mutation_prob: float, generator: torch.Generator) -> torch.Tensor:
    flips = torch.rand(signs.shape, generator=generator, device=signs.device) < mutation_prob
    out = signs.clone()
    out[flips] *= -1
    return out


def _crossover(a: torch.Tensor, b: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    take_a = torch.rand(a.shape, generator=generator, device=a.device) < 0.5
    return torch.where(take_a, a, b)


def evolutionary_sign_search(
    sign_sizes: Dict[str, int],
    evaluate: Callable[[Dict[str, torch.Tensor]], float],
    cfg: HadamardSearchConfig = HadamardSearchConfig(),
) -> Tuple[Dict[str, torch.Tensor], float, List[float]]:
    """Search sign vectors for the rotations named in ``sign_sizes``.

    Args:
        sign_sizes: rotation name -> dimension, e.g. ``{"Qe": 1280, "<layer>": 64}``.
            One sign vector is searched per entry, all mutated jointly since they are
            scored by a single end-to-end loss.
        evaluate: takes a ``{name: sign vector}`` candidate and returns its loss (lower
            is better). Must be deterministic across calls - use the same calibration
            batches every time, or candidates cannot be compared.
        cfg: search hyperparameters.

    Returns:
        ``(best_signs, best_loss, history)``, where history is the best loss per
        generation, starting with the initial population.
    """
    if cfg.survivors < 1 or cfg.survivors > cfg.population:
        raise ValueError(f"survivors must be in [1, population]; got {cfg.survivors}/{cfg.population}")

    generator = torch.Generator()
    generator.manual_seed(cfg.seed if cfg.seed is not None else torch.initial_seed() & 0xFFFFFFFF)

    # The all-ones candidate is the plain (unrandomized) Hadamard - a meaningful baseline
    # and a guarantee the search never returns something worse than it.
    population: List[Dict[str, torch.Tensor]] = [
        {name: torch.ones(size, dtype=torch.float64) for name, size in sign_sizes.items()}
    ]
    while len(population) < cfg.population:
        population.append({name: random_signs(size, generator) for name, size in sign_sizes.items()})

    scored: List[Tuple[float, Dict[str, torch.Tensor]]] = []
    history: List[float] = []

    for generation in range(cfg.generations + 1):
        # Survivors carry their score over; only the new candidates are evaluated.
        for candidate in population:
            loss = evaluate(candidate)
            scored.append((loss, candidate))
        scored.sort(key=lambda item: item[0])
        scored = scored[: cfg.survivors]
        history.append(scored[0][0])
        if cfg.verbose:
            gen_label = "init" if generation == 0 else f"gen {generation}"
            print(f"[hadamard-search] {gen_label}: best={scored[0][0]:.6f} "
                  f"top{cfg.survivors}={[round(s, 6) for s, _ in scored]}")

        if generation == cfg.generations:
            break

        # Breed the next generation from the survivors.
        population = []
        while len(population) < cfg.population - cfg.survivors:
            i = int(torch.randint(len(scored), (1,), generator=generator).item())
            parent = scored[i][1]
            if cfg.crossover and len(scored) > 1:
                j = int(torch.randint(len(scored), (1,), generator=generator).item())
                other = scored[j][1]
                child = {name: _crossover(parent[name], other[name], generator) for name in sign_sizes}
            else:
                child = {name: parent[name].clone() for name in sign_sizes}
            population.append({name: _mutate(child[name], cfg.mutation_prob, generator) for name in sign_sizes})

    best_loss, best_signs = scored[0]
    return best_signs, best_loss, history


def make_batch_evaluator(model, loss_fn, batches, params: Dict[str, torch.Tensor],
                         block_sizes: Optional[Dict[str, Optional[int]]] = None) -> Callable:
    """Build the ``evaluate`` callback: install the candidate, then score it.

    ``batches`` is materialized up front by the callers so every candidate sees exactly
    the same data - otherwise the losses are not comparable and the search is noise.
    """
    @torch.no_grad()
    def evaluate(signs: Dict[str, torch.Tensor]) -> float:
        write_rotations_(params, signs, block_sizes)
        total = 0.0
        for batch in batches:
            total += float(loss_fn(model, batch))
        return total / max(len(batches), 1)

    return evaluate


def check_search_is_meaningful(activation_bits: int, quantize_weights: bool = False) -> None:
    """Guard against searching a model whose forward has nothing to quantize.

    Rotations are exactly output-preserving in full precision, so unless the forward fake
    quantizes *something* - the activations, the weights, or both - every candidate scores
    identically and the search just returns its first entry.
    """
    if activation_bits >= 16 and not quantize_weights:
        raise ValueError(
            "Hadamard search needs a quantized forward pass to have anything to optimize, "
            f"but activation_bits={activation_bits} and weight quantization is off. Every "
            "orthogonal rotation gives the same loss at full precision. Set activation_bits "
            "to 4 or 8, or enable weight quantization for a weight-only search."
        )
