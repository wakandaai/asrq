"""A (1 + lambda) evolutionary search over the sign vectors of a randomized Hadamard R1.

The residual-stream rotation is searched within the family

    R1 = diag(s1) @ H @ diag(s2),    s1, s2 in {-1, +1}^n,

with H a fixed Hadamard matrix of the stream's width (block-diagonal when R1 is), while each attention
module's R2 stays the random Hadamard the rotation search starts from. Every member is orthogonal, so the
full-precision model is the same function for every candidate and only the quantization error differs; the
fitness is the KL divergence from the full-precision model to the model with fake-quantized activations,
the objective the Cayley SGD search minimises.

The stream is rotated as ``x @ R1 = x @ diag(s1) @ H @ diag(s2)``: s1 flips input channels before H mixes
them, and decides where each outlier lands, while s2 flips the signs of the rotated coordinates. A
symmetric quantizer commutes with a per-coordinate sign flip -- a group's largest magnitude is unchanged
and rounding is odd -- and every layer reading the stream quantizes its input symmetrically and folds s2
into its weight, so with symmetric activation quantization s2 does not change the fitness. With asymmetric
quantization it does: a flip moves a group's minimum and maximum. ``mutate="auto"``, the default, therefore
flips only s1 when activations are quantized symmetrically and both vectors when asymmetrically.

A weight-only search (see learn_rotations) folds R1 into the weights and rounds them in groups along each
row. On the output side, ``R1.T @ W``, s2 negates whole rows, which the weight grid commutes with. On the input
side, ``W @ R1``, it negates single entries within a group, and neither weight grid commutes with that: the
asymmetric one offsets by the group minimum, and the symmetric one (Humming's) has one more negative code than
positive ones, placed on whichever extreme dominates. For a weight-only search ``mutate="auto"`` therefore
flips both vectors.

One generation, as in EvoPress's multi-step selection:

1. The parent is mutated into ``offspring`` children, each by flipping ``flips`` positions drawn uniformly
   from the mutated sign vectors (of every stream).
2. Stage 1 scores every child on a random subset of ``stage_samples[0]`` calibration samples, and the best
   ``survivors[0]`` go on.
3. Stage 2 scores those on a new random subset of ``stage_samples[1]`` samples, and the best
   ``survivors[1]`` go on.
4. Stage 3 scores the remaining children on the full evaluation set, the first ``stage_samples[2]``
   samples, the same set the parent was scored on. The best child replaces the parent only if its fitness
   is lower, so the parent's fitness never increases.

The subsets are drawn anew each generation and shared by all children within a stage, so children are
compared on the same data while no small subset is selected against repeatedly. Few flips keep a child
close to its parent, which is what lets a small subset rank children that differ by little.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from asrq.transforms.rotation.hadamard_utils import matmul_hadU

SignVectors = Dict[Optional[str], Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class EvolutionConfig:
    """Settings of evolve_signs; see the module docstring.

    Args:
        generations: Number of generations.
        offspring: Children per generation (lambda).
        survivors: Children kept after stage 1 and after stage 2.
        stage_samples: Calibration samples scored in stages 1, 2 and 3; None chooses them from the
            calibration set's size with default_stage_samples.
        flips: Sign positions flipped per mutation.
        mutate: ``"s1_s2"`` flips positions of both sign vectors, ``"s1"`` only of s1, and ``"auto"`` chooses
            ``"s1"`` for symmetric activation quantization and ``"s1_s2"`` for asymmetric; see resolve_mutate.
        seed: Seed of the initial signs, the mutations and the stage subsets.
        cache_teacher: Keep the full-precision logits of every calibration batch on the CPU, instead of
            recomputing them for every candidate. They do not depend on the rotation.
    """

    generations: int = 8
    offspring: int = 16
    survivors: Tuple[int, int] = (4, 2)
    stage_samples: Optional[Tuple[int, int, int]] = None
    flips: int = 2
    mutate: str = "auto"
    seed: int = 0
    cache_teacher: bool = True
    history: List[dict] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.survivors = tuple(self.survivors)
        if self.stage_samples is not None:
            self.stage_samples = tuple(self.stage_samples)
        if self.mutate not in ("auto", "s1_s2", "s1"):
            raise ValueError(f"mutate must be 'auto', 's1_s2' or 's1', got {self.mutate!r}")
        if len(self.survivors) != 2 or not self.offspring >= self.survivors[0] >= self.survivors[1] >= 1:
            raise ValueError(
                f"need offspring >= survivors[0] >= survivors[1] >= 1, got {self.offspring} and {self.survivors}"
            )
        if self.stage_samples is not None and len(self.stage_samples) != 3:
            raise ValueError(f"stage_samples needs three sizes, got {self.stage_samples}")
        if self.flips < 1:
            raise ValueError(f"flips must be at least 1, got {self.flips}")

    @classmethod
    def from_mapping(cls, settings: Optional[Mapping]) -> "EvolutionConfig":
        return cls(**dict(settings or {}))

    def resolve_mutate(self, activation_symmetric: bool) -> None:
        """Replace ``mutate="auto"`` by the sign vectors that change the fitness: s1 alone for symmetric
        activation quantization, which s2 does not affect, and both for asymmetric."""
        if self.mutate == "auto":
            self.mutate = "s1" if activation_symmetric else "s1_s2"

    def settings(self) -> dict:
        """The configuration without the run's history, as saved with the rotation."""
        settings = asdict(self)
        settings.pop("history")
        return settings


def default_stage_samples(num_samples: int) -> Tuple[int, int, int]:
    """Stage sizes for a calibration set: 8, 16, 64 for 64 samples; 8, 64, 256 for 256; 16, 64, 768 for 768.

    The last stage is the whole set; the first two are capped by it.
    """
    if num_samples <= 64:
        first, second = 8, 16
    elif num_samples <= 256:
        first, second = 8, 64
    else:
        first, second = 16, 64
    return min(first, num_samples), min(second, num_samples), num_samples


def _hadamard(width: int) -> torch.Tensor:
    """A normalized ``(width, width)`` Hadamard in float64, rescaled so its rows have unit norm exactly.

    matmul_hadU divides by a single-precision sqrt(width), which leaves a width that is not a power of two
    (whisper-large-v3's 1280) orthogonal only to ~3e-8.
    """
    H = matmul_hadU(torch.eye(width, dtype=torch.float64))
    return H / H[0].norm()


def hadamard_basis(width: int, block_size: Optional[int], device) -> torch.Tensor:
    """The fixed H of R1 = diag(s1) @ H @ diag(s2): a normalized Hadamard, block-diagonal with blocks of
    block_size when given. float64, ``(width, width)``."""
    if block_size is None:
        return _hadamard(width).to(device)
    if width % block_size:
        raise ValueError(f"block_size {block_size} does not divide width {width}")
    return torch.block_diag(*[_hadamard(block_size)] * (width // block_size)).to(device)


def signed_hadamard(H: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """``diag(s1) @ H @ diag(s2)``, without forming the diagonal matrices."""
    return s1.to(H).unsqueeze(1) * H * s2.to(H).unsqueeze(0)


def random_sign_vectors(widths: Mapping[Optional[str], int], generator: torch.Generator, device) -> SignVectors:
    """Independent random ``(s1, s2)`` per stream, float32 on device."""
    def signs(width):
        return (torch.randint(0, 2, (width,), generator=generator) * 2 - 1).to(device=device, dtype=torch.float32)
    return {stream: (signs(width), signs(width)) for stream, width in widths.items()}


def mutate_signs(parent: SignVectors, flips: int, mutate: str, generator: torch.Generator) -> SignVectors:
    """A copy of parent with ``flips`` distinct positions flipped, drawn uniformly over the mutated vectors.

    With ``mutate="s1_s2"`` the positions are drawn from every stream's s1 and s2 together, with ``"s1"``
    from every stream's s1.
    """
    vectors = [(stream, which) for stream in parent for which in ((0, 1) if mutate == "s1_s2" else (0,))]
    sizes = [parent[stream][which].numel() for stream, which in vectors]
    total = sum(sizes)
    if flips > total:
        raise ValueError(f"cannot flip {flips} of {total} positions")
    child = {stream: (s1.clone(), s2.clone()) for stream, (s1, s2) in parent.items()}
    offsets = torch.tensor([0, *sizes]).cumsum(0)
    for position in torch.randperm(total, generator=generator)[:flips].tolist():
        index = int(torch.searchsorted(offsets, position, right=True)) - 1
        stream, which = vectors[index]
        child[stream][which][position - int(offsets[index])] *= -1
    return child


def _subset(sample_counts: Sequence[int], samples: int, generator: Optional[torch.Generator]) -> List[int]:
    """Batch indices covering at least ``samples`` samples: a random subset, or the leading batches."""
    order = range(len(sample_counts)) if generator is None else torch.randperm(len(sample_counts), generator=generator).tolist()
    chosen, covered = [], 0
    for index in order:
        if covered >= samples:
            break
        chosen.append(index)
        covered += sample_counts[index]
    return sorted(chosen)


def evolve_signs(
    parent: SignVectors,
    fitness: Callable[[SignVectors, List[int]], float],
    sample_counts: Sequence[int],
    config: EvolutionConfig,
    log: Callable[[str], None] = print,
) -> Tuple[SignVectors, float]:
    """Run the (1 + lambda) search from parent and return the final parent and its fitness.

    Args:
        parent: The initial ``{stream: (s1, s2)}``.
        fitness: ``fn(signs, batch_indices) -> float``, the mean KL of the model rotated by signs over the
            listed calibration batches. Lower is better.
        sample_counts: Samples in each calibration batch.
        config: The search settings. Each generation's record is appended to ``config.history``.
        log: Called with one line per generation.
    """
    if config.mutate == "auto":
        raise ValueError("resolve mutate='auto' with EvolutionConfig.resolve_mutate before the search")
    total = sum(sample_counts)
    stages = config.stage_samples or default_stage_samples(total)
    generator = torch.Generator().manual_seed(config.seed)
    evaluation_set = _subset(sample_counts, min(stages[2], total), None)
    parent_fitness = fitness(parent, evaluation_set)
    log(f"evolution: initial fitness {parent_fitness:.6f} on {sum(sample_counts[i] for i in evaluation_set)} samples; "
        f"stages {stages}, {config.offspring} offspring, survivors {config.survivors}, {config.flips} flips of {config.mutate}")
    config.history.append({"generation": 0, "fitness": parent_fitness, "accepted": True})
    for generation in range(1, config.generations + 1):
        start = time.time()
        children = [mutate_signs(parent, config.flips, config.mutate, generator) for _ in range(config.offspring)]
        for stage, keep in ((0, config.survivors[0]), (1, config.survivors[1])):
            subset = _subset([sample_counts[i] for i in evaluation_set], stages[stage], generator)
            batches = [evaluation_set[i] for i in subset]
            scores = [fitness(child, batches) for child in children]
            ranked = sorted(range(len(children)), key=scores.__getitem__)
            children = [children[i] for i in ranked[:keep]]
        scores = [fitness(child, evaluation_set) for child in children]
        best = min(range(len(children)), key=scores.__getitem__)
        accepted = scores[best] < parent_fitness
        if accepted:
            parent, parent_fitness = children[best], scores[best]
        config.history.append({
            "generation": generation, "fitness": parent_fitness, "best_child": scores[best], "accepted": accepted,
        })
        log(f"  generation {generation}/{config.generations}: parent {parent_fitness:.6f}, best child "
            f"{scores[best]:.6f}{' (accepted)' if accepted else ''}, {time.time() - start:.1f}s")
    return parent, parent_fitness


def stage_cost(config: EvolutionConfig, num_samples: int) -> int:
    """Calibration samples scored per generation, a measure of a generation's cost."""
    stages = config.stage_samples or default_stage_samples(num_samples)
    return (config.offspring * stages[0] + config.survivors[0] * stages[1]
            + config.survivors[1] * min(stages[2], num_samples))

