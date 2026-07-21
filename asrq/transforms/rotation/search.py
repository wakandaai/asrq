# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

import matplotlib.pyplot as plt
import torch

from asrq.transforms.rotation.hadamard_utils import matmul_hadU


def normalized_hadamard_matrix(size: int) -> torch.Tensor:
    """Return a deterministic normalized Hadamard-like orthogonal matrix."""
    eye = torch.eye(size, dtype=torch.float64)
    return matmul_hadU(eye)


@dataclass
class SignHadamardCandidate:
    """Search candidate for a signed Hadamard family R = D0 H D1."""

    s0: torch.Tensor
    s1: torch.Tensor

    def clone(self) -> "SignHadamardCandidate":
        return SignHadamardCandidate(self.s0.clone(), self.s1.clone())

    def to_rotation(
        self,
        base_h: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        s0 = self.s0.to(device=device, dtype=base_h.dtype)
        s1 = self.s1.to(device=device, dtype=base_h.dtype)
        left = s0.unsqueeze(1)
        right = s1.unsqueeze(0)
        rotation = left * base_h.to(device=device) * right
        return rotation.to(dtype=dtype)


@dataclass
class RotationSearchSite:
    """Model-agnostic searchable rotation site."""

    site_id: str
    block_id: str
    dimension: int
    base_h: torch.Tensor
    current_candidate: SignHadamardCandidate
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RotationSearchParams:
    population_size: int = 8
    elite_count: int = 2
    parent_pool_fraction: float = 0.5
    generations: int = 16
    patience: int = 4
    mutate_both_probability: float = 0.1
    large_mutation_probability: float = 0.1
    small_mutation_min: int = 1
    small_mutation_max: int = 2
    medium_mutation_min: int = 4
    medium_mutation_max: int = 8
    large_mutation_fraction: float = 0.25
    seed: int = 42
    verbose: bool = True


@dataclass
class RotationSearchHistory:
    site_id: str
    generation_indices: list[int] = field(default_factory=list)
    best_scores: list[float] = field(default_factory=list)
    committed_score: float | None = None


@dataclass
class RotationSearchResult:
    best_candidates: dict[str, SignHadamardCandidate]
    histories: dict[str, RotationSearchHistory]


class RotationSearchAdapter(Protocol):
    def sites(self) -> Sequence[RotationSearchSite]:
        ...

    def refresh_caches(self) -> None:
        ...

    def score_site_candidate(
        self,
        site: RotationSearchSite,
        candidate: SignHadamardCandidate,
    ) -> float:
        ...

    def commit_site_candidate(
        self,
        site: RotationSearchSite,
        candidate: SignHadamardCandidate,
    ) -> None:
        ...


def random_sign_candidate(
    dimension: int,
    generator: torch.Generator,
) -> SignHadamardCandidate:
    return SignHadamardCandidate(
        _random_sign_vector(dimension, generator),
        _random_sign_vector(dimension, generator),
    )


def _random_sign_vector(dimension: int, generator: torch.Generator) -> torch.Tensor:
    vector = torch.randint(0, 2, (dimension,), generator=generator, dtype=torch.int8)
    return vector.mul_(2).sub_(1) #multply by 2 and subtract 1.


def _mutation_flip_count(
    dimension: int,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> int:
    large_roll = torch.rand(1, generator=generator).item()
    if large_roll < params.large_mutation_probability:
        return max(1, int(round(dimension * params.large_mutation_fraction)))

    medium_roll = torch.rand(1, generator=generator).item()
    if medium_roll < 0.5:
        low = min(params.small_mutation_min, dimension)
        high = min(params.small_mutation_max, dimension)
    else:
        low = min(params.medium_mutation_min, dimension)
        high = min(params.medium_mutation_max, dimension)

    if high < low:
        high = low
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


def mutate_candidate(
    candidate: SignHadamardCandidate,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> SignHadamardCandidate:
    mutated = candidate.clone()
    mutate_both = torch.rand(1, generator=generator).item() < params.mutate_both_probability
    targets = ("s0", "s1") if mutate_both else (("s0",) if torch.rand(1, generator=generator).item() < 0.5 else ("s1",))

    for attr in targets:
        vector = getattr(mutated, attr)
        flips = _mutation_flip_count(vector.numel(), params, generator)
        indices = torch.randperm(vector.numel(), generator=generator)[:flips]
        vector[indices] = -vector[indices]

    return mutated


def _rank_weights(count: int) -> torch.Tensor:
    return torch.linspace(count, 1, count, dtype=torch.float64)


def _sample_parent(
    ranked: Sequence[tuple[SignHadamardCandidate, float]],
    params: RotationSearchParams,
    generator: torch.Generator,
) -> SignHadamardCandidate:
    pool_size = max(params.elite_count, int(round(len(ranked) * params.parent_pool_fraction)))
    pool = ranked[:pool_size]
    weights = _rank_weights(len(pool))
    index = int(torch.multinomial(weights, 1, generator=generator).item())
    return pool[index][0]


def _initialize_population(
    site: RotationSearchSite,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> list[SignHadamardCandidate]:
    population = [site.current_candidate.clone()]
    while len(population) < params.population_size:
        population.append(random_sign_candidate(site.dimension, generator))
    return population


def run_rotation_search(
    adapter: RotationSearchAdapter,
    params: RotationSearchParams,
) -> RotationSearchResult:
    """Run fixed-population mutation-only search over all adapter sites."""
    generator = torch.Generator()
    generator.manual_seed(params.seed)

    best_candidates: dict[str, SignHadamardCandidate] = {}
    histories: dict[str, RotationSearchHistory] = {}
    adapter.refresh_caches()

    for site in adapter.sites():
        population = _initialize_population(site, params, generator)
        best_score = float("inf")
        best_candidate = site.current_candidate.clone()
        stagnant_generations = 0
        history = RotationSearchHistory(site_id=site.site_id)
        histories[site.site_id] = history

        if params.verbose:
            print(f"[rotation-search] site={site.site_id} start population={params.population_size}")

        for generation in range(params.generations):
            ranked = sorted(
                (
                    (candidate, adapter.score_site_candidate(site, candidate))
                    for candidate in population
                ),
                key=lambda item: item[1],
            )
            current_best_candidate, current_best_score = ranked[0]
            history.generation_indices.append(generation)
            history.best_scores.append(float(current_best_score))
            if current_best_score + 1e-12 < best_score:
                best_score = current_best_score
                best_candidate = current_best_candidate.clone()
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if params.verbose:
                print(
                    f"[rotation-search] site={site.site_id} generation={generation + 1}/{params.generations} "
                    f"best_nmse={current_best_score:.6f} stagnant={stagnant_generations}"
                )

            if stagnant_generations >= params.patience:
                if params.verbose:
                    print(f"[rotation-search] site={site.site_id} early-stop patience={params.patience}")
                break

            next_population = [candidate.clone() for candidate, _ in ranked[: params.elite_count]]
            while len(next_population) < params.population_size:
                parent = _sample_parent(ranked, params, generator)
                next_population.append(mutate_candidate(parent, params, generator))
            population = next_population

        adapter.commit_site_candidate(site, best_candidate)
        site.current_candidate = best_candidate.clone()
        best_candidates[site.site_id] = best_candidate.clone()
        history.committed_score = float(best_score)
        adapter.refresh_caches()

        if params.verbose:
            print(f"[rotation-search] site={site.site_id} committed_nmse={best_score:.6f}")

    return RotationSearchResult(best_candidates=best_candidates, histories=histories)


def save_rotation_search_artifacts(
    result: RotationSearchResult,
    rotation_path: str,
) -> dict[str, str]:
    """Save search history tensors and a progress plot next to the learned rotations."""
    rotation_file = Path(rotation_path)
    history_path = rotation_file.with_name(f"{rotation_file.stem}_search_history.pt")
    plot_path = rotation_file.with_name(f"{rotation_file.stem}_search_progress.png")

    serializable = {
        site_id: {
            "generation_indices": history.generation_indices,
            "best_scores": history.best_scores,
            "committed_score": history.committed_score,
        }
        for site_id, history in result.histories.items()
    }
    torch.save(serializable, history_path)

    plt.figure(figsize=(12, 6))
    for site_id, history in result.histories.items():
        if history.best_scores:
            plt.plot(history.generation_indices, history.best_scores, alpha=0.35, linewidth=1.0, label=site_id)
    plt.xlabel("Generation")
    plt.ylabel("Best NMSE")
    plt.title("Rotation Search Progress")
    if len(result.histories) <= 10:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {"history_path": str(history_path), "plot_path": str(plot_path)}
