# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import matplotlib.pyplot as plt
import torch
import os
import time
from asrq.transforms.rotation.hadamard_utils import matmul_hadU
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

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
class ThreeSignHadamardCandidate:
    """Search candidate for a richer signed Hadamard family R = D0 H D1 H D2."""

    s0: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor

    def clone(self) -> "ThreeSignHadamardCandidate":
        return ThreeSignHadamardCandidate(self.s0.clone(), self.s1.clone(), self.s2.clone())

    def to_rotation(
        self,
        base_h: torch.Tensor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        h = base_h.to(device=device)
        left = self.s0.to(device=device, dtype=base_h.dtype).unsqueeze(1) * h
        middle = self.s1.to(device=device, dtype=base_h.dtype).unsqueeze(1) * h
        right = self.s2.to(device=device, dtype=base_h.dtype).unsqueeze(0)
        rotation = left @ middle
        rotation = rotation * right
        return rotation.to(dtype=dtype)


@dataclass
class PairedThreeSignHadamardCandidate:
    """Joint search candidate for paired global rotations, e.g. (Qe, Qd)."""

    qe: ThreeSignHadamardCandidate
    qd: ThreeSignHadamardCandidate

    def clone(self) -> "PairedThreeSignHadamardCandidate":
        return PairedThreeSignHadamardCandidate(self.qe.clone(), self.qd.clone())


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
class GlobalRotationSearchSite:
    """Single searchable global rotation site for Qe."""

    site_id: str
    dimension: int
    base_h: torch.Tensor
    current_candidate: Any
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
    global_score_metric: str = "task_loss"
    seed: int = 42
    verbose: bool = True


@dataclass
class RotationSearchHistory:
    site_id: str
    generation_indices: list[int] = field(default_factory=list)
    best_scores: list[float] = field(default_factory=list)
    generation_durations_sec: list[float] = field(default_factory=list)
    running_best_scores: list[float] = field(default_factory=list)
    population_scores: list[list[float]] = field(default_factory=list)
    elite_scores: list[list[float]] = field(default_factory=list)
    improved_flags: list[bool] = field(default_factory=list)
    stagnant_generation_counts: list[int] = field(default_factory=list)
    generation_mutation_summaries: list[dict[str, Any]] = field(default_factory=list)
    committed_score: float | None = None


@dataclass
class RotationSearchResult:
    best_candidates: dict[str, SignHadamardCandidate]
    histories: dict[str, RotationSearchHistory]


@dataclass
class AlternatingSearchParams:
    q2_params: RotationSearchParams
    qe_params: RotationSearchParams
    q2_refine_params: RotationSearchParams
    outer_rounds: int
    outer_patience: int
    qe_min_delta: float


@dataclass
class AlternatingRotationSearchResult:
    local_result: RotationSearchResult
    global_history: list[RotationSearchHistory]
    best_global_candidate: Any
    best_global_score: float
    outer_rounds_completed: int
    local_history_rounds: list[dict[str, Any]] = field(default_factory=list)


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


class GlobalRotationSearchAdapter(Protocol):
    def global_site(self) -> GlobalRotationSearchSite:
        ...

    def refresh_global_caches(self) -> None:
        ...

    def score_global_candidate(
        self,
        candidate: Any,
    ) -> float:
        ...

    def commit_global_candidate(
        self,
        candidate: Any,
    ) -> None:
        ...

    def initialize_global_population(
        self,
        params: "RotationSearchParams",
        generator: torch.Generator,
    ) -> list[Any]:
        ...

    def mutate_global_candidate(
        self,
        candidate: Any,
        params: "RotationSearchParams",
        generator: torch.Generator,
    ) -> Any:
        ...

    def rotation_state(self) -> dict[str, torch.Tensor]:
        ...


def _serialize_history(history: RotationSearchHistory) -> dict[str, Any]:
    return {
        "site_id": history.site_id,
        "generation_indices": history.generation_indices,
        "best_scores": history.best_scores,
        "generation_durations_sec": history.generation_durations_sec,
        "running_best_scores": history.running_best_scores,
        "population_scores": history.population_scores,
        "elite_scores": history.elite_scores,
        "improved_flags": history.improved_flags,
        "stagnant_generation_counts": history.stagnant_generation_counts,
        "generation_mutation_summaries": history.generation_mutation_summaries,
        "committed_score": history.committed_score,
    }


def _serialize_round_histories(round_histories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized_rounds: list[dict[str, Any]] = []
    for round_history in round_histories:
        serialized_rounds.append(
            {
                "round_index": round_history["round_index"],
                "histories": {
                    site_id: _serialize_history(history)
                    for site_id, history in round_history["histories"].items()
                },
            }
        )
    return serialized_rounds


def random_sign_candidate(
    dimension: int,
    generator: torch.Generator,
) -> SignHadamardCandidate:
    return SignHadamardCandidate(
        _random_sign_vector(dimension, generator),
        _random_sign_vector(dimension, generator),
    )


def random_three_sign_candidate(
    dimension: int,
    generator: torch.Generator,
) -> ThreeSignHadamardCandidate:
    return ThreeSignHadamardCandidate(
        _random_sign_vector(dimension, generator),
        _random_sign_vector(dimension, generator),
        _random_sign_vector(dimension, generator),
    )


def random_paired_three_sign_candidate(
    dimension: int,
    generator: torch.Generator,
) -> PairedThreeSignHadamardCandidate:
    return PairedThreeSignHadamardCandidate(
        qe=random_three_sign_candidate(dimension, generator),
        qd=random_three_sign_candidate(dimension, generator),
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


def _flip_vector_entries(
    vector: torch.Tensor,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> int:
    flips = _mutation_flip_count(vector.numel(), params, generator)
    indices = torch.randperm(vector.numel(), generator=generator)[:flips]
    vector[indices] = -vector[indices]
    return flips


def mutate_candidate(
    candidate: SignHadamardCandidate,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> SignHadamardCandidate:
    mutated, _ = _mutate_candidate_with_metadata(candidate, params, generator)
    return mutated


def _mutate_candidate_with_metadata(
    candidate: SignHadamardCandidate,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> tuple[SignHadamardCandidate, dict[str, Any]]:
    mutated = candidate.clone()
    mutate_both = torch.rand(1, generator=generator).item() < params.mutate_both_probability
    targets = ("s0", "s1") if mutate_both else (("s0",) if torch.rand(1, generator=generator).item() < 0.5 else ("s1",))
    flip_counts: dict[str, int] = {}

    for attr in targets:
        vector = getattr(mutated, attr)
        flip_counts[attr] = _flip_vector_entries(vector, params, generator)

    return mutated, {
        "origin": "mutation",
        "mutated_attrs": list(targets),
        "flip_counts": flip_counts,
        "total_flips": sum(flip_counts.values()),
    }


def mutate_three_sign_candidate(
    candidate: ThreeSignHadamardCandidate,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> ThreeSignHadamardCandidate:
    mutated, _ = _mutate_three_sign_candidate_with_metadata(candidate, params, generator)
    return mutated


def _mutate_three_sign_candidate_with_metadata(
    candidate: ThreeSignHadamardCandidate,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> tuple[ThreeSignHadamardCandidate, dict[str, Any]]:
    mutated = candidate.clone()
    rolls = torch.rand(2, generator=generator)
    if rolls[0].item() < params.mutate_both_probability:
        targets = ("s0", "s1", "s2")
    else:
        selector = rolls[1].item()
        if selector < (1.0 / 3.0):
            targets = ("s0",)
        elif selector < (2.0 / 3.0):
            targets = ("s1",)
        else:
            targets = ("s2",)
    flip_counts: dict[str, int] = {}

    for attr in targets:
        vector = getattr(mutated, attr)
        flip_counts[attr] = _flip_vector_entries(vector, params, generator)

    return mutated, {
        "origin": "mutation",
        "mutated_attrs": list(targets),
        "flip_counts": flip_counts,
        "total_flips": sum(flip_counts.values()),
    }


def mutate_paired_three_sign_candidate(
    candidate: PairedThreeSignHadamardCandidate,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> PairedThreeSignHadamardCandidate:
    mutated, _ = _mutate_paired_three_sign_candidate_with_metadata(candidate, params, generator)
    return mutated


def _mutate_paired_three_sign_candidate_with_metadata(
    candidate: PairedThreeSignHadamardCandidate,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> tuple[PairedThreeSignHadamardCandidate, dict[str, Any]]:
    mutated = candidate.clone()
    mutate_both = torch.rand(1, generator=generator).item() < params.mutate_both_probability
    component_summaries: dict[str, Any] = {}
    if mutate_both:
        mutated.qe, component_summaries["qe"] = _mutate_three_sign_candidate_with_metadata(mutated.qe, params, generator)
        mutated.qd, component_summaries["qd"] = _mutate_three_sign_candidate_with_metadata(mutated.qd, params, generator)
    elif torch.rand(1, generator=generator).item() < 0.5:
        mutated.qe, component_summaries["qe"] = _mutate_three_sign_candidate_with_metadata(mutated.qe, params, generator)
    else:
        mutated.qd, component_summaries["qd"] = _mutate_three_sign_candidate_with_metadata(mutated.qd, params, generator)
    total_flips = sum(summary.get("total_flips", 0) for summary in component_summaries.values())
    return mutated, {
        "origin": "mutation",
        "mutated_components": list(component_summaries.keys()),
        "component_summaries": component_summaries,
        "total_flips": total_flips,
    }


def _summarize_population_metadata(population_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    origins: dict[str, int] = {}
    parent_ranks: dict[int, int] = {}
    mutation_attr_counts: dict[str, int] = {}
    total_flips: list[int] = []
    for metadata in population_metadata:
        origin = metadata.get("origin", "unknown")
        origins[origin] = origins.get(origin, 0) + 1
        if "parent_rank" in metadata:
            rank = int(metadata["parent_rank"])
            parent_ranks[rank] = parent_ranks.get(rank, 0) + 1
        for attr in metadata.get("mutated_attrs", []):
            mutation_attr_counts[attr] = mutation_attr_counts.get(attr, 0) + 1
        for summary in metadata.get("component_summaries", {}).values():
            for attr in summary.get("mutated_attrs", []):
                key = f"paired.{attr}"
                mutation_attr_counts[key] = mutation_attr_counts.get(key, 0) + 1
        if "total_flips" in metadata:
            total_flips.append(int(metadata["total_flips"]))

    summary: dict[str, Any] = {
        "origins": origins,
        "parent_rank_counts": parent_ranks,
        "mutation_attr_counts": mutation_attr_counts,
    }
    if total_flips:
        summary["mean_total_flips"] = float(sum(total_flips) / len(total_flips))
        summary["max_total_flips"] = int(max(total_flips))
    return summary


def _initialize_population_metadata(population_size: int) -> list[dict[str, Any]]:
    metadata = [{"origin": "incumbent"}]
    while len(metadata) < population_size:
        metadata.append({"origin": "random"})
    return metadata


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


def _sample_global_parent(
    ranked: Sequence[tuple[ThreeSignHadamardCandidate, float]],
    params: RotationSearchParams,
    generator: torch.Generator,
) -> ThreeSignHadamardCandidate:
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


def _initialize_global_population(
    site: GlobalRotationSearchSite,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> list[ThreeSignHadamardCandidate]:
    population = [site.current_candidate.clone()]
    while len(population) < params.population_size:
        population.append(random_three_sign_candidate(site.dimension, generator))
    return population


def _initialize_global_population_signed(
    site: GlobalRotationSearchSite,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> list[ThreeSignHadamardCandidate]:
    population = [site.current_candidate.clone()]
    while len(population) < params.population_size:
        population.append(random_sign_candidate(site.dimension, generator))
    return population

def _default_initialize_global_population(
    site: GlobalRotationSearchSite,
    params: RotationSearchParams,
    generator: torch.Generator,
) -> list[Any]:
    current = site.current_candidate
    if isinstance(current, ThreeSignHadamardCandidate):
        return _initialize_global_population(site, params, generator)
    if isinstance(current, PairedThreeSignHadamardCandidate):
        population: list[PairedThreeSignHadamardCandidate] = [current.clone()]
        while len(population) < params.population_size:
            population.append(random_paired_three_sign_candidate(site.dimension, generator))
        return population
    if isinstance(current, SignHadamardCandidate):
        return _initialize_global_population_signed(site, params, generator)
    raise TypeError(f"Unsupported global candidate type: {type(current)!r}")


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
        population_metadata = _initialize_population_metadata(len(population))
        best_score = float("inf")
        best_candidate = site.current_candidate.clone()
        stagnant_generations = 0
        history = RotationSearchHistory(site_id=site.site_id)
        histories[site.site_id] = history

        if params.verbose:
            print(f"[rotation-search] site={site.site_id} start population={params.population_size}")

        for generation in range(params.generations):
            generation_start = time.perf_counter()
            ranked = sorted(
                (
                    (candidate, adapter.score_site_candidate(site, candidate))
                    for candidate in population
                ),
                key=lambda item: item[1],
            )
            generation_duration = time.perf_counter() - generation_start
            current_best_candidate, current_best_score = ranked[0]
            elite_scores = [float(score) for _, score in ranked[: params.elite_count]]
            improved = current_best_score + 1e-12 < best_score
            history.generation_indices.append(generation)
            history.best_scores.append(float(current_best_score))
            history.generation_durations_sec.append(generation_duration)
            history.population_scores.append([float(score) for _, score in ranked])
            history.elite_scores.append(elite_scores)
            history.improved_flags.append(improved)
            history.generation_mutation_summaries.append(_summarize_population_metadata(population_metadata))
            if improved:
                best_score = current_best_score
                best_candidate = current_best_candidate.clone()
                stagnant_generations = 0
            else:
                stagnant_generations += 1
            history.running_best_scores.append(float(best_score))
            history.stagnant_generation_counts.append(stagnant_generations)

            if params.verbose:
                print(
                    f"[rotation-search] site={site.site_id} generation={generation + 1}/{params.generations} "
                    f"best_score={current_best_score:.6f} stagnant={stagnant_generations}"
                )

            if stagnant_generations >= params.patience:
                if params.verbose:
                    print(f"[rotation-search] site={site.site_id} early-stop patience={params.patience}")
                break

            next_population = [candidate.clone() for candidate, _ in ranked[: params.elite_count]]
            next_population_metadata = [
                {"origin": "elite", "source_rank": idx}
                for idx, _ in enumerate(ranked[: params.elite_count])
            ]
            while len(next_population) < params.population_size:
                parent = _sample_parent(ranked, params, generator)
                parent_rank = next(
                    idx for idx, (ranked_candidate, _) in enumerate(ranked) if ranked_candidate is parent
                )
                child, child_metadata = _mutate_candidate_with_metadata(parent, params, generator)
                child_metadata["parent_rank"] = parent_rank
                next_population.append(child)
                next_population_metadata.append(child_metadata)
            population = next_population
            population_metadata = next_population_metadata

        adapter.commit_site_candidate(site, best_candidate)
        site.current_candidate = best_candidate.clone()
        best_candidates[site.site_id] = best_candidate.clone()
        history.committed_score = float(best_score)
        adapter.refresh_caches()

        if params.verbose:
            print(f"[rotation-search] site={site.site_id} committed_score={best_score:.6f}")

    return RotationSearchResult(best_candidates=best_candidates, histories=histories)


def run_global_rotation_search(
    adapter: GlobalRotationSearchAdapter,
    site: GlobalRotationSearchSite,
    params: RotationSearchParams,
) -> tuple[Any, float, RotationSearchHistory]:
    generator = torch.Generator()
    generator.manual_seed(params.seed)
    print("running global search")

    adapter.refresh_global_caches()
    print("done setting global caches")

    if hasattr(adapter, "initialize_global_population"):
        population = adapter.initialize_global_population(params, generator)
    else:
        population = _default_initialize_global_population(site, params, generator)
    population_metadata = _initialize_population_metadata(len(population))
    best_candidate = site.current_candidate.clone()
    best_score = adapter.score_global_candidate(best_candidate)
    
    stagnant_generations = 0
    history = RotationSearchHistory(site_id=site.site_id)

    for generation in range(params.generations):
        generation_start = time.perf_counter()
        population_results = []
        for candidate in tqdm(population):            
            population_results.append((candidate, adapter.score_global_candidate(candidate)))
        # start = time.perf_counter()
        for i in range(len(population_results)):
            print(f"candidate {i} fitness {population_results[i][1]} ")

        # with ProcessPoolExecutor(max_workers=6) as executor:
        #     population_results = list(executor.map(adapter.score_global_candidate, population) )
        # print(f"Multi {time.time() - start}")        
        # results = 

        ranked = sorted(population_results, key=lambda item: item[1] )
        generation_duration = time.perf_counter() - generation_start
        # ranked = sorted(
        #     (
        #         (candidate, adapter.score_global_candidate(candidate))
        #         for candidate in population
        #     ),
        #     key=lambda item: item[1],
        # )
        current_best_candidate, current_best_score = ranked[0]
        print(f"global_generation: {generation} current_best_score: {ranked}")
        elite_scores = [float(score) for _, score in ranked[: params.elite_count]]
        improved = current_best_score + 1e-12 < best_score
        
        history.generation_indices.append(generation)
        history.best_scores.append(float(current_best_score))
        history.generation_durations_sec.append(generation_duration)
        history.population_scores.append([float(score) for _, score in ranked])
        history.elite_scores.append(elite_scores)
        history.improved_flags.append(improved)
        history.generation_mutation_summaries.append(_summarize_population_metadata(population_metadata))
        if improved:
            best_score = current_best_score
            best_candidate = current_best_candidate.clone()
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        history.running_best_scores.append(float(best_score))
        history.stagnant_generation_counts.append(stagnant_generations)

        if stagnant_generations >= params.patience:
            break

        next_population = [candidate.clone() for candidate, _ in ranked[: params.elite_count]]
        next_population_metadata = [
            {"origin": "elite", "source_rank": idx}
            for idx, _ in enumerate(ranked[: params.elite_count])
        ]
        while len(next_population) < params.population_size:
            parent = _sample_global_parent(ranked, params, generator)
            parent_rank = next(
                idx for idx, (ranked_candidate, _) in enumerate(ranked) if ranked_candidate is parent
            )
            if hasattr(adapter, "mutate_global_candidate"):
                child = adapter.mutate_global_candidate(parent, params, generator)
                child_metadata = {"origin": "mutation", "parent_rank": parent_rank}
                next_population.append(child)
                next_population_metadata.append(child_metadata)
            else:
                if isinstance(parent, ThreeSignHadamardCandidate):
                    child, child_metadata = _mutate_three_sign_candidate_with_metadata(parent, params, generator)
                elif isinstance(parent, PairedThreeSignHadamardCandidate):
                    child, child_metadata = _mutate_paired_three_sign_candidate_with_metadata(parent, params, generator)
                elif isinstance(parent, SignHadamardCandidate):
                    child, child_metadata = _mutate_candidate_with_metadata(parent, params, generator)
                else:
                    raise TypeError(f"Unsupported global candidate type: {type(parent)!r}")
                child_metadata["parent_rank"] = parent_rank
                next_population.append(child)
                next_population_metadata.append(child_metadata)
        population = next_population
        population_metadata = next_population_metadata

    adapter.commit_global_candidate(best_candidate)
    site.current_candidate = best_candidate.clone()
    history.committed_score = float(best_score)
    adapter.refresh_global_caches()
    return best_candidate, best_score, history


def run_alternating_rotation_search(
    local_adapter: RotationSearchAdapter,
    global_adapter: GlobalRotationSearchAdapter,
    params: AlternatingSearchParams,
    save_path: str | None = None,
) -> AlternatingRotationSearchResult:
    local_result = run_rotation_search(local_adapter, params.q2_params)
    local_history_rounds: list[dict[str, Any]] = [
        {
            "round_index": 0,
            "histories": local_result.histories,
        }
    ]
    global_site = global_adapter.global_site()
    best_global_candidate = global_site.current_candidate.clone()
    best_global_score = global_adapter.score_global_candidate(best_global_candidate)
    outer_stagnation = 0
    outer_rounds_completed = 0
    global_history: list[RotationSearchHistory] = []
    
    save_dir = Path(save_path).parent if save_path is not None else None
    save_name = Path(save_path).stem if save_path is not None else None

    print(f"="*20)
    print(f"Starting Alternating Search!")

    for round_idx in range(params.outer_rounds):
        print(f"="*20)
        global_epoch_time = 0
        local_epoch_time = 0
        print(f"Round {round_idx + 1}")
        start_time = time.perf_counter()
        qe_candidate, qe_score, qe_history = run_global_rotation_search(global_adapter, global_site, params.qe_params)
        global_epoch_time  = time.perf_counter() - start_time

        qe_history.site_id = f"{qe_history.site_id}.round_{round_idx + 1}"
        global_history.append(qe_history)
        outer_rounds_completed = round_idx + 1

        if qe_score < best_global_score - params.qe_min_delta:
            best_global_score = qe_score
            best_global_candidate = qe_candidate.clone()
            outer_stagnation = 0
            start_time = time.perf_counter()
            local_result = run_rotation_search(local_adapter, params.q2_refine_params)
            local_epoch_time  = time.perf_counter() - start_time
            local_history_rounds.append(
                {
                    "round_index": round_idx + 1,
                    "histories": local_result.histories,
                }
            )
        else:
            outer_stagnation += 1
            if outer_stagnation >= params.outer_patience:
                break
        
                    
        if save_dir is not None and save_name is not None:
            to_save: dict[str, Any] = {}
            if hasattr(global_adapter, "rotation_state"):
                to_save.update(global_adapter.rotation_state())
            elif hasattr(global_adapter, "qe_param"):
                to_save["Qe"] = global_adapter.qe_param.data.detach().cpu()
            if hasattr(local_adapter, "q2_params"):
                to_save["Q2s"] = {k: v.data.detach().cpu() for k, v in local_adapter.q2_params.items()}
            to_save["global_histories"] = [_serialize_history(history) for history in global_history]
            to_save["local_history_rounds"] = _serialize_round_histories(local_history_rounds)
            to_save["local_epoch_time"] = local_epoch_time
            to_save["global_epoch_time"] = global_epoch_time
            torch.save(to_save, os.path.join(save_dir, f"{save_name}_{round_idx}.pt"))
            print(f" saved to {save_name}_{round_idx}.pt")
        
    return AlternatingRotationSearchResult(
        local_result=local_result,
        global_history=global_history,
        local_history_rounds=local_history_rounds,
        best_global_candidate=best_global_candidate,
        best_global_score=best_global_score,
        outer_rounds_completed=outer_rounds_completed,
    )


def save_rotation_search_artifacts(
    result: RotationSearchResult,
    rotation_path: str,
) -> dict[str, str]:
    """Save search history tensors and a progress plot next to the learned rotations."""
    rotation_file = Path(rotation_path)
    history_path = rotation_file.with_name(f"{rotation_file.stem}_search_history.pt")
    plot_path = rotation_file.with_name(f"{rotation_file.stem}_search_progress.png")

    serializable = {
        site_id: _serialize_history(history)
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


def save_alternating_search_artifacts(
    result: AlternatingRotationSearchResult,
    rotation_path: str,
) -> dict[str, str]:
    rotation_file = Path(rotation_path)
    local = save_rotation_search_artifacts(result.local_result, rotation_path)
    global_history_path = rotation_file.with_name(f"{rotation_file.stem}_global_search_history.pt")
    global_plot_path = rotation_file.with_name(f"{rotation_file.stem}_global_search_progress.png")

    serialized_global = [_serialize_history(history) for history in result.global_history]
    torch.save(serialized_global, global_history_path)
    torch.save(
        {
            "site_histories": {
                site_id: _serialize_history(history)
                for site_id, history in result.local_result.histories.items()
            },
            "round_histories": _serialize_round_histories(result.local_history_rounds),
        },
        local["history_path"],
    )

    plt.figure(figsize=(12, 6))
    for history in result.global_history:
        if history.best_scores:
            plt.plot(
                history.generation_indices,
                history.best_scores,
                alpha=0.7,
                linewidth=1.5,
                label=history.site_id,
            )
    plt.xlabel("Generation")
    plt.ylabel("Best Global Score")
    plt.title("Global Rotation Search Progress")
    if len(result.global_history) <= 10 and result.global_history:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(global_plot_path)
    plt.close()

    return {
        "local_history_path": local["history_path"],
        "local_plot_path": local["plot_path"],
        "global_history_path": str(global_history_path),
        "global_plot_path": str(global_plot_path),
    }
