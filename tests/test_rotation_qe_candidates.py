import sys
import types
import unittest
from importlib import import_module
from pathlib import Path
from typing import Optional

import torch


ROOT = Path(__file__).resolve().parents[1]
ASRQ_ROOT = ROOT / "asrq"


def _stub_package(name: str, path: Optional[Path] = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_stub_package("matplotlib")
_stub_module("matplotlib.pyplot")
_stub_package("asrq", ASRQ_ROOT)
_stub_package("asrq.transforms", ASRQ_ROOT / "transforms")
_stub_package("asrq.transforms.rotation", ASRQ_ROOT / "transforms" / "rotation")

search_module = import_module("asrq.transforms.rotation.search")
RotationSearchParams = search_module.RotationSearchParams
RotationSearchSite = search_module.RotationSearchSite
ThreeSignHadamardCandidate = search_module.ThreeSignHadamardCandidate
normalized_hadamard_matrix = search_module.normalized_hadamard_matrix
run_global_rotation_search = search_module.run_global_rotation_search


class DummyGlobalAdapter:
    def __init__(self, site: RotationSearchSite, target: ThreeSignHadamardCandidate) -> None:
        self._site = site
        self._target = target
        self.refresh_count = 0
        self.commits: list[ThreeSignHadamardCandidate] = []

    def global_site(self) -> RotationSearchSite:
        return self._site

    def refresh_global_caches(self) -> None:
        self.refresh_count += 1

    def score_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> float:
        return float(
            (candidate.s0 != self._target.s0).sum()
            + (candidate.s1 != self._target.s1).sum()
            + (candidate.s2 != self._target.s2).sum()
        )

    def commit_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> None:
        self.commits.append(candidate.clone())


class RotationGlobalQeCandidateTests(unittest.TestCase):
    def test_three_sign_hadamard_candidate_builds_square_rotation(self) -> None:
        base_h = normalized_hadamard_matrix(8)
        candidate = ThreeSignHadamardCandidate(
            s0=torch.ones(8, dtype=torch.int8),
            s1=-torch.ones(8, dtype=torch.int8),
            s2=torch.ones(8, dtype=torch.int8),
        )

        rotation = candidate.to_rotation(base_h, device="cpu", dtype=torch.float32)

        self.assertEqual(rotation.shape, (8, 8))

    def test_global_search_commits_candidate_and_records_history(self) -> None:
        base_h = normalized_hadamard_matrix(8)
        start = ThreeSignHadamardCandidate(
            s0=torch.ones(8, dtype=torch.int8),
            s1=torch.ones(8, dtype=torch.int8),
            s2=torch.ones(8, dtype=torch.int8),
        )
        target = start.clone()
        target.s0[0] *= -1
        target.s1[1] *= -1
        target.s2[2] *= -1
        site = RotationSearchSite(
            site_id="global.site",
            block_id="global.block",
            dimension=8,
            base_h=base_h,
            current_candidate=start,
        )
        adapter = DummyGlobalAdapter(site, target)
        params = RotationSearchParams(
            population_size=10,
            elite_count=2,
            parent_pool_fraction=0.5,
            generations=12,
            patience=4,
            seed=3,
            verbose=False,
        )

        best_candidate, best_score, history = run_global_rotation_search(adapter, site, params)

        self.assertTrue(adapter.commits)
        self.assertLessEqual(best_score, adapter.score_global_candidate(start))
        self.assertEqual(len(history.best_scores), len(history.generation_indices))
        self.assertGreaterEqual(adapter.refresh_count, 2)
        self.assertIsInstance(best_candidate, ThreeSignHadamardCandidate)


if __name__ == "__main__":
    unittest.main()
