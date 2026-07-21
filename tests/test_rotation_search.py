import unittest

import torch

from asrq.transforms.rotation.search import (
    RotationSearchParams,
    RotationSearchResult,
    RotationSearchSite,
    SignHadamardCandidate,
    normalized_hadamard_matrix,
    random_sign_candidate,
    run_rotation_search,
)


class DummyAdapter:
    def __init__(self, site: RotationSearchSite, target: SignHadamardCandidate) -> None:
        self._site = site
        self._target = target
        self.refresh_count = 0
        self.commits: list[SignHadamardCandidate] = []

    def sites(self):
        return [self._site]

    def refresh_caches(self) -> None:
        self.refresh_count += 1

    def score_site_candidate(
        self,
        site: RotationSearchSite,
        candidate: SignHadamardCandidate,
    ) -> float:
        return float((candidate.s0 != self._target.s0).sum() + (candidate.s1 != self._target.s1).sum())

    def commit_site_candidate(
        self,
        site: RotationSearchSite,
        candidate: SignHadamardCandidate,
    ) -> None:
        self.commits.append(candidate.clone())


class RotationSearchTests(unittest.TestCase):
    def test_signed_hadamard_candidate_builds_orthogonal_rotation(self) -> None:
        base_h = normalized_hadamard_matrix(8)
        candidate = SignHadamardCandidate(
            s0=torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.int8),
            s1=torch.tensor([-1, 1, -1, 1, -1, 1, -1, 1], dtype=torch.int8),
        )

        rotation = candidate.to_rotation(base_h, device="cpu", dtype=torch.float64)
        should_be_identity = rotation.T @ rotation

        self.assertTrue(torch.allclose(should_be_identity, torch.eye(8, dtype=torch.float64), atol=1e-6))

    def test_search_commits_improved_candidate_and_refreshes_context(self) -> None:
        generator = torch.Generator().manual_seed(11)
        start = random_sign_candidate(8, generator)
        target = SignHadamardCandidate(s0=start.s0.clone(), s1=start.s1.clone())
        target.s0[:2] *= -1
        target.s1[4:6] *= -1

        site = RotationSearchSite(
            site_id="dummy.site",
            block_id="dummy.block",
            dimension=8,
            base_h=normalized_hadamard_matrix(8),
            current_candidate=start.clone(),
        )
        adapter = DummyAdapter(site, target)
        params = RotationSearchParams(
            population_size=10,
            elite_count=2,
            parent_pool_fraction=0.5,
            generations=24,
            patience=6,
            seed=7,
        )

        result = run_rotation_search(adapter, params)

        self.assertIn("dummy.site", result.best_candidates)
        best = result.best_candidates["dummy.site"]
        self.assertLessEqual(
            adapter.score_site_candidate(site, best),
            adapter.score_site_candidate(site, start),
        )
        self.assertTrue(adapter.commits)
        self.assertGreaterEqual(adapter.refresh_count, 2)

    def test_search_returns_per_site_history(self) -> None:
        generator = torch.Generator().manual_seed(3)
        start = random_sign_candidate(8, generator)
        target = SignHadamardCandidate(s0=start.s0.clone(), s1=start.s1.clone())
        target.s0[0] *= -1

        site = RotationSearchSite(
            site_id="history.site",
            block_id="history.block",
            dimension=8,
            base_h=normalized_hadamard_matrix(8),
            current_candidate=start.clone(),
        )
        adapter = DummyAdapter(site, target)
        params = RotationSearchParams(population_size=6, elite_count=1, generations=4, patience=4, seed=5)

        result = run_rotation_search(adapter, params)

        self.assertIsInstance(result, RotationSearchResult)
        self.assertIn("history.site", result.best_candidates)
        self.assertIn("history.site", result.histories)
        self.assertGreaterEqual(len(result.histories["history.site"].best_scores), 1)
        self.assertEqual(
            len(result.histories["history.site"].best_scores),
            len(result.histories["history.site"].generation_indices),
        )


if __name__ == "__main__":
    unittest.main()
