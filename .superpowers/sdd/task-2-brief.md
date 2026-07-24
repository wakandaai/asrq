### Task 2: Add Global Qe Candidate Family and Alternating Search Orchestrator

**Files:**
- Modify: `asrq/asrq/transforms/rotation/search.py`
- Test: `asrq/tests/test_rotation_qe_candidates.py`
- Test: `asrq/tests/test_rotation_alternating_search.py`

**Interfaces:**
- Consumes: `RotationSearchParams`, `RotationSearchSite`, existing `SignHadamardCandidate`
- Produces:
  - `ThreeSignHadamardCandidate`
  - `GlobalRotationSearchAdapter`
  - `AlternatingSearchParams`
  - `run_global_rotation_search(adapter, site, params) -> tuple[candidate, score, history]`
  - `run_alternating_rotation_search(local_adapter, global_adapter, params) -> AlternatingRotationSearchResult`

- [ ] **Step 1: Write failing tests for `Qe` candidate family and outer-loop control**

```python
def test_three_sign_hadamard_candidate_builds_square_rotation():
    base_h = normalized_hadamard_matrix(8)
    candidate = ThreeSignHadamardCandidate(
        s0=torch.ones(8, dtype=torch.int8),
        s1=-torch.ones(8, dtype=torch.int8),
        s2=torch.ones(8, dtype=torch.int8),
    )
    rotation = candidate.to_rotation(base_h, device="cpu", dtype=torch.float32)
    assert rotation.shape == (8, 8)


def test_alternating_search_skips_q2_refine_when_qe_does_not_improve():
    local_adapter = FakeLocalAdapter()
    global_adapter = FakeGlobalAdapter(scores=[1.0, 0.99995])
    result = run_alternating_rotation_search(local_adapter, global_adapter, params)
    assert result.outer_rounds_completed == 1
    assert local_adapter.refine_calls == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest asrq/tests/test_rotation_qe_candidates.py asrq/tests/test_rotation_alternating_search.py -v`

Expected: FAIL because `ThreeSignHadamardCandidate` and the alternating orchestrator do not exist.

- [ ] **Step 3: Add `Qe` candidate representation and mutation**

```python
@dataclass
class ThreeSignHadamardCandidate:
    s0: torch.Tensor
    s1: torch.Tensor
    s2: torch.Tensor

    def clone(self) -> "ThreeSignHadamardCandidate":
        return ThreeSignHadamardCandidate(self.s0.clone(), self.s1.clone(), self.s2.clone())

    def to_rotation(self, base_h: torch.Tensor, *, device, dtype) -> torch.Tensor:
        left = self.s0.to(device=device, dtype=base_h.dtype).unsqueeze(1)
        mid = self.s1.to(device=device, dtype=base_h.dtype)
        right = self.s2.to(device=device, dtype=base_h.dtype).unsqueeze(0)
        h = base_h.to(device=device)
        return ((left * h) @ torch.diag(mid) @ h * right).to(dtype=dtype)
```

- [ ] **Step 4: Add global adapter protocol and alternating outer loop**

```python
class GlobalRotationSearchAdapter(Protocol):
    def global_site(self) -> RotationSearchSite:
        ...

    def refresh_global_caches(self) -> None:
        ...

    def score_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> float:
        ...

    def commit_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> None:
        ...


@dataclass
class AlternatingSearchParams:
    q2_params: RotationSearchParams
    qe_params: RotationSearchParams
    q2_refine_params: RotationSearchParams
    outer_rounds: int
    outer_patience: int
    qe_min_delta: float
```

- [ ] **Step 5: Implement the alternating loop with early-stop and history**

```python
for round_idx in range(params.outer_rounds):
    qe_candidate, qe_score, qe_history = run_global_rotation_search(...)
    qe_improved = qe_score < best_qe_score - params.qe_min_delta
    if qe_improved:
        best_qe_score = qe_score
        outer_stagnation = 0
        run_rotation_search(local_adapter, params.q2_refine_params)
    else:
        outer_stagnation += 1
        if outer_stagnation >= params.outer_patience:
            break
```

- [ ] **Step 6: Run tests to verify search primitives pass**

Run: `pytest asrq/tests/test_rotation_qe_candidates.py asrq/tests/test_rotation_alternating_search.py -v`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add asrq/asrq/transforms/rotation/search.py asrq/tests/test_rotation_qe_candidates.py asrq/tests/test_rotation_alternating_search.py
git commit -m "feat: add alternating q2 qe search engine"
```

