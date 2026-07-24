# Alternating Q2/Qe Rotation Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current local `Q2` search into a configurable alternating search pipeline that runs `Q2 -> Qe -> Q2` with early stopping, using local NMSE for `Q2` and global fake-quant task loss for `Qe`.

**Architecture:** Keep the existing per-site search engine for local `Q2` search, then add a second global search path for `Qe` using a separate evaluator interface. Wrap both inside an outer alternating loop with configurable round count, per-stage budgets, and early-stop logic based on meaningful `Qe` improvement.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf, existing ASRQ rotation search utilities, NeMo Parakeet evaluation/loss path, Hugging Face Whisper teacher-forced loss path, matplotlib for saved search plots.

## Global Constraints

- Preserve the current `transform.type=search` entrypoint; do not break single-stage `Q2`-only search.
- Keep the first implementation limited to Whisper and Parakeet.
- Reuse the existing signed-Hadamard family for `Q2`: `R = D0 H D1`.
- Introduce a richer signed-Hadamard family for `Qe`: `R = D0 H D1 H D2`.
- `Q2` search metric remains full-block NMSE under fake quantization.
- `Qe` search metric must be model-native task loss under fake quantization.
- Parakeet global metric: fake-quant CTC loss.
- Whisper global metric: fake-quant teacher-forced seq2seq cross-entropy loss.
- Outer-loop refinement should skip the trailing `Q2` stage when `Qe` does not improve by at least `qe_min_delta`.
- Save histories and plots for both local and global search stages.

---

## File Structure

- Modify: `asrq/asrq/configs/transform/rotation.yaml`
  - Add alternating-search config knobs for outer rounds and per-stage budgets.
- Modify: `asrq/asrq/transforms/rotation/base.py`
  - Parse new config fields and route between legacy single-stage search and alternating search.
- Modify: `asrq/asrq/transforms/rotation/search.py`
  - Add `Qe` candidate family, global search runner, alternating outer loop orchestration, and richer history serialization.
- Modify: `asrq/asrq/transforms/rotation/parakeet_ctc_utils.py`
  - Add Parakeet global `Qe` evaluator and alternating-search adapter wiring.
- Modify: `asrq/asrq/transforms/rotation/whisper_utils.py`
  - Add Whisper global `Qe` evaluator and alternating-search adapter wiring.
- Modify: `asrq/asrq/transforms/rotation/__init__.py`
  - Export any new alternating-search entrypoints.
- Modify: `asrq/runners.md`
  - Document practical commands for alternating search.
- Modify: `asrq/README.md`
  - Document the new alternating-search design and config knobs.
- Create: `asrq/tests/test_rotation_alternating_search.py`
  - Unit tests for outer-loop control flow and early-stop logic.
- Create: `asrq/tests/test_rotation_qe_candidates.py`
  - Unit tests for `Qe` candidate family and mutation behavior.

### Task 1: Add Alternating-Search Config and Public API

**Files:**
- Modify: `asrq/asrq/configs/transform/rotation.yaml`
- Modify: `asrq/asrq/transforms/rotation/base.py`
- Test: `asrq/tests/test_rotation_alternating_search.py`

**Interfaces:**
- Consumes: `RotationTransformConfig`, existing `RotationSearchParams`
- Produces:
  - `RotationTransformConfig.outer_rounds: int`
  - `RotationTransformConfig.outer_patience: int`
  - `RotationTransformConfig.qe_min_delta: float`
  - `RotationTransformConfig.q2_refine_generations: int`
  - `RotationTransformConfig.qe_generations: int`
  - `RotationTransformConfig.qe_patience: int`
  - `RotationTransformConfig.search_mode: str`
  - `RotationTransformConfig.qe_search_params(seed: int) -> RotationSearchParams`
  - `RotationTransformConfig.q2_refine_search_params(seed: int) -> RotationSearchParams`

- [ ] **Step 1: Write the failing config/dispatch test**

```python
def test_rotation_transform_config_exposes_alternating_search_knobs():
    cfg = OmegaConf.create(
        {
            "name": "rotation",
            "type": "search",
            "search_mode": "alternating",
            "outer_rounds": 3,
            "outer_patience": 1,
            "qe_min_delta": 1e-3,
            "generations": 16,
            "q2_refine_generations": 6,
            "qe_generations": 10,
            "patience": 4,
            "qe_patience": 3,
            "population_size": 8,
            "elite_count": 2,
            "parent_pool_fraction": 0.5,
            "mutate_both_probability": 0.1,
            "large_mutation_probability": 0.1,
            "small_mutation_min": 1,
            "small_mutation_max": 2,
            "medium_mutation_min": 4,
            "medium_mutation_max": 8,
            "large_mutation_fraction": 0.25,
            "num_samples": 128,
            "epochs": 1,
            "learning_rate": 0.01,
            "batch_size": 1,
            "learn_rotation": True,
            "use": True,
            "path": "",
        }
    )
    rotation_cfg = RotationTransformConfig(cfg)
    assert rotation_cfg.search_mode == "alternating"
    assert rotation_cfg.outer_rounds == 3
    assert rotation_cfg.qe_generations == 10
    assert rotation_cfg.q2_refine_generations == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest asrq/tests/test_rotation_alternating_search.py::test_rotation_transform_config_exposes_alternating_search_knobs -v`

Expected: FAIL because the new config fields and helper methods do not exist.

- [ ] **Step 3: Add config fields and helper constructors**

```python
class RotationTransformConfig(TransformConfig):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.search_mode = getattr(cfg, "search_mode", "local_q2")
        self.outer_rounds = getattr(cfg, "outer_rounds", 1)
        self.outer_patience = getattr(cfg, "outer_patience", 1)
        self.qe_min_delta = getattr(cfg, "qe_min_delta", 0.0)
        self.q2_refine_generations = getattr(cfg, "q2_refine_generations", self.generations)
        self.qe_generations = getattr(cfg, "qe_generations", self.generations)
        self.qe_patience = getattr(cfg, "qe_patience", self.patience)

    def qe_search_params(self, seed: int) -> RotationSearchParams:
        return RotationSearchParams(
            population_size=self.population_size,
            elite_count=self.elite_count,
            parent_pool_fraction=self.parent_pool_fraction,
            generations=self.qe_generations,
            patience=self.qe_patience,
            mutate_both_probability=self.mutate_both_probability,
            large_mutation_probability=self.large_mutation_probability,
            small_mutation_min=self.small_mutation_min,
            small_mutation_max=self.small_mutation_max,
            medium_mutation_min=self.medium_mutation_min,
            medium_mutation_max=self.medium_mutation_max,
            large_mutation_fraction=self.large_mutation_fraction,
            seed=seed,
        )
```

- [ ] **Step 4: Update transform dispatch to keep old behavior by default**

```python
if self.cfg.type == "search" and self.cfg.search_mode == "alternating":
    obtain_rotations_for_parakeet_search(
        ...,
        search_mode="alternating",
        q2_params=self.cfg.search_params(seed),
        qe_params=self.cfg.qe_search_params(seed + 1),
        q2_refine_params=self.cfg.q2_refine_search_params(seed + 2),
        outer_rounds=self.cfg.outer_rounds,
        outer_patience=self.cfg.outer_patience,
        qe_min_delta=self.cfg.qe_min_delta,
    )
elif self.cfg.type == "search":
    obtain_rotations_for_parakeet_search(...existing local q2 args...)
```

- [ ] **Step 5: Run tests to verify config/dispatch passes**

Run: `pytest asrq/tests/test_rotation_alternating_search.py::test_rotation_transform_config_exposes_alternating_search_knobs -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add asrq/asrq/configs/transform/rotation.yaml asrq/asrq/transforms/rotation/base.py asrq/tests/test_rotation_alternating_search.py
git commit -m "feat: add alternating rotation search config"
```

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

### Task 3: Add Parakeet Global Qe Search Adapter

**Files:**
- Modify: `asrq/asrq/transforms/rotation/parakeet_ctc_utils.py`
- Test: `asrq/tests/test_rotation_alternating_search.py`

**Interfaces:**
- Consumes:
  - `run_rotation_search`
  - `run_alternating_rotation_search`
  - `set_rotation_fake_quant_state`
  - existing Parakeet calibration dataset / collate / `parakeet_ctc_loss_fn`
- Produces:
  - `_ParakeetGlobalQeSearchAdapter`
  - `obtain_rotations_for_parakeet_search(..., search_mode: str, q2_params, qe_params, q2_refine_params, outer_rounds, outer_patience, qe_min_delta, ...)`

- [ ] **Step 1: Write a failing test for Parakeet alternating dispatch**

```python
def test_parakeet_search_uses_global_qe_loss_when_search_mode_is_alternating():
    adapter = FakeParakeetGlobalAdapter()
    score = adapter.score_global_candidate(fake_candidate)
    assert isinstance(score, float)
    assert adapter.loss_name == "ctc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest asrq/tests/test_rotation_alternating_search.py::test_parakeet_search_uses_global_qe_loss_when_search_mode_is_alternating -v`

Expected: FAIL because the global adapter and alternating branch are missing.

- [ ] **Step 3: Build a Parakeet global `Qe` scorer around fake-quant CTC loss**

```python
class _ParakeetGlobalQeSearchAdapter:
    def score_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> float:
        self._apply_qe_candidate(candidate)
        set_rotation_fake_quant_state(
            self.model,
            enabled=True,
            activation_bits=self.activation_bits,
            weight_bits=self.weight_bits,
        )
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.calibration_batches:
                total_loss += float(parakeet_ctc_loss_fn(self.model, _clone_batch(batch)).item())
        set_rotation_fake_quant_state(self.model, enabled=False, activation_bits=self.activation_bits, weight_bits=self.weight_bits)
        return total_loss / len(self.calibration_batches)
```

- [ ] **Step 4: Wire `Q2 -> Qe -> Q2` into `obtain_rotations_for_parakeet_search`**

```python
if search_mode == "alternating":
    local_adapter = _ParakeetRotationSearchAdapter(...)
    global_adapter = _ParakeetGlobalQeSearchAdapter(...)
    result = run_alternating_rotation_search(
        local_adapter,
        global_adapter,
        AlternatingSearchParams(...),
    )
else:
    result = run_rotation_search(local_adapter, q2_params)
```

- [ ] **Step 5: Save both local and global histories**

```python
artifacts = save_rotation_search_artifacts(result.local_result, save_path)
save_global_rotation_search_artifacts(result.global_histories, save_path)
```

- [ ] **Step 6: Run focused tests**

Run: `pytest asrq/tests/test_rotation_alternating_search.py -k parakeet -v`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add asrq/asrq/transforms/rotation/parakeet_ctc_utils.py asrq/tests/test_rotation_alternating_search.py
git commit -m "feat: add parakeet global qe search"
```

### Task 4: Add Whisper Global Qe Search Adapter

**Files:**
- Modify: `asrq/asrq/transforms/rotation/whisper_utils.py`
- Modify: `asrq/asrq/transforms/rotation/__init__.py`
- Test: `asrq/tests/test_rotation_alternating_search.py`

**Interfaces:**
- Consumes:
  - existing Whisper calibration dataset / `whisper_loss_fn`
  - `set_rotation_fake_quant_state`
  - alternating search engine from Task 2
- Produces:
  - `_WhisperGlobalQeSearchAdapter`
  - alternating-search path in `obtain_rotations_for_whisper_search`

- [ ] **Step 1: Write a failing test for Whisper global metric selection**

```python
def test_whisper_global_qe_metric_uses_teacher_forced_ce():
    adapter = FakeWhisperGlobalAdapter()
    score = adapter.score_global_candidate(fake_candidate)
    assert isinstance(score, float)
    assert adapter.loss_name == "teacher_forced_ce"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest asrq/tests/test_rotation_alternating_search.py::test_whisper_global_qe_metric_uses_teacher_forced_ce -v`

Expected: FAIL because the Whisper global adapter does not exist.

- [ ] **Step 3: Build the Whisper global `Qe` scorer**

```python
class _WhisperGlobalQeSearchAdapter:
    def score_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> float:
        self._apply_qe_candidate(candidate)
        set_rotation_fake_quant_state(
            self.model,
            enabled=True,
            activation_bits=self.activation_bits,
            weight_bits=self.weight_bits,
        )
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.calibration_batches:
                total_loss += float(whisper_loss_fn(self.model, _clone_tree(batch)).item())
        set_rotation_fake_quant_state(self.model, enabled=False, activation_bits=self.activation_bits, weight_bits=self.weight_bits)
        return total_loss / len(self.calibration_batches)
```

- [ ] **Step 4: Wire alternating search into the Whisper entrypoint**

```python
if search_mode == "alternating":
    local_adapter = _WhisperRotationSearchAdapter(...)
    global_adapter = _WhisperGlobalQeSearchAdapter(...)
    result = run_alternating_rotation_search(...)
else:
    result = run_rotation_search(local_adapter, q2_params)
```

- [ ] **Step 5: Export any new entrypoint or helper needed by `rotation/base.py`**

```python
from .whisper_utils import obtain_rotations_for_whisper_search
```

- [ ] **Step 6: Run focused tests**

Run: `pytest asrq/tests/test_rotation_alternating_search.py -k whisper -v`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add asrq/asrq/transforms/rotation/whisper_utils.py asrq/asrq/transforms/rotation/__init__.py asrq/tests/test_rotation_alternating_search.py
git commit -m "feat: add whisper global qe search"
```

### Task 5: Update Artifacts, Docs, and Runner Guidance

**Files:**
- Modify: `asrq/asrq/transforms/rotation/search.py`
- Modify: `asrq/README.md`
- Modify: `asrq/runners.md`
- Test: manual smoke commands only

**Interfaces:**
- Consumes: alternating-search results and histories from Tasks 2-4
- Produces:
  - saved `*_global_search_history.pt`
  - saved `*_global_search_progress.png`
  - runner commands for `search_mode=alternating`

- [ ] **Step 1: Write the history/plot extension test**

```python
def test_alternating_search_saves_local_and_global_histories(tmp_path):
    result = make_fake_alternating_result()
    artifacts = save_alternating_search_artifacts(result, str(tmp_path / "rotation.pt"))
    assert artifacts["local_history_path"].endswith("_search_history.pt")
    assert artifacts["global_history_path"].endswith("_global_search_history.pt")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest asrq/tests/test_rotation_alternating_search.py::test_alternating_search_saves_local_and_global_histories -v`

Expected: FAIL because alternating artifact saving does not exist.

- [ ] **Step 3: Extend artifact saving for alternating search**

```python
def save_alternating_search_artifacts(result, rotation_path: str) -> dict[str, str]:
    local = save_rotation_search_artifacts(result.local_result, rotation_path)
    global_history_path = rotation_file.with_name(f"{rotation_file.stem}_global_search_history.pt")
    global_plot_path = rotation_file.with_name(f"{rotation_file.stem}_global_search_progress.png")
    torch.save(serialize_global_histories(result.global_histories), global_history_path)
    plot_global_histories(result.global_histories, global_plot_path)
    return {
        "local_history_path": local["history_path"],
        "local_plot_path": local["plot_path"],
        "global_history_path": str(global_history_path),
        "global_plot_path": str(global_plot_path),
    }
```

- [ ] **Step 4: Update docs and runner commands**

```markdown
transform.type=search
transform.search_mode=alternating
transform.outer_rounds=3
transform.outer_patience=1
transform.qe_min_delta=1e-3
transform.qe_generations=8
transform.q2_refine_generations=6
```

- [ ] **Step 5: Run smoke verification**

Run:

```bash
python -m unittest discover -s asrq/tests -p "test_rotation_alternating_search.py"
python -m py_compile asrq/asrq/transforms/rotation/search.py asrq/asrq/transforms/rotation/parakeet_ctc_utils.py asrq/asrq/transforms/rotation/whisper_utils.py asrq/asrq/transforms/rotation/base.py
```

Expected:
- unit tests PASS
- `py_compile` exits successfully

- [ ] **Step 6: Commit**

```bash
git add asrq/asrq/transforms/rotation/search.py asrq/README.md asrq/runners.md asrq/tests/test_rotation_alternating_search.py
git commit -m "docs: add alternating q2 qe rotation search usage"
```

## Self-Review

- Spec coverage:
  - Alternating `Q2 -> Qe -> Q2` loop: covered in Task 2.
  - Skip/break on no meaningful `Qe` improvement: covered in Task 2.
  - Model-native global metrics: Parakeet in Task 3, Whisper in Task 4.
  - Configurable number of outer rounds and per-stage budgets: covered in Task 1.
  - Saved progress artifacts and runner docs: covered in Task 5.
- Placeholder scan:
  - No `TODO`/`TBD` placeholders remain.
  - Each task has explicit files, interfaces, commands, and expected behavior.
- Type consistency:
  - `RotationSearchParams` remains the per-stage local/global search budget object.
  - `AlternatingSearchParams` owns the outer-loop knobs and stage param bundles.
  - `ThreeSignHadamardCandidate` is distinct from `SignHadamardCandidate`.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-21-alternating-q2-qe-search.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
