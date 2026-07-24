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

