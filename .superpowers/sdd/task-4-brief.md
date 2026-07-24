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

