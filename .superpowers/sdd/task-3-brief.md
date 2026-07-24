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

