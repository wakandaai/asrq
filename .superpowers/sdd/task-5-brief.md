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
