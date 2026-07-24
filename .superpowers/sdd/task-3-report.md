# Task 3 Report

## Status

DONE_WITH_CONCERNS

## Scope

- Modified `/home/blessedg/asrq/asrq/transforms/rotation/parakeet_ctc_utils.py`
- Modified `/home/blessedg/asrq/tests/test_rotation_alternating_search.py`

## What Changed

- Added Parakeet global Qe fake-quant CTC scoring support through `_ParakeetGlobalQeSearchAdapter`.
- Added alternating-search dispatch in `obtain_rotations_for_parakeet_search(...)` while preserving the existing local-Q2 path when `search_mode != "alternating"`.
- Accepted the task-brief keyword names `qe_params` and `q2_refine_params`, while keeping compatibility with the already-wired local callers that still pass `qe_search_params` and `q2_refine_search_params`.
- Added focused tests for:
  - global Qe scoring via Parakeet fake-quant CTC loss
  - alternating dispatch and global-history artifact save
- Added `_save_global_rotation_search_artifacts(...)` so alternating runs persist global search history next to the local history artifacts.

## TDD Notes

1. Added the new Parakeet-focused tests first in `tests/test_rotation_alternating_search.py`.
2. Initial focused execution attempts exposed environment issues before feature failures:
   - `pytest` not on PATH in the default shell
   - `/usr/bin/python` missing `pytest`
   - local PyTorch import failure (`libtorch_cuda.so: undefined symbol: ncclMemFree`)
3. Switched to the repo’s documented `conda activate asrq` runner under the approved `srun` job.
4. First meaningful red failure:
   - `TypeError: obtain_rotations_for_parakeet_search() got an unexpected keyword argument 'qe_params'`
5. Implemented the minimal production changes to satisfy the failing Parakeet alternating-search contract.
6. Re-ran the focused Parakeet tests to green.

## Test Commands

### Red-phase command from the brief

```bash
srun --jobid=107159 --overlap bash -lc 'source ~/.bashrc && conda activate asrq && cd /home/blessedg/asrq && python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py::test_parakeet_search_uses_global_qe_loss_when_search_mode_is_alternating -v'
```

Observed result:
- after fixing the test import harness, this single test passed

### Focused verification command for Task 3

```bash
srun --jobid=107159 --overlap bash -lc 'source ~/.bashrc && conda activate asrq && cd /home/blessedg/asrq && python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py -k parakeet -v'
```

Observed result:
- `2 passed, 3 deselected in 4.80s`

## Concerns

- I only ran the focused Parakeet slice requested for this task, not the broader rotation suite.
- `parakeet_ctc_utils.py` already contained in-flight uncommitted Task 3-related edits when I started; I preserved them and layered the minimal contract fixes on top rather than rewriting that section.
