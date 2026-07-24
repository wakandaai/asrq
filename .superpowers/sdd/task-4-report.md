Status: DONE_WITH_CONCERNS

Summary
- Added focused Whisper alternating-search tests in `tests/test_rotation_alternating_search.py`.
- Updated `asrq/transforms/rotation/whisper_utils.py` to fall back to CPU when CUDA is unavailable in the Whisper search path so the alternating dispatch test can execute in this environment.
- `asrq/transforms/rotation/__init__.py` did not require a change because `obtain_rotations_for_whisper_search` was already exported in the working tree.

TDD Notes
- I first added a failing focused Whisper test for alternating dispatch and a focused global-Qe metric test.
- The intended red-phase command from the brief could not run in the default shell environment because:
  - `pytest` was not installed for `/usr/bin/python`
  - that interpreter also imported a broken CUDA `torch` build (`undefined symbol: ncclMemFree`)
- I switched to the existing `asrq` conda environment and reran the focused tests there.
- The global-Qe metric test passed immediately because `_WhisperGlobalQeSearchAdapter` and the teacher-forced CE scoring path were already present in the working tree before my change.
- The alternating-dispatch Whisper test failed first with `RuntimeError: No CUDA GPUs are available` from the hard-coded `device = "cuda"` path in `obtain_rotations_for_whisper_search`.
- After changing that entrypoint to use `device = "cuda" if torch.cuda.is_available() else "cpu"`, the Whisper-focused tests passed.

Files Changed
- `asrq/transforms/rotation/whisper_utils.py`
- `tests/test_rotation_alternating_search.py`

Files Inspected But Not Changed
- `asrq/transforms/rotation/__init__.py`

Test Commands Run
1. `pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py::test_whisper_global_qe_metric_uses_teacher_forced_ce -v`
   - Failed immediately: `pytest: command not found`
2. `python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py::test_whisper_global_qe_metric_uses_teacher_forced_ce -v`
   - Failed immediately: `No module named pytest`
3. `bash -lc 'source /home/blessedg/miniconda3/etc/profile.d/conda.sh && conda activate asrq && python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py::test_whisper_global_qe_metric_uses_teacher_forced_ce -v'`
   - Passed: `1 passed`
4. `bash -lc 'source /home/blessedg/miniconda3/etc/profile.d/conda.sh && conda activate asrq && python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py -k whisper -v'`
   - First run failed: `test_whisper_obtain_rotations_dispatches_to_alternating_search` hit `RuntimeError: No CUDA GPUs are available`
5. `bash -lc 'source /home/blessedg/miniconda3/etc/profile.d/conda.sh && conda activate asrq && python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py -k whisper -v'`
   - Passed after the CPU fallback change: `2 passed, 6 deselected, 1 warning`
6. `bash -lc 'source /home/blessedg/miniconda3/etc/profile.d/conda.sh && conda activate asrq && python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py -v'`
   - First run exposed a test-stub regression in `matplotlib.pyplot.figure`
7. `bash -lc 'source /home/blessedg/miniconda3/etc/profile.d/conda.sh && conda activate asrq && python -m pytest /home/blessedg/asrq/tests/test_rotation_alternating_search.py -v'`
   - Passed after expanding the pyplot stub: `8 passed, 1 warning`

Concerns
- Strict red/green was only partially achievable because part of Task 4 had already been implemented in the working tree before I started; the new global-Qe metric test did not fail first.
- The successful focused runs depend on the existing `asrq` conda environment rather than the default system Python.
- The final focused file run still emits one warning from `torch.cuda` / NVML initialization in the environment, but the tests pass.
