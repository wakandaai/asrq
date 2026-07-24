Status: DONE
Files changed:
- /home/blessedg/asrq/asrq/configs/transform/rotation.yaml
- /home/blessedg/asrq/asrq/transforms/rotation/base.py
- /home/blessedg/asrq/tests/test_rotation_alternating_search.py

Tests run:
1. bash -lc 'source ~/.bashrc && conda activate asrq && cd /home/blessedg/asrq && python -m unittest discover -s tests -p "test_rotation_alternating_search.py"'
   Result: PASS (2 tests, 0.005s)
2. bash -lc 'source ~/.bashrc && conda activate asrq && python -m py_compile /home/blessedg/asrq/asrq/transforms/rotation/base.py /home/blessedg/asrq/tests/test_rotation_alternating_search.py'
   Result: PASS

Notes:
- Added alternating-search config knobs and stage-specific RotationSearchParams helpers.
- Preserved existing search behavior by default with search_mode=local_q2.
- Alternating-only dispatch args are gated behind search_mode == "alternating" so current local-Q2 entrypoints remain runtime-safe.
