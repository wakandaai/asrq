# Experiment scripts

One bash script per experiment, each opening with a comment block that says what the experiment answers, what
it varies, what it holds fixed and where its output goes. Scripts call `asrq/rot-exp.py` (learn a rotation) or
`asrq/exp.py` (quantize and evaluate) and nothing else, so an experiment is reproducible by reading one file.

```bash
bash scripts/<name>.sh                      # foreground, with the default conda and project paths
bash scripts/<name>.sh ~/miniconda3/bin/activate ~/asrq   # or give them as arguments
nohup bash scripts/<name>.sh > outputs/pipeline/<name>.log 2>&1 &   # detached, survives the shell
```

Conventions:

- **`method=<label>`** on every run: it names the results directory under `results/evaluations/` and the wandb
  run, so a run's settings are recoverable from its name and its `config.yaml`.
- **`quantized_path=null`** for sweeps (a saved model is several GB); `auto` when the same weights will be
  evaluated more than once.
- **Evaluation batch size** per model: Whisper 64, Parakeet and Canary-Qwen 128.
- **`MODELS=...`** and similar environment variables let a script run a subset, e.g. `MODELS=whisper bash
  scripts/<name>.sh`.
- Runs print everything they produce. Only `tr '\r' '\n'` is applied, which turns tqdm's redraws into separate
  lines so the log stays line-based; filter it when reading:

```bash
tail -f outputs/pipeline/<name>.log | grep -E "^WER|generation |Traceback"   # live
grep -E "^\[|^WER" outputs/pipeline/<name>.log                               # results afterwards
```
