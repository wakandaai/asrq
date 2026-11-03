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
tail -f outputs/pipeline/<name>.log | grep -E "^WER|  step |Traceback"   # live
grep -E "^\[|^WER" outputs/pipeline/<name>.log                               # results afterwards
```

Recovery after quantization:

- `scale_recovery` (`quantizer.scale_recovery`) gives each layer a transformer block's norms feed a
  per-input-channel scale, fitted inside GPTQ and folded into the norm.
- `weight_recovery` (`quantizer.weight_recovery`) refits the Linear a rotation inserts after a Conformer
  block's output norm, so the quantized block reproduces the full-precision block's output.
- `layer_weight_recovery` (`quantizer.layer_weight_recovery`) refits every quantized weight the same way, each
  one just before it is quantized, against the output it had before anything in its block was quantized. It
  also makes each block quantize its layers in the order they run rather than the order they are defined.

`scale_recovery` only applies to weight-only runs: it folds a per-input-channel scale into a norm, which the
activation quantizer would then have to round, and where a rotation spread the outliers across those channels a
per-channel scale pulls them apart again. `asrq/experiment.py` turns it off whenever `activation_bits` is below
16 and prints that it did, so a run's recorded config says what happened.
