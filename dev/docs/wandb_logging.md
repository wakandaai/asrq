# Weights & Biases logging

Off by default. Everything goes through `asrq/tracking.py`, which is a no-op until a run is open, so nothing
else imports wandb and the code runs unchanged with logging off or wandb missing.

```bash
pip install ".[wandb]"

python asrq/rot-exp.py model=whisper activation_bits=4 transform.search=evolution \
    transform.num_samples=128 transform.path=outputs/rotation/evo128_whisper_a4.pt \
    wandb.enabled=true wandb.project=asrq wandb.group=w4a4-search

python asrq/exp.py model=whisper transform=rotation transform.path=outputs/rotation/evo128_whisper_a4.pt \
    quantizer=gptq activation_bits=4 quantizer.scale_recovery=true quantizer.block_output_refit=true \
    wandb.enabled=true wandb.group=w4a4-full method=w4a4_evo128
```

| Setting | Meaning |
|---|---|
| `wandb.enabled` | `false` by default; nothing is logged and wandb is never imported |
| `wandb.project`, `wandb.entity` | where the run goes (`asrq` by default) |
| `wandb.group` | groups the runs of one experiment, e.g. a search and the run that uses its rotation |
| `wandb.tags` | free tags, e.g. `[w2, canary]` |
| `wandb.mode` | `online`, `offline` or `disabled` |

**One run per script.** `rot-exp.py` opens a run with `job_type: rotation`, `exp.py` one with `job_type:
quantize`. The run's name defaults to `<model>-<method>`, and the whole resolved config is stored as the run
config, so every setting is recoverable from the run.

A metric group that repeats carries its own counter as its x axis (`search/generation`, `search/step`,
`scale_recovery/group`, `refit/block`), so the charts are indexed by what actually varies.

---

## Rotation search (`rot-exp.py`)

**Both searches:**

| Key | What |
|---|---|
| `rotation/logit_error_norm_folding`, `..._rotation_patching`, `..._rotation_folding` | the verification passes' relative logit error; these must stay tiny, and they catch a broken rewrite |
| `rotation/initial_objective` | the KL at the random Hadamard start, the baseline the search improves on |
| `rotation/objective`, `rotation/activation_bits` | what is being minimised and against which activation quantization |
| `rotation/seconds`, `rotation/path` | wall time and the checkpoint written |

**Evolutionary search** (`transform.search=evolution`), per generation on `search/generation`:

| Key | What |
|---|---|
| `search/fitness` | the parent's KL after the generation; never increases |
| `search/best_child` | the best child's KL, which shows how close the generation came when it was rejected |
| `search/accepted` | 1 when the parent was replaced, so the acceptance rate is visible |
| `search/candidates_seen` | the archive's size, i.e. distinct candidates scored so far |
| `search/generation_seconds` | cost per generation |

Summary: `search/name`, `search/stages`, `search/offspring`, `search/flips`, `search/mutate`,
`search/samples`, `search/samples_per_generation`, `search/initial_fitness`, `search/final_fitness`,
`search/improvement`.

**Cayley SGD** (`transform.search=cayley`), every `log_every` steps on `search/step`:

| Key | What |
|---|---|
| `search/loss`, `search/loss_min`, `search/loss_max` | the objective over the window |
| `search/learning_rate` | the schedule's current rate |
| `search/orthogonality_drift` | `|R1ᵀR1 − I|`, which must stay near zero or the rewrite stops being exact |
| `search/epoch` | the epoch the step belongs to |

Summary: `search/name`, `search/steps`, `search/epoch_<n>_loss`, `search/final_loss`.

---

## Quantization and evaluation (`exp.py`)

**The run's config** also records what the applied rotation checkpoint says about itself: `rotation_file`,
`rotation_model`, `rotation_search`, `rotation_search_settings`, `rotation_activation_quantization`. So a
quantization run shows which search produced its rotation without opening the file.

**Quantization:**

| Key | What |
|---|---|
| `quantize/loaded_from_cache` | whether `quantized_path` was reused instead of quantizing |
| `quantize/seconds` | wall time of the quantization pass, when it ran |

**Scale recovery** (`quantizer.scale_recovery=true`), per block on `scale_recovery/group`:

| Key | What |
|---|---|
| `scale_recovery/error_ratio`, `..._max` | the block's mean and worst `L(s)/L(1)`: how much of the quantized layers' output error the scales removed |
| `scale_recovery/scale_min`, `scale_recovery/scale_max` | the range of the fitted scales; values far from 1 mean the grid was badly matched |
| `scale_recovery/scales` | a histogram of the scales |

**Block output refitting** (`quantizer.block_output_refit=true`), per block on `refit/block`:

| Key | What |
|---|---|
| `refit/error_before`, `refit/error_after` | the block's output error relative to the full-precision output's energy, before and after the refit |
| `refit/error_ratio` | after / before, so a chart over blocks shows where the refit helps |

**Evaluation**, per dataset split:

| Key | What |
|---|---|
| `eval/wer`, `eval/rtfx`, `eval/dataset` | the split's WER, real-time factor and name |
| `eval/<dataset>.<split>/wer`, `.../rtfx` | the same in the summary, so runs are comparable in a table |
| `eval/wer_mean`, `eval/splits` | the mean WER over the evaluated splits |

---

## Comparing runs

- **Group a search with the runs that use its rotation** (`wandb.group`), so a rotation's KL trajectory sits
  beside the WER it produced.
- **The summary table** carries the settings that matter (through the run config) plus `eval/*/wer`, which is
  the comparison the experiments are about.
- **The per-block charts** answer where a recipe helps: `scale_recovery/error_ratio` and `refit/error_ratio`
  against the block index show whether early or late blocks dominate the error.
