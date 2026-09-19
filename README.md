# ASRQ — ASR Quantization Toolkit

ASRQ quantizes large pre-trained speech recognition models to low bit widths, down to 2-bit weights and 4-bit
weights and activations (W4A4), while keeping WER close to full precision. It combines rotation and scaling
transforms, GPTQ, closed-form corrections after GPTQ, and real low-bit inference through
[humming](https://github.com/ldfrancis/humming) kernels.

## Supported models

| Config name | Model | Architecture |
|---|---|---|
| `whisper` | `openai/whisper-large-v3` | Transformer encoder-decoder (Hugging Face) |
| `parakeet` | `nvidia/parakeet-ctc-1.1b` | Conformer encoder, CTC head (NeMo) |
| `canary_qwen` | `nvidia/canary-qwen-2.5b` | Conformer encoder + Qwen3 LLM decoder (NeMo SALM, LoRA merged) |

## Results

Full LibriSpeech test-clean / test-other WER (%). GPTQ is calibrated on 2048 utterances, and the rotations
come from the evolutionary Hadamard search.

| Model | Full precision | W2 weight-only + block output refit | W4A4 + norm tweak |
|---|---|---|---|
| Parakeet-CTC 1.1B | 1.83 / 3.53 | 2.52 / 5.29 | 1.83 / 3.75 |
| Canary-Qwen 2.5B | 1.57 / 3.06 | 3.34 / 5.85 | 1.69 / 3.22 |
| Whisper large-v3 | 1.94 / 3.88 | 2.42 / 5.36 | 2.11 / 4.53 |

- **W2:** 2-bit asymmetric weights in groups of 128, activations left at 16 bits.
- **W4A4:** 4-bit symmetric weights in groups of 128. Activations are 4-bit per token, except `attn_out` and
  `fc2`, which use groups of 128.

---

## Installation

```bash
git submodule update --init third_party/open_asr_leaderboard third_party/humming
pip install -e .
pip install "nemo-toolkit[asr]"
pip install "transformers==4.57.6" "datasets==3.6.0" evaluate lhotse soundfile scikit-learn
pip install -e third_party/humming --no-deps
```

- **`third_party/open_asr_leaderboard`:** supplies the English text normalizer used for WER. Its pinned commit
  decides what counts as a match, so re-run the baselines after bumping it.
- **`third_party/humming`:** a fork of humming (branch `hadamard-sign-seed`). Its kernels generate the random
  Hadamard signs from a seed, and ASRQ's rotation checkpoints store that seed. Install it over the
  `humming-kernels` package. Its kernels are JIT-compiled on first use, which needs `ninja` and a CUDA toolkit.

## Calibration data

Calibration uses LibriSpeech `train.clean.360` audio, paired with each model's own transcripts. Build the set
once per model; the audio is selected on the first run and shared after that:

```bash
python -m asrq.calibration.build --model openai/whisper-large-v3 \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
python -m asrq.calibration.build --model nvidia/parakeet-ctc-1.1b \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
python -m asrq.calibration.build --model nvidia/canary-qwen-2.5b \
    --calibration-dir outputs/calibration/librispeech_train_clean_360_2048
```

`calibration.path` in `asrq/configs/config.yaml` points at this directory. If it's empty, calibration streams
LibriSpeech with its reference text instead.

---

## Workflow

The two entry points share one Hydra config (`asrq/configs/config.yaml`):

1. **`asrq/rot-exp.py`** learns a rotation and saves it to `transform.path`.
2. **`asrq/exp.py`** takes a model through these steps:
   1. loads it;
   2. applies the saved transform;
   3. quantizes it, or loads a previously quantized copy;
   4. optionally converts it to humming kernels;
   5. evaluates it.

The same activation settings (`activation_bits`, `activation_group_size`, `activation_symmetric`,
`activation_groupwise_roles`) and weight settings (`quantizer.*`) are passed to the rotation search and to
evaluation. So a rotation is searched against the quantization it will be evaluated with. Use the same
overrides for both commands.

### 1. Learn a rotation (`rot-exp.py`)

The rotation transform:
- folds each LayerNorm into an RMSNorm and rotates the residual stream by R1;
- rotates each attention module's value/output path by a head-wise R2;
- inserts an online Hadamard with random signs before every `fc2` layer (feed-forward down projection,
  Conformer `pointwise_conv2`, Qwen `down_proj`).

Canary-Qwen has two R1 streams, one for the Conformer encoder and one for the LLM. Two search methods are
available (`transform.search`):

| `search` | What is searched | Cost |
|---|---|---|
| `cayley` | R1 and the R2s, from random Hadamards, by Cayley SGD on the KL objective | `num_samples`, `epochs`, `learning_rate`, `batch_size` |
| `evolution` | R1 = diag(s1) · H · diag(s2), with (1+λ) evolution over the sign vectors; R2 stays a random Hadamard | `transform.evolution.*`; 8 generations ≈ Cayley's time on Whisper |

```bash
# W4A4: evolutionary search against 4-bit activations
python asrq/rot-exp.py model=parakeet activation_bits=4 transform.search=evolution \
    transform.num_samples=64 transform.path=outputs/rotation/parakeet_a4.pt

# W4A4: Cayley SGD
python asrq/rot-exp.py model=whisper activation_bits=4 transform.search=cayley \
    transform.num_samples=800 transform.learning_rate=0.5 transform.path=outputs/rotation/whisper_a4_cayley.pt

# W2 weight-only: evolutionary search against GPTQ-quantized weights
python asrq/rot-exp.py model=canary_qwen activation_bits=16 quantizer=gptq quantizer.bits=2 \
    quantizer.symmetric=False quantizer.group_size=128 transform.weight_only=true \
    transform.weight_only_quantizer=gptq transform.search=evolution transform.num_samples=64 \
    transform.path=outputs/rotation/canary_qwen_w2.pt
```

`evolution.generations: 0` saves the random Hadamard starting point without searching, which gives a
random-rotation baseline.

#### How the search quantizes candidates

Every candidate rotation computes the same full-precision function, so only the quantization error differs
between candidates. The fitness is the KL divergence from the full-precision model's logits to the quantized
model's, on the calibration samples. The full-precision logits are computed once and cached
(`evolution.cache_teacher`).

**Weight-activation search** (`activation_bits` < 16, Cayley or evolution):
- The rotations are patched into each layer's forward pass rather than folded into the weights, so R1 and R2 can
  change every step.
- Every layer's input is fake-quantized with the same quantizer that evaluation attaches.
  - The quantized roles are q, k, v, fc1, attn_out and fc2.
  - The roles in `activation_groupwise_roles` get one scale per group; the rest get one per token.
  - `activation_symmetric` picks symmetric or asymmetric activation quantization.
- Weights stay in full precision during the search.

**Weight-only search** (`transform.weight_only=true`, evolution only, activations in full precision). Each
candidate is applied exactly as a finished rotation would be:
1. The norm-folded original weights are restored. They are kept in pinned CPU memory.
2. R1, the R2s and the online Hadamards are folded into the weights.
3. The weights are quantized with `quantizer.bits`, `quantizer.group_size` and `quantizer.symmetric`, by
   `transform.weight_only_quantizer`:
   - **`rtn`:** round to nearest, on the same grid the RTN quantizer uses.
   - **`gptq`:** GPTQ on Hessians collected once, from the model with R1 set to the identity.
     - Layers that read the rotated residual stream (q, k, v, fc1) see the input X·R1. For a candidate, their
       Hessian is therefore R1ᵀ·H·R1, and it is re-factored per candidate.
     - Every other layer's Hessian doesn't depend on R1, so it is factored once.
     - Layers with the same shape are quantized together by a batched GPTQ solver.
   - **`null`:** use the experiment's quantizer.

The fitness then runs the deployed model's forward pass, with quantized weights and no patching.

#### Evolution details

One generation follows EvoPress's multi-step selection:
1. The parent produces `offspring` children, each with `flips` sign positions flipped.
2. Stage 1 scores every child on a few samples, and the best `survivors[0]` go on.
3. Stage 2 scores those on more samples, and the best `survivors[1]` go on.
4. Stage 3 scores the rest on the full set.

The best child replaces the parent only if its KL is lower. `stage_samples: null` picks 8/16/N samples when
N ≤ 64, 8/64/N when N ≤ 256, and 16/64/N above that.

`evolution.mutate` chooses which sign vectors are flipped:
- `s1`: only s1.
- `s1_s2`: both s1 and s2.
- `auto` (the default) picks:
  - `s1` for symmetric activation quantization, where a sign flip after H cannot change a symmetric quantizer's
    error;
  - `s1_s2` for asymmetric activation quantization and for every weight-only search.

The online Hadamard's random signs are stored as a single seed (`hadamard_sign_seed`). The humming fork
regenerates the same signs inside its kernel, so the fused weights and the kernel always agree.

### 2. Quantize and evaluate (`exp.py`)

```bash
# W4A4: GPTQ W4 on the saved rotation, fake 4-bit activations, full LibriSpeech
python asrq/exp.py model=parakeet quantizer=gptq activation_bits=4 \
    transform.path=outputs/rotation/parakeet_a4.pt \
    "eval_datasets=[{dataset:librispeech,split:test.clean},{dataset:librispeech,split:test.other}]"

# W2 weight-only, asymmetric, groups of 128
python asrq/exp.py model=canary_qwen quantizer=gptq quantizer.bits=2 quantizer.symmetric=False \
    quantizer.group_size=128 activation_bits=16 transform.path=outputs/rotation/canary_qwen_w2.pt

# No transform, RTN
python asrq/exp.py model=whisper quantizer=rtn transform=none
```

- **Quantizers:** `quantizer=gptq` (block by block, calibrated) and `quantizer=rtn` (round to nearest).
  `quantizer.bits`, `quantizer.group_size` and `quantizer.symmetric` set the weight grid.
- **Activations:** quantization is simulated (fake-quantized) during evaluation. `activation_bits: 16` turns it
  off.
- **Evaluation data:** `eval_datasets` lists the Open ASR Leaderboard datasets and splits to evaluate (all
  seven by default). `eval_batches: N` evaluates only the first N batches of each split, which is useful for
  quick checks.
- **Results:** results go to `results/evaluations/<model>_<method>_..._results.csv`, next to a copy of the
  config. Set `method=<label>` to name a run.

#### Reusing quantized models

With `quantized_path: auto` (the default), exp.py saves the quantized model to
`outputs/quantized/<model>-<hash>.pt`. A later run with the same settings loads that file instead of quantizing
again.

- **What the hash covers:** the model, the whole quantizer config, the calibration set, the transform config and
  the contents of the rotation file. Changing any of them quantizes again into a new file.
- **What it leaves out:** evaluation and inference settings. You can re-evaluate the same weights with a
  different `inference`, `eval_dtype` or dataset.
- **Options:**
  - `quantized_path=/some/file.pt` uses a specific file.
  - `quantized_path=null` never saves or loads.
  - Deleting the file forces a re-quantization.
- **Disk use:** files keep the model's own dtype, so a float32 model costs about 4 bytes per parameter.

### 3. Corrections after GPTQ: norm tweaking and block output refitting

Both corrections run inside the GPTQ block loop, right after each block is quantized. Each is solved in closed
form from statistics that GPTQ's calibration pass already collects. They are independent of each other; the
results above use one or the other. Derivations, limits and full results are in
[docs/norm_tweaking_and_output_refit.md](docs/norm_tweaking_and_output_refit.md).

**Norm tweaking** (`quantizer.norm_tweak=true`, `asrq/quantizers/norm_tweak.py`) rescales each norm that feeds
the block's quantized layers. The per-channel scale s makes the quantized layers reproduce the full-precision
outputs:

    min_s  Σ_l ‖X·W_lᵀ − X·diag(s)·Q_lᵀ‖²   ⇒   (H ⊙ Σ_l Q_lᵀQ_l + λI) s = diag(H · Σ_l W_lᵀQ_l) + λ

- **Symbols:** H = XᵀX is the GPTQ Hessian. W_l and Q_l are layer l's full-precision and quantized weights.
- **λ:** `norm_tweak_ridge` × mean diag, which pulls s toward 1.
- **Scope:** the fit covers weight error only.
- **Cost:** none at inference, since the scale is folded into the norm.

**Block output refitting** (`quantizer.block_output_refit=true`, `asrq/quantizers/output_refit.py`) refits the
linear layer at each block's output. It maps the quantized block's input to that layer, Z, onto the
full-precision block's output, Y:

    [W, b] = (Yᵀ·Za + λ·[W0, b0]) · (Zaᵀ·Za + λI)⁻¹,   Za = [Z, 1]

- **What it corrects:** any error that is linear in Z, including the error of layers that no norm feeds.
- **Conformer blocks** (Parakeet, and Canary-Qwen's encoder): the identity linear that rotation inserts after
  each block's output norm is refit, so inference costs nothing extra.
- **Transformer blocks** (Whisper, the Qwen LLM): set `quantizer.block_output_refit_insert=true` to give each
  block an identity `output_linear`. It stays unquantized, which costs one fp16 d×d matmul per block.
- **λ:** `block_output_refit_ridge` pulls the fit toward the current weights.

```bash
# W2 with block output refitting (inserted layers for the transformer blocks)
python asrq/exp.py model=whisper quantizer=gptq quantizer.bits=2 quantizer.symmetric=False \
    quantizer.group_size=128 activation_bits=16 quantizer.block_output_refit=true \
    quantizer.block_output_refit_insert=true transform.path=outputs/rotation/whisper_w2.pt

# W4A4 with norm tweaking
python asrq/exp.py model=whisper quantizer=gptq activation_bits=4 quantizer.norm_tweak=true \
    transform.path=outputs/rotation/whisper_a4.pt
```

On full LibriSpeech, refitting is better at W2 (Canary-Qwen test-other 5.85 vs 7.49 with norm tweaking). At
W4A4 the two are within about 0.1 WER, and norm tweaking has no inference cost.

### 4. Real low-bit inference and speedups (humming)

By default (`inference: fake`), quantization is simulated in fp16 or bf16, which is what the WER numbers use.
To measure speed, `inference=humming` replaces every activation-quantized layer with an `ASRQLinear` that runs
humming's low-bit matmul:
- packed weights;
- per-token or group-wise activation quantization, from the same config;
- the fc2 online Hadamard and its seeded signs fused into the same kernel call.

Weights must be symmetric.

Use it with float16 and a CUDA-graph transcription function. humming's per-call Python overhead dominates eager
decoding, and in bfloat16 every layer converts to float16 and back:

```bash
python asrq/exp.py model=parakeet quantizer=gptq activation_bits=4 transform.path=outputs/rotation/parakeet_a4.pt \
    inference=humming eval_dtype=float16 model.generate_fn=generate_parakeet_cuda_graphs
```

| Model | `model.generate_fn` | W4A4 vs fp16 (both graphed, float16) |
|---|---|---|
| Whisper | `generate_whisper_cuda_graphs` | 1.22× at batch 1, 1.15× at batch 64 |
| Parakeet | `generate_parakeet_cuda_graphs` | 1.31× at batch 1, 1.14× at batch 128 |
| Canary-Qwen | `generate_canaryqwen_cuda_graphs` | 1.45× at batch 1, 1.07× at batch 128 |

W2A16 Whisper decoding is 1.14× fp16 at batch 1, and its linear weights shrink from 2801 MiB to 373 MiB.
humming's WER matches fake quantization within noise.

### Scaling transform

`transform=scaling` is a SmoothQuant-style per-channel scaling on the same layers that rotation targets. Where a
nonlinearity feeds the layer, it inserts an `InputScale` module after the activation. The scales are searched in
exp.py against the configured weight and activation quantization, and saved to `transform.path`:

```bash
python asrq/exp.py model=parakeet quantizer=gptq activation_bits=8 transform=scaling \
    transform.path=outputs/scaling/parakeet_a8.pt
```

---

## Configuration

`asrq/configs/` holds the Hydra config groups, and any value can be overridden on the command line.

| File | Main settings |
|---|---|
| `config.yaml` | Calibration, activation quantization, `quantized_path`, `inference`, `eval_datasets`, `eval_batches`, `eval_dtype` |
| `model/*.yaml` | `exclude_modules`, `eval_batch_size`, `generate_fn`, `quantize_block_output_linear` (Conformer) |
| `quantizer/gptq.yaml` | `bits`, `group_size`, `symmetric`, `percdamp`, `norm_tweak*`, `block_output_refit*` |
| `quantizer/rtn.yaml` | `bits`, `group_size`, `symmetric` |
| `transform/rotation.yaml` | `search`, `evolution.*`, `weight_only`, `weight_only_quantizer`, `hadamard_block_size`, `learn_r2`, `fc2_online_hadamard`, `path` |
| `transform/scaling.yaml` | `type`, `obtain_scales`, `path` |
| `transform/none.yaml` | No transform |

## Project structure

```
asrq/
├── exp.py, rot-exp.py         # Hydra entry points (quantize/evaluate, learn a rotation)
├── experiment.py              # run_experiment, learn_rotation_experiment, quantized-model caching
├── configs/                   # config.yaml, model/, quantizer/, transform/
├── calibration/               # calibration set builder and loader
├── core/
│   ├── model.py               # ModelQ: quantize, corrections, save/load, humming conversion
│   └── linear.py              # ASRQLinear (humming low-bit layer)
├── models/
│   ├── transformers/whisper.py
│   └── nemo/parakeet_ctc.py, canary_qwen.py
├── quantizers/
│   ├── gptq.py, gptq_solver.py, rtn.py, weight_rounding.py
│   ├── activation.py          # activation fake quantization, shared by the search and evaluation
│   ├── norm_tweak.py          # closed-form norm tweaking
│   └── output_refit.py        # block output refitting
├── transforms/
│   ├── rotation/              # rotation transform, Cayley SGD, evolutionary Hadamard search, per-model layers
│   └── scaling/               # scaling transform and per-model layers
└── evaluation/                # Open ASR Leaderboard evaluation, CUDA-graph transcription per model
tests/                         # unit tests; `-m integration` runs small end-to-end pipelines per model
third_party/                   # open_asr_leaderboard (normalizer), humming (fork with seeded Hadamard signs)
```

## Tests

```bash
pytest -m "not integration" tests
pytest -m integration tests
```

The integration tests need a GPU and the cached models. They learn a rotation, then quantize and evaluate each
model on 8 utterances.
