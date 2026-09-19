# Closed-form norm tweaking & block output refitting

Two corrections applied after GPTQ that recover quantization error without gradient steps. Both run inside
the GPTQ block loop right after a block's layers are quantized, and both are solved in closed form from
statistics the calibration pass already produces. They are independent: each has its own flag, and each can
be used alone.

| | Norm tweaking | Block output refitting |
|---|---|---|
| Code | `asrq/quantizers/norm_tweak.py` | `asrq/quantizers/output_refit.py` |
| Flag | `quantizer.norm_tweak=true` | `quantizer.block_output_refit=true` (plus `block_output_refit_insert=true` for transformer blocks) |
| Parameters per block | one scale vector (length d) per norm | a d×(d+1) weight and bias |
| What it corrects | the layers the norm feeds (q/k/v, fc1, gate/up) | the whole block's output, including layers no norm feeds (o_proj, fc2, pointwise_conv2, down_proj) |
| Inference cost | none (the scale lives in the norm) | none for Conformer blocks; one unquantized d×d matmul per transformer block |
| Best at | W4A4 | W2 |

---

## Convention

`nn.Linear` computes `Y = X @ W.T + b` on batched **row** vectors: `X: (tokens, in)`, `W: (out, in)`.
Every derivation below is in this form. `diag(s)` is the diagonal matrix of a vector `s`, `⊙` is the
elementwise product, and `H = X.T @ X` is the GPTQ Hessian (up to GPTQ's normalisation) of the layers
reading `X`.

---

## 1. Closed-form norm tweaking

### Idea

A norm feeds one or more linear layers `l`, with full-precision weights `W_l` and, after GPTQ, quantized
weights `Q_l`. All of them read the same input `X`, the norm's output. Multiplying the norm's output by a
per-channel scale `s` gives the layers `X @ diag(s)`. Choose `s` so the quantized layers reproduce the
full-precision outputs as closely as possible:

```
L(s) = Σ_l || X @ W_l.T  −  X @ diag(s) @ Q_l.T ||_F²
```

The bias of each layer appears on both sides and cancels.

### Derivation

`X @ diag(s) @ Q_l.T = X @ (Q_l @ diag(s)).T`, so the error of layer `l` is `X @ E_l.T` with

```
E_l(s) = W_l − Q_l @ diag(s)            (out_l, d)
L(s)   = Σ_l tr(E_l @ H @ E_l.T)
```

Column `j` of `Q_l`, `q_lj`, is the only part of `Q_l @ diag(s)` that `s_j` touches:
`∂E_l / ∂s_j = −q_lj @ e_j.T`. Then

```
∂L/∂s_j = −2 Σ_l tr(q_lj @ e_j.T @ H @ E_l.T) = −2 Σ_l (H @ E_l.T @ Q_l)_jj
```

Setting every derivative to zero:

```
diag(H @ Σ_l W_l.T @ Q_l)  =  diag(H @ diag(s) @ Σ_l Q_l.T @ Q_l)
```

The right side's `j`-th entry is `Σ_k H_jk s_k (Q.T Q)_kj`, which is linear in `s` with matrix
`H ⊙ Σ_l Q_l.T @ Q_l` (`Q.T Q` is symmetric). So `L` is quadratic in `s`, and its minimum solves the
`d × d` system

```
G s = b,     G = H ⊙ Σ_l Q_l.T @ Q_l,     b = diag(H @ Σ_l W_l.T @ Q_l)
```

### Ridge

A channel the calibration barely excites has a tiny row in `G`, and its `s_j` would be set by noise. Adding
`λ ||s − 1||²` pulls every scale toward 1 (no change):

```
(G + λI) s = b + λ·1,        λ = norm_tweak_ridge × mean(diag(G))       (default ridge 0.01)
```

### Applying the scale

`scale_norm_output(norm, s)`:
- **Norm with an affine weight (and bias):** multiply both by `s`. The whole output `γ ⊙ x̂ + β` is scaled.
- **Scale-free norm:** a rotation folds γ into the next layers and leaves the norm scale-free. Here `s` is
  registered as a new `weight` buffer, which the norm multiplies its output by. `s` then acts in the rotated
  basis, which is where the quantized layers' inputs live.

The scale is part of the norm, so inference costs nothing extra, fake or humming.

### Where it is applied

Only norms whose layers are **all** quantized in the current block (`norm_tweak_targets()` per model):

| Model | Norm → layers |
|---|---|
| Whisper (encoder and decoder) | `self_attn_layer_norm` → q/k/v; `final_layer_norm` → fc1; decoder `encoder_attn_layer_norm` → cross-attention q only (k/v read the encoder output) |
| Parakeet, Canary-Qwen encoder | `norm_feed_forward1` → ff1.linear1; `norm_self_att` → q/k/v; `norm_conv` → pointwise_conv1; `norm_feed_forward2` → ff2.linear1 |
| Canary-Qwen LLM | `input_layernorm` → q/k/v_proj; `post_attention_layernorm` → gate/up_proj |

### In the GPTQ loop

1. The calibration pass accumulates each layer's Hessian as GPTQ always does.
2. **Before** GPTQ, `capture_norm_tweaks` copies the first target layer's `H`, which GPTQ discards when it
   quantizes, and every target layer's full-precision `W_l`.
3. GPTQ quantizes the block's layers.
4. `apply_norm_tweaks` reads the quantized `Q_l`, solves for `s` and scales the norm. The log reports
   `L(s) / L(1)`, the output error left relative to no tweak.

### Limits

- **Weight error only.** Activations are not quantized during GPTQ, so the fit does not see activation
  error. With quantized activations, `s` also changes what the activation quantizer rounds, which is not
  linear in `s`. A gradient variant through fake-quantized activations was tried and removed: it was hard to
  tune and gained little (Canary-Qwen W4A4 1.56 / 2.65 closed form vs 1.54 / 2.57 gradient).
- **`H` comes from the pre-quantization pass.** A norm deeper in a block (e.g. `norm_conv`, after the
  attention module) reads a residual stream that changes slightly once the earlier sublayers are quantized.
  The fit uses the unquantized sublayers' inputs.
- **d degrees of freedom per norm.** Layers no norm feeds (attention output, fc2, pointwise_conv2,
  down_proj) are not corrected. The measured output-error ratios are modest: a mean of 0.92–0.95 per norm
  at W2.

---

## 2. Block output refitting

### Idea

A block that ends in a linear layer computes its output as `Z @ W.T + b`, where `Z` is that layer's input.
Once the block's other layers are quantized, `Z` differs from the full-precision block's, and so does the
output. Refitting `W` and `b` to map the **quantized** block's `Z` onto the **full-precision** block output
`Y` corrects whatever part of the block's error is linear in `Z`. That includes the error of layers no norm
feeds.

### Derivation

With `Za = [Z, 1]` (tokens × (d+1)) and `Wa = [W, b]` (out × (d+1)), the ridge least-squares fit pulled
toward the current weights `Wa0` is

```
Wa = argmin || Za @ Wa.T − Y ||_F² + λ || Wa − Wa0 ||_F²
```

Setting the gradient to zero, `Wa @ (Za.T @ Za + λI) = Y.T @ Za + λ Wa0`, so

```
Wa = (Y.T @ Za + λ Wa0) @ (Za.T @ Za + λI)⁻¹,        λ = block_output_refit_ridge × mean(diag(Z.T @ Z))
```

(default ridge 0.01). Only the sums `Za.T @ Za` ((d+1)², float64) and `Y.T @ Za` (out × (d+1), float64)
are needed, accumulated token by token. The output error before and after the refit is computed from the
same sums plus `Σ Y²`:

```
|| Za @ Wa.T − Y ||² = Σ (Wa @ Za.T Za ⊙ Wa) − 2 Σ (Wa ⊙ Y.T Za) + Σ Y²
```

The log reports the after/before error ratio.

### Which layer is refit

`block_output_layer(block)`:

- **Conformer blocks** (Parakeet, Canary-Qwen encoder): the rotation already inserts an identity `Linear`
  after every block's output norm (`<block>.norm_out.1`, all blocks but the last). It carries that norm's
  folded centering, scale and shift, and is unquantized by default (`quantize_block_output_linear: False`).
  This linear is refit, so inference costs nothing extra. Parakeet refits 41 of its 42 blocks.
- **Transformer blocks** (Whisper encoder and decoder, Qwen LLM): these end in a residual sum, not a linear
  layer. With `block_output_refit_insert=true`, `insert_block_output_linear` gives each block an identity
  child `output_linear`, applied to the block's output by a forward hook. It is excluded from quantization
  (`should_quantize_module`), so it costs one fp16 `d × d` matmul per block at inference.
- **Quantized output layers are not refit:** the refit would overwrite their quantized weights.

Without `insert`, only Conformer blocks are refit. For Canary-Qwen this leaves the LLM uncorrected, which
at W2 stays collapsed (55.7 / 41.6 WER vs 5.17 / 5.83 with insert, 256 utterances).

### In the GPTQ loop

1. In the calibration pass that collects the Hessians, the full-precision block's output `Y` for each
   calibration sample is stored on the CPU. The block's input is the output of the already quantized
   previous blocks, as in sequential GPTQ.
2. GPTQ quantizes the block's layers (and norm tweaking runs, if enabled).
3. `refit_block_output` re-runs the quantized block on the same inputs. A forward hook on the output layer
   accumulates `(Z, Y)` into `OutputRefit`, which then solves for and writes `W, b`.
4. The refit block's outputs become the next block's inputs.

Because `Y` and `Z` come from the same block input, the refit corrects **this block's own** error. It does
not undo error inherited from earlier blocks.

### Save and load

The inserted `output_linear`s are part of the model's state. `save_quantized` records their block names and
widths, and `load_quantized` recreates them before loading the state dict.

### Limits

- **Weight error only**, for the same reason as norm tweaking.
- **Memory:** the targets hold one full-precision block output per calibration sample on the CPU while a
  block is processed, roughly `samples × tokens × d × 4` bytes in float32.
- **Transformer blocks cost inference time:** the inserted linears are dense, unquantized matmuls, one per
  block. Their latency has not been measured yet.

---

## 3. Results

WER % test-clean / test-other.

### 256 utterances

Same rotation, GPTQ on 256 calibration utterances, W2 = asymmetric g128 with a random Hadamard, W4A4 =
learned rotation with `attn_out`/`fc2` in groups of 128.

| Model | Setting | None | Norm tweak | Refit (+ insert) |
|---|---|---|---|---|
| Parakeet | W2 | 2.96 / 4.80 | **2.54** / 4.42 | 2.65 / 4.47 |
| Canary-Qwen | W2 | 52.6 / 36.0 | 5.55 / 7.74 | **5.17 / 5.83** |
| Whisper | W2 | 2.44 / 7.62 | 3.03 / 5.59 | 3.10 / **5.10** |
| Parakeet | W4A4 | 2.08 / 3.12 | 1.84 / 3.00 | 1.86 / **2.97** |
| Canary-Qwen | W4A4 | 1.71 / 2.72 | 1.56 / 2.65 | 1.56 / **2.62** |
| Whisper | W4A4 | 2.39 / 4.15 | **2.24** / 4.41 | 2.26 / 4.35 |

Mean output-error ratios logged in these runs:
- **Norm tweak:** 0.92–0.95 per norm at W2.
- **Refit, Conformer `norm_out.1`:** 0.63–0.70.
- **Refit, inserted transformer layers:** 0.48–0.61.

### Full LibriSpeech

GPTQ on 2048 calibration utterances, evolutionary Hadamard rotations.

| Model | Setting | Norm tweak | Refit (+ insert) | Full precision |
|---|---|---|---|---|
| Parakeet | W2 | 2.54 / 5.38 | **2.52 / 5.29** | 1.83 / 3.53 |
| Canary-Qwen | W2 | 4.29 / 7.49 | **3.34 / 5.85** | 1.57 / 3.06 |
| Whisper | W2 | 2.64 / 6.11 | **2.42 / 5.36** | 1.94 / 3.88 |
| Parakeet | W4A4 | **1.83** / 3.75 | 1.89 / 3.75 | 1.83 / 3.53 |
| Canary-Qwen | W4A4 | 1.69 / **3.22** | **1.64** / 3.33 | 1.57 / 3.06 |
| Whisper | W4A4 | 2.11 / 4.53 | **2.10 / 4.44** | 1.94 / 3.88 |

### Reading the results

- **W2: refit is better.** At W2 the error compounds through depth: Canary-Qwen goes from ~50% WER to ~5%
  with either correction. The refit's d×(d+1) parameters per block also absorb the error of layers no norm
  feeds.
- **W4A4: the two are within about 0.1 WER.** Norm tweaking needs no extra layer, so it is the default choice
  there. On Whisper, refit gives 2.10 / 4.44 against norm tweaking's 2.11 / 4.53, but it pays for an inserted
  unquantized linear per block.
- **Whisper W4A4 at 256 utterances:** neither correction beat no correction on test-other.

---

## 4. Usage

```bash
# Norm tweaking (any model)
python asrq/exp.py model=parakeet quantizer=gptq activation_bits=4 quantizer.norm_tweak=true \
    transform.path=outputs/rotation/parakeet_a4.pt

# Block output refitting; insert gives transformer blocks an output linear
python asrq/exp.py model=canary_qwen quantizer=gptq quantizer.bits=2 quantizer.symmetric=False \
    quantizer.group_size=128 activation_bits=16 quantizer.block_output_refit=true \
    quantizer.block_output_refit_insert=true transform.path=outputs/rotation/canary_qwen_w2.pt
```

| Setting (`quantizer.*`) | Default | Meaning |
|---|---|---|
| `norm_tweak` | `False` | enable norm tweaking |
| `norm_tweak_ridge` | `0.01` | pull of `s` toward 1, relative to mean diag(G) |
| `block_output_refit` | `False` | enable block output refitting |
| `block_output_refit_ridge` | `0.01` | pull toward the current weights, relative to mean diag(ZᵀZ) |
| `block_output_refit_insert` | `False` | give transformer blocks an unquantized identity `output_linear` to refit |

Both settings are part of the quantizer config, so they are part of the fingerprint of a saved quantized
model (`quantized_path`).

Tests: `tests/test_norm_tweak.py` checks the closed form against the least-squares solution, the ridge
behaviour, norm scaling and error reduction after GPTQ. `tests/test_output_refit.py` checks the refit's
statistics, the ridge, the reported error reduction, that a quantized output layer is skipped, and the
inserted layer.
