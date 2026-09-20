# Scale recovery & block output refitting

Two corrections that recover quantization error without gradient steps. Both run in the GPTQ block loop and use
only what the calibration pass already produces. They cover different block types and are normally both enabled:
scale recovery recovers transformer blocks (Whisper, the Qwen LLM), and the refit recovers Conformer blocks
(Parakeet, Canary-Qwen's encoder), which already end in a linear layer.

| | Scale recovery | Block output refitting |
|---|---|---|
| Code | `asrq/quantizers/scale_recovery.py` | `asrq/quantizers/output_refit.py` |
| Flag | `quantizer.scale_recovery=true` | `quantizer.block_output_refit=true` |
| When | inside GPTQ, as each column is rounded | after the block's GPTQ |
| Parameters per block | one scale vector (length d) per norm | a d×(d+1) weight and bias |
| Blocks | transformer (Whisper encoder and decoder, Qwen LLM) | Conformer (Parakeet, Canary-Qwen encoder) |
| What it corrects | the layers a norm feeds (q/k/v, fc1, gate/up) | the whole block's output, including layers no norm feeds (o_proj, fc2, pointwise_conv2) |
| Inference cost | none (the scale lives in the norm) | none (the layer already exists) |

---

## Convention

`nn.Linear` computes `Y = X @ W.T + b` on batched **row** vectors: `X: (tokens, in)`, `W: (out, in)`.
Every derivation below is in this form. `diag(s)` is the diagonal matrix of a vector `s`, `⊙` is the
elementwise product, and `H = X.T @ X` is the GPTQ Hessian (up to GPTQ's normalisation) of the layers
reading `X`.

---

## 1. Scale recovery inside GPTQ

### Idea

Multiplying a quantized layer's input by a per-channel scale `s` gives `X @ diag(s)`, which is the same as
multiplying column `j` of its quantized weight `Q` by `s_j`:

```
X @ diag(s) @ Q.T = X @ (Q @ diag(s)).T
```

So the grid gains one free degree of freedom per input channel while `Q` stays on the grid. Nothing is added to
the model: `diag(s)` is folded into whatever produces that input, channel by channel. The scale is fitted while
GPTQ quantizes the layers.

### The fit

GPTQ quantizes one column at a time. When it reaches column `j`, it has already pushed the rounding error of
the earlier columns into it, so the column it rounds is the **updated** column `w̃_j`: what GPTQ wants column `j`
to be at that point. After rounding it to `q_j`, the scale that best stretches `q_j` onto `w̃_j` is ordinary
least squares on one column:

```
s_j = argmin_s ||w̃_j − s·q_j||²  =  ⟨w̃_j, q_j⟩ / ⟨q_j, q_j⟩
```

GPTQ then compensates the error that is left, `w̃_j − s_j·q_j` instead of `w̃_j − q_j`, in the later columns, as
it always does. Nothing else in GPTQ changes: `q_j` is stored on the grid, and `s` goes to the norm.

The choice of `s_j` has to happen inside the loop. Fitting the same `s` after GPTQ (against the recorded `w̃`) is
inconsistent: the later columns have already compensated the unscaled error, so they then over-correct. At W2
this collapsed Whisper (34 / 70 WER) and cost Parakeet and Canary-Qwen several points.

### Ridge

A pull toward `s_j = 1` (no change) keeps poorly determined columns in place. With `c_j = 1 / U_jj²`, the weight
GPTQ's objective gives column `j` (`U` the upper Cholesky factor of the damped inverse Hessian):

```
s_j = (c_j ⟨w̃_j, q_j⟩ + λ) / (c_j ⟨q_j, q_j⟩ + λ),        λ = scale_recovery_ridge × mean_j(c_j ||w_j||²)
```

(default ridge 0.01). The result is a weighted average of the least-squares fit and 1.

### Where the scale goes

The scale is folded into the norm feeding the layer: its affine weight and bias are multiplied by `s`, or, when a
rotation has left it scale-free, `s` is registered as a new `weight` buffer and acts in the rotated basis. Only
layers a norm feeds are scaled.

### Layers sharing a norm

A norm has one scale per channel, so the layers it feeds (q/k/v, or gate/up) must share `s`. Two options,
`quantizer.scale_recovery_method`:

- **`lockstep`** (default): the layers read the same input, so they have the same Hessian, and GPTQ treats each
  row of a weight independently. Their weights are stacked row-wise and quantized as one matrix, which gives
  each layer exactly its own GPTQ result, and `s_j` is fitted over all of their rows at once:
  `s_j = Σ_l ⟨w̃_lj, q_lj⟩ / Σ_l ⟨q_lj, q_lj⟩`.
- **`average`**: each layer is quantized on its own with its own scales `s_lj`, and the norm gets
  `Σ_l ⟨q_lj,q_lj⟩ s_lj / Σ_l ⟨q_lj,q_lj⟩`. For one column this is the lockstep formula, but each layer
  compensated the error of its own scale, not the shared one, so from the second column on the updated weights,
  and the scales, drift apart (on random weights, 63 of 64 columns differ, by up to 0.09).

At W2 on 256 utterances they are within noise of each other (test-other: Parakeet 5.20 vs 5.32, Canary-Qwen 6.73
vs 6.34, Whisper 5.74 vs 5.93, lockstep vs average).

### Where it is applied

Only norms whose layers are **all** quantized in the current block (`scale_recovery_targets()` per model):

| Model | Norm → layers |
|---|---|
| Whisper (encoder and decoder) | `self_attn_layer_norm` → q/k/v; `final_layer_norm` → fc1; decoder `encoder_attn_layer_norm` → cross-attention q only (k/v read the encoder output) |
| Canary-Qwen LLM | `input_layernorm` → q/k/v_proj; `post_attention_layernorm` → gate/up_proj |

In the block loop, `capture_scale_recovery` lists these targets, `ModelQ.quantize_layers` quantizes each
norm's layers together (and every other layer on its own), and `apply_scale_recovery` folds the scales in. The log reports,
per norm, `L(s) / L(1)`: the layers' output error with the scale relative to the same grid weights without it.

### Limits

- **Weight error only.** Activations are not quantized during GPTQ, so the fit does not see activation
  error, and with quantized activations `s` also changes what the activation quantizer rounds.
- **d degrees of freedom per norm.** Layers no norm feeds (attention output, fc2, pointwise_conv2, down_proj) are not corrected.
- **Lower layer error is not always lower WER.** On Parakeet at W2, the in-loop fit gives every layer a lower
  held-out output error than a closed-form fit after GPTQ (0.84 vs 0.93 of plain GPTQ), yet a higher WER.

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
- **Quantizing those inserted layers** (`model.quantize_block_output_linear=true`) refits them first and
  quantizes them afterwards, so they keep the correction and still end up on the grid; their Hessian is
  collected in the refit pass, from the quantized block's inputs. Their error lands straight in the residual
  stream and nothing downstream corrects it, so `quantizer.block_output_linear_bits` can keep them wider than
  the rest. On Parakeet at W2 (256 utterances), where the 41 layers weigh 86 MB in fp16 against about 275 MB
  for the encoder's W2 weights:

  | Their bits | Size | Refit, then quantized | Quantized, no refit |
  |---|---|---|---|
  | fp16 (default) | 86 MB | 2.65 / **4.47** | 2.96 / 4.80 |
  | W8 | 43 MB | **2.54** / 4.68 | 2.87 / 4.65 |
  | W4 | 21.5 MB | 2.91 / 4.96 | 3.43 / 5.20 |
  | W2 | 11 MB | 4.08 / 6.40 | 3.49 / 5.51 |

  W8 halves their size for about 0.2 WER on test-other. Below that the layers' own error dominates, and at W2 the
  refit becomes counterproductive: it moves the weights away from the near-identity map the rotation put there,
  which a 2-bit grid represents badly, and that costs more than the correction gains. The crossover sits between
  W4 and W2.
- **Transformer blocks** end in a residual sum, not a linear layer, and are left to scale recovery. Giving them an
  inserted layer to refit was tried and removed; the results are below.
- **Quantized output layers are not refit:** the refit would overwrite their quantized weights.

For Canary-Qwen only the encoder is refit; its LLM needs scale recovery, without which W2 stays collapsed
(55.7 / 41.6 WER, 256 utterances).

### In the GPTQ loop

1. In the calibration pass that collects the Hessians, the full-precision block's output `Y` for each
   calibration sample is stored on the CPU. The block's input is the output of the already quantized
   previous blocks, as in sequential GPTQ.
2. GPTQ quantizes the block's layers (and scale recovery runs, if enabled).
3. `refit_block_output` re-runs the quantized block on the same inputs. A forward hook on the output layer
   accumulates `(Z, Y)` into `OutputRefit`, which then solves for and writes `W, b`.
4. The refit block's outputs become the next block's inputs.

Because `Y` and `Z` come from the same block input, the refit corrects **this block's own** error. It does
not undo error inherited from earlier blocks.

### Limits

- **Weight error only**, for the same reason as scale recovery.
- **Memory:** the targets hold one full-precision block output per calibration sample on the CPU while a
  block is processed, roughly `samples × tokens × d × 4` bytes in float32.
- **Conformer blocks only:** a block with no output layer cannot be refit.

---

## 3. Results

WER % test-clean / test-other.

### 256 utterances

GPTQ on 256 calibration utterances; W2 = asymmetric g128 with a random Hadamard, W4A4 = learned rotation with
`attn_out`/`fc2` in groups of 128.

W2, both corrections enabled, so each block type gets the one that applies to it:

| Model | Recovered blocks | None | Recipe |
|---|---|---|---|
| Parakeet | refit, 41 Conformer layers | 2.96 / 4.80 | 2.65 / 4.47 |
| Whisper | scale recovery, 160 transformer norms | 2.44 / 7.62 | 3.14 / 5.74 |
| Canary-Qwen | refit 31 encoder layers + scale recovery on 56 LLM norms | 52.6 / 36.0 | **3.21 / 5.20** |

Canary-Qwen is the only model both corrections apply to, and the split beats either alone by a wide margin: its
best single-correction W2 results were 4.08 / 6.73 (scale recovery everywhere) and 5.17 / 5.83 (refit everywhere,
with a layer inserted into the LLM blocks).

The W4A4 recipe has not been run. Earlier W4A4 runs, with refitting in every block type: Parakeet 1.86 / 2.97,
Canary-Qwen 1.56 / 2.62, Whisper 2.26 / 4.35, against 2.08 / 3.12, 1.71 / 2.72 and 2.39 / 4.15 uncorrected.

Mean output-error ratios logged in these runs:
- **Scale recovery:** 0.16–0.37 of the same grid weights without the scale (not comparable with the refit's ratio,
  which is relative to the unrefit block).
- **Refit, Conformer `norm_out.1`:** 0.63–0.70.

### Full LibriSpeech

GPTQ on 2048 calibration utterances, evolutionary Hadamard rotations. Scale recovery has not been run at full
scale.

| Model | Setting | Refit (+ insert) | Full precision |
|---|---|---|---|
| Parakeet | W2 | 2.52 / 5.29 | 1.83 / 3.53 |
| Canary-Qwen | W2 | 3.34 / 5.85 | 1.57 / 3.06 |
| Whisper | W2 | 2.42 / 5.36 | 1.94 / 3.88 |
| Parakeet | W4A4 | 1.89 / 3.75 | 1.83 / 3.53 |
| Canary-Qwen | W4A4 | 1.64 / 3.33 | 1.57 / 3.06 |
| Whisper | W4A4 | 2.10 / 4.44 | 1.94 / 3.88 |

### Folding into other layers (removed)

A scale can also be folded into a preceding layer's **rows**: an attention output projection's into `v_proj`
(its input is a weighted average over tokens of `v_proj`'s rows), and a SwiGLU down projection's into `up_proj`
(the product is elementwise). Both folds are exact, verified to 3e-16 on a real attention module, and free, and
under Qwen's grouped-query attention the `o_proj` channels reading one value channel are tied to a single scale.

They were implemented and removed: W2 on 256 utterances measured flat to worse, although the local output error
dropped sharply (a cross-attention `v_proj` group logged a ratio of 0.001).

| Model | norms only | with row folds |
|---|---|---|
| Whisper | 3.14 / **5.74** | **3.10** / 6.31 |
| Canary-Qwen (with refit) | **3.21 / 5.20** | 3.32 / 5.40 |

Same pattern as the in-loop fit on Parakeet: lower per-layer output error does not imply lower WER.

### Inserted layers for transformer blocks (removed)

Before scale recovery moved inside GPTQ, transformer blocks were given an unquantized identity `output_linear` to
refit, optionally as a low-rank correction `x + (x @ A) @ B + b` fitted by reduced-rank regression. Both are
removed: the layer cost one d×d matmul per block at inference (about +65% of a W2 Qwen block's weight bytes), and
low rank did not save that cost safely. W2, 256 utterances, dense insert against rank d/4, d/8, d/16:

| Rank | Whisper (d 1280) | Canary-Qwen (LLM d 2048) |
|---|---|---|
| dense | 3.10 / 5.10 | 5.17 / 5.83 |
| d/4 | 2.33 / 5.49 | 5.34 / 8.08 |
| d/8 | 2.85 / 5.76 | 13.02 / 13.19 |
| d/16 | 2.72 / 6.00 | 15.24 / 14.27 |

Canary-Qwen's LLM error is not concentrated in a few directions, so truncating loses most of the correction and
it compounds over 28 blocks. Whisper tolerates low rank, and is better on test-clean at every rank, but its
test-other degrades as the rank falls.

### Reading the results

- **Both corrections rescue W2 where error compounds through depth:** Canary-Qwen goes from ~50% WER to 4–5%.
- **Refit is the stronger W2 correction on test-other**, on all three models. Its d×(d+1) parameters per
  block also absorb the error of layers no norm feeds.
- **Scale recovery costs nothing at inference.** On Canary-Qwen it gives the best W2 test-clean (4.08).

---

## 4. Usage

```bash
# Scale recovery (any model)
python asrq/exp.py model=canary_qwen quantizer=gptq quantizer.bits=2 quantizer.symmetric=False \
    quantizer.group_size=128 activation_bits=16 quantizer.scale_recovery=true \
    transform.path=outputs/rotation/canary_qwen_w2.pt

# Both: scale recovery for the LLM, block output refitting for the Conformer encoder
python asrq/exp.py model=canary_qwen quantizer=gptq quantizer.bits=2 quantizer.symmetric=False \
    quantizer.group_size=128 activation_bits=16 quantizer.scale_recovery=true quantizer.block_output_refit=true \
    transform.path=outputs/rotation/canary_qwen_w2.pt
```

| Setting (`quantizer.*`) | Default | Meaning |
|---|---|---|
| `scale_recovery` | `False` | enable scale recovery |
| `scale_recovery_ridge` | `0.01` | pull of `s` toward 1, relative to the mean weighted column energy |
| `scale_recovery_method` | `lockstep` | how a norm's layers share `s`: `lockstep` or `average` |
| `block_output_refit` | `False` | enable block output refitting |
| `block_output_refit_ridge` | `0.01` | pull toward the current weights, relative to mean diag(ZᵀZ) |

Both settings are part of the quantizer config, so they are part of the fingerprint of a saved quantized
model (`quantized_path`).

Tests: `tests/test_scale_recovery.py` checks that GPTQ without the scale is unchanged and that stacked rows give
each layer its own GPTQ result, that the scale lowers the output error and the ridge holds it at 1, norm
scaling, and the grouped quantization of a norm's layers. `tests/test_output_refit.py` checks the refit's
statistics, the ridge, the reported error reduction, and that a quantized output layer is skipped.
