# LayerNorm folding & pointwise-conv performance

Reference notes for the `asrq/transforms/rotation/` refactor.

Measurements: RTX A6000 (sm_86), torch 2.14.0+cu126, driver 570.195.03, `torch.utils.benchmark`
median of `blocked_autorange`. All equivalences verified numerically against the original module.

---

## 1. LayerNorm -> RMSNorm folding

### The identity

For a row vector `x` of width `n`, let `M = I - (1/n) * 11^T` be the centering matrix.
`M` is symmetric and idempotent (`M^2 = M`, `M^T = M`), and `x @ M == x - mean(x)`.

Because `mean((x - mean(x))^2) == mean((x @ M)^2)`, LayerNorm decomposes exactly as:

```
LayerNorm(x) = RMSNorm(x @ M) * gamma + beta
```

So a LayerNorm becomes an RMSNorm if the centering `M` and the shift `beta` are pushed into
the surrounding layers, leaving only `gamma` in the norm.

### Convention

`nn.Linear` computes `Y = X @ W.T + b` on batched **row** vectors, `X: (..., in)`, `W: (out, in)`.
State every derivation in this form. The textbook column-vector form (`y = W @ x`) is the transpose
dual and reads as if `M` were applied to the batched activation tensor, which is dimensionally
impossible (`M @ X` with `X: (batch, dim)`).

### Fold 1 — centering into the *previous layer*

```
(X @ W1.T + b1) @ M = X @ (W1.T @ M) + b1 @ M
                    = X @ (M @ W1).T + M @ b1        (M symmetric)
```

So `W1 <- M @ W1` and `b1 <- M @ b1`.

Note `W @ v == v @ W.T` for 1-D `v`, so the bias form is the same either way — but say so
explicitly rather than leaving it ambiguous.

### Fold 2 — shift into the *next layer*

```
(RMSNorm(Xc) * gamma + beta) @ W2.T + b2
  = (RMSNorm(Xc) * gamma) @ W2.T + (beta @ W2.T + b2)
```

So `b2 <- b2 + beta @ W2.T` (create the bias if the layer has none).

### Fold 3 — scale into the next layer (RMSNorm -> pure normalization)

```
(normed * gamma) @ W.T + b = normed @ (W @ diag(gamma)).T + b
```

So `W <- W * gamma`, broadcast over the **in_features** axis. Bias unchanged.

### Ordering constraint

Fold 2 must run **before** fold 3 on the same next layer. Fold 2 writes the shift into the
next layer's *bias* using its original weight; fold 3 then scales the *weight*, which does not
disturb it. Reversed, fold 2 would use the already-scaled weight and be wrong.

### Pointwise Conv1d next layers

A pointwise `Conv1d` stores its weight as `(out, in, 1)` and applies it along the **channel**
axis, so `weight[:, :, 0]` is the equivalent matrix and all three folds apply unchanged.

Two traps:

- `weight * gamma` on a 3-D conv weight broadcasts `gamma: (in,)` against the trailing **kernel**
  axis, silently producing `(out, in, in)` instead of scaling channels. Take the 2-D view first.
- `squeeze(-1)` returns a **view** sharing storage. Write through it **in place** (`copy_`, `mul_`).
  Rebinding (`layer.weight.data = <2-D tensor>`) would reshape a conv weight to 2-D. The `Linear`
  path survives both mistakes, so this only breaks when a conv hits it.

Depthwise convs (`kernel_size > 1`, or `groups > 1`) cannot be folded this way — the helper
raises rather than silently mis-folding, which matters because a Conformer's depthwise conv
(`kernel_size=31`, `groups=channels`) sits directly between the two pointwise ones.

### Open issue: the previous layer in a residual network

In NeMo's `ConformerLayer`, `norm_conv`'s next layer is `pointwise_conv1` (fine), but its
**previous layer is the residual stream**, not a single layer. There is no single `linear1` to absorb
`M` into. Every layer writing into that residual would have to be centered together, which is why
this transform is usually applied globally across the block rather than per-norm.

---

## 2. Pointwise Conv1d vs Linear — performance

### Isolated op (fp16)

| shape | conv1d `(N,C,L)` | linear `(N,L,C)` | transpose+linear+transpose |
|---|---|---|---|
| N1 L500 512->1024 | 45.5 us | **18.0 us (2.53x)** | 36.6 us (1.24x) |
| N8 L500 512->1024 | 98.4 us | **53.1 us (1.85x)** | 107.6 us (0.91x) |
| N8 L500 1024->1024 | 147.9 us | **93.7 us (1.58x)** | 185.6 us (0.80x) |
| N32 L1000 1024->2048 | 2092.8 us | **1237.4 us (1.69x)** | 2073.0 us (1.01x) |

cuDNN's 1x1 path peaks near 64 TFLOP/s; the GEMM reaches 108. On **CPU** the two are within ~6% —
the gap is GPU-only.

**Do not cite these as end-to-end numbers.** See below.

### Real `ConformerConvolution` module (fp16)

| config | d512 B8 T500 | d1024 B8 T500 | d1024 B32 T1000 |
|---|---|---|---|
| NeMo original (conv1d) | 1.00x | 1.00x | 1.00x |
| linear | 1.17x | 1.08x | 1.15x |
| **linear + `.contiguous()` before pw2** | **1.24x** | **1.14x** | **1.21x** |
| linear + `.contiguous()` before depthwise | 1.12x | 1.03x | 1.12x |
| both | 1.18x | 1.09x | 1.16x |

Only **1.1-1.25x**, not the isolated 1.5-2.5x. Component breakdown (fp16, d512 B8 T500,
module = 325.5 us):

```
pointwise_conv1 (conv)   127.9 us   39.3%    -> as Linear: 53.6 us  16.5%
pointwise_conv2 (conv)    55.3 us   17.0%
depthwise_conv (k=31)    123.8 us   38.0%    <- irreducible, never a matmul
batch_norm                31.9 us    9.8%    <- memory-bound
glu                       20.9 us    6.4%
activation (Swish)        11.7 us    3.6%
transpose+contiguous      26.9 us    8.3%
```

The depthwise conv is 27-38% and the elementwise ops another ~20%, so Amdahl caps the win near
1.2x. Across a full Conformer layer (the conv module is one of four sub-blocks) the model-level
effect is smaller still.

**Convert for the `ASRQLinear` quantization path. The ~1.2x is a side benefit, not a justification.**

---

## 3. Contiguity, not transposes

`transpose` emits **no kernel** — it is pure stride metadata, and `x.transpose(1,2).transpose(1,2)`
returns metadata identical to `x`. What costs is handing a kernel non-contiguous memory.

GEMM on a transposed view (fp16):

| pw2 shape | contiguous | transposed view | `.contiguous()` first |
|---|---|---|---|
| B8 T500 D512 | 48.1 us | 65.0 us (1.35x) | 50.9 us (1.06x) |
| B32 T1000 D1024 | 621.9 us | 1313.3 us (2.11x) | 1083.0 us (1.74x) |

Rules that follow:

- **cuBLAS GEMM** wants contiguous. An explicit `.contiguous()` is *cheaper* than letting cuBLAS
  handle strided input.
- **cuDNN conv** is the opposite — it reads strided input more cheaply than a copy costs. Forcing
  contiguity before the depthwise conv makes things *worse*.
- `Conv1d` allocates a fresh contiguous output, absorbing its input's non-contiguity.
  `Linear` + `transpose` does not, and `glu` **propagates its input's stride permutation**.

That last point is why swapping the pointwise convs silently hands the depthwise conv a strided
input the original never had:

```
DROP-IN:   after transpose back   (8,1024,500)  contig=False  strides=(512000, 1, 1024)
           after glu(dim=1)       (8, 512,500)  contig=False  strides=(256000, 1,  512)
ORIGINAL:  after pointwise_conv1  (8,1024,500)  contig=True   strides=(512000, 500, 1)
           after glu(dim=1)       (8, 512,500)  contig=True   strides=(256000, 500, 1)
```

Real, but benign — see the table above. The depthwise conv's *output* is contiguous either way.

### Recommended rewrite

NeMo's transposes sit at the outer edges of `ConformerConvolution.forward`, so `pointwise_conv2`
receives `(B,D,T)`. Both pointwise convs can become `Linear` with the transposes **relocated, not
added**:

```python
def forward(self, x):               # (B, T, D)
    x = self.pw1(x)                 # (B, T, 2D)   x arrives contiguous
    x = F.glu(x, dim=-1)            # (B, T, D)    dim=-1, not dim=1
    x = x.transpose(1, 2)           # (B, D, T)    leave non-contiguous
    x = self.depthwise_conv(x)
    x = self.batch_norm(x)
    x = self.activation(x)
    x = x.transpose(1, 2)           # (B, T, D)
    x = x.contiguous()              # materialize for the GEMM
    return self.pw2(x)              # (B, T, D)    already the return layout
```

Two transposes, same as the original. No trailing transpose after `pw2` — it is consumed by moving
the Linear past it. `pad_mask` needs `unsqueeze(-1)` instead of `unsqueeze(1)` on the `(B,T,C)` side.

A plain drop-in swap (leaving NeMo's transposes in place) measures the same everywhere except
batch-1 fp32 (1.15x vs 1.33x), because `pw1`'s inner transpose cancels the outer one exactly.

---

## 4. Benchmarking gotcha

`torch.backends.cudnn.allow_tf32` defaults to **True** while `torch.backends.cuda.matmul.allow_tf32`
defaults to **False**. Any fp32 conv-vs-linear comparison must set both explicitly, or conv silently
gets tensor cores the matmul is denied and wins by ~1.33x — inverting the result.

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

With TF32 matched, the fp32 module numbers are 1.11-1.33x, consistent with fp16.

Also note torch wheels bundle their own CUDA runtime, so the system toolkit at `/usr/local/cuda`
is irrelevant — only the **driver** version gates which build will initialize. A cu126/cu128 build
runs on driver 570 (CUDA 12.8); a cu130 build needs 580+.
