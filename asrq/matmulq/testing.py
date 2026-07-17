import torch
from asrq.matmulq.base import mmq

M = 64
N = 16000
K = 32000

# mymatmul_launcher picks its grid size (and hence how much workspace/locks
# it needs) from a runtime occupancy query, not just M/N/K -- so these sizes
# come from the C++ side rather than being recomputed here. An undersized
# buffer causes an out-of-bounds workspace write that can silently corrupt
# the locks buffer and hang the kernel.
workspace_size, locks_size = mmq.mymatmul_workspace_sizes(M, N, K)
workspace = torch.zeros(workspace_size, device="cuda").float()
locks = torch.zeros(locks_size, device="cuda").int()

A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(N, K, dtype=torch.float16, device="cuda")*10

# cuBLAS's fp16 matmul can itself use reduced-precision (not full fp32)
# accumulation internally for speed when this flag is left at its default of
# True -- at K=32000 that gives `ref` its own non-trivial accumulation noise,
# which isn't a fair "ground truth" to diff a full-fp32-accumulate kernel
# against. Force full precision for the reference only.
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
ref = torch.matmul(A, B.T)
out = mmq.mymatmul(A, B, workspace, locks)

# Combined absolute+relative tolerance (torch.allclose's atol + rtol*|ref|),
# not pure relative error. Pure relative error breaks down whenever the true
# sum is small due to catastrophic cancellation among K terms (e.g. K=32000
# terms up to magnitude ~30 each summing to a "true" value near zero): cuBLAS
# and this kernel add those terms in different orders, so fp32 rounding
# differs slightly along the way -- the *absolute* error stays bounded by a
# roughly-fixed noise floor for a given K, but dividing that same small
# absolute noise by a near-zero reference blows up the relative error even
# though nothing is actually wrong. atol below is sized for K in the tens of
# thousands; shrink it for much smaller K if you want a tighter check.
diff = (ref - out).abs()
atol, rtol = 1.0, 1e-2
tol = atol + rtol * ref.abs()
bad = diff > tol
max_diff = diff.max().item()
print(f"Max absolute difference between reference and mymatmul: {max_diff}")
print(f"Elements outside atol={atol} + rtol={rtol}*|ref|: {bad.sum().item()} / {ref.numel()}")
if bad.any():
    print("Error: mymatmul output differs from reference by more than the allowed tolerance")

# # GPU Metrics
# gpu_metrics = mmq.get_gpu_metrics()
# for k, v in gpu_metrics.items():
#     print(f"{k:<30}: {v}")