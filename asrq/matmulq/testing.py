import torch
from asrq.matmulq.base import mmq

M = 1
N = 1024
K = 1024

# workspace
workspace = torch.zeros((M, N), device="cuda").int()

A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(N, K, dtype=torch.float16, device="cuda")
ref = torch.matmul(A, B.T)
out = mmq.mymatmul(A, B, workspace)

diff = (ref - out).abs()
max_diff = diff.max().item()
print(f"Max difference between reference and mymatmul: {max_diff}")

# GPU Metrics
gpu_metrics = mmq.get_gpu_metrics()
for k, v in gpu_metrics.items():
    print(f"{k:<30}: {v}")