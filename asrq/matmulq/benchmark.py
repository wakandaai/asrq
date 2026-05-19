import torch
from asrq.matmulq.base import mmq


def bench_cuda(m, n, k, reps=200, warmup=50):
    """Benchmark custom CUDA matmul vs PyTorch on GPU."""
    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(n, k, dtype=torch.float16, device="cuda")
    B_t = B.t().contiguous()

    # Warm up
    for _ in range(warmup):
        mmq.matmul_16x16(A, B) # 16x16 
        torch.mm(A, B_t)
    torch.cuda.synchronize()

    # 16x16 kernel
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        mmq.matmul_16x16(A, B)
    e.record()
    torch.cuda.synchronize()
    custom_ms = s.elapsed_time(e) / reps

    # 4x16 kernel

    # 2x16 kernel

    # 4x4 kernel

    # 4x8 kernel

    # pytorch 16x16 kernel
    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(reps):
        torch.mm(A, B_t)
    e2.record()
    torch.cuda.synchronize()
    pytorch_ms = s2.elapsed_time(e2) / reps



    return custom_ms, pytorch_ms


def bench_cpu(m, n, k, reps=50, warmup=10):
    """Benchmark custom CPU matmul vs PyTorch on CPU."""
    A = torch.randn(m, k, dtype=torch.float32)
    B = torch.randn(n, k, dtype=torch.float32)
    B_t = B.t().contiguous()

    import time

    for _ in range(warmup):
        mmq.matmul_cpu(A, B_t)
        torch.mm(A, B_t)

    t0 = time.perf_counter()
    for _ in range(reps):
        mmq.matmul_cpu(A, B_t)
    custom_ms = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        torch.mm(A, B_t)
    pytorch_ms = (time.perf_counter() - t0) / reps * 1000

    return custom_ms, pytorch_ms


def tflops(m, n, k, ms):
    return 2 * m * n * k / (ms * 1e-3) / 1e12


def run_cuda_benchmark():
    print("\n=== CUDA Benchmark (custom kernel vs torch.mm) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6}  {'16x16':>12} {'16x16 TF/s':>12} {'Speedup':>9} {'PyTorch':>12} {'PyTorch TF/s':>13}  {'Regime'}")
    print("-" * 105)
    shapes = [
        (1,    4096, 4096, "memory-bound"),
        (4,    4096, 4096, "memory-bound"),
        (16,   4096, 4096, "memory-bound"),
        (64,   4096, 4096, "transitional"),
        (512,  4096, 4096, "compute-bound"),
        (2048, 4096, 4096, "compute-bound"),
        (4096, 4096, 4096, "compute-bound"),
        (8192, 8192, 8192, "compute-bound"),
    ]
    for m, n, k, regime in shapes:
        custom_ms, pytorch_ms = bench_cuda(m, n, k)
        custom_tf = tflops(m, n, k, custom_ms)
        pytorch_tf = tflops(m, n, k, pytorch_ms)
        print(f"{m:>6} {n:>6} {k:>6}  {custom_ms:>11.4f}ms {custom_tf:>11.3f} {pytorch_ms/custom_ms:>8.2f}x {pytorch_ms:>11.4f}ms {pytorch_tf:>12.3f}  {regime}")


def run_cpu_benchmark():
    print("\n=== CPU Benchmark (custom kernel vs torch.mm fp32) ===")
    print(f"{'M':>6} {'N':>6} {'K':>6}  {'16x16':>12} {'16x16 GF/s':>12} {'PyTorch':>12} {'PyTorch GF/s':>13} {'Speedup':>9}")
    print("-" * 95)
    shapes = [
        (16,  512,  512),
        (64,  1024, 1024),
        (128, 2048, 2048),
        (512, 4096, 4096),
    ]
    for m, n, k in shapes:
        custom_ms, pytorch_ms = bench_cpu(m, n, k)
        custom_gf = tflops(m, n, k, custom_ms) * 1000  # GFLOP/s
        pytorch_gf = tflops(m, n, k, pytorch_ms) * 1000
        print(f"{m:>6} {n:>6} {k:>6}  {custom_ms:>11.4f}ms {custom_gf:>11.3f}  {pytorch_ms:>11.4f}ms {pytorch_gf:>12.3f}  {pytorch_ms/custom_ms:>8.2f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run GPU benchmark.")
    else:
        # Correctness check before benchmarking
        print("Running correctness check...")
        M, N, K = 512, 512, 512
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
        B_t = B.t().contiguous()
        ref = torch.mm(A, B_t)
        out = mmq.matmul_16x16(A, B)
        max_err = (ref - out).abs().max().item()
        print(f"Max absolute error vs torch.mm: {max_err:.6f}")
        assert max_err < 0.1, f"Correctness check failed! max_err={max_err}"
        print("Correctness check passed.\n")

        run_cuda_benchmark()
