import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small-M configs (decode / GEMV regime)
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        # Large-M configs (prefill / compute-bound)
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def w4a4_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables dictate how much memory to step to get to the next row/col
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters (provided by @triton.autotune)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 1. L2 Cache Optimization (Standard Triton Grouping)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 2. Setup Pointers — use tl.multiple_of to signal BLOCK-aligned offsets to the compiler
    offs_am = tl.multiple_of((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, BLOCK_SIZE_M)
    offs_bn = tl.multiple_of((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 3. FP32 accumulator for FP8 wgmma (H100 FP8 = 2x INT8 throughput on sm_90a)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 4. Main K-loop over the packed (K/2) dimension.
    K_packed = K // 2
    for k in range(0, tl.cdiv(K_packed, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Unpack INT4 -> INT8 via sign-extending bit shifts, then cast to FP8 (e4m3).
        # INT4 values are in [-8, 7]; FP8 e4m3 exactly represents integers in [-8, 8].
        # H100 FP8 wgmma (wgmma.mma_async...e4m3) runs at 3958 TOPS vs 1979 TOPS for INT8.
        a0 = ((a << 4).to(tl.int8) >> 4).to(tl.float16).to(tl.float8e4nv)   # lower nibble
        a1 = (a >> 4).to(tl.int8).to(tl.float16).to(tl.float8e4nv)           # upper nibble
        b0 = ((b << 4).to(tl.int8) >> 4).to(tl.float16).to(tl.float8e4nv)
        b1 = (b >> 4).to(tl.int8).to(tl.float16).to(tl.float8e4nv)

        # Two fused FP8 dots with FP32 accumulation
        accumulator = tl.dot(a0, b0, accumulator, out_dtype=tl.float32)
        accumulator = tl.dot(a1, b1, accumulator, out_dtype=tl.float32)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 5. Store Output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    # Cast FP32 accumulator back to int32 for output (FP32 exactly represents ints up to 2^24;
    # max partial sum = 8*8*4096 = 262144 which is well within this range)
    tl.store(c_ptrs, accumulator.to(tl.int32), mask=c_mask)


# ==========================================================
# Python Wrapper Function
# ==========================================================
def matmul_w4a4(a, b):
    # Check shapes and types
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == torch.int8 and b.dtype == torch.int8, "Inputs must be packed int8"
    
    M, K_packed = a.shape
    K_packed_b, N = b.shape
    
    # Real K is double the packed K
    K = K_packed * 2 
    
    # Allocate output tensor (int32 accumulation)
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    
    # Grid launch function
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    w4a4_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        # Block sizes are selected automatically by @triton.autotune
    )
    return c


# ==========================================================
# Test and Verification
# ==========================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    
    M, N, K = 4096, 4096, 4096

    # For the test we multiply two random matrices in both pytorch and custom kernel
    A = torch.randint(-7, 7, (M, K), dtype=torch.int8, device='cuda')
    B = torch.randint(-7, 7, (N, K), dtype=torch.int8, device='cuda')

    A_packed = (A[:, ::2] & 0x0F) | ((A[:, 1::2] & 0x0F) << 4)
    B_packed = (B[:, ::2] & 0x0F) | ((B[:, 1::2] & 0x0F) << 4)
    triton_output = matmul_w4a4(A_packed, B_packed.t())

    print(f"Triton Output Shape: {triton_output.shape}")
    print(f"Triton Output Dtype: {triton_output.dtype}")
    print("Success! The kernel executed.")

    def bench(m, n, k, reps=200, warmup=50):
        a_full  = torch.randint(-7, 7, (m, k), dtype=torch.int8, device='cuda')
        b_full  = torch.randint(-7, 7, (n, k), dtype=torch.int8, device='cuda')
        a_pk = (a_full[:, ::2] & 0x0F) | ((a_full[:, 1::2] & 0x0F) << 4)
        b_pk = (b_full[:, ::2] & 0x0F) | ((b_full[:, 1::2] & 0x0F) << 4)
        b_pk_t = b_pk.t().contiguous()
        b_full_t = b_full.t().contiguous()

        # _int_mm requires M > 16; fall back to fp16 mm for small batch
        if m > 16:
            pt_fn = lambda: torch._int_mm(a_full, b_full_t)
            pt_label = "int_mm"
        else:
            a_fp = a_full.to(torch.float16)
            b_fp = b_full_t.to(torch.float16)
            pt_fn = lambda: torch.mm(a_fp, b_fp)
            pt_label = "fp16mm"

        for _ in range(warmup):
            matmul_w4a4(a_pk, b_pk_t)
            pt_fn()
        torch.cuda.synchronize()

        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps): matmul_w4a4(a_pk, b_pk_t)
        e.record(); torch.cuda.synchronize()
        t_ms = s.elapsed_time(e) / reps

        s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s2.record()
        for _ in range(reps): pt_fn()
        e2.record(); torch.cuda.synchronize()
        pt_ms = s2.elapsed_time(e2) / reps

        return t_ms, pt_ms, pt_label

    print(f"\n{'M':>6} {'N':>6} {'K':>6}  {'Triton':>10} {'PyTorch':>10} {'Speedup':>9}  {'Regime'}")
    print("-" * 72)
    shapes = [
        # Memory-bandwidth-bound (decode inference)
        (1,    4096, 4096, "memory-bound"),
        (4,    4096, 4096, "memory-bound"),
        (16,   4096, 4096, "memory-bound"),
        (64,   4096, 4096, "transitional"),
        # Compute-bound (prefill / training)
        (512,  4096, 4096, "compute-bound"),
        (2048, 4096, 4096, "compute-bound"),
        (4096, 4096, 4096, "compute-bound"),
    ]
    for m, n, k, regime in shapes:
        t_ms, pt_ms, pt_label = bench(m, n, k)
        print(f"{m:>6} {n:>6} {k:>6}  {t_ms:>9.4f}ms {pt_ms:>9.4f}ms {pt_ms/t_ms:>8.2f}x  {regime} (vs {pt_label})")