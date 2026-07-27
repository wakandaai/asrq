#include "common.h"
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>

// W4A16: fp16 activations x per-channel-quantized packed-int4 weights, both
// symmetric (matmul_kernel_w4a16) and asymmetric/zero-point
// (matmul_kernel_w4a16_asym) variants. Both are register-resident designs:
// there is no dequantized-B tile in shared memory at all -- packed bytes are
// cp.async'd into a small staging buffer and each thread dequantizes
// straight into the registers fed to mma_m16n8k16, using the LOP3 tricks
// below and the mma.sync B-fragment tau/g mapping documented in each
// kernel's compute().

// Fast int4->fp16 dequant, the same bit trick used by Marlin/AWQ-style W4A16
// kernels: nibbles are packed using an *excess-8* (offset-binary) encoding --
// raw_nibble = signed_value + 8, so raw_nibble is 0..15 with no sign
// extension needed -- rather than two's complement, specifically so this
// trick applies directly. lop3.b32 computes (a & b) | c in a single
// instruction (immediate 0xea encodes that 3-input truth table); with
// b = 0x000f000f (mask out everything except two nibbles, one per 16-bit
// half) and c = 0x64006400 (the fp16 bit pattern for 1024.0, repeated in
// both halves), the result's two halves are the fp16 values 1024+nibble --
// exact, since mantissa bits 0..3 at fp16's exponent-25 (2^10) scale
// contribute exactly 2^0..2^3 to the value, matching the nibble's own bit
// weights. A single packed __hsub2 against (1032, 1032) = (1024+8, 1024+8)
// then recovers signed_value = nibble - 8 for both halves at once, instead
// of a per-element sign-extend (shift trick) + int-to-float hardware
// conversion (__int2half_rn) for each of the 8 nibbles individually.
template<int SHIFT>
__device__ __forceinline__
half2 dequant_group_s4(uint32_t packed_word) {
    constexpr uint32_t MASK = 0x000f000f;
    constexpr uint32_t MAGIC = 0x64006400;
    uint32_t shifted = packed_word >> SHIFT;
    uint32_t lo;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(lo) : "r"(shifted), "n"(MASK), "n"(MAGIC), "n"(0xea));
    const half2 BIAS = __halves2half2(__float2half_rn(1032.0f), __float2half_rn(1032.0f));
    return __hsub2(*reinterpret_cast<half2*>(&lo), BIAS);
}

// Dequantizes the 2 excess-8-encoded nibbles packed in one byte (low nibble
// = even element, high nibble = odd element) into a half2 {even, odd},
// by placing the low nibble at bits[0:4) and the high nibble at bits[16:20)
// of a 32-bit word (matching dequant_group_s4<0>'s expected "one nibble per
// 16-bit half" input layout) and reusing that same LOP3 trick.
__device__ __forceinline__
half2 dequant_byte_s4(uint8_t byte) {
    uint32_t word = (uint32_t(byte) & 0xF) | ((uint32_t(byte) & 0xF0) << 12);
    return dequant_group_s4<0>(word);
}

// Asymmetric (zero-point) dequant: unlike the symmetric kernels above, the
// packed code is plain UNSIGNED (0..15 for int4, 0..3 for int2) -- the
// zero-point absorbs the centering, so there's no excess-K encoding to undo.
// dequant(code) = scale*code + zero. Reusing the same LOP3 "(code | 1024.0)"
// trick gives raw_fp16 = 1024+code exactly, same as the symmetric dequants
// above -- but UNLIKE those, this subtracts the bias (1024, exact, since
// 1024 and 1024+code are both exactly representable and small integers
// subtract exactly in fp16) *before* multiplying by scale, not after: an
// earlier version tried folding the "-1024" into a single FMA as
// raw_fp16*scale + (zero - 1024*scale), which is algebraically equivalent
// but numerically broken -- 1024*scale and raw_fp16*scale are both roughly
// the same (large) magnitude while their difference (the actual small
// dequantized value) is not, so fp16's ~10 mantissa bits mostly cancel out
// computing it (measured: real, large errors, not just rounding noise).
// Extracting the exact small-integer code first avoids ever forming that
// oversized intermediate, at the cost of one extra half2 op (hsub2 then
// hfma2, instead of a single hfma2).
__device__ __forceinline__
half2 dequant_byte_s4_asym(uint8_t byte, half2 scale2, half2 zero2) {
    constexpr uint32_t MASK = 0x000f000f;
    constexpr uint32_t MAGIC = 0x64006400;
    uint32_t word = (uint32_t(byte) & 0xF) | ((uint32_t(byte) & 0xF0) << 12);
    uint32_t lo;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(lo) : "r"(word), "n"(MASK), "n"(MAGIC), "n"(0xea));
    const half2 BIAS1024 = __halves2half2(__float2half_rn(1024.0f), __float2half_rn(1024.0f));
    half2 code = __hsub2(*reinterpret_cast<half2*>(&lo), BIAS1024);
    return __hfma2(code, scale2, zero2);
}


// Multi-stage cp.async pipeline, mirroring matmul_kernel_data_parallel's
// prologue/steady-state/drain structure. Unlike an earlier version of this
// kernel, there's no separate dequantized B tile in shared memory at all:
// the packed int4 weights are cp.async'd as raw bytes into a per-stage
// staging buffer (1/4 the size fp16 B would need) and compute() below reads
// straight out of that buffer, dequantizing directly into registers right
// before each mma call (see the tau/g comment in compute()) -- no unpack
// pass, no extra barrier for it, and a smaller per-stage shared-memory
// footprint than materializing dequantized B would need. The per-channel
// scale is not part of this K-loop at all: since scale[n] doesn't depend on
// k, it's factored out of the reduction entirely and applied once in the
// epilogue below.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, bool CACHE_B = true>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a16(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                          const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    constexpr int MMA_K = 16;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_BLOCK_N = cdiv(N, BLOCK_N);
    const int NUM_BLOCK_K = cdiv(K, BLOCK_K);

    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;

    extern __shared__ __half shm[];
    const uint32_t shm_u32 = cvta_shared(shm);
    constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(__half);
    constexpr int Bq_size = CACHE_B ? BLOCK_N * (BLOCK_K / 2) : 0;
    constexpr int stage_size = A_size + Bq_size;
    const uint32_t A_shm_base = shm_u32;
    const uint32_t Bq_shm_base = A_shm_base + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const __half* Ap = A + off_m * K;
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            if constexpr (CACHE_B) {
                global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 2);
                Bp += BLOCK_K / 2;
            }
            Ap += BLOCK_K;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        // tau/g are this thread's position within mma.sync.m16n8k16's
        // documented B-operand fragment layout: thread lane_id owns local
        // k-offsets {2*tau, 2*tau+1, 2*tau+8, 2*tau+9} (tau = lane_id % 4)
        // at column g = lane_id / 4 within one MMA_N=8 tile. Those 4
        // k-offsets fall into exactly 2 packed bytes (2tau,2tau+1 share one
        // byte; 2tau+8,2tau+9 share the next, 4 bytes over), so each thread
        // needs only 2 shared-memory byte reads + 2 cheap LOP3 dequants per
        // (k,n) MMA tile -- reading straight out of the packed cp.async
        // staging buffer and building its own B fragment, instead of a
        // hardware ldmatrix fetch from a separately-materialized,
        // swizzled fp16 tile (which is also why Bq_shm has no swizzle:
        // nothing here goes through ldmatrix).
        const int tau = lane_id % 4;
        const int g = lane_id / 4;

        auto compute = [&](int k_iter){
            const uint32_t A_stage_base = A_shm_thread + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_stage_base + m * MMA_M * BLOCK_K * sizeof(__half);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            if constexpr (CACHE_B) {
                const uint32_t Bq_stage_base = Bq_shm_base + (k_iter % NUM_STAGES) * stage_size;
                for(int k = 0; k < NUM_MMA_K; k++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                        const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 2);
                        const uint32_t byte_addr_lo = row_base + k * (MMA_K / 2) + tau;
                        const uint32_t byte_addr_hi = byte_addr_lo + 4;
                        uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                        uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                        half2 lo_pair = dequant_byte_s4(byte_lo);
                        half2 hi_pair = dequant_byte_s4(byte_hi);
                        B_reg[k][n][0] = *reinterpret_cast<uint32_t*>(&lo_pair);
                        B_reg[k][n][1] = *reinterpret_cast<uint32_t*>(&hi_pair);
                    }
                }
            } else {
                // No shared-memory staging for B at all: with NUM_WARP_M ==
                // 1 (the only case this path is used for -- see
                // w4a16_matmul_launcher), there's no cross-warp reuse of B
                // within a tile to lose by skipping the cache, so each
                // thread just reads its 2 bytes straight out of global
                // memory (Bp, unchanged by load_AB above since it's never
                // called for B here) and dequantizes them directly.
                // Out-of-bounds N/K reads are clamped to a safe in-row
                // address rather than zero-filled like the cached path:
                // clamped reads land on a real (if wrong) byte instead of
                // a true zero, but that's fine here since A's own zero-fill
                // at the same K position multiplies the contribution to
                // zero regardless of what B's padding decodes to (excess-8
                // encoding means a raw zero byte doesn't dequantize to zero
                // anyway, so this path never relied on that to begin with).
                for(int k = 0; k < NUM_MMA_K; k++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                        const int safe_n_col = (n_col < valid_width) ? n_col : 0;
                        const uint8_t* row_ptr = Bp + safe_n_col * (K / 2);
                        const int abs_k = k_iter * BLOCK_K + k * MMA_K;
                        const int abs_k_lo = abs_k + 2 * tau;
                        const int abs_k_hi = abs_k + 2 * tau + 8;
                        const int byte_col_lo = (abs_k_lo < K) ? abs_k_lo / 2 : 0;
                        const int byte_col_hi = (abs_k_hi < K) ? abs_k_hi / 2 : 0;
                        half2 lo_pair = dequant_byte_s4(row_ptr[byte_col_lo]);
                        half2 hi_pair = dequant_byte_s4(row_ptr[byte_col_hi]);
                        B_reg[k][n][0] = *reinterpret_cast<uint32_t*>(&lo_pair);
                        B_reg[k][n][1] = *reinterpret_cast<uint32_t*>(&hi_pair);
                    }
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k16(A_reg[k][m], B_reg[k][n], acc[m][n]);
                    }
                }
            }
        };

        for(int stage=0; stage < NUM_STAGES - 1; stage++){
            load_AB(stage);
        }

        for(int k = 0; k < NUM_BLOCK_K - (NUM_STAGES - 1); k++){
            __syncthreads();
            load_AB(k + NUM_STAGES - 1);
            cp_async_wait_group<NUM_STAGES - 1>();
            __syncthreads();
            compute(k);
        }

        for(int k = std::max(0, NUM_BLOCK_K - (NUM_STAGES - 1)); k < NUM_BLOCK_K; k++){
            __syncthreads();
            cp_async_wait_all();
            __syncthreads();
            compute(k);
        }
        __syncthreads();

        for(int m=0; m<NUM_MMA_M; m++){
            for(int n=0; n<NUM_MMA_N; n++){
                const int row = m * MMA_M + (lane_id / 4);
                const int col = n * MMA_N + (lane_id % 4) * 2;
                const int local_row = warp_id_m * WARP_M + row;
                const int local_col = warp_id_n * WARP_N + col;
                float *regs = acc[m][n];
                const int n0 = off_n + local_col;
                const int n1 = off_n + local_col + 1;
                const float s0 = (n0 < N) ? __half2float(scales[n0]) : 0.0f;
                const float s1 = (n1 < N) ? __half2float(scales[n1]) : 0.0f;
                const float bias0 = (bias && n0 < N) ? __half2float(bias[n0]) : 0.0f;
                const float bias1 = (bias && n1 < N) ? __half2float(bias[n1]) : 0.0f;
                __half* p0 = Cp + row * N + col;
                __half* p1 = Cp + (row + 8) * N + col;
                if (local_row < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p0) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p0)[0] = __float22half2_rn(make_float2(regs[0] * s0 + bias0, regs[1] * s1 + bias1));
                        } else {
                            p0[0] = __float2half_rn(regs[0] * s0 + bias0);
                            p0[1] = __float2half_rn(regs[1] * s1 + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p0[0] = __float2half_rn(regs[0] * s0 + bias0);
                    }
                }
                if (local_row + 8 < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p1) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p1)[0] = __float22half2_rn(make_float2(regs[2] * s0 + bias0, regs[3] * s1 + bias1));
                        } else {
                            p1[0] = __float2half_rn(regs[2] * s0 + bias0);
                            p1[1] = __float2half_rn(regs[3] * s1 + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p1[0] = __float2half_rn(regs[2] * s0 + bias0);
                    }
                }
            }
        }
    }
}

constexpr int W4_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, bool CACHE_B = true>
void w4a16_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                   int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (CACHE_B ? (BLOCK_N * (BLOCK_K / 2)) : 0);
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w4a16<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES, CACHE_B>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a16_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                            const int M, const int N, const int K, cudaStream_t stream){
    // Tried CACHE_B=false for this tier (BLOCK_M=16, NUM_WARP_M=1, so no
    // cross-warp-M reuse of B to lose by skipping its shared-memory cache
    // -- see the CACHE_B=false branch in matmul_kernel_w4a16's compute()):
    // measured a clear regression (e.g. M=16 dropped from ~3.3x to ~1.7x
    // over PyTorch fp16). The scalar, unpipelined direct-from-global reads
    // expose global memory latency that cp.async's multi-stage pipelining
    // was hiding -- that benefit outweighs the caching overhead it was
    // meant to avoid, even with no reuse to lose. Left CACHE_B itself in
    // place (default true) as a documented dead end, not wired into any
    // tier's dispatch.
    if (M <= 16)
        w4a16_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a16_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, bias, C, M, N, K, stream);
    else
        w4a16_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}


// Asymmetric (zero-point) counterpart to matmul_kernel_w4a16. Same overall
// structure (register-resident dequant straight out of the packed cp.async
// staging buffer, no dequantized B tile in shared memory, same mma.sync
// tau/g B-fragment mapping) -- the only real differences are: Bq holds
// plain UNSIGNED codes (no excess-8 encoding, since zeros[n] absorbs the
// centering), dequant_byte_s4_asym's FMA needs scale/adj_zero broadcast
// pairs (hoisted per n, since they don't depend on k), and the epilogue no
// longer multiplies by scale at all -- see dequant_byte_s4_asym's comment
// for why the zero-point can't be deferred to the epilogue the way scale
// alone can in the symmetric kernel.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a16_asym(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                               const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    constexpr int MMA_K = 16;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_BLOCK_N = cdiv(N, BLOCK_N);
    const int NUM_BLOCK_K = cdiv(K, BLOCK_K);

    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;

    extern __shared__ __half shm[];
    const uint32_t shm_u32 = cvta_shared(shm);
    constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(__half);
    constexpr int Bq_size = BLOCK_N * (BLOCK_K / 2);
    constexpr int stage_size = A_size + Bq_size;
    const uint32_t A_shm_base = shm_u32;
    const uint32_t Bq_shm_base = A_shm_base + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const __half* Ap = A + off_m * K;
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 2);
            Bp += BLOCK_K / 2;
            Ap += BLOCK_K;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;

        auto compute = [&](int k_iter){
            const uint32_t A_stage_base = A_shm_thread + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_stage_base + m * MMA_M * BLOCK_K * sizeof(__half);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            const uint32_t Bq_stage_base = Bq_shm_base + (k_iter % NUM_STAGES) * stage_size;
            for(int n=0; n<NUM_MMA_N; n++){
                const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                const int safe_n_col = (n_col < valid_width) ? n_col : 0;
                const int global_n = off_n + safe_n_col;
                const half2 scale2 = __halves2half2(scales[global_n], scales[global_n]);
                const half2 zero2 = __halves2half2(zeros[global_n], zeros[global_n]);
                const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 2);
                for(int k = 0; k < NUM_MMA_K; k++){
                    const uint32_t byte_addr_lo = row_base + k * (MMA_K / 2) + tau;
                    const uint32_t byte_addr_hi = byte_addr_lo + 4;
                    uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                    uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                    half2 lo_pair = dequant_byte_s4_asym(byte_lo, scale2, zero2);
                    half2 hi_pair = dequant_byte_s4_asym(byte_hi, scale2, zero2);
                    B_reg[k][n][0] = *reinterpret_cast<uint32_t*>(&lo_pair);
                    B_reg[k][n][1] = *reinterpret_cast<uint32_t*>(&hi_pair);
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k16(A_reg[k][m], B_reg[k][n], acc[m][n]);
                    }
                }
            }
        };

        for(int stage=0; stage < NUM_STAGES - 1; stage++){
            load_AB(stage);
        }

        for(int k = 0; k < NUM_BLOCK_K - (NUM_STAGES - 1); k++){
            __syncthreads();
            load_AB(k + NUM_STAGES - 1);
            cp_async_wait_group<NUM_STAGES - 1>();
            __syncthreads();
            compute(k);
        }

        for(int k = std::max(0, NUM_BLOCK_K - (NUM_STAGES - 1)); k < NUM_BLOCK_K; k++){
            __syncthreads();
            cp_async_wait_all();
            __syncthreads();
            compute(k);
        }
        __syncthreads();

        // No per-column scale here (unlike the symmetric epilogue): the
        // affine dequant (scale*code + zero) was already fully applied
        // in-loop, above, so acc already holds the correctly-dequantized
        // accumulation -- just add bias (if any) and convert to fp16,
        // exactly like the plain fp16 kernel's epilogue.
        for(int m=0; m<NUM_MMA_M; m++){
            for(int n=0; n<NUM_MMA_N; n++){
                const int row = m * MMA_M + (lane_id / 4);
                const int col = n * MMA_N + (lane_id % 4) * 2;
                const int local_row = warp_id_m * WARP_M + row;
                const int local_col = warp_id_n * WARP_N + col;
                float *regs = acc[m][n];
                const int n0 = off_n + local_col;
                const int n1 = off_n + local_col + 1;
                const float bias0 = (bias && n0 < N) ? __half2float(bias[n0]) : 0.0f;
                const float bias1 = (bias && n1 < N) ? __half2float(bias[n1]) : 0.0f;
                __half* p0 = Cp + row * N + col;
                __half* p1 = Cp + (row + 8) * N + col;
                if (local_row < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p0) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p0)[0] = __float22half2_rn(make_float2(regs[0] + bias0, regs[1] + bias1));
                        } else {
                            p0[0] = __float2half_rn(regs[0] + bias0);
                            p0[1] = __float2half_rn(regs[1] + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p0[0] = __float2half_rn(regs[0] + bias0);
                    }
                }
                if (local_row + 8 < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p1) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p1)[0] = __float22half2_rn(make_float2(regs[2] + bias0, regs[3] + bias1));
                        } else {
                            p1[0] = __float2half_rn(regs[2] + bias0);
                            p1[1] = __float2half_rn(regs[3] + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p1[0] = __float2half_rn(regs[2] + bias0);
                    }
                }
            }
        }
    }
}

constexpr int W4_ASYM_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w4a16_asym_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                        int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (BLOCK_N * (BLOCK_K / 2));
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4_ASYM_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w4a16_asym<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, zeros, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a16_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                                 const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16)
        w4a16_asym_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a16_asym_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else
        w4a16_asym_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, zeros, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}


constexpr int W4A16_GROUP_SIZE = 128;

// Groupwise (per-channel-per-128-k-group) symmetric W4A16: same
// register-resident design as matmul_kernel_w4a16 above, but the scale now
// varies every GROUP_SIZE=128 k-values instead of being constant across the
// whole K reduction -- so unlike the plain per-channel kernel, it can't be
// deferred to a single post-reduction epilogue multiply (same reasoning as
// the asymmetric kernel's zero-point, which also can't be deferred: a
// single K-reduction here sums contributions scaled by DIFFERENT group
// scales, so each group's contribution has to be scaled before it's added
// into the accumulator, not after). BLOCK_K=64 evenly divides GROUP_SIZE=128
// (a static_assert enforces this), so every load_AB/compute() call's entire
// BLOCK_K span falls within exactly one group -- the group's scale is
// hoisted once per n (not per k), exactly like the asymmetric kernel hoists
// its scale/zero per n. Reuses dequant_byte_s4 (the same excess-8 LOP3
// dequant matmul_kernel_w4a16 uses) unchanged, just multiplying its
// unscaled result by the group's scale via __hmul2 instead of leaving it
// for the epilogue.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, int GROUP_SIZE = W4A16_GROUP_SIZE>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a16_group(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                                const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    static_assert(GROUP_SIZE % BLOCK_K == 0, "GROUP_SIZE must be a multiple of BLOCK_K");
    constexpr int MMA_K = 16;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_BLOCK_N = cdiv(N, BLOCK_N);
    const int NUM_BLOCK_K = cdiv(K, BLOCK_K);
    const int NUM_GROUPS = K / GROUP_SIZE;

    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;

    extern __shared__ __half shm[];
    const uint32_t shm_u32 = cvta_shared(shm);
    constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(__half);
    constexpr int Bq_size = BLOCK_N * (BLOCK_K / 2);
    constexpr int stage_size = A_size + Bq_size;
    const uint32_t A_shm_base = shm_u32;
    const uint32_t Bq_shm_base = A_shm_base + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const __half* Ap = A + off_m * K;
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 2);
            Bp += BLOCK_K / 2;
            Ap += BLOCK_K;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;

        auto compute = [&](int k_iter){
            const uint32_t A_stage_base = A_shm_thread + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_stage_base + m * MMA_M * BLOCK_K * sizeof(__half);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            const uint32_t Bq_stage_base = Bq_shm_base + (k_iter % NUM_STAGES) * stage_size;
            const int group_id = (k_iter * BLOCK_K) / GROUP_SIZE;
            for(int n=0; n<NUM_MMA_N; n++){
                const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                const int safe_n_col = (n_col < valid_width) ? n_col : 0;
                const int global_n = off_n + safe_n_col;
                const __half s = scales[global_n * NUM_GROUPS + group_id];
                const half2 scale2 = __halves2half2(s, s);
                const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 2);
                for(int k = 0; k < NUM_MMA_K; k++){
                    const uint32_t byte_addr_lo = row_base + k * (MMA_K / 2) + tau;
                    const uint32_t byte_addr_hi = byte_addr_lo + 4;
                    uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                    uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                    half2 lo_pair = __hmul2(dequant_byte_s4(byte_lo), scale2);
                    half2 hi_pair = __hmul2(dequant_byte_s4(byte_hi), scale2);
                    B_reg[k][n][0] = *reinterpret_cast<uint32_t*>(&lo_pair);
                    B_reg[k][n][1] = *reinterpret_cast<uint32_t*>(&hi_pair);
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k16(A_reg[k][m], B_reg[k][n], acc[m][n]);
                    }
                }
            }
        };

        for(int stage=0; stage < NUM_STAGES - 1; stage++){
            load_AB(stage);
        }

        for(int k = 0; k < NUM_BLOCK_K - (NUM_STAGES - 1); k++){
            __syncthreads();
            load_AB(k + NUM_STAGES - 1);
            cp_async_wait_group<NUM_STAGES - 1>();
            __syncthreads();
            compute(k);
        }

        for(int k = std::max(0, NUM_BLOCK_K - (NUM_STAGES - 1)); k < NUM_BLOCK_K; k++){
            __syncthreads();
            cp_async_wait_all();
            __syncthreads();
            compute(k);
        }
        __syncthreads();

        // No per-column scale here (unlike matmul_kernel_w4a16's epilogue):
        // the group scale was already applied in-loop, above -- bias still
        // needs adding here, same as every other epilogue.
        for(int m=0; m<NUM_MMA_M; m++){
            for(int n=0; n<NUM_MMA_N; n++){
                const int row = m * MMA_M + (lane_id / 4);
                const int col = n * MMA_N + (lane_id % 4) * 2;
                const int local_row = warp_id_m * WARP_M + row;
                const int local_col = warp_id_n * WARP_N + col;
                float *regs = acc[m][n];
                const int n0 = off_n + local_col;
                const int n1 = off_n + local_col + 1;
                const float bias0 = (bias && n0 < N) ? __half2float(bias[n0]) : 0.0f;
                const float bias1 = (bias && n1 < N) ? __half2float(bias[n1]) : 0.0f;
                __half* p0 = Cp + row * N + col;
                __half* p1 = Cp + (row + 8) * N + col;
                if (local_row < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p0) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p0)[0] = __float22half2_rn(make_float2(regs[0] + bias0, regs[1] + bias1));
                        } else {
                            p0[0] = __float2half_rn(regs[0] + bias0);
                            p0[1] = __float2half_rn(regs[1] + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p0[0] = __float2half_rn(regs[0] + bias0);
                    }
                }
                if (local_row + 8 < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p1) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p1)[0] = __float22half2_rn(make_float2(regs[2] + bias0, regs[3] + bias1));
                        } else {
                            p1[0] = __float2half_rn(regs[2] + bias0);
                            p1[1] = __float2half_rn(regs[3] + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p1[0] = __float2half_rn(regs[2] + bias0);
                    }
                }
            }
        }
    }
}

constexpr int W4_GROUPWISE_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w4a16_group_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                         int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (BLOCK_N * (BLOCK_K / 2));
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4_GROUPWISE_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w4a16_group<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a16_group_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                                  const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16)
        w4a16_group_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a16_group_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, bias, C, M, N, K, stream);
    else
        w4a16_group_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}


// Asymmetric (zero-point) counterpart to matmul_kernel_w4a16_group above --
// same relationship as matmul_kernel_w4a16_asym has to matmul_kernel_w4a16:
// Bq holds plain UNSIGNED codes (no excess-8 encoding, since zeros absorbs
// the centering) and the epilogue no longer multiplies by scale at all,
// since the affine dequant (scale*code + zero) is already fully applied
// in-loop. The only difference from matmul_kernel_w4a16_asym is that both
// scale AND zero now vary per (n, group) instead of per n -- but since
// BLOCK_K=64 evenly divides GROUP_SIZE=128 (a static_assert enforces this,
// same as matmul_kernel_w4a16_group), every load_AB/compute() call's entire
// BLOCK_K span still falls within exactly one group, so both are hoisted
// once per n exactly as matmul_kernel_w4a16_asym already does, just indexed
// by (n, group_id) instead of just n.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, int GROUP_SIZE = W4A16_GROUP_SIZE>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a16_group_asym(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                                     const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    static_assert(GROUP_SIZE % BLOCK_K == 0, "GROUP_SIZE must be a multiple of BLOCK_K");
    constexpr int MMA_K = 16;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_BLOCK_N = cdiv(N, BLOCK_N);
    const int NUM_BLOCK_K = cdiv(K, BLOCK_K);
    const int NUM_GROUPS = K / GROUP_SIZE;

    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;

    extern __shared__ __half shm[];
    const uint32_t shm_u32 = cvta_shared(shm);
    constexpr int A_size = BLOCK_M * BLOCK_K * sizeof(__half);
    constexpr int Bq_size = BLOCK_N * (BLOCK_K / 2);
    constexpr int stage_size = A_size + Bq_size;
    const uint32_t A_shm_base = shm_u32;
    const uint32_t Bq_shm_base = A_shm_base + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const __half* Ap = A + off_m * K;
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 2);
            Bp += BLOCK_K / 2;
            Ap += BLOCK_K;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;

        auto compute = [&](int k_iter){
            const uint32_t A_stage_base = A_shm_thread + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_stage_base + m * MMA_M * BLOCK_K * sizeof(__half);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            const uint32_t Bq_stage_base = Bq_shm_base + (k_iter % NUM_STAGES) * stage_size;
            const int group_id = (k_iter * BLOCK_K) / GROUP_SIZE;
            for(int n=0; n<NUM_MMA_N; n++){
                const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                const int safe_n_col = (n_col < valid_width) ? n_col : 0;
                const int global_n = off_n + safe_n_col;
                const int sz_idx = global_n * NUM_GROUPS + group_id;
                const half2 scale2 = __halves2half2(scales[sz_idx], scales[sz_idx]);
                const half2 zero2 = __halves2half2(zeros[sz_idx], zeros[sz_idx]);
                const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 2);
                for(int k = 0; k < NUM_MMA_K; k++){
                    const uint32_t byte_addr_lo = row_base + k * (MMA_K / 2) + tau;
                    const uint32_t byte_addr_hi = byte_addr_lo + 4;
                    uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                    uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                    half2 lo_pair = dequant_byte_s4_asym(byte_lo, scale2, zero2);
                    half2 hi_pair = dequant_byte_s4_asym(byte_hi, scale2, zero2);
                    B_reg[k][n][0] = *reinterpret_cast<uint32_t*>(&lo_pair);
                    B_reg[k][n][1] = *reinterpret_cast<uint32_t*>(&hi_pair);
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k16(A_reg[k][m], B_reg[k][n], acc[m][n]);
                    }
                }
            }
        };

        for(int stage=0; stage < NUM_STAGES - 1; stage++){
            load_AB(stage);
        }

        for(int k = 0; k < NUM_BLOCK_K - (NUM_STAGES - 1); k++){
            __syncthreads();
            load_AB(k + NUM_STAGES - 1);
            cp_async_wait_group<NUM_STAGES - 1>();
            __syncthreads();
            compute(k);
        }

        for(int k = std::max(0, NUM_BLOCK_K - (NUM_STAGES - 1)); k < NUM_BLOCK_K; k++){
            __syncthreads();
            cp_async_wait_all();
            __syncthreads();
            compute(k);
        }
        __syncthreads();

        for(int m=0; m<NUM_MMA_M; m++){
            for(int n=0; n<NUM_MMA_N; n++){
                const int row = m * MMA_M + (lane_id / 4);
                const int col = n * MMA_N + (lane_id % 4) * 2;
                const int local_row = warp_id_m * WARP_M + row;
                const int local_col = warp_id_n * WARP_N + col;
                float *regs = acc[m][n];
                const int n0 = off_n + local_col;
                const int n1 = off_n + local_col + 1;
                const float bias0 = (bias && n0 < N) ? __half2float(bias[n0]) : 0.0f;
                const float bias1 = (bias && n1 < N) ? __half2float(bias[n1]) : 0.0f;
                __half* p0 = Cp + row * N + col;
                __half* p1 = Cp + (row + 8) * N + col;
                if (local_row < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p0) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p0)[0] = __float22half2_rn(make_float2(regs[0] + bias0, regs[1] + bias1));
                        } else {
                            p0[0] = __float2half_rn(regs[0] + bias0);
                            p0[1] = __float2half_rn(regs[1] + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p0[0] = __float2half_rn(regs[0] + bias0);
                    }
                }
                if (local_row + 8 < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p1) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p1)[0] = __float22half2_rn(make_float2(regs[2] + bias0, regs[3] + bias1));
                        } else {
                            p1[0] = __float2half_rn(regs[2] + bias0);
                            p1[1] = __float2half_rn(regs[3] + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p1[0] = __float2half_rn(regs[2] + bias0);
                    }
                }
            }
        }
    }
}

constexpr int W4_GROUP_ASYM_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w4a16_group_asym_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                              int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (BLOCK_N * (BLOCK_K / 2));
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4_GROUP_ASYM_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w4a16_group_asym<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, zeros, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a16_group_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                                       const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16)
        w4a16_group_asym_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a16_group_asym_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else
        w4a16_group_asym_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, zeros, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}
