#include "common.h"
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>

// W2A16: fp16 activations x per-channel-quantized packed-int2 weights, both
// symmetric (matmul_kernel_w2a16) and asymmetric/zero-point
// (matmul_kernel_w2a16_asym) variants. Same register-resident design as
// asrq_4x16.cu's W4A16 kernels (see that file's top comment and each
// kernel's compute() here for the mma.sync B-fragment tau/g mapping),
// narrowed to int2: 4 codes/byte instead of int4's 2, so a thread's needed
// codes no longer land in their own byte each -- see byte_idx_lo/hi and
// bit_offset below.

// Same LOP3 trick, narrowed to 2-bit codes for W2A16: codes are packed using
// an excess-2 (offset-binary) encoding -- raw_code = signed_value + 2, so
// raw_code is 0..3 (signed range [-2, 1]) -- rather than two's complement,
// again so the trick applies directly. Unlike int4 (2 codes/byte), int4
// packs 4 codes/byte here, so a single byte can hold 2 codes that are NOT
// the pair this call wants -- bit_offset (0 or 4) selects which of the two
// 2-code pairs within the byte to dequantize, extracting them to bits[0:2)
// and bits[2:4) before reusing the same (a&b)|c / MAGIC=1024.0 / bias
// pattern as dequant_byte_s4, just with a narrower 0x0003 mask and a
// smaller bias (1024+2 instead of 1024+8).
__device__ __forceinline__
half2 dequant_byte_s2_pair(uint8_t byte, int bit_offset) {
    constexpr uint32_t MASK = 0x00030003;
    constexpr uint32_t MAGIC = 0x64006400;
    const uint32_t pair = (uint32_t(byte) >> bit_offset) & 0xF;
    const uint32_t word = (pair & 0x3) | ((pair & 0xC) << 14);
    uint32_t lo;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(lo) : "r"(word), "n"(MASK), "n"(MAGIC), "n"(0xea));
    const half2 BIAS = __halves2half2(__float2half_rn(1026.0f), __float2half_rn(1026.0f));
    return __hsub2(*reinterpret_cast<half2*>(&lo), BIAS);
}

// Asymmetric (zero-point) counterpart to dequant_byte_s2_pair above: same
// bit_offset-selected 2-code extraction, but the packed code is plain
// UNSIGNED (0..3, no excess-2 encoding -- the zero-point absorbs the
// centering instead), and an exact bias-1024 subtraction happens before the
// FMA against (scale, zero) rather than being folded into it: 1024*scale and
// raw_fp16*scale are both roughly the same (large) magnitude while their
// difference (the actual small dequantized value) is not, so computing that
// difference directly in fp16 loses most of its ~10 mantissa bits to
// cancellation. Extracting the exact small-integer code first (via __hsub2
// against 1024, exact since both operands are small integers) avoids ever
// forming that oversized intermediate.
__device__ __forceinline__
half2 dequant_byte_s2_pair_asym(uint8_t byte, int bit_offset, half2 scale2, half2 zero2) {
    constexpr uint32_t MASK = 0x00030003;
    constexpr uint32_t MAGIC = 0x64006400;
    const uint32_t pair = (uint32_t(byte) >> bit_offset) & 0xF;
    const uint32_t word = (pair & 0x3) | ((pair & 0xC) << 14);
    uint32_t lo;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
                 : "=r"(lo) : "r"(word), "n"(MASK), "n"(MAGIC), "n"(0xea));
    const half2 BIAS1024 = __halves2half2(__float2half_rn(1024.0f), __float2half_rn(1024.0f));
    half2 code = __hsub2(*reinterpret_cast<half2*>(&lo), BIAS1024);
    return __hfma2(code, scale2, zero2);
}


template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w2a16(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
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
    constexpr int Bq_size = BLOCK_N * (BLOCK_K / 4);
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
        const uint8_t* Bp = Bq + off_n * (K / 4);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 4>(Bp, K / 4, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 4);
            Ap += BLOCK_K;
            Bp += BLOCK_K / 4;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;
        const int byte_idx_lo = tau / 2;
        const int byte_idx_hi = tau / 2 + 2;
        const int bit_offset = (tau % 2) * 4;

        auto compute = [&](int k_iter){
            const uint32_t A_stage_base = A_shm_thread + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_stage_base + m * MMA_M * BLOCK_K * sizeof(__half);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            const uint32_t Bq_stage_base = Bq_shm_base + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int n=0; n<NUM_MMA_N; n++){
                    const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                    const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 4);
                    const uint32_t byte_addr_lo = row_base + k * (MMA_K / 4) + byte_idx_lo;
                    const uint32_t byte_addr_hi = row_base + k * (MMA_K / 4) + byte_idx_hi;
                    uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                    uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                    half2 lo_pair = dequant_byte_s2_pair(byte_lo, bit_offset);
                    half2 hi_pair = dequant_byte_s2_pair(byte_hi, bit_offset);
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

constexpr int W2_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w2a16_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                   int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (BLOCK_N * (BLOCK_K / 4));
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W2_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w2a16<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w2a16_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                            const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16)
        w2a16_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, bias, C, M, N, K, stream);
    else if (M <= 64)
        w2a16_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, bias, C, M, N, K, stream);
    else
        w2a16_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}


// Asymmetric (zero-point) counterpart to matmul_kernel_w2a16 above: same
// register-resident design (no dequantized B tile in shared memory, same
// mma.sync B-fragment tau/g mapping), but Bq holds plain UNSIGNED 2-bit
// codes (no excess-2 encoding, since zeros[n] absorbs the centering),
// dequant_byte_s2_pair_asym's FMA needs scale/zero broadcast pairs (hoisted
// per n, since they don't depend on k), and the epilogue no longer
// multiplies by scale at all -- the affine dequant (scale*code + zero) is
// already fully applied in-loop, so acc holds the correctly-dequantized
// accumulation directly.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w2a16_asym(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
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
    constexpr int Bq_size = BLOCK_N * (BLOCK_K / 4);
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
        const uint8_t* Bp = Bq + off_n * (K / 4);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 4>(Bp, K / 4, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 4);
            Ap += BLOCK_K;
            Bp += BLOCK_K / 4;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;
        const int byte_idx_lo = tau / 2;
        const int byte_idx_hi = tau / 2 + 2;
        const int bit_offset = (tau % 2) * 4;

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
                const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 4);
                for(int k = 0; k < NUM_MMA_K; k++){
                    const uint32_t byte_addr_lo = row_base + k * (MMA_K / 4) + byte_idx_lo;
                    const uint32_t byte_addr_hi = row_base + k * (MMA_K / 4) + byte_idx_hi;
                    uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                    uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                    half2 lo_pair = dequant_byte_s2_pair_asym(byte_lo, bit_offset, scale2, zero2);
                    half2 hi_pair = dequant_byte_s2_pair_asym(byte_hi, bit_offset, scale2, zero2);
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

constexpr int W2_ASYM_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w2a16_asym_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                        int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (BLOCK_N * (BLOCK_K / 4));
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W2_ASYM_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w2a16_asym<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, zeros, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w2a16_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                                 const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16)
        w2a16_asym_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else if (M <= 64)
        w2a16_asym_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else
        w2a16_asym_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, zeros, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}


constexpr int W2A16_GROUP_SIZE = 128;

// Groupwise (per-channel-per-128-k-group) symmetric W2A16 -- same relationship
// to matmul_kernel_w2a16 above as matmul_kernel_w4a16_group (asrq_4x16.cu)
// has to matmul_kernel_w4a16: the scale now varies every GROUP_SIZE=128
// k-values instead of being constant across the whole K reduction, so it's
// applied in-loop (hoisted once per n, since BLOCK_K=64 evenly divides
// GROUP_SIZE=128 -- a static_assert enforces this -- so every
// load_AB/compute() call's BLOCK_K span falls within exactly one group)
// instead of deferred to the epilogue. Reuses dequant_byte_s2_pair
// unchanged, just multiplying its unscaled result by the group's scale via
// __hmul2.
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, int GROUP_SIZE = W2A16_GROUP_SIZE>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w2a16_group(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
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
    constexpr int Bq_size = BLOCK_N * (BLOCK_K / 4);
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
        const uint8_t* Bp = Bq + off_n * (K / 4);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 4>(Bp, K / 4, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 4);
            Ap += BLOCK_K;
            Bp += BLOCK_K / 4;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;
        const int byte_idx_lo = tau / 2;
        const int byte_idx_hi = tau / 2 + 2;
        const int bit_offset = (tau % 2) * 4;

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
                const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 4);
                for(int k = 0; k < NUM_MMA_K; k++){
                    const uint32_t byte_addr_lo = row_base + k * (MMA_K / 4) + byte_idx_lo;
                    const uint32_t byte_addr_hi = row_base + k * (MMA_K / 4) + byte_idx_hi;
                    uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                    uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                    half2 lo_pair = __hmul2(dequant_byte_s2_pair(byte_lo, bit_offset), scale2);
                    half2 hi_pair = __hmul2(dequant_byte_s2_pair(byte_hi, bit_offset), scale2);
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

        // No per-column scale here (unlike matmul_kernel_w2a16's epilogue):
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

constexpr int W2_GROUPWISE_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w2a16_group_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                         int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (BLOCK_N * (BLOCK_K / 4));
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W2_GROUPWISE_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w2a16_group<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w2a16_group_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias, __half* C,
                                  const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16)
        w2a16_group_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, bias, C, M, N, K, stream);
    else if (M <= 64)
        w2a16_group_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, bias, C, M, N, K, stream);
    else
        w2a16_group_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}


// Asymmetric (zero-point) counterpart to matmul_kernel_w2a16_group above --
// same relationship as matmul_kernel_w2a16_asym has to matmul_kernel_w2a16
// (see asrq_4x16.cu's matmul_kernel_w4a16_group_asym for the general
// reasoning, which is identical here just narrowed to int2): Bq holds plain
// UNSIGNED codes, scale AND zero vary per (n, group) instead of per n, and
// both are hoisted once per n since BLOCK_K=64 evenly divides
// GROUP_SIZE=128 (a static_assert enforces this).
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, int GROUP_SIZE = W2A16_GROUP_SIZE>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w2a16_group_asym(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
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
    constexpr int Bq_size = BLOCK_N * (BLOCK_K / 4);
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
        const uint8_t* Bp = Bq + off_n * (K / 4);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm_base + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm_base + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 4>(Bp, K / 4, Bq_shm_base + stage_id * stage_size, tid, valid_width, valid_k / 4);
            Ap += BLOCK_K;
            Bp += BLOCK_K / 4;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;
        const int byte_idx_lo = tau / 2;
        const int byte_idx_hi = tau / 2 + 2;
        const int bit_offset = (tau % 2) * 4;

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
                const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 4);
                for(int k = 0; k < NUM_MMA_K; k++){
                    const uint32_t byte_addr_lo = row_base + k * (MMA_K / 4) + byte_idx_lo;
                    const uint32_t byte_addr_hi = row_base + k * (MMA_K / 4) + byte_idx_hi;
                    uint8_t byte_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_lo)));
                    uint8_t byte_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(byte_addr_hi)));
                    half2 lo_pair = dequant_byte_s2_pair_asym(byte_lo, bit_offset, scale2, zero2);
                    half2 hi_pair = dequant_byte_s2_pair_asym(byte_hi, bit_offset, scale2, zero2);
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

constexpr int W2_GROUP_ASYM_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w2a16_group_asym_launch(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                              int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int stage_size = (BLOCK_M * BLOCK_K * sizeof(__half))
                              + (BLOCK_N * (BLOCK_K / 4));
    constexpr int shm_size = stage_size * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W2_GROUP_ASYM_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w2a16_group_asym<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, Bq, scales, zeros, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w2a16_group_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros, const __half* bias, __half* C,
                                       const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16)
        w2a16_group_asym_launch<16, 64, 64, 1, 4, 12>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else if (M <= 64)
        w2a16_group_asym_launch<64, 64, 64, 2, 4, 4>(A, Bq, scales, zeros, bias, C, M, N, K, stream);
    else
        w2a16_group_asym_launch<128, 64, 64, 4, 4, 3>(A, Bq, scales, zeros, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}


