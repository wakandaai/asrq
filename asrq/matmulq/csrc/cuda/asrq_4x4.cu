#include "common.h"
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>

// W4A4: BOTH A and B quantized to int4, symmetric, and run through the
// hardware int4 tensor core (mma_m16n8k64_s4, common.h) directly -- unlike
// every WxA16 kernel (asrq_4x16.cu, asrq_2x16.cu), there is no dequantization to fp16 anywhere in
// the K-loop. Nibbles are plain two's-complement signed values (range
// [-8,7]), packed 2/byte (low nibble = even k, high nibble = odd k): this is
// NOT the excess-8 encoding the WxA16 dequant helpers use -- that encoding
// exists only to make the LOP3 fp16-dequant trick exact, and there's no
// fp16 dequant here for it to serve. A raw signed nibble is exactly what
// mma.sync's s4 operand expects, so cp.async'd packed bytes feed the MMA
// with no transformation at all. The K-reduction therefore accumulates in
// int32; scale_A (one value per M row) and scale_B (one value per N
// channel, matching the WxA16 convention) are applied once each in the
// epilogue as scale_A[m]*scale_B[n]*acc[m,n], then rounded to fp16.
//
// A's per-row scale (rather than a single scalar for all of A) is what the
// caller means by "per tensor quantization" with "scales of size M": each
// row of A (i.e. each token/M-index) gets its own scale, same shape
// convention as B's existing per-channel scales.
//
// Tile geometry: an int4 MMA_K=64 chunk is 32 bytes/row -- identical to
// fp16's MMA_K=16 chunk (16 halfs x 2B) -- so with BLOCK_K counted in int4
// elements and set 4x the fp16 kernel's BLOCK_K (256 here vs. 64 there),
// every row-byte-width, swizzle_better<> pattern, and k*32 XOR address
// trick used by matmul_kernel_data_parallel's fp16 A/B tiles carries over
// unchanged for BOTH operands here (unlike the WxA16 kernels (asrq_4x16.cu, asrq_2x16.cu), which only
// reuse this geometry for A -- B there skips ldmatrix entirely). Only the
// loader (global_to_shared_async_bytes_swizzled instead of the __half
// loader) and the MMA op (s4 x s4 -> s32 instead of f16 x f16 -> f32)
// differ from matmul_kernel_data_parallel (asrq_16x16.cu).
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a4(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                         const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    constexpr int MMA_K = 64;
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
    constexpr int A_size = BLOCK_M * (BLOCK_K / 2);
    constexpr int B_size = BLOCK_N * (BLOCK_K / 2);
    constexpr int AB_size = A_size + B_size;
    const uint32_t A_shm = shm_u32;
    const uint32_t B_shm = A_shm + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const uint8_t* Ap = Aq + off_m * (K / 2);
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K / 2>(A_offm, lane_id / 16);
        const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
        const uint32_t B_shm_thread = B_shm + swizzle_better<BLOCK_K / 2>(B_offn, (lane_id % 16) / 8);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k_bytes = std::min(BLOCK_K, K - k_iter * BLOCK_K) / 2;
            global_to_shared_async_bytes_swizzled<TB_SIZE, BLOCK_M, BLOCK_K / 2>(Ap, K / 2, A_shm + stage_id * AB_size, tid, valid_height, valid_k_bytes);
            global_to_shared_async_bytes_swizzled<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, B_shm + stage_id * AB_size, tid, valid_width, valid_k_bytes);
            Ap += BLOCK_K / 2;
            Bp += BLOCK_K / 2;
            cp_async_commit_group();
        };

        int32_t acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        auto compute = [&](int k_iter){
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                   A_addr += m * MMA_M * (BLOCK_K / 2);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int n=0; n<NUM_MMA_N; n+=2){
                    uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                    B_addr += n * MMA_N * (BLOCK_K / 2);
                    ldmatrix_x4(B_reg[k][n], B_addr ^(k * 32));
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k64_s4(A_reg[k][m], B_reg[k][n], acc[m][n]);
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
                int32_t *regs = acc[m][n];
                const int m0 = off_m + local_row;
                const int m1 = off_m + local_row + 8;
                const int n0 = off_n + local_col;
                const int n1 = off_n + local_col + 1;
                const float sa0 = (m0 < M) ? __half2float(scales_A[m0]) : 0.0f;
                const float sa1 = (m1 < M) ? __half2float(scales_A[m1]) : 0.0f;
                const float sb0 = (n0 < N) ? __half2float(scales_B[n0]) : 0.0f;
                const float sb1 = (n1 < N) ? __half2float(scales_B[n1]) : 0.0f;
                const float bias0 = (bias && n0 < N) ? __half2float(bias[n0]) : 0.0f;
                const float bias1 = (bias && n1 < N) ? __half2float(bias[n1]) : 0.0f;
                __half* p0 = Cp + row * N + col;
                __half* p1 = Cp + (row + 8) * N + col;
                if (local_row < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p0) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p0)[0] = __float22half2_rn(make_float2(regs[0] * sa0 * sb0 + bias0, regs[1] * sa0 * sb1 + bias1));
                        } else {
                            p0[0] = __float2half_rn(regs[0] * sa0 * sb0 + bias0);
                            p0[1] = __float2half_rn(regs[1] * sa0 * sb1 + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p0[0] = __float2half_rn(regs[0] * sa0 * sb0 + bias0);
                    }
                }
                if (local_row + 8 < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p1) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p1)[0] = __float22half2_rn(make_float2(regs[2] * sa1 * sb0 + bias0, regs[3] * sa1 * sb1 + bias1));
                        } else {
                            p1[0] = __float2half_rn(regs[2] * sa1 * sb0 + bias0);
                            p1[1] = __float2half_rn(regs[3] * sa1 * sb1 + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p1[0] = __float2half_rn(regs[2] * sa1 * sb0 + bias0);
                    }
                }
            }
        }
    }
}

constexpr int W4A4_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w4a4_launch(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                  int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int shm_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2) * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4A4_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w4a4<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a4_matmul_launcher(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                           const int M, const int N, const int K, cudaStream_t stream){
    // BLOCK_K = 256 int4 elements (128 bytes/row packed) for every tier, to
    // match the row-byte-width (and hence swizzle_better<> threshold/
    // pattern) that matmul_kernel_data_parallel's fp16 tiles use at its own
    // BLOCK_K=64 -- see matmul_kernel_w4a4's comment. NUM_STAGES is kept
    // lower than the WxA16 kernels' at the same BLOCK_M tier because both A
    // and B tiles are cached here (the WxA16 kernels only cache a
    // dequant-free packed B, and A is fp16 -- both smaller than caching an
    // ldmatrix-ready packed tile of both operands), so shared memory per
    // stage is larger.
    if (M <= 16)
        w4a4_launch<16, 64, 256, 1, 4, 6>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a4_launch<64, 64, 256, 2, 4, 4>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else
        w4a4_launch<128, 64, 256, 4, 4, 3>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}

// W4A4 with GROUPWISE weight quantization: B's scale now varies every
// GROUP_SIZE=128 k-values (one scale per (channel, group), scales_B shape
// (N, K/128)) instead of being one scalar per output channel; A stays
// per-token (one scale per M row), unchanged from matmul_kernel_w4a4.
// matmul_kernel_w4a4's BLOCK_K=256 spans TWO 128-wide groups, so this
// kernel instead uses BLOCK_K=128==GROUP_SIZE (halving BLOCK_K also halves
// NUM_MMA_K, from 4 down to 2, since MMA_K=64 is unchanged) so that every
// outer k_iter of the main loop covers exactly one group -- same trick
// matmul_kernel_w4a8_group (asrq_4x8.cu) uses to fold a groupwise B into an
// otherwise int32-accumulating tensor-core MMA: reset the int32 accumulator
// after every k_iter and fold that group's contribution into a float
// accumulator scaled by that group's scale_B, instead of scaling the whole
// K-reduction once in the epilogue.
constexpr int W4A4_GROUP_SIZE = 128;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, int GROUP_SIZE = W4A4_GROUP_SIZE>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a4_group(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                               const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    static_assert(GROUP_SIZE == BLOCK_K, "W4A4 groupwise kernel requires BLOCK_K == GROUP_SIZE (one outer k_iter per group)");
    constexpr int MMA_K = 64;
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
    constexpr int A_size = BLOCK_M * (BLOCK_K / 2);
    constexpr int B_size = BLOCK_N * (BLOCK_K / 2);
    constexpr int AB_size = A_size + B_size;
    const uint32_t A_shm = shm_u32;
    const uint32_t B_shm = A_shm + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const uint8_t* Ap = Aq + off_m * (K / 2);
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K / 2>(A_offm, lane_id / 16);
        const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
        const uint32_t B_shm_thread = B_shm + swizzle_better<BLOCK_K / 2>(B_offn, (lane_id % 16) / 8);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k_bytes = std::min(BLOCK_K, K - k_iter * BLOCK_K) / 2;
            global_to_shared_async_bytes_swizzled<TB_SIZE, BLOCK_M, BLOCK_K / 2>(Ap, K / 2, A_shm + stage_id * AB_size, tid, valid_height, valid_k_bytes);
            global_to_shared_async_bytes_swizzled<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, B_shm + stage_id * AB_size, tid, valid_width, valid_k_bytes);
            Ap += BLOCK_K / 2;
            Bp += BLOCK_K / 2;
            cp_async_commit_group();
        };

        int32_t acc[NUM_MMA_M][NUM_MMA_N][4] = {};
        float fp_acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        auto compute = [&](int k_iter){
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                   A_addr += m * MMA_M * (BLOCK_K / 2);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int n=0; n<NUM_MMA_N; n+=2){
                    uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                    B_addr += n * MMA_N * (BLOCK_K / 2);
                    ldmatrix_x4(B_reg[k][n], B_addr ^(k * 32));
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k64_s4(A_reg[k][m], B_reg[k][n], acc[m][n]);
                    }
                }
            }
        };

        // See matmul_kernel_w4a8_group's identical helper for why this is
        // needed (scale_B is no longer constant across all of K).
        auto accumulate_group = [&](int group_id){
            for(int m=0; m<NUM_MMA_M; m++){
                for(int n=0; n<NUM_MMA_N; n++){
                    const int col = n * MMA_N + (lane_id % 4) * 2;
                    const int local_col = warp_id_n * WARP_N + col;
                    const int n0 = off_n + local_col;
                    const int n1 = off_n + local_col + 1;
                    const float sb0 = (n0 < N) ? __half2float(scales_B[(size_t)n0 * NUM_GROUPS + group_id]) : 0.0f;
                    const float sb1 = (n1 < N) ? __half2float(scales_B[(size_t)n1 * NUM_GROUPS + group_id]) : 0.0f;
                    int32_t* regs = acc[m][n];
                    fp_acc[m][n][0] += regs[0] * sb0;
                    fp_acc[m][n][1] += regs[1] * sb1;
                    fp_acc[m][n][2] += regs[2] * sb0;
                    fp_acc[m][n][3] += regs[3] * sb1;
                    regs[0] = regs[1] = regs[2] = regs[3] = 0;
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
            accumulate_group(k);
        }

        for(int k = std::max(0, NUM_BLOCK_K - (NUM_STAGES - 1)); k < NUM_BLOCK_K; k++){
            __syncthreads();
            cp_async_wait_all();
            __syncthreads();
            compute(k);
            accumulate_group(k);
        }
        __syncthreads();

        for(int m=0; m<NUM_MMA_M; m++){
            for(int n=0; n<NUM_MMA_N; n++){
                const int row = m * MMA_M + (lane_id / 4);
                const int col = n * MMA_N + (lane_id % 4) * 2;
                const int local_row = warp_id_m * WARP_M + row;
                const int local_col = warp_id_n * WARP_N + col;
                float *regs = fp_acc[m][n];
                const int m0 = off_m + local_row;
                const int m1 = off_m + local_row + 8;
                const int n0 = off_n + local_col;
                const int n1 = off_n + local_col + 1;
                const float sa0 = (m0 < M) ? __half2float(scales_A[m0]) : 0.0f;
                const float sa1 = (m1 < M) ? __half2float(scales_A[m1]) : 0.0f;
                const float bias0 = (bias && n0 < N) ? __half2float(bias[n0]) : 0.0f;
                const float bias1 = (bias && n1 < N) ? __half2float(bias[n1]) : 0.0f;
                __half* p0 = Cp + row * N + col;
                __half* p1 = Cp + (row + 8) * N + col;
                if (local_row < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p0) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p0)[0] = __float22half2_rn(make_float2(regs[0] * sa0 + bias0, regs[1] * sa0 + bias1));
                        } else {
                            p0[0] = __float2half_rn(regs[0] * sa0 + bias0);
                            p0[1] = __float2half_rn(regs[1] * sa0 + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p0[0] = __float2half_rn(regs[0] * sa0 + bias0);
                    }
                }
                if (local_row + 8 < valid_height) {
                    if (local_col + 1 < valid_width) {
                        if ((reinterpret_cast<uintptr_t>(p1) & 0x3) == 0) {
                            reinterpret_cast<__half2*>(p1)[0] = __float22half2_rn(make_float2(regs[2] * sa1 + bias0, regs[3] * sa1 + bias1));
                        } else {
                            p1[0] = __float2half_rn(regs[2] * sa1 + bias0);
                            p1[1] = __float2half_rn(regs[3] * sa1 + bias1);
                        }
                    } else if (local_col < valid_width) {
                        p1[0] = __float2half_rn(regs[2] * sa1 + bias0);
                    }
                }
            }
        }
    }
}

constexpr int W4A4_GROUP_M_GROUPWISE = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w4a4_group_launch(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                        int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int shm_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2) * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4A4_GROUP_M_GROUPWISE, NUM_SMS});

    auto kernel = matmul_kernel_w4a4_group<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a4_group_matmul_launcher(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                                 const int M, const int N, const int K, cudaStream_t stream){
    // BLOCK_K=128==GROUP_SIZE for every tier (half of w4a4_matmul_launcher's
    // BLOCK_K=256 -- see matmul_kernel_w4a4_group's top comment for why),
    // NUM_STAGES doubled at each tier relative to w4a4_matmul_launcher to
    // keep roughly the same total shared-memory footprint / pipeline depth
    // now that each stage is half the size.
    if (M <= 16)
        w4a4_group_launch<16, 64, 128, 1, 4, 12>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a4_group_launch<64, 64, 128, 2, 4, 8>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else
        w4a4_group_launch<128, 64, 128, 4, 4, 6>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}
