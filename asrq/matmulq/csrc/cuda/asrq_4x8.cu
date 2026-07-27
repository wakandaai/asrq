#include "common.h"
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>

// W4A8: int8 activations x per-channel-quantized packed-int4 weights, run
// through the hardware int8 tensor core (mma_m16n8k32_s8, common.h). Unlike
// the WxA16 kernels (asrq_4x16.cu, asrq_2x16.cu), which dequantize B
// straight into fp16 registers for an f16 MMA, and unlike matmul_kernel_w4a4
// (asrq_4x4.cu), which needs no unpacking at all since both operands are
// already int4, here B has to become actual int8 VALUES before it can feed
// an int8 tensor core -- there is no "int4 x int8" MMA instruction. B is
// unpacked register-resident, straight out of the packed cp.async staging
// buffer, exactly like the WxA16 kernels' B handling -- no separate
// unpacked-int8 tile in shared memory at all, unlike an earlier version of
// this kernel which materialized one and fed it through ldmatrix.
//
// The per-thread (row, col) mapping for mma.sync.m16n8k32's B operand
// (col-major, 32 rows/K x 8 cols/N) is the natural doubling of
// mma.sync.m16n8k16's f16 B-fragment mapping (tau=lane%4, g=lane/4, k-offsets
// {2*tau, 2*tau+1, 2*tau+8, 2*tau+9}, used by the WxA16 kernels): int8 packs
// 4 elements/register instead of fp16's 2, and K is 32 instead of 16, so
// each of the 2 registers holds 4 k-values instead of 2, and the "second
// half" offset doubles from 8 to 16. Thread lane_id owns k-offsets
// {4*tau, 4*tau+1, 4*tau+2, 4*tau+3} (register b0) and
// {4*tau+16, ..., 4*tau+19} (register b1) at column g = lane_id/4. Each
// register's 4 k-values fall into exactly 2 adjacent packed bytes (byte
// column 2*tau and 2*tau+1 for b0; 2*tau+8 and 2*tau+9 for b1), so building
// both registers costs 4 shared-memory byte reads + 4 cheap sign-extends
// per (k,n) MMA tile.
//
// A is int8, plain two's-complement bytes (not packed -- int8 already is
// the tensor-core's native element size), quantized with one scale per row
// (per M / per token, matching W4A4's convention). B is packed int4, two
// two's-complement nibbles per byte (low nibble = even k, high nibble = odd
// k) -- NOT excess-8 encoded like the WxA16 kernels' B, since there's no
// LOP3 fp16-dequant trick here either: unpacking a nibble just sign-extends
// it to a full int8 value directly. B has one scale per output channel (per
// N), matching the WxA16/W4A4 convention. The K-reduction accumulates in
// int32; scale_A[m]*scale_B[n] is applied once per output element in the
// epilogue, then rounded to fp16 -- identical structure to matmul_kernel_w4a4.
__device__ __forceinline__
void unpack_s4_pair_to_s8(uint8_t byte, int8_t &lo, int8_t &hi) {
    lo = static_cast<int8_t>(byte << 4) >> 4;
    hi = static_cast<int8_t>(byte) >> 4;
}
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a8(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                         const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    constexpr int MMA_K = 32;
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
    // A: int8, 1 byte/elem, swizzled (ldmatrix-ready, same geometry as the
    // fp16/W4A4 kernels' A tile since a BLOCK_K=128 row is 128 bytes either
    // way). Bq_packed: raw packed-nibble staging, unswizzled -- read
    // straight out of by compute() below, never through ldmatrix.
    constexpr int A_size = BLOCK_M * BLOCK_K;
    constexpr int Bq_packed_size = BLOCK_N * (BLOCK_K / 2);
    constexpr int stage_size = A_size + Bq_packed_size;
    const uint32_t A_shm = shm_u32;
    const uint32_t Bq_packed_shm = A_shm + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const uint8_t* Ap = reinterpret_cast<const uint8_t*>(Aq) + off_m * K;
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async_bytes_swizzled<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, Bq_packed_shm + stage_id * stage_size, tid, valid_width, valid_k / 2);
            Ap += BLOCK_K;
            Bp += BLOCK_K / 2;
            cp_async_commit_group();
        };

        int32_t acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;

        auto compute = [&](int k_iter){
            const uint32_t A_stage_base = A_shm_thread + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_stage_base + m * MMA_M * BLOCK_K;
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            const uint32_t Bq_stage_base = Bq_packed_shm + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int n=0; n<NUM_MMA_N; n++){
                    const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                    const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 2) + k * (MMA_K / 2);
                    const uint32_t addr_b0_lo = row_base + 2 * tau;
                    const uint32_t addr_b0_hi = addr_b0_lo + 1;
                    const uint32_t addr_b1_lo = addr_b0_lo + 8;
                    const uint32_t addr_b1_hi = addr_b0_lo + 9;
                    uint8_t byte_b0_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b0_lo)));
                    uint8_t byte_b0_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b0_hi)));
                    uint8_t byte_b1_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b1_lo)));
                    uint8_t byte_b1_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b1_hi)));

                    int8_t b0_0, b0_1, b0_2, b0_3;
                    unpack_s4_pair_to_s8(byte_b0_lo, b0_0, b0_1);
                    unpack_s4_pair_to_s8(byte_b0_hi, b0_2, b0_3);
                    int8_t b1_0, b1_1, b1_2, b1_3;
                    unpack_s4_pair_to_s8(byte_b1_lo, b1_0, b1_1);
                    unpack_s4_pair_to_s8(byte_b1_hi, b1_2, b1_3);

                    B_reg[k][n][0] = static_cast<uint32_t>(static_cast<uint8_t>(b0_0))
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b0_1)) << 8)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b0_2)) << 16)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b0_3)) << 24);
                    B_reg[k][n][1] = static_cast<uint32_t>(static_cast<uint8_t>(b1_0))
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b1_1)) << 8)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b1_2)) << 16)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b1_3)) << 24);
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k32_s8(A_reg[k][m], B_reg[k][n], acc[m][n]);
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

constexpr int W4A8_GROUP_M = 8;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w4a8_launch(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                  int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int shm_size = (BLOCK_M * BLOCK_K + BLOCK_N * (BLOCK_K / 2)) * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4A8_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w4a8<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a8_matmul_launcher(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                           const int M, const int N, const int K, cudaStream_t stream){
    // BLOCK_K = 128 int8 elements (128 bytes/row) for every tier, matching
    // the row-byte-width matmul_kernel_data_parallel's fp16 tiles use at its
    // own BLOCK_K=64 (16 halfs x 2B = 32B * 4 MMA_K-chunks = 128B) -- see
    // this file's top comment. NUM_STAGES can go deeper than an earlier
    // version of this kernel allowed, now that B is dequantized
    // register-resident (see matmul_kernel_w4a8's comment) instead of via a
    // separate unpacked-int8 shared-memory tile -- each stage now holds only
    // two buffers (A, packed-B staging) instead of three.
    if (M <= 16)
        w4a8_launch<16, 64, 128, 1, 4, 8>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a8_launch<64, 64, 128, 2, 4, 4>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else
        w4a8_launch<128, 64, 128, 4, 4, 3>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}

// W4A8 with GROUPWISE weight quantization: B's scale now varies every
// GROUP_SIZE=128 k-values (one scale per (channel, group), scales_B shape
// (N, K/128)) instead of being one scalar per output channel; A stays
// per-token (one scale per M row), unchanged from matmul_kernel_w4a8. Since
// BLOCK_K is fixed at 128 == GROUP_SIZE for every tier below, each outer
// k_iter of the main loop covers EXACTLY one group -- so instead of
// matmul_kernel_w4a8's "accumulate all of K in one int32 register, scale
// once in the epilogue" (only valid because scale_B is constant across the
// whole K-reduction there), this kernel resets the int32 accumulator after
// every k_iter and folds that group's contribution into a float
// accumulator (scaled by that group's scale_B), the same trick
// matmul_kernel_w4a16_group (asrq_4x16.cu) uses for its own groupwise
// dequant, adapted to an int32-accumulating tensor-core MMA instead of a
// register-resident fp16 dequant.
constexpr int W4A8_GROUP_SIZE = 128;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES, int GROUP_SIZE = W4A8_GROUP_SIZE>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_w4a8_group(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                               const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
    static_assert(GROUP_SIZE == BLOCK_K, "W4A8 groupwise kernel requires BLOCK_K == GROUP_SIZE (one outer k_iter per group)");
    constexpr int MMA_K = 32;
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
    constexpr int A_size = BLOCK_M * BLOCK_K;
    constexpr int Bq_packed_size = BLOCK_N * (BLOCK_K / 2);
    constexpr int stage_size = A_size + Bq_packed_size;
    const uint32_t A_shm = shm_u32;
    const uint32_t Bq_packed_shm = A_shm + A_size;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    for (int tile = blockIdx.x; tile < NUM_TILES; tile += gridDim.x) {
        int block_m, block_n;
        tile_scheduler_l2(tile, NUM_BLOCK_M, NUM_BLOCK_N, GROUP_M, block_m, block_n);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const uint8_t* Ap = reinterpret_cast<const uint8_t*>(Aq) + off_m * K;
        const uint8_t* Bp = Bq + off_n * (K / 2);
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K>(A_offm, lane_id / 16);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async_bytes_swizzled<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm + stage_id * stage_size, tid, valid_height, valid_k);
            global_to_shared_async_bytes<TB_SIZE, BLOCK_N, BLOCK_K / 2>(Bp, K / 2, Bq_packed_shm + stage_id * stage_size, tid, valid_width, valid_k / 2);
            Ap += BLOCK_K;
            Bp += BLOCK_K / 2;
            cp_async_commit_group();
        };

        int32_t acc[NUM_MMA_M][NUM_MMA_N][4] = {};
        float fp_acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        const int tau = lane_id % 4;
        const int g = lane_id / 4;

        auto compute = [&](int k_iter){
            const uint32_t A_stage_base = A_shm_thread + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_stage_base + m * MMA_M * BLOCK_K;
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            const uint32_t Bq_stage_base = Bq_packed_shm + (k_iter % NUM_STAGES) * stage_size;
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int n=0; n<NUM_MMA_N; n++){
                    const int n_col = warp_id_n * WARP_N + n * MMA_N + g;
                    const uint32_t row_base = Bq_stage_base + n_col * (BLOCK_K / 2) + k * (MMA_K / 2);
                    const uint32_t addr_b0_lo = row_base + 2 * tau;
                    const uint32_t addr_b0_hi = addr_b0_lo + 1;
                    const uint32_t addr_b1_lo = addr_b0_lo + 8;
                    const uint32_t addr_b1_hi = addr_b0_lo + 9;
                    uint8_t byte_b0_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b0_lo)));
                    uint8_t byte_b0_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b0_hi)));
                    uint8_t byte_b1_lo = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b1_lo)));
                    uint8_t byte_b1_hi = *reinterpret_cast<uint8_t*>(__cvta_shared_to_generic(static_cast<size_t>(addr_b1_hi)));

                    int8_t b0_0, b0_1, b0_2, b0_3;
                    unpack_s4_pair_to_s8(byte_b0_lo, b0_0, b0_1);
                    unpack_s4_pair_to_s8(byte_b0_hi, b0_2, b0_3);
                    int8_t b1_0, b1_1, b1_2, b1_3;
                    unpack_s4_pair_to_s8(byte_b1_lo, b1_0, b1_1);
                    unpack_s4_pair_to_s8(byte_b1_hi, b1_2, b1_3);

                    B_reg[k][n][0] = static_cast<uint32_t>(static_cast<uint8_t>(b0_0))
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b0_1)) << 8)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b0_2)) << 16)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b0_3)) << 24);
                    B_reg[k][n][1] = static_cast<uint32_t>(static_cast<uint8_t>(b1_0))
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b1_1)) << 8)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b1_2)) << 16)
                                   | (static_cast<uint32_t>(static_cast<uint8_t>(b1_3)) << 24);
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k32_s8(A_reg[k][m], B_reg[k][n], acc[m][n]);
                    }
                }
            }
        };

        // Folds the just-computed group's int32 acc into fp_acc (scaled by
        // that group's per-(channel,group) scale_B), then resets acc so the
        // next group's MMAs start from zero -- see this kernel's top
        // comment for why (scale_B is no longer constant across all of K).
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

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void w4a8_group_launch(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                        int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int shm_size = (BLOCK_M * BLOCK_K + BLOCK_N * (BLOCK_K / 2)) * NUM_STAGES;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);
    const int GROUP_M = std::min({NUM_BLOCK_M, W4A8_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_w4a8_group<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void w4a8_group_matmul_launcher(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B, const __half* bias, __half* C,
                                 const int M, const int N, const int K, cudaStream_t stream){
    // Same BLOCK_K=128=GROUP_SIZE tiers as w4a8_matmul_launcher (see
    // matmul_kernel_w4a8_group's top comment for why BLOCK_K must equal
    // GROUP_SIZE here).
    if (M <= 16)
        w4a8_group_launch<16, 64, 128, 1, 4, 8>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else if (M <= 64)
        w4a8_group_launch<64, 64, 128, 2, 4, 4>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);
    else
        w4a8_group_launch<128, 64, 128, 4, 4, 3>(Aq, Bq, scales_A, scales_B, bias, C, M, N, K, stream);

    CUDA_CHECK(cudaGetLastError());
}
