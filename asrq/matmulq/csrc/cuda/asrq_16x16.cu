#include "common.h"
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>

// fp16 x fp16 -> fp16 data-parallel GEMM: no stream-K, no workspace/locks --
// each output tile's full K reduction is owned start-to-finish by a single
// threadblock via a grid-stride loop over tiles (tile_scheduler_l2, in
// common.h, picks the L2-friendly tile order). This is the baseline design
// the WxA16/W4A4 kernels in the other asrq_*.cu files all derive from.

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel_data_parallel(const __half* A, const __half* B, const __half* bias, __half* C, const int M, const int N, const int K, const int NUM_TILES, const int GROUP_M) {
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
    constexpr int B_size = BLOCK_N * BLOCK_K * sizeof(__half);
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
        const __half* Ap = A + off_m * K;
        const __half* Bp = B + off_n * K;
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);
        const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
        const uint32_t B_shm_thread = B_shm + swizzle_better<BLOCK_K * sizeof(__half)>(B_offn, (lane_id % 16)/8);

        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);

        auto load_AB = [&](int k_iter){
            const int stage_id = k_iter % NUM_STAGES;
            const int valid_k = std::min(BLOCK_K, K - k_iter * BLOCK_K);
            global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm + stage_id * AB_size, tid, valid_height, valid_k);
            global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(Bp, K, B_shm + stage_id * AB_size, tid, valid_width, valid_k);
            Ap += BLOCK_K;
            Bp += BLOCK_K;
            cp_async_commit_group();
        };

        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        auto compute = [&](int k_iter){
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                   A_addr += m * MMA_M * BLOCK_K * sizeof(__half);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            }
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int n=0; n<NUM_MMA_N; n+=2){
                    uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                    B_addr += n * MMA_N * BLOCK_K * sizeof(__half);
                    ldmatrix_x4(B_reg[k][n], B_addr ^(k * 32));
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


constexpr int DP_GROUP_M = 32;

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
void dp_launch(const __half* A, const __half* B, const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(__half) * NUM_STAGES;
    int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_TILES = NUM_BLOCK_M * cdiv(N, BLOCK_N);
    const int NUM_SMS = std::min(num_sms(), NUM_TILES);

    const int GROUP_M = std::min({NUM_BLOCK_M, DP_GROUP_M, NUM_SMS});

    auto kernel = matmul_kernel_data_parallel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<NUM_SMS, TB_SIZE, shm_size, stream>>>(A, B, bias, C, M, N, K, NUM_TILES, GROUP_M);
}

void mymatmul_launcher(const __half* A, const __half* B, const __half* bias, __half* C, const int M, const int N, const int K, cudaStream_t stream){
    if (M <= 16){
        if(N <= 1024)
            dp_launch<16, 64, 64, 1, 4, 4>(A, B, bias, C, M, N, K, stream);
        else
            dp_launch<16, 128, 64, 1, 4, 2>(A, B, bias, C, M, N, K, stream);
    }
    else if(M <= 64){
        if(N <= 1024)
            dp_launch<64, 64, 64, 2, 4, 4>(A, B, bias, C, M, N, K, stream);
        else
            dp_launch<64, 128, 64, 2, 4, 2>(A, B, bias, C, M, N, K, stream);
    }
    else{
        if(N <= 1024)
            dp_launch<128, 64, 64, 4, 4, 2>(A, B, bias, C, M, N, K, stream);
        else
            dp_launch<128, 128, 64, 4, 4, 2>(A, B, bias, C, M, N, K, stream);
    }

    CUDA_CHECK(cudaGetLastError());
}
