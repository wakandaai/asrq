#include "common.h"
#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>


constexpr int MMA_M = 16;
constexpr int MMA_N = 8;

// constexpr int MMA_M = 1;
// constexpr int MMA_N = 1;

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do
      // Guarantee that subsequent writes by this threadblock will be visible globally.
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible globally, while releasing the barrier.
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val));
  }
}

__device__
void tile_scheduler(int tile_idx, int num_block_m, int num_block_n, int& block_m, int& block_n) {
    block_m = tile_idx / num_block_n;
    block_n = tile_idx % num_block_n;
}

template<int TB_SIZE, int HEIGHT, int WIDTH>
__device__ static
void global_to_shared_async(const __half* in, int in_stride, uint32_t out, int tid, int valid_height=HEIGHT){
    constexpr int num_elems = 16 / sizeof(__half);
    // Total 16-byte vectors needed to cover the tile, spread over num_iters
    // passes of TB_SIZE threads each. Using cdiv (not a plain divide) matters
    // whenever the tile is smaller than one full pass -- e.g. an 8x16 B tile
    // is only 16 vectors, less than a 32-thread block's single pass of 32,
    // so a plain "/" truncates to 0 iterations and the tile is never loaded.
    constexpr int total_vecs = (HEIGHT * WIDTH) / num_elems;
    constexpr int num_iters = cdiv(total_vecs, TB_SIZE);

    for(int iter=0; iter<num_iters; iter++){
        const int vec_idx = iter * TB_SIZE + tid;
        if (vec_idx >= total_vecs) continue;
        const int idx = vec_idx * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        uint32_t dst_addr = out + swizzle_better<WIDTH * sizeof(__half)>(row, col / num_elems);
        const bool valid = row < valid_height;
        const __half *src_ptr = valid ? (in + row * in_stride + col) : in;
        cp_async_pred(dst_addr, src_ptr, valid);
    }
}


template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N, int NUM_STAGES>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__
void matmul_kernel(const __half* A, const __half* B, __half* C, float* workspace, int* locks, const int M, const int N, const int K, const int NUM_TILES, const int KS_FSM) {
    constexpr int MMA_K = 16;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_BLOCK_N = cdiv(N, BLOCK_N);
    const int NUM_BLOCK_K = cdiv(K, BLOCK_K);
    const int SM_ID = blockIdx.x;
    const int NUM_SMS = gridDim.x;

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

    // Flattened (tile, k-block) index space, split into NUM_SMS contiguous
    // chunks of KS_FSM iterations each. A chunk may span a partial tile at
    // either end -- that's the whole point of stream-K.
    int processed_k = 0;
    int total_k = (NUM_TILES * NUM_BLOCK_K);
    int start_k = (blockIdx.x * KS_FSM);
    int endk = std::min(start_k + KS_FSM, total_k);
    int k = start_k;

    // end_tile must be derived from this SM's last INCLUDED index (endk-1),
    // not from the exclusive upper bound -- otherwise an SM whose range ends
    // exactly on a tile boundary (endk a multiple of NUM_BLOCK_K) spuriously
    // enters one extra tile iteration with zero assigned work in it, and
    // then wrongly qualifies as that tile's "finishing SM".
    const int start_tile = start_k / NUM_BLOCK_K;
    const int end_tile = (endk - 1) / NUM_BLOCK_K;

    uint32_t A_reg[NUM_MMA_K][NUM_MMA_M][4];
    uint32_t B_reg[NUM_MMA_K][NUM_MMA_N][2];

    // Max number of SMs that can ever contribute to a single tile: the tile's
    // own k-splits, plus one extra for the case where the tile isn't aligned
    // to a KS_FSM boundary.
    const int MAX_SMS_PER_TILE = cdiv(NUM_BLOCK_K, KS_FSM) + 1;
    bool preloaded = false;

    for(int tile=start_tile; (tile<=end_tile && tile < NUM_TILES); tile++){
        int block_m, block_n;
        tile_scheduler(tile, NUM_BLOCK_M, NUM_BLOCK_N, block_m, block_n);

        // Which SMs (in blockIdx.x space) touch this tile, and where this SM
        // falls in that sequence. Contributions arrive in blockIdx.x order,
        // so `local_slot` is also the workspace slot this SM writes its
        // partial to -- the SM at first_sm_for_tile writes slot 0, the next
        // writes slot 1, etc. The SM at last_sm_for_tile never writes a
        // partial: it's the one that finishes the tile and reduces.
        const int tile_k_start = tile * NUM_BLOCK_K;
        const int tile_k_end = tile_k_start + NUM_BLOCK_K;
        const int first_sm_for_tile = tile_k_start / KS_FSM;
        const int last_sm_for_tile = (tile_k_end - 1) / KS_FSM;
        const int num_sms_for_tile = last_sm_for_tile - first_sm_for_tile + 1;
        const int local_slot = SM_ID - first_sm_for_tile;
        int* tile_lock = locks + tile;

        // Never prefetch past this SM's own assigned range, or past the end
        // of the current tile -- Ap/Bp are only valid within this tile.
        const int k_upper = std::min(endk, tile_k_end);

        const int off_m = block_m * BLOCK_M;
        const int off_n = block_n * BLOCK_N;
        const __half* Ap = A + off_m * K + (start_k % NUM_BLOCK_K) * BLOCK_K;
        const __half* Bp = B + off_n * K + (start_k % NUM_BLOCK_K) * BLOCK_K;
        __half *Cp = C + (off_m + warp_id_m * WARP_M) * N + (off_n + warp_id_n * WARP_N);

        const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
        const uint32_t A_shm_thread = A_shm + swizzle_better<BLOCK_K * sizeof(__half)>(A_offm, lane_id / 16);
        const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8) + (lane_id / 16) * 8;
        const uint32_t B_shm_thread = B_shm + swizzle_better<BLOCK_K * sizeof(__half)>(B_offn, (lane_id % 16)/8);

        // Load A and B tiles into shared memory, perform computation, and write results to C
        // Load A and B tiles into shared memory:
        const int valid_height = std::min(BLOCK_M, M - off_m);
        const int valid_width = std::min(BLOCK_N, N - off_n);
        // `do_load=false` still commits an (empty) cp.async group: wait_group's
        // "N groups remain pending" counter has to keep advancing every stage
        // regardless of whether there's real work left to fetch, or a later
        // wait_group<NUM_STAGES-1>() can be satisfied by groups that were
        // never even issued -- i.e. it returns without actually waiting for
        // the in-flight copy the upcoming compute() depends on.
        auto load_AB = [&](int k_iter, bool do_load){
            if (do_load) {
                const int stage_id = k_iter % NUM_STAGES;
                global_to_shared_async<TB_SIZE, BLOCK_M, BLOCK_K>(Ap, K, A_shm + stage_id * AB_size, tid, valid_height);
                global_to_shared_async<TB_SIZE, BLOCK_N, BLOCK_K>(Bp, K, B_shm + stage_id * AB_size, tid);

                Ap += BLOCK_K;
                Bp += BLOCK_K;
            }
            cp_async_commit_group();
        };

        // Fresh accumulator for every tile -- a single SM can walk through
        // several tiles in this loop, and each one must start from zero.
        float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

        // Computation
        auto compute = [&](int k_iter){
            for(int k = 0; k < NUM_MMA_K; k++){
                for(int m=0; m<NUM_MMA_M; m++){
                   uint32_t A_addr = A_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                   A_addr += m * MMA_M * BLOCK_K * sizeof(__half);
                   ldmatrix_x4(A_reg[k][m], A_addr ^(k * 32));
                }
            
                for(int n=0; n<NUM_MMA_N; n++){
                    uint32_t B_addr = B_shm_thread + (k_iter % NUM_STAGES) * AB_size;
                    B_addr += n * MMA_N * BLOCK_K * sizeof(__half);
                    ldmatrix_x2(B_reg[k][n], B_addr ^(k * 32));
                }
            
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        mma_m16n8k16(A_reg[k][m], B_reg[k][n], acc[m][n]);
                    }
                }
            }
        };

        // Pipeline fill for every tile -- Ap/Bp just got rebased to this
        // tile's own A/B block, so the shared-memory stages have to be
        // refilled from scratch regardless of whether this is the SM's
        // first tile or a later one.
        
        // if(!preloaded)
        for(int stage=k; (stage < k + NUM_STAGES - 1); stage++){
            load_AB(stage, stage < k_upper);
        }
        // preloaded = true;

        for(; !(start_k != k && k % NUM_BLOCK_K == 0 ) && (processed_k < KS_FSM); k++, processed_k++) {
            // pre-load AB tile for one future stage, if there's still one left to fetch
            const int k_to_preload = k + NUM_STAGES - 1;
            load_AB(k_to_preload, k_to_preload < k_upper);
            cp_async_wait_group<NUM_STAGES - 1>();

            // Perform computation:
            compute(k);

        }
        start_k = k;

        // Write results to C:
        if(k % NUM_BLOCK_K == 0){ // final reduction
            barrier_acquire(tile_lock, num_sms_for_tile - 1);
            for(int s=0; s<num_sms_for_tile-1; s++){
                for(int m=0; m<NUM_MMA_M; m++){
                    for(int n=0; n<NUM_MMA_N; n++){
                        // Every term in this offset (MMA_M*BLOCK_N, MMA_N,
                        // lane_id*4) is a multiple of 4 floats, and workspace
                        // itself is at least 256-byte aligned, so this is
                        // always 16-byte aligned: one float4 load instead of
                        // four scalar float loads.
                        const float4 partial = reinterpret_cast<const float4*>(
                            workspace + (tile * (MAX_SMS_PER_TILE * BLOCK_M * BLOCK_N)) + s * (BLOCK_M * BLOCK_N) + (warp_id_m * WARP_M) * BLOCK_N + (warp_id_n * WARP_N) + m * MMA_M * BLOCK_N + n * MMA_N + lane_id*4
                        )[0];
                        acc[m][n][0] += partial.x;
                        acc[m][n][1] += partial.y;
                        acc[m][n][2] += partial.z;
                        acc[m][n][3] += partial.w;
                    }
                }
            }
            for(int m=0; m<NUM_MMA_M; m++){
                for(int n=0; n<NUM_MMA_N; n++){
                    const int row = m * MMA_M + (lane_id / 4);
                    const int col = n * MMA_N + (lane_id % 4) * 2;
                    const int local_row = warp_id_m * WARP_M + row;
                    const int local_col = warp_id_n * WARP_N + col;
                    float *regs = acc[m][n];
                    // The vectorized half2 store needs a 4-byte-aligned address.
                    // col is always even, but row*N can land on an odd half
                    // when N is odd, shifting alignment row-to-row -- check
                    // the actual pointer rather than assume N is even.
                    __half* p0 = Cp + row * N + col;
                    __half* p1 = Cp + (row + 8) * N + col;
                    if (local_row < valid_height) {
                        if (local_col + 1 < valid_width) {
                            if ((reinterpret_cast<uintptr_t>(p0) & 0x3) == 0) {
                                reinterpret_cast<__half2*>(p0)[0] = __float22half2_rn(make_float2(regs[0], regs[1]));
                            } else {
                                p0[0] = __float2half_rn(regs[0]);
                                p0[1] = __float2half_rn(regs[1]);
                            }
                        } else if (local_col < valid_width) {
                            p0[0] = __float2half_rn(regs[0]);
                        }
                    }
                    if (local_row + 8 < valid_height) {
                        if (local_col + 1 < valid_width) {
                            if ((reinterpret_cast<uintptr_t>(p1) & 0x3) == 0) {
                                reinterpret_cast<__half2*>(p1)[0] = __float22half2_rn(make_float2(regs[2], regs[3]));
                            } else {
                                p1[0] = __float2half_rn(regs[2]);
                                p1[1] = __float2half_rn(regs[3]);
                            }
                        } else if (local_col < valid_width) {
                            p1[0] = __float2half_rn(regs[2]);
                        }
                    }
                }
            }

            // Reset the lock so the same workspace/locks buffers can be reused
            // by a later launch without a CPU-side memset.
            barrier_release(tile_lock, /*reset=*/true);

        }
        else{ // partial reduction
            // TILE, SM, BLOCK_M, BLOCK_N,
            float* workspace_ptr = workspace + tile * (MAX_SMS_PER_TILE * BLOCK_M * BLOCK_N) + local_slot * (BLOCK_M * BLOCK_N) + (warp_id_m * WARP_M) * BLOCK_N + (warp_id_n * WARP_N);
            for(int i=0; i<NUM_MMA_M; i++){
                for(int j=0; j<NUM_MMA_N; j++){
                    // Same 16-byte-aligned offset as the read side above:
                    // one float4 store instead of four scalar float stores.
                    // Built from acc's scalars rather than reinterpreting
                    // acc[i][j] itself, since a plain float[4] has no
                    // alignas(16) guarantee to make that cast safe.
                    float4* ws4 = reinterpret_cast<float4*>(workspace_ptr + i * MMA_M * WARP_N + j * MMA_N + lane_id*4);
                    *ws4 = make_float4(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                }
            }

            // release
            barrier_release(tile_lock);
        }
    }
}


int num_sms() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.multiProcessorCount;
}


// Tile shape + launch-size derivation, shared verbatim between the actual
// launcher and mymatmul_workspace_sizes(). These two MUST agree on
// workspace_size/locks_size -- the caller allocates workspace/locks from
// the latter, and the kernel indexes into them using the former's NUM_SMS
// (which depends on a runtime occupancy query, not just NUM_TILES/NUM_BLOCK_K,
// so it can't be recomputed correctly from Python without duplicating this
// whole function). A mismatch here means an out-of-bounds workspace write
// that can go on to corrupt the locks buffer and hang the stream-K fixup
// barrier forever.
struct MatmulLaunchParams {
    int BLOCK_M, BLOCK_N, BLOCK_K;
    int TB_SIZE, shm_size;
    int NUM_BLOCK_M, NUM_BLOCK_N, NUM_BLOCK_K;
    int NUM_SMS, NUM_TILES, KS_FSM, MAX_SMS_PER_TILE;
    int workspace_size, locks_size;
};

MatmulLaunchParams compute_matmul_launch_params(int M, int N, int K) {
    constexpr int BLOCK_M = 16, BLOCK_N = 8, BLOCK_K = 16;
    constexpr int NUM_WARP_M = 1, NUM_WARP_N = 1;
    constexpr int NUM_STAGES = 5;
    const int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    const int NUM_BLOCK_M = cdiv(M, BLOCK_M);
    const int NUM_BLOCK_N = cdiv(N, BLOCK_N);
    const int NUM_BLOCK_K = cdiv(K, BLOCK_K);
    const int shm_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(__half) * NUM_STAGES;

    auto kernel = matmul_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N, NUM_STAGES>;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

    // The stream-K fixup barrier (barrier_acquire/barrier_release) requires
    // every block in the grid to be resident and running concurrently: a
    // "finishing" SM spins waiting on a "partial" SM's release, and if the
    // scheduler hasn't even dispatched that block yet, it spins forever. A
    // plain <<<>>> launch does not guarantee full-grid concurrent residency
    // -- grid_size <= SM_count merely makes it *likely* in practice, not
    // guaranteed by the CUDA execution model -- so this has to go through
    // cudaLaunchCooperativeKernel, which does guarantee it (and fails the
    // launch instead of hanging if the grid can't actually fit).
    int max_active_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks_per_sm, kernel, TB_SIZE, shm_size));
    const int NUM_SMS = std::min(num_sms(), max_active_blocks_per_sm * num_sms());

    const int NUM_TILES = NUM_BLOCK_M * NUM_BLOCK_N;
    const int KS_FSM = cdiv(NUM_BLOCK_K * NUM_TILES, NUM_SMS);
    const int MAX_SMS_PER_TILE = cdiv(NUM_BLOCK_K, KS_FSM) + 1;

    MatmulLaunchParams p;
    p.BLOCK_M = BLOCK_M; p.BLOCK_N = BLOCK_N; p.BLOCK_K = BLOCK_K;
    p.TB_SIZE = TB_SIZE; p.shm_size = shm_size;
    p.NUM_BLOCK_M = NUM_BLOCK_M; p.NUM_BLOCK_N = NUM_BLOCK_N; p.NUM_BLOCK_K = NUM_BLOCK_K;
    p.NUM_SMS = NUM_SMS; p.NUM_TILES = NUM_TILES; p.KS_FSM = KS_FSM;
    p.MAX_SMS_PER_TILE = MAX_SMS_PER_TILE;
    p.workspace_size = NUM_TILES * MAX_SMS_PER_TILE * BLOCK_M * BLOCK_N;
    p.locks_size = NUM_TILES;
    return p;
}

// Required sizes for the workspace/locks buffers mymatmul_launcher will
// index into for this (M, N, K). Call this to allocate them -- don't
// recompute the formula in Python; see compute_matmul_launch_params().
void mymatmul_workspace_sizes(int M, int N, int K, int* workspace_size, int* locks_size) {
    MatmulLaunchParams p = compute_matmul_launch_params(M, N, K);
    *workspace_size = p.workspace_size;
    *locks_size = p.locks_size;
}

void mymatmul_launcher(const __half* A, const __half* B, __half* C, float* workspace, int* locks,const int M, const int N, const int K, cudaStream_t stream){
    MatmulLaunchParams p = compute_matmul_launch_params(M, N, K);
    auto kernel = matmul_kernel<16, 8, 16, 1, 1, 5>;
    const int NUM_SMS = p.NUM_SMS, NUM_TILES = p.NUM_TILES, KS_FSM = p.KS_FSM;

    void* kernel_args[] = {
        (void*)&A, (void*)&B, (void*)&C, (void*)&workspace, (void*)&locks,
        (void*)&M, (void*)&N, (void*)&K, (void*)&NUM_TILES, (void*)&KS_FSM,
    };
    // Must run on the caller's stream, not the default/null stream: A, B,
    // workspace, locks, and C are all produced/consumed by other work (e.g.
    // torch ops) ordered on that stream, and CUDA does not implicitly
    // synchronize across streams. Launching on stream 0 here raced the
    // kernel against whatever was still filling those buffers -- invisible
    // for small tensors (the fill finishes before the kernel gets
    // scheduled), a real silent-corruption bug for large ones (e.g. a
    // multi-hundred-MB B that takes measurably long to fill).
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel, dim3(NUM_SMS), dim3(p.TB_SIZE), kernel_args, p.shm_size, stream));
    CUDA_CHECK(cudaGetLastError());
    std::cout << "---Stats---\n";
    int key_width = 20;
    std::cout << std::left << std::setw(key_width) << "KS_FSM:" << KS_FSM << "\n";
    std::cout << std::left << std::setw(key_width) << "NUM_TILES:" << NUM_TILES << "\n";
    std::cout << std::left << std::setw(key_width) << "NUM_SMS:" << NUM_SMS << "\n";
    std::cout << std::left << std::setw(key_width) << "shm_size:" << p.shm_size << "\n";
    std::cout << std::left << std::setw(key_width) << "NUM_BLOCK_M:" << p.NUM_BLOCK_M << "\n";
    std::cout << std::left << std::setw(key_width) << "NUM_BLOCK_N:" << p.NUM_BLOCK_N << "\n";
    std::cout << std::left << std::setw(key_width) << "NUM_BLOCK_K:" << p.NUM_BLOCK_K << "\n";
    std::cout << "---------------------\n";

}
