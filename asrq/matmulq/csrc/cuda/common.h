#include <cstdint>
#include <iostream>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ ":" << __LINE__ << std::endl;        \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

__host__ __device__ inline
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

constexpr int WARP_SIZE = 32;


// convert generic address (C++ address, 64-bit) to shared state space address (32-bit)
// all PTX instructions expect share memory address to be in shared state space (not 100%)
__device__ inline
uint32_t cvta_shared(const void *ptr) { return static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); }

__device__ inline
void ldmatrix_x1(uint32_t reg[1], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];"
              : "=r"(reg[0])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x2(uint32_t reg[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
              : "=r"(reg[0]), "=r"(reg[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t reg[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
              : "r"(addr));
}

__device__ inline
void mma_m16n8k16(const uint32_t A[4], const uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, "  // D
               "{%4, %5, %6, %7}, "  // A
               "{%8, %9}, "          // B
               "{%0, %1, %2, %3};"   // C
              : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]));
}

// Tried an f16-accumulate variant of this MMA (mma.sync...f16.f16.f16.f16,
// C/D packed as 2 f16x2 registers instead of 4 scalar f32) to see if
// narrower accumulator registers would speed up the fp16/W4A16/W2A16
// kernels. Measured on this GPU: no throughput gain from the MMA
// instruction itself (same or slightly *slower* at large compute-bound
// shapes, e.g. 8194 vs 8983 GFLOP/s at M=8192,N=K=4096 -- accumulate width
// doesn't change tensor-core issue rate on this architecture), while
// per-K-step rounding to fp16 measurably hurt accuracy at larger K (several
// existing correctness-sweep shapes failed that passed with f32 accumulate).
// Reverted; not worth the precision loss for zero speed benefit.

// Signed int8 tensor-core MMA (sm_80+): 16x8x32 with int32 accumulate.
// A is 16x32 row-major s8, B is 32x8 col-major s8, C/D are 16x8 s32.
// Fragment register counts match mma_m16n8k16 (A=4, B=2, C/D=4): each 32-bit
// A/B register packs 4 signed bytes, so one MMA consumes K=32 int8 = 32 B,
// the same K-byte-stride as the fp16 m16n8k16 MMA. This lets the int8 kernel
// reuse the fp16 loader/ldmatrix/swizzle geometry verbatim.
__device__ inline
void mma_m16n8k32_s8(const uint32_t A[4], const uint32_t B[2], int32_t D[4]) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0, %1, %2, %3}, "  // D
               "{%4, %5, %6, %7}, "  // A
               "{%8, %9}, "          // B
               "{%0, %1, %2, %3};"   // C
              : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]));
}

// Signed int4 tensor-core MMA (sm_80+): 16x8x64 with int32 accumulate.
// A is 16x64 row-major s4, B is 64x8 col-major s4, C/D are 16x8 s32.
// Fragment register counts match mma_m16n8k16 (A=4, B=2, C/D=4): each 32-bit
// A/B register packs 8 signed nibbles, so one MMA consumes K=64 int4 = 32 B,
// the same K-byte-stride as the fp16 m16n8k16 MMA. This is what lets the int4
// kernel reuse the fp16 loader/ldmatrix/swizzle geometry verbatim.
__device__ inline
void mma_m16n8k64_s4(const uint32_t A[4], const uint32_t B[2], int32_t D[4]) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
               "{%0, %1, %2, %3}, "  // D
               "{%4, %5, %6, %7}, "  // A
               "{%8, %9}, "          // B
               "{%0, %1, %2, %3};"   // C
              : "+r"(D[0]), "+r"(D[1]), "+r"(D[2]), "+r"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]));
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy
__device__ inline
void cp_async(uint32_t dst, const void *src) {
  // .ca means cache to L1 and L2. .cg means cache to L2 only.
  // .cg only accepts cp-size=16
  // .ca results in significantly slower kernel, probably because it uses up L1 resources
  // + additional copy, which is unnecessary, since we already manually cache it in shared memory.
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
};

// Predicated 16-byte cp.async with zero-fill on false predicate.
// Uses the cp.async src-size variant: if src_size < cp-size, the remaining
// destination bytes are zero-filled. src_size=0 -> entire 16 bytes zero.
// Use this for the boundary tiles of a matmul where some rows/cols are OOB.
__device__ inline
void cp_async_pred(uint32_t dst, const void *src, bool pred) {
  int src_size = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               ::"r"(dst), "l"(src), "r"(src_size));
}

// Same as cp_async_pred, but tags the L2 line with an evict_first eviction
// priority instead of the default (evict_normal): the line becomes the
// first candidate L2 picks when it needs to make room, so it doesn't
// meaningfully persist there. Use this for data that's read exactly once
// and never reused across blocks/SMs, to avoid it displacing data that IS
// being reused (e.g. B in the marlin-style kernel, which streams through
// once per (block_n,k) while A gets re-read by every block_m iteration on
// every SM). Note: L2::no_allocate is not accepted as a primary eviction
// priority by ptxas on this toolchain/arch, so evict_first is the closest
// available approximation of "don't cache this."
__device__ inline
void cp_async_pred_evict_first(uint32_t dst, const void *src, bool pred) {
  uint64_t policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
  int src_size = pred ? 16 : 0;
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;"
               ::"r"(dst), "l"(src), "r"(src_size), "l"(policy));
}

__device__ inline
void cp_async_commit_group() { asm volatile("cp.async.commit_group;"); };

template <int N>
__device__ inline
void cp_async_wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); };

__device__ inline
void cp_async_wait_all() { asm volatile("cp.async.wait_all;"); };

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / std::max(128 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

// STRIDE in bytes, col in the units of 16-byte
template <int STRIDE>
__device__ static
uint32_t swizzle_better(uint32_t row, uint32_t col) {
  if constexpr (STRIDE >= 128)
    col ^= (row % 8) / std::max(128 / STRIDE, 1);
  return row * STRIDE + col * 16;
}

template <typename T, typename... Args>
void launch_kernel(T *kernel, int num_blocks, int block_size, int shm_size, Args... args) {
  if (shm_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));

  kernel<<<num_blocks, block_size, shm_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Shared across every matmul_kernel_* (asrq_16x16.cu, asrq_4x16.cu,
// asrq_2x16.cu, asrq_4x4.cu): tile scheduling, cp.async tile loaders, and the
// host-side SM count query. Each kernel's own file has the design comments
// specific to how it uses these; this header just holds the one copy shared
// by all of them.

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;

// L2-aware serpentine/boustrophedon tile scheduler: groups GROUP_M
// consecutive block_m values together and sweeps all block_n within a group
// before advancing (alternating sweep direction each group), so that tiles
// processed close together in time/across concurrently-running blocks share
// A/B rows and stay resident in L2 instead of thrashing it.
__device__ __forceinline__
void tile_scheduler_l2(int tile_idx, int num_block_m, int num_block_n, int GROUP_M, int& block_m, int& block_n) {
    const int num_pid_in_group = GROUP_M * num_block_n;
    const int group_id = tile_idx / num_pid_in_group;
    const int first_pid_m = group_id * GROUP_M;
    int group_size_m = num_block_m - first_pid_m;
    if (group_size_m > GROUP_M) group_size_m = GROUP_M;
    const int tile_in_group = tile_idx % num_pid_in_group;
    block_m = first_pid_m + (tile_in_group % group_size_m);
    const int n_in_group = tile_in_group / group_size_m;
    block_n = (group_id % 2 == 0) ? n_in_group : (num_block_n - 1 - n_in_group);
}

// Async-copies a HEIGHT x WIDTH tile of __half elements from global to
// shared memory, cp.async-style, using swizzle_better<> addressing so a
// later ldmatrix fetch from the same tile is bank-conflict-free.
template<int TB_SIZE, int HEIGHT, int WIDTH, bool EVICT_FIRST = false>
__device__ static
void global_to_shared_async(const __half* in, int in_stride, uint32_t out, int tid, int valid_height=HEIGHT, int valid_width=WIDTH){
    constexpr int num_elems = 16 / sizeof(__half);
    constexpr int total_vecs = (HEIGHT * WIDTH) / num_elems;
    constexpr int num_iters = cdiv(total_vecs, TB_SIZE);

    for(int iter=0; iter<num_iters; iter++){
        const int vec_idx = iter * TB_SIZE + tid;
        if (vec_idx >= total_vecs) continue;
        const int idx = vec_idx * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;
        uint32_t dst_addr = out + swizzle_better<WIDTH * sizeof(__half)>(row, col / num_elems);
        const bool valid = row < valid_height && (col + num_elems) <= valid_width;
        const __half *src_ptr = valid ? (in + row * in_stride + col) : in;
        if constexpr (EVICT_FIRST) {
            cp_async_pred_evict_first(dst_addr, src_ptr, valid);
        } else {
            cp_async_pred(dst_addr, src_ptr, valid);
        }
    }
}

// Async-copies a HEIGHT x WIDTH_BYTES tile of raw bytes from global to
// shared memory, row-major, WITHOUT swizzling. Used by the WxA16 kernels'
// packed-weight B tile, which is read back out with plain per-thread
// addressed loads (see each kernel's compute() for the mma.sync B-fragment
// mapping), never through ldmatrix -- so there's no ldmatrix-specific
// bank-conflict pattern to arrange for here.
template<int TB_SIZE, int HEIGHT, int WIDTH_BYTES>
__device__ static
void global_to_shared_async_bytes(const uint8_t* in, int in_stride_bytes, uint32_t out, int tid, int valid_height=HEIGHT, int valid_width_bytes=WIDTH_BYTES){
    constexpr int vec_bytes = 16;
    constexpr int total_vecs = (HEIGHT * WIDTH_BYTES) / vec_bytes;
    constexpr int num_iters = cdiv(total_vecs, TB_SIZE);

    for(int iter=0; iter<num_iters; iter++){
        const int vec_idx = iter * TB_SIZE + tid;
        if (vec_idx >= total_vecs) continue;
        const int idx = vec_idx * vec_bytes;
        const int row = idx / WIDTH_BYTES;
        const int col = idx % WIDTH_BYTES;
        uint32_t dst_addr = out + row * WIDTH_BYTES + col;
        const bool valid = row < valid_height && (col + vec_bytes) <= valid_width_bytes;
        const uint8_t *src_ptr = valid ? (in + row * in_stride_bytes + col) : in;
        cp_async_pred(dst_addr, src_ptr, valid);
    }
}

// Same swizzled-tile geometry as global_to_shared_async above (16-byte
// cp.async vectors, swizzle_better<WIDTH_BYTES> addressing), but for raw
// packed bytes instead of __half elements. Used by matmul_kernel_w4a4 to
// stage BOTH A and B tiles through ldmatrix -- unlike
// global_to_shared_async_bytes above (unswizzled, since those never go
// through ldmatrix), this one must match the swizzle pattern ldmatrix's
// addressing expects, exactly like the __half loader does for fp16 tiles.
template<int TB_SIZE, int HEIGHT, int WIDTH_BYTES>
__device__ static
void global_to_shared_async_bytes_swizzled(const uint8_t* in, int in_stride_bytes, uint32_t out, int tid, int valid_height=HEIGHT, int valid_width_bytes=WIDTH_BYTES){
    constexpr int vec_bytes = 16;
    constexpr int total_vecs = (HEIGHT * WIDTH_BYTES) / vec_bytes;
    constexpr int num_iters = cdiv(total_vecs, TB_SIZE);

    for(int iter=0; iter<num_iters; iter++){
        const int vec_idx = iter * TB_SIZE + tid;
        if (vec_idx >= total_vecs) continue;
        const int idx = vec_idx * vec_bytes;
        const int row = idx / WIDTH_BYTES;
        const int col = idx % WIDTH_BYTES;
        uint32_t dst_addr = out + swizzle_better<WIDTH_BYTES>(row, col / vec_bytes);
        const bool valid = row < valid_height && (col + vec_bytes) <= valid_width_bytes;
        const uint8_t *src_ptr = valid ? (in + row * in_stride_bytes + col) : in;
        cp_async_pred(dst_addr, src_ptr, valid);
    }
}

inline int num_sms() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.multiProcessorCount;
}