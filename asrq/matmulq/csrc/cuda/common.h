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