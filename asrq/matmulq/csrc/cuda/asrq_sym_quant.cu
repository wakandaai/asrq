#include "common.h"
#include <algorithm>
#include <cstdint>
#include <cuda_fp16.h>
#include <iostream>

// Per-row (per-token) symmetric activation quantization, fp16 -> int4/int8.
// This is the piece a real low-bit-matmul inference pipeline needs that a
// PRE-quantized-weight kernel (W4A4, W4A8) doesn't: weights are quantized
// once, offline, but activations arrive in fp16 every forward pass and have
// to be quantized on the fly. Both variants here share one design: one
// thread block per row, a single vectorized (128-bit, 8 halfs/load) pass
// over that row cached into shared memory while tracking a running
// max-abs, a fast block-wide max reduction (warp shuffles, then one more
// shuffle round across per-warp maxima), then a second pass -- reading
// back from the shared-memory cache instead of global memory -- that
// quantizes and writes the packed/unpacked output. So each row of A is
// read from global memory exactly once and written exactly once,
// regardless of quantization width.
//
// Codes are plain two's-complement (matching matmul_kernel_w4a4's and
// matmul_kernel_w4a8's A convention) -- NOT excess-K encoded, since neither
// consumer kernel dequantizes A through an LOP3 fp16 trick; A feeds its
// tensor core directly as a raw signed integer.

constexpr int QUANT_TB_SIZE = 256;
constexpr int QUANT_VEC = 8; // 8 halfs = 16 bytes per vectorized load/store chunk

__device__ __forceinline__
float block_reduce_max(float local_max) {
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    __shared__ float warp_max[QUANT_TB_SIZE / 32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) warp_max[warp] = local_max;
    __syncthreads();

    float v = (threadIdx.x < QUANT_TB_SIZE / 32) ? warp_max[threadIdx.x] : 0.0f;
    if (warp == 0) {
        for (int offset = (QUANT_TB_SIZE / 32) / 2; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

__launch_bounds__(QUANT_TB_SIZE)
__global__
void quantize_sym_int8_kernel(const __half* A, int8_t* Aq, __half* scales, int K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ __half row_cache[];

    const float4* A_row = reinterpret_cast<const float4*>(A + row * K);
    float4* cache_row = reinterpret_cast<float4*>(row_cache);
    const int vecs = K / QUANT_VEC;

    float local_max = 0.0f;
    for (int v = tid; v < vecs; v += QUANT_TB_SIZE) {
        float4 chunk = A_row[v];
        cache_row[v] = chunk;
        const __half* h = reinterpret_cast<const __half*>(&chunk);
        #pragma unroll
        for (int i = 0; i < QUANT_VEC; i++)
            local_max = fmaxf(local_max, fabsf(__half2float(h[i])));
    }

    const float row_max = block_reduce_max(local_max);
    const float scale = fmaxf(row_max, 1e-8f) / 127.0f;
    const float inv_scale = 1.0f / scale;
    if (tid == 0) scales[row] = __float2half_rn(scale);

    int8_t* Aq_row = Aq + row * K;
    for (int idx = tid; idx < K; idx += QUANT_TB_SIZE) {
        int q = __float2int_rn(__half2float(row_cache[idx]) * inv_scale);
        q = max(-127, min(127, q));
        Aq_row[idx] = static_cast<int8_t>(q);
    }
}

__launch_bounds__(QUANT_TB_SIZE)
__global__
void quantize_sym_int4_kernel(const __half* A, uint8_t* Aq, __half* scales, int K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ __half row_cache[];

    const float4* A_row = reinterpret_cast<const float4*>(A + row * K);
    float4* cache_row = reinterpret_cast<float4*>(row_cache);
    const int vecs = K / QUANT_VEC;

    float local_max = 0.0f;
    for (int v = tid; v < vecs; v += QUANT_TB_SIZE) {
        float4 chunk = A_row[v];
        cache_row[v] = chunk;
        const __half* h = reinterpret_cast<const __half*>(&chunk);
        #pragma unroll
        for (int i = 0; i < QUANT_VEC; i++)
            local_max = fmaxf(local_max, fabsf(__half2float(h[i])));
    }

    const float row_max = block_reduce_max(local_max);
    const float scale = fmaxf(row_max, 1e-8f) / 7.0f;
    const float inv_scale = 1.0f / scale;
    if (tid == 0) scales[row] = __float2half_rn(scale);

    // Two's complement, 2 codes/byte (low nibble = even k, high nibble = odd
    // k) -- matching matmul_kernel_w4a4/w4a8's B packing convention.
    uint8_t* Aq_row = Aq + row * (K / 2);
    for (int idx = tid; idx < K / 2; idx += QUANT_TB_SIZE) {
        int q0 = __float2int_rn(__half2float(row_cache[2 * idx]) * inv_scale);
        int q1 = __float2int_rn(__half2float(row_cache[2 * idx + 1]) * inv_scale);
        q0 = max(-8, min(7, q0));
        q1 = max(-8, min(7, q1));
        uint8_t nibble0 = static_cast<uint8_t>(q0) & 0xF;
        uint8_t nibble1 = static_cast<uint8_t>(q1) & 0xF;
        Aq_row[idx] = nibble0 | (nibble1 << 4);
    }
}

void quantize_sym_int8_launcher(const __half* A, int8_t* Aq, __half* scales, int M, int K, cudaStream_t stream) {
    const int shm_size = K * sizeof(__half);
    auto kernel = quantize_sym_int8_kernel;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<M, QUANT_TB_SIZE, shm_size, stream>>>(A, Aq, scales, K);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_sym_int4_launcher(const __half* A, uint8_t* Aq, __half* scales, int M, int K, cudaStream_t stream) {
    const int shm_size = K * sizeof(__half);
    auto kernel = quantize_sym_int4_kernel;
    if (shm_size > 48'000)
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size));
    kernel<<<M, QUANT_TB_SIZE, shm_size, stream>>>(A, Aq, scales, K);
    CUDA_CHECK(cudaGetLastError());
}
