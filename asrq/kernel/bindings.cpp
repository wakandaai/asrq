#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

// -------------------------------------------------------------------------
// Helper Macros and Inline Functions
// -------------------------------------------------------------------------

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __line__ << std::endl; \
        exit(1); \
    } \
}

// Unpack a 4-bit integer from a byte and apply sign extension
inline __host__ __device__ int8_t unpack_int4(uint8_t val, bool upper) {
    int8_t v = upper ? (val >> 4) : (val & 0x0F);
    // Sign extension for 4-bit Two's Complement (Range: -8 to 7)
    return v > 7 ? v - 16 : v;
}

// Pack two 8-bit integers (assumed to be in range -8 to 7) into one byte
inline __host__ __device__ uint8_t pack_int4(int8_t lower, int8_t upper) {
    return (uint8_t)(lower & 0x0F) | ((uint8_t)(upper & 0x0F) << 4);
}

// -------------------------------------------------------------------------
// CPU Implementations
// -------------------------------------------------------------------------

// Dynamically quantize activations symmetrically from FP32/FP16 to INT4
// A is M x K, A_int4 is M x (K/2)
void quantize_activations_cpu(const float* A, int M, int K, uint8_t* A_int4, float* scales) {
    for (int i = 0; i < M; ++i) {
        float max_abs = 0.0f;
        // 1. Find Max Absolute Value
        for (int k = 0; k < K; ++k) {
            max_abs = std::max(max_abs, std::abs(A[i * K + k]));
        }
        
        // 2. Compute symmetric scale (mapping max to 7)
        float scale = max_abs / 7.0f;
        if (scale == 0.0f) scale = 1e-9f; // Prevent division by zero
        scales[i] = scale;

        // 3. Quantize and Pack
        for (int k = 0; k < K; k += 2) {
            int8_t q0 = std::clamp((int)std::round(A[i * K + k] / scale), -8, 7);
            int8_t q1 = std::clamp((int)std::round(A[i * K + k + 1] / scale), -8, 7);
            A_int4[i * (K / 2) + (k / 2)] = pack_int4(q0, q1);
        }
    }
}

// CPU INT4 Matmul
// A_int4 is M x (K/2)
// W_int4_T is N x (K/2) -> Transposed for optimal contiguous access along K
void matmul_int4_cpu(const uint8_t* A_int4, const uint8_t* W_int4_T, 
                     const float* scale_A, const float* scale_W, 
                     float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t accumulator = 0;
            
            // Loop through K in steps of 2 (1 byte = 2 elements)
            for (int k_idx = 0; k_idx < K / 2; ++k_idx) {
                uint8_t a_val = A_int4[i * (K / 2) + k_idx];
                uint8_t w_val = W_int4_T[j * (K / 2) + k_idx];
                
                int8_t a0 = unpack_int4(a_val, false);
                int8_t a1 = unpack_int4(a_val, true);
                
                int8_t w0 = unpack_int4(w_val, false);
                int8_t w1 = unpack_int4(w_val, true);
                
                accumulator += (a0 * w0) + (a1 * w1);
            }
            
            // Dequantize and assign
            C[i * N + j] = accumulator * scale_A[i] * scale_W[j];
        }
    }
}

// -------------------------------------------------------------------------
// GPU (CUDA) Implementations
// -------------------------------------------------------------------------

// Simple thread-per-row quantization kernel
// Note: For production, use a warp-level reduction (e.g., via CUB) to parallelize the K dimension.
__global__ void quantize_activations_kernel(const float* __restrict__ A, int M, int K, 
                                            uint8_t* __restrict__ A_int4, float* __restrict__ scales) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float max_abs = 0.0f;
        
        // 1. Find Max Abs
        for (int k = 0; k < K; ++k) {
            float val = fabsf(A[row * K + k]);
            if (val > max_abs) max_abs = val;
        }
        
        // 2. Scale
        float scale = max_abs / 7.0f;
        if (scale == 0.0f) scale = 1e-9f;
        scales[row] = scale;
        
        // 3. Quantize and pack
        for (int k = 0; k < K; k += 2) {
            int q0 = max(-8, min(7, (int)roundf(A[row * K + k] / scale)));
            int q1 = max(-8, min(7, (int)roundf(A[row * K + k + 1] / scale)));
            A_int4[row * (K / 2) + (k / 2)] = pack_int4(q0, q1);
        }
    }
}

// Scale INT32 CUTLASS GEMM output back to FP32 using row/col scales
__global__ void scale_int32_to_float_kernel(const int32_t* __restrict__ C_int32,
                                            const float* __restrict__ scale_A,
                                            const float* __restrict__ scale_W,
                                            float* __restrict__ C_fp32,
                                            int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        C_fp32[row * N + col] = static_cast<float>(C_int32[row * N + col]) * scale_A[row] * scale_W[col];
    }
}

// -------------------------------------------------------------------------
// PyTorch Extension Bindings
// -------------------------------------------------------------------------

torch::Tensor int4_matmul_cuda(torch::Tensor A, torch::Tensor W_int4_T, torch::Tensor scale_W) {
    // Basic PyTorch Tensor checks
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(W_int4_T.is_cuda(), "W_int4_T must be a CUDA tensor");
    TORCH_CHECK(scale_W.is_cuda(), "scale_W must be a CUDA tensor");
    
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(W_int4_T.is_contiguous(), "W_int4_T must be contiguous");
    TORCH_CHECK(scale_W.is_contiguous(), "scale_W must be contiguous");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = W_int4_T.size(0);

    TORCH_CHECK(W_int4_T.size(1) == K / 2, "W_int4_T shape mismatch: expected (N, K/2)");
    TORCH_CHECK(scale_W.size(0) == N, "scale_W shape mismatch: expected (N)");

    // Allocate intermediate and output tensors natively on the PyTorch allocator
    auto options_uint8 = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
    auto options_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto options_int32 = torch::TensorOptions().dtype(torch::kInt32).device(A.device());

    torch::Tensor A_int4 = torch::empty({M, K / 2}, options_uint8);
    torch::Tensor scale_A = torch::empty({M}, options_fp32);
    torch::Tensor C_int32 = torch::empty({M, N}, options_int32);
    torch::Tensor C = torch::empty({M, N}, options_fp32);

    // 1. Launch dynamic quantization kernel
    int threads_quant = 256;
    int blocks_quant = (M + threads_quant - 1) / threads_quant;
    quantize_activations_kernel<<<blocks_quant, threads_quant>>>(
        A.data_ptr<float>(), M, K, 
        A_int4.data_ptr<uint8_t>(), scale_A.data_ptr<float>()
    );
    
    // 2. Launch CUTLASS INT4 GEMM
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t, cutlass::layout::RowMajor,
        cutlass::int4b_t, cutlass::layout::ColumnMajor,
        int32_t, cutlass::layout::RowMajor,
        int32_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm90 // Target Hopper architecture
    >;

    Gemm gemm_op;
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    Gemm::Arguments args(
        problem_size,
        {reinterpret_cast<cutlass::int4b_t*>(A_int4.data_ptr<uint8_t>()), K},
        {reinterpret_cast<cutlass::int4b_t*>(W_int4_T.data_ptr<uint8_t>()), K},
        {C_int32.data_ptr<int32_t>(), N},
        {C_int32.data_ptr<int32_t>(), N},
        {1, 0}
    );

    cutlass::Status status = gemm_op(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));

    // 3. Launch Scaling Kernel
    dim3 threads_scale(16, 16);
    dim3 blocks_scale((N + threads_scale.x - 1) / threads_scale.x, 
                      (M + threads_scale.y - 1) / threads_scale.y);
    scale_int32_to_float_kernel<<<blocks_scale, threads_scale>>>(
        C_int32.data_ptr<int32_t>(), scale_A.data_ptr<float>(), 
        scale_W.data_ptr<float>(), C.data_ptr<float>(), M, N
    );

    return C;
}

// Expose the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_int4", &int4_matmul_cuda, "INT4 Symmetric Matmul with CUTLASS (CUDA)");
}