#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include <vector>
// #include "cutlass/cutlass.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)   \
  CHECK_CUDA(x);         \
  CHECK_CONTIGUOUS(x)

// Shared by every matmul wrapper below: validates an optional (N,) f16 bias
// tensor and returns the raw pointer the kernel epilogues expect (nullptr
// when no bias was passed), so bias gets added inside the kernel's own
// output write instead of a separate elementwise CUDA kernel afterward.
static const __half* get_bias_ptr(const c10::optional<torch::Tensor>& bias_opt, int64_t N) {
    if (!bias_opt.has_value()) return nullptr;
    const auto& bias = bias_opt.value();
    CHECK_INPUT(bias);
    TORCH_CHECK(bias.dtype() == torch::kFloat16, "bias must be float16");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N, "bias must have shape (N,); got ", bias.sizes());
    return reinterpret_cast<const __half*>(bias.data_ptr<at::Half>());
}

// Forward declarations from .cu files
// matmul_sm89: tuned for sm_89 (Ada). Used as the default fp16 matmul.
// void matmul_sm89_launcher (const __half* A, const __half* B, __half* C, int M, int N, int K);
// Data-parallel fp16 GEMM: no stream-K, no workspace/locks -- each output
// tile's full K reduction is owned start-to-finish by a single threadblock
// via a grid-stride loop. Tile shape is chosen internally per (M,N,K); see
// compute_matmul_launch_params() in asrq.cu.
// Every launcher below now takes an optional bias pointer (nullptr = no
// bias), added to each output element in the kernel's own epilogue -- see
// each matmul_kernel_*'s bias0/bias1 computation in its .cu file. This
// matches cuBLAS/cuBLASLt's fused bias epilogue instead of requiring a
// separate elementwise add kernel afterward (which would cost a full extra
// memory-bound pass over the (M,N) output, and loses to cuBLAS's own fused
// path in a head-to-head nn.Linear comparison).
void mymatmul_launcher(const __half* A, const __half* B, const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// W4A16: f16 activations x per-channel-symmetric packed-int4 weights.
// A (M,K) f16; Bq (N,K/2) uint8 packed excess-8 (offset-binary) nibbles --
// nibble = signed_value + 8, range 0..15, NOT two's complement -- (low
// nibble = even k, high nibble = odd k); scales (N,) f16 per-output-channel
// scale. Returns (M,N) f16 = A @ dequant(Bq).T (+ bias).
void w4a16_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias,
                           __half* C, int M, int N, int K, cudaStream_t stream);

// W2A16: f16 activations x per-channel-symmetric packed-int2 weights.
// A (M,K) f16; Bq (N,K/4) uint8 packed excess-2 (offset-binary) 2-bit codes,
// 4 per byte -- code = signed_value + 2, range 0..3, NOT two's complement --
// (code j, j=0..3, at bits[2j:2j+2), covering element 4*byte_index+j);
// scales (N,) f16 per-output-channel scale. Returns (M,N) f16 =
// A @ dequant(Bq).T (+ bias).
void w2a16_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias,
                           __half* C, int M, int N, int K, cudaStream_t stream);

// W4A16 groupwise: same excess-8 packed-int4 weights as w4a16_matmul, but
// scales (N, K/128) f16 -- one scale per (channel, 128-wide K group) instead
// of one per channel. Returns (M,N) f16 = A @ dequant_group(Bq).T (+ bias).
// Requires K % 128 == 0 (in addition to w4a16_matmul's own K % 32 == 0).
void w4a16_group_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias,
                                  __half* C, int M, int N, int K, cudaStream_t stream);

// W2A16 groupwise: same excess-2 packed-int2 weights as w2a16_matmul, but
// scales (N, K/128) f16 -- one scale per (channel, 128-wide K group).
// Returns (M,N) f16 = A @ dequant_group(Bq).T (+ bias). Requires
// K % 128 == 0 (in addition to w2a16_matmul's own K % 64 == 0).
void w2a16_group_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* bias,
                                  __half* C, int M, int N, int K, cudaStream_t stream);

// W4A16 groupwise asymmetric (zero-point): same PLAIN UNSIGNED packed-int4
// weights as w4a16_asym_matmul, but scales, zeros (N, K/128) f16 -- one
// scale/zero per (channel, 128-wide K group) instead of one per channel.
// Returns (M,N) f16 = A @ (scales*code + zeros).T (grouped) (+ bias).
// Requires K % 128 == 0 (in addition to w4a16_asym_matmul's own K % 32 == 0).
void w4a16_group_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros,
                                       const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// W2A16 groupwise asymmetric (zero-point): same PLAIN UNSIGNED packed-int2
// weights as w2a16_asym_matmul, but scales, zeros (N, K/128) f16 -- one
// scale/zero per (channel, 128-wide K group). Returns (M,N) f16 =
// A @ (scales*code + zeros).T (grouped) (+ bias). Requires K % 128 == 0 (in
// addition to w2a16_asym_matmul's own K % 64 == 0).
void w2a16_group_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros,
                                       const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// W4A16 asymmetric (zero-point): A (M,K) f16; Bq (N,K/2) uint8 packed PLAIN
// UNSIGNED nibbles (0..15, no excess-8 encoding -- zeros[n] absorbs the
// centering); scales, zeros (N,) f16 per-output-channel. Returns (M,N) f16 =
// A @ (scales[:,None]*dequant_code(Bq) + zeros[:,None]).T (+ bias).
void w4a16_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros,
                                const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// W2A16 asymmetric (zero-point): same convention as w4a16_asym above, but
// Bq (N,K/4) uint8 packed plain unsigned 2-bit codes (0..3), 4 per byte.
void w2a16_asym_matmul_launcher(const __half* A, const uint8_t* Bq, const __half* scales, const __half* zeros,
                                const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// W4A4: BOTH A and B packed int4, symmetric, run through the hardware
// int4xint4 tensor core directly (int32 accumulate) -- no fp16
// dequantization anywhere in the K-loop, unlike the WxA16 kernels above.
// Nibbles are plain two's-complement signed values (range [-8,7]), packed
// 2/byte (low nibble = even k, high nibble = odd k) -- NOT excess-8 encoded,
// since there's no LOP3 fp16-dequant trick to serve here. Aq (M,K/2) uint8;
// Bq (N,K/2) uint8; scales_A (M,) f16 per-row (per-token) scale for A;
// scales_B (N,) f16 per-channel scale for B. Returns (M,N) f16 =
// (scales_A[:,None] * scales_B[None,:]) * (dequant_code(Aq) @ dequant_code(Bq).T) (+ bias).
void w4a4_matmul_launcher(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B,
                          const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// W4A8: int8 activations x packed-int4 weights, run through the hardware
// int8 tensor core (B is unpacked register-resident straight out of the
// packed cp.async staging buffer, since there's no int4xint8 MMA
// instruction). Aq (M,K) int8, plain two's-complement bytes; Bq (N,K/2)
// uint8 packed two's-complement int4 nibbles (NOT excess-8 encoded), 2/byte
// (low nibble = even k, high nibble = odd k); scales_A (M,) f16 per-row
// scale for A; scales_B (N,) f16 per-channel scale for B. Returns (M,N)
// f16 = (scales_A[:,None] * scales_B[None,:]) * (Aq @ dequant_code(Bq).T) (+ bias).
void w4a8_matmul_launcher(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B,
                          const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// W4A4/W4A8 with GROUPWISE weight quantization: same operand layout as
// w4a4_matmul/w4a8_matmul above, except scales_B is (N, K/128) instead of
// (N,) -- one scale per (output channel, 128-wide K-group) instead of one
// per channel, matching the WxA16 kernels' *_group_matmul convention. A's
// quantization (per-row/per-token scales_A) is unchanged.
void w4a4_group_matmul_launcher(const uint8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B,
                                 const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);
void w4a8_group_matmul_launcher(const int8_t* Aq, const uint8_t* Bq, const __half* scales_A, const __half* scales_B,
                                 const __half* bias, __half* C, int M, int N, int K, cudaStream_t stream);

// Per-row (per-token) symmetric activation quantization, fp16 -> int4/int8,
// plain two's-complement codes (matching W4A4/W4A8's A convention -- no
// excess-K encoding, since neither consumer kernel dequantizes A through an
// LOP3 trick). A (M,K) f16; Aq_int8 (M,K) int8 or Aq_int4 (M,K/2) uint8
// packed 2/byte; scales (M,) f16, dequant(A[m,k]) = scales[m] * code.
void quantize_sym_int8_launcher(const __half* A, int8_t* Aq, __half* scales, int M, int K, cudaStream_t stream);
void quantize_sym_int4_launcher(const __half* A, uint8_t* Aq, __half* scales, int M, int K, cudaStream_t stream);

// ---------------------------------------------------------------------------
// CPU (AVX2/FMA) quantized matmuls -- asrq/matmulq/csrc/x86/qmatmul.{h,cpp}.
// All tensors here are plain fp32 (no fp16 storage on the CPU side -- see
// qmatmul.h's top comment for why), packed weight codes are TWO'S COMPLEMENT
// for symmetric / PLAIN UNSIGNED for asymmetric (not the GPU kernels'
// excess-K encoding, which exists only to serve their LOP3 trick). One
// dispatcher each for the WxA16 (A stays fp32) and WxA8 (A quantized to
// int8 first) cases; every wbits/symmetric/groupwise combination funnels
// through whichever of these two matches its activation width.
// void wxa16_cpu_matmul_launcher(int wbits, bool symmetric, bool groupwise,
//                                 const float* A, const uint8_t* Bq, const float* scales, const float* zeros,
//                                 const float* bias, float* C, int M, int N, int K);
// void wxa8_cpu_matmul_launcher(int wbits, bool symmetric, bool groupwise,
//                                const int8_t* Aq, const float* scales_A,
//                                const uint8_t* Bq, const float* scales_B, const float* zeros_B,
//                                const float* bias, float* C, int M, int N, int K);
// void quantize_sym_int8_cpu_launcher(const float* A, int8_t* Aq, float* scales, int M, int K);



// // matmul_int4_sm89: hand-written int4 x int4 -> int32 tensor-core GEMM.
// // A, B are packed int4 (2/byte) reinterpreted as uint16 words; K is the
// // unpacked int4 contraction length.
// void matmul_int4_sm89_launcher(const uint16_t* A, const uint16_t* B, int32_t* C,
//                                int M, int N, int K);
// // Fused int4 GEMM + per-channel rescale: Y = sA[m]*sB[n]*(A @ B^T), f16 out.
// void matmul_int4_rescale_sm89_launcher(const uint16_t* A, const uint16_t* B,
//                                        const float* sA, const float* sB, __half* Y,
//                                        int M, int N, int K);
// // matmul_int8_sm89: hand-written int8 x int8 -> int32 tensor-core GEMM.
// // A, B are int8 reinterpreted as uint16 words; K is the int8 contraction length.
// void matmul_int8_sm89_launcher(const uint16_t* A, const uint16_t* B, int32_t* C,
//                                int M, int N, int K);
// // matmul_wxax_sm89: fused f16-activation x grouped-per-channel quantized-weight
// // GEMM. A f16 (M,K); W packed BITS-bit words (N, K*BITS/16); sW (N, K/128) f16.
// // qA (M, K*BITS/16) uint16 and sA (M,) f32 are scratch filled internally (packed
// // quantized activations + per-token scale). Y f16 (M,N). bits is 4 or 8.
// void matmul_wxax_sm89_launcher(const __half* A, const uint16_t* W, const __half* sW,
//                                uint16_t* qA, float* sA, __half* Y,
//                                int M, int N, int K, int bits, cudaStream_t stream);
// // matmul_f16xint4: f16 activation x grouped-int4 weight -> f16. The int4 weights
// // are dequantized to f16 (w = sB*q + zB, unsigned nibble q, groupsize 128) and
// // the contraction runs as an f16 tensor-core GEMM. A f16 (M,K); B packed uint8
// // (N, K/2); sB,zB f16 (N, K/128); C f16 (M,N). Requires K%128==0, N%128==0.
// void matmul_f16xint4(const __half* A, const uint8_t* B, __half* sB, __half* zB,
//                      __half* C, int M, int N, int K);
// // matmul_w4a16_sym: W4A16 with per-channel symmetric (signed int4) weights.
// // A f16 (M,K); B signed packed int4 (N,K/2); sB f32 (N,) per-channel; C f16.
// void matmul_w4a16_sym_launcher(const __half* A, const uint8_t* B, const float* sB,
//                                __half* C, int M, int N, int K);
// // matmul_f16xint2: W2A16, asymmetric grouped (group=128). A f16 (M,K); B packed
// // unsigned int2 (N,K/4); sB,zB f16 (N,K/128); C f16. w = sB*q + zB, q in [0,3].
// void matmul_f16xint2(const __half* A, const uint8_t* B, __half* sB, __half* zB,
//                      __half* C, int M, int N, int K);
// cutlass::Status cutlass_int8_matmul_launcher(const int8_t* A, const int8_t* B, int32_t* C,
//                                               int M, int N, int K, cudaStream_t stream);
// // cublasStatus_t is forward-declarable as an enum; just declare via cublas header.
// #include <cublas_v2.h>
// cublasStatus_t cublaslt_int8_matmul_launcher(const int8_t* A, const int8_t* B, int32_t* C,
//                                              int M, int N, int K, cudaStream_t stream);
// cublasStatus_t cublaslt_int4_matmul_launcher(const int8_t* A_packed, const int8_t* B_packed,
//                                              int32_t* C, int M, int N, int K,
//                                              cudaStream_t stream);
// void per_token_sym_quant_launcher_f32(const float* X, int8_t* Q, float* scale,
//                                       int rows, int C, cudaStream_t stream);
// // per_token int4 quantize+pack (from matmul_wxax_sm89.cu): A f16 (M,K) ->
// // qA (M, K/4) uint16 words (matmul_int4 layout) + sA (M,) f32 per-token scale.
// void per_token_quant_int4_launcher(const __half* A, uint16_t* qA, float* sA,
//                                    int M, int K, cudaStream_t stream);
// void per_token_sym_quant_launcher_f16(const __half* X, int8_t* Q, float* scale,
//                                       int rows, int C, cudaStream_t stream);
// void rescale_int32_rowcol_launcher_f16(const int32_t* C, const float* sA, const float* sB,
//                                        __half* Y, int M, int N, cudaStream_t stream);
// void rescale_int32_rowcol_launcher_f32(const int32_t* C, const float* sA, const float* sB,
//                                        float* Y, int M, int N, cudaStream_t stream);
// void unpack_int4_to_int8_launcher(const uint8_t* Bp, int8_t* B,
//                                   size_t n_packed_bytes, cudaStream_t stream);

// ---- GPU metrics ----------------------------------------------------------
//
// Tensor cores per SM by compute capability (NVIDIA hardware tables):
//   7.0 (Volta V100)            : 8 TCs / SM
//   7.5 (Turing T4/RTX 20xx)    : 8 TCs / SM
//   8.0 (Ampere A100)           : 4 TCs / SM
//   8.6 (Ampere RTX 30xx, A40)  : 4 TCs / SM
//   8.9 (Ada     RTX 40xx, L40) : 4 TCs / SM
//   9.0 (Hopper H100/H200)      : 4 TCs / SM
//  10.x (Blackwell B100/B200)   : 4 TCs / SM
// 0 indicates "unknown / no tensor cores".
static int tensor_cores_per_sm(int major, int minor) {
    if (major == 7) return 8;
    if (major == 8) return 4;
    if (major == 9) return 4;
    if (major >= 10) return 4;
    return 0;
}

py::dict get_gpu_metrics(int64_t device = -1) {
    int dev = static_cast<int>(device);
    if (dev < 0) {
        TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "cudaGetDevice failed");
    }
    cudaDeviceProp prop{};
    TORCH_CHECK(cudaGetDeviceProperties(&prop, dev) == cudaSuccess,
                "cudaGetDeviceProperties failed for device ", dev);

    const int tc_per_sm = tensor_cores_per_sm(prop.major, prop.minor);

    py::dict d;
    d["device"]                  = dev;
    d["name"]                    = std::string(prop.name);
    d["computeCapability"]       = py::make_tuple(prop.major, prop.minor);
    d["smCount"]                 = prop.multiProcessorCount;
    d["tensorCoresPerSm"]        = tc_per_sm;
    d["totalTensorCores"]        = tc_per_sm * prop.multiProcessorCount;
    d["sharedMemPerSM"]          = static_cast<int64_t>(prop.sharedMemPerMultiprocessor);
    d["sharedMemPerBlock"]       = static_cast<int64_t>(prop.sharedMemPerBlock);
    d["sharedMemPerBlockOptin"]  = static_cast<int64_t>(prop.sharedMemPerBlockOptin);
    d["l2CacheSize"]             = prop.l2CacheSize;             // bytes
    d["globalMemSize"]           = static_cast<int64_t>(prop.totalGlobalMem);
    d["maxThreadsPerSM"]         = prop.maxThreadsPerMultiProcessor;
    d["maxThreadsPerBlock"]      = prop.maxThreadsPerBlock;
    d["regsPerSM"]               = prop.regsPerMultiprocessor;
    d["regsPerBlock"]            = prop.regsPerBlock;
    d["maxRegsPerThread"]        = 255;                          // hardware limit
    d["warpSize"]                = prop.warpSize;
    d["clockRateKHz"]            = prop.clockRate;
    d["memoryClockRateKHz"]      = prop.memoryClockRate;
    d["memoryBusWidthBits"]      = prop.memoryBusWidth;
    return d;
}

// torch::Tensor matmul_16x16(torch::Tensor A, torch::Tensor B) {
//     CHECK_INPUT(A);
//     CHECK_INPUT(B);
//     TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
//     TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
//     TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
//     TORCH_CHECK(A.size(1) == B.size(1),
//                 "Inner dimensions must match: A is (M, K), B is (N, K); "
//                 "got A=(", A.size(0), ",", A.size(1), ") B=(",
//                 B.size(0), ",", B.size(1), ")");

//     int M = A.size(0), K = A.size(1), N = B.size(0);
//     TORCH_CHECK((K % 8) == 0,
//                 "matmul_16x16 requires K to be a multiple of 8 (cp.async 16-byte vectors); got K=", K);
//     auto C = torch::empty({M, N}, A.options());

//     // matmul_16x16 is now an alias for the sm_89-tuned kernel; the legacy
//     // Hopper launcher (matmul_16x16_launcher) was removed.
//     matmul_sm89_launcher(
//         reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//         reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
//         reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
//         M, N, K
//     );
//     return C;
// }

torch::Tensor mymatmul(const torch::Tensor& A, const torch::Tensor& B,
                        const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1),
                "Inner dimensions must match: A is (M, K), B is (N, K); "
                "got A=(", A.size(0), ",", A.size(1), ") B=(",
                B.size(0), ",", B.size(1), ")");

    int M = A.size(0), K = A.size(1), N = B.size(0);
    TORCH_CHECK((K % 8) == 0,
                "mymatmul requires K to be a multiple of 8 (cp.async 16-byte vectors); got K=", K);
    auto C = torch::empty({M, N}, A.options());

    mymatmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a16_matmul(const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
                            const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed excess-8 int4 nibbles, 2 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 1, "scales must be 1D (N,)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    // Bq's packed rows are cp.async'd 16 bytes at a time with a row stride
    // of K/2 bytes; that stride must itself be a multiple of 16 bytes for
    // every row's start to stay 16-byte aligned, i.e. K % 32 == 0 -- a
    // stricter requirement than the fp16 kernel's K % 8.
    TORCH_CHECK((K % 32) == 0,
                "w4a16_matmul requires K to be a multiple of 32 (cp.async "
                "16-byte-aligned rows over packed int4 weights); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N,
                "scales must have shape (N,); got (", scales.size(0), ")");

    auto C = torch::empty({M, N}, A.options());
    w4a16_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w2a16_matmul(const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
                            const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed excess-2 int2 codes, 4 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 1, "scales must be 1D (N,)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    // Bq's packed rows are cp.async'd 16 bytes at a time with a row stride
    // of K/4 bytes; that stride must itself be a multiple of 16 bytes for
    // every row's start to stay 16-byte aligned, i.e. K % 64 == 0 -- a
    // stricter requirement than w4a16_matmul's K % 32 (int2 packs 4 codes
    // per byte instead of int4's 2, so the row is 4x narrower for the same K).
    TORCH_CHECK((K % 64) == 0,
                "w2a16_matmul requires K to be a multiple of 64 (cp.async "
                "16-byte-aligned rows over packed int2 weights); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 4,
                "Bq must have shape (N, K/4); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N,
                "scales must have shape (N,); got (", scales.size(0), ")");

    auto C = torch::empty({M, N}, A.options());
    w2a16_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a16_group_matmul(const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
                                  const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed excess-8 int4 nibbles, 2 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 2, "scales must be 2D (N, K/128)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    TORCH_CHECK((K % 32) == 0,
                "w4a16_group_matmul requires K to be a multiple of 32 (cp.async "
                "16-byte-aligned rows over packed int4 weights); got K=", K);
    // The kernel hoists one scale per (n, group) across an entire BLOCK_K=64
    // tile, which only makes sense if every such tile falls inside exactly
    // one group -- true as long as K % 128 == 0 (GROUP_SIZE=128 is a
    // multiple of every BLOCK_K this kernel dispatches, 64).
    TORCH_CHECK((K % 128) == 0,
                "w4a16_group_matmul requires K to be a multiple of 128 (the "
                "fixed group size); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == K / 128,
                "scales must have shape (N, K/128); got (", scales.size(0), ",", scales.size(1), ")");

    auto C = torch::empty({M, N}, A.options());
    w4a16_group_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w2a16_group_matmul(const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
                                  const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed excess-2 int2 codes, 4 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 2, "scales must be 2D (N, K/128)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    TORCH_CHECK((K % 64) == 0,
                "w2a16_group_matmul requires K to be a multiple of 64 (cp.async "
                "16-byte-aligned rows over packed int2 weights); got K=", K);
    TORCH_CHECK((K % 128) == 0,
                "w2a16_group_matmul requires K to be a multiple of 128 (the "
                "fixed group size); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 4,
                "Bq must have shape (N, K/4); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == K / 128,
                "scales must have shape (N, K/128); got (", scales.size(0), ",", scales.size(1), ")");

    auto C = torch::empty({M, N}, A.options());
    w2a16_group_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a16_asym_matmul(const torch::Tensor& A, const torch::Tensor& Bq,
                                 const torch::Tensor& scales, const torch::Tensor& zeros,
                                 const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed unsigned int4 codes, 2 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zeros.dtype() == torch::kFloat16, "zeros must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 1 && zeros.dim() == 1, "scales and zeros must be 1D (N,)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    TORCH_CHECK((K % 32) == 0,
                "w4a16_asym_matmul requires K to be a multiple of 32 (cp.async "
                "16-byte-aligned rows over packed int4 weights); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N, "scales must have shape (N,); got (", scales.size(0), ")");
    TORCH_CHECK(zeros.size(0) == N, "zeros must have shape (N,); got (", zeros.size(0), ")");

    auto C = torch::empty({M, N}, A.options());
    w4a16_asym_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w2a16_asym_matmul(const torch::Tensor& A, const torch::Tensor& Bq,
                                 const torch::Tensor& scales, const torch::Tensor& zeros,
                                 const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed unsigned int2 codes, 4 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zeros.dtype() == torch::kFloat16, "zeros must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 1 && zeros.dim() == 1, "scales and zeros must be 1D (N,)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    TORCH_CHECK((K % 64) == 0,
                "w2a16_asym_matmul requires K to be a multiple of 64 (cp.async "
                "16-byte-aligned rows over packed int2 weights); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 4,
                "Bq must have shape (N, K/4); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N, "scales must have shape (N,); got (", scales.size(0), ")");
    TORCH_CHECK(zeros.size(0) == N, "zeros must have shape (N,); got (", zeros.size(0), ")");

    auto C = torch::empty({M, N}, A.options());
    w2a16_asym_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a16_group_asym_matmul(const torch::Tensor& A, const torch::Tensor& Bq,
                                       const torch::Tensor& scales, const torch::Tensor& zeros,
                                       const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed unsigned int4 codes, 2 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zeros.dtype() == torch::kFloat16, "zeros must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 2 && zeros.dim() == 2, "scales and zeros must be 2D (N, K/128)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    TORCH_CHECK((K % 32) == 0,
                "w4a16_group_asym_matmul requires K to be a multiple of 32 (cp.async "
                "16-byte-aligned rows over packed int4 weights); got K=", K);
    TORCH_CHECK((K % 128) == 0,
                "w4a16_group_asym_matmul requires K to be a multiple of 128 (the "
                "fixed group size); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == K / 128,
                "scales must have shape (N, K/128); got (", scales.size(0), ",", scales.size(1), ")");
    TORCH_CHECK(zeros.size(0) == N && zeros.size(1) == K / 128,
                "zeros must have shape (N, K/128); got (", zeros.size(0), ",", zeros.size(1), ")");

    auto C = torch::empty({M, N}, A.options());
    w4a16_group_asym_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w2a16_group_asym_matmul(const torch::Tensor& A, const torch::Tensor& Bq,
                                       const torch::Tensor& scales, const torch::Tensor& zeros,
                                       const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(A);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales);
    CHECK_INPUT(zeros);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed unsigned int2 codes, 4 per byte)");
    TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
    TORCH_CHECK(zeros.dtype() == torch::kFloat16, "zeros must be float16");
    TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");
    TORCH_CHECK(scales.dim() == 2 && zeros.dim() == 2, "scales and zeros must be 2D (N, K/128)");

    int M = A.size(0), K = A.size(1), N = Bq.size(0);
    TORCH_CHECK((K % 64) == 0,
                "w2a16_group_asym_matmul requires K to be a multiple of 64 (cp.async "
                "16-byte-aligned rows over packed int2 weights); got K=", K);
    TORCH_CHECK((K % 128) == 0,
                "w2a16_group_asym_matmul requires K to be a multiple of 128 (the "
                "fixed group size); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 4,
                "Bq must have shape (N, K/4); got Bq=(", Bq.size(0), ",", Bq.size(1),
                ") with K=", K);
    TORCH_CHECK(scales.size(0) == N && scales.size(1) == K / 128,
                "scales must have shape (N, K/128); got (", scales.size(0), ",", scales.size(1), ")");
    TORCH_CHECK(zeros.size(0) == N && zeros.size(1) == K / 128,
                "zeros must have shape (N, K/128); got (", zeros.size(0), ",", zeros.size(1), ")");

    auto C = torch::empty({M, N}, A.options());
    w2a16_group_asym_matmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zeros.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a4_matmul(const torch::Tensor& Aq, const torch::Tensor& Bq,
                           const torch::Tensor& scales_A, const torch::Tensor& scales_B,
                           const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(Aq);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales_A);
    CHECK_INPUT(scales_B);
    TORCH_CHECK(Aq.dtype() == torch::kUInt8, "Aq must be uint8 (packed signed int4 codes, 2 per byte)");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed signed int4 codes, 2 per byte)");
    TORCH_CHECK(scales_A.dtype() == torch::kFloat16, "scales_A must be float16");
    TORCH_CHECK(scales_B.dtype() == torch::kFloat16, "scales_B must be float16");
    TORCH_CHECK(Aq.dim() == 2 && Bq.dim() == 2, "Aq and Bq must be 2D");
    TORCH_CHECK(scales_A.dim() == 1, "scales_A must be 1D (M,)");
    TORCH_CHECK(scales_B.dim() == 1, "scales_B must be 1D (N,)");

    int M = Aq.size(0), K = Aq.size(1) * 2, N = Bq.size(0);
    // Both Aq and Bq's packed rows are cp.async'd 16 bytes at a time with a
    // row stride of K/2 bytes; that stride must be a multiple of 16 bytes
    // for every row's start to stay 16-byte aligned, i.e. K % 32 == 0 --
    // same requirement as w4a16_matmul.
    TORCH_CHECK((K % 32) == 0,
                "w4a4_matmul requires K to be a multiple of 32 (cp.async "
                "16-byte-aligned rows over packed int4 weights); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Aq K=", K, " Bq=(", Bq.size(0), ",", Bq.size(1), ")");
    TORCH_CHECK(scales_A.size(0) == M, "scales_A must have shape (M,); got (", scales_A.size(0), ")");
    TORCH_CHECK(scales_B.size(0) == N, "scales_B must have shape (N,); got (", scales_B.size(0), ")");

    auto C = torch::empty({M, N}, scales_A.options());
    w4a4_matmul_launcher(
        Aq.data_ptr<uint8_t>(),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scales_B.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a8_matmul(const torch::Tensor& Aq, const torch::Tensor& Bq,
                           const torch::Tensor& scales_A, const torch::Tensor& scales_B,
                           const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(Aq);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales_A);
    CHECK_INPUT(scales_B);
    TORCH_CHECK(Aq.dtype() == torch::kInt8, "Aq must be int8 (plain two's-complement bytes, unpacked)");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed signed int4 codes, 2 per byte)");
    TORCH_CHECK(scales_A.dtype() == torch::kFloat16, "scales_A must be float16");
    TORCH_CHECK(scales_B.dtype() == torch::kFloat16, "scales_B must be float16");
    TORCH_CHECK(Aq.dim() == 2 && Bq.dim() == 2, "Aq and Bq must be 2D");
    TORCH_CHECK(scales_A.dim() == 1, "scales_A must be 1D (M,)");
    TORCH_CHECK(scales_B.dim() == 1, "scales_B must be 1D (N,)");

    int M = Aq.size(0), K = Aq.size(1), N = Bq.size(0);
    // Aq (int8, 1B/elem) needs row stride K % 16 == 0 for 16-byte-aligned
    // cp.async rows; Bq (packed int4, K/2 B/elem) needs K % 32 == 0, the
    // stricter of the two -- same requirement as w4a16_matmul/w4a4_matmul.
    TORCH_CHECK((K % 32) == 0,
                "w4a8_matmul requires K to be a multiple of 32 (cp.async "
                "16-byte-aligned rows over both int8 activations and packed "
                "int4 weights); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Aq K=", K, " Bq=(", Bq.size(0), ",", Bq.size(1), ")");
    TORCH_CHECK(scales_A.size(0) == M, "scales_A must have shape (M,); got (", scales_A.size(0), ")");
    TORCH_CHECK(scales_B.size(0) == N, "scales_B must have shape (N,); got (", scales_B.size(0), ")");

    auto C = torch::empty({M, N}, scales_A.options());
    w4a8_matmul_launcher(
        Aq.data_ptr<int8_t>(),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scales_B.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a4_group_matmul(const torch::Tensor& Aq, const torch::Tensor& Bq,
                                 const torch::Tensor& scales_A, const torch::Tensor& scales_B,
                                 const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(Aq);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales_A);
    CHECK_INPUT(scales_B);
    TORCH_CHECK(Aq.dtype() == torch::kUInt8, "Aq must be uint8 (packed signed int4 codes, 2 per byte)");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed signed int4 codes, 2 per byte)");
    TORCH_CHECK(scales_A.dtype() == torch::kFloat16, "scales_A must be float16");
    TORCH_CHECK(scales_B.dtype() == torch::kFloat16, "scales_B must be float16");
    TORCH_CHECK(Aq.dim() == 2 && Bq.dim() == 2, "Aq and Bq must be 2D");
    TORCH_CHECK(scales_A.dim() == 1, "scales_A must be 1D (M,)");
    TORCH_CHECK(scales_B.dim() == 2, "scales_B must be 2D (N, K/128)");

    int M = Aq.size(0), K = Aq.size(1) * 2, N = Bq.size(0);
    // The groupwise kernel hoists one scale per (n, group) across an entire
    // BLOCK_K=128 tile, which only makes sense if every such tile falls
    // inside exactly one group -- true as long as K % 128 == 0 (the fixed
    // group size). cp.async alignment (K % 32 == 0) is subsumed by this.
    TORCH_CHECK((K % 128) == 0,
                "w4a4_group_matmul requires K to be a multiple of 128 (the "
                "fixed group size); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Aq K=", K, " Bq=(", Bq.size(0), ",", Bq.size(1), ")");
    TORCH_CHECK(scales_A.size(0) == M, "scales_A must have shape (M,); got (", scales_A.size(0), ")");
    TORCH_CHECK(scales_B.size(0) == N && scales_B.size(1) == K / 128,
                "scales_B must have shape (N, K/128); got (", scales_B.size(0), ",", scales_B.size(1), ")");

    auto C = torch::empty({M, N}, scales_A.options());
    w4a4_group_matmul_launcher(
        Aq.data_ptr<uint8_t>(),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scales_B.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

torch::Tensor w4a8_group_matmul(const torch::Tensor& Aq, const torch::Tensor& Bq,
                                 const torch::Tensor& scales_A, const torch::Tensor& scales_B,
                                 const c10::optional<torch::Tensor>& bias = c10::nullopt) {
    CHECK_INPUT(Aq);
    CHECK_INPUT(Bq);
    CHECK_INPUT(scales_A);
    CHECK_INPUT(scales_B);
    TORCH_CHECK(Aq.dtype() == torch::kInt8, "Aq must be int8 (plain two's-complement bytes, unpacked)");
    TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8 (packed signed int4 codes, 2 per byte)");
    TORCH_CHECK(scales_A.dtype() == torch::kFloat16, "scales_A must be float16");
    TORCH_CHECK(scales_B.dtype() == torch::kFloat16, "scales_B must be float16");
    TORCH_CHECK(Aq.dim() == 2 && Bq.dim() == 2, "Aq and Bq must be 2D");
    TORCH_CHECK(scales_A.dim() == 1, "scales_A must be 1D (M,)");
    TORCH_CHECK(scales_B.dim() == 2, "scales_B must be 2D (N, K/128)");

    int M = Aq.size(0), K = Aq.size(1), N = Bq.size(0);
    // Same BLOCK_K==GROUP_SIZE==128 requirement as w4a4_group_matmul (see
    // matmul_kernel_w4a8_group's top comment).
    TORCH_CHECK((K % 128) == 0,
                "w4a8_group_matmul requires K to be a multiple of 128 (the "
                "fixed group size); got K=", K);
    TORCH_CHECK(Bq.size(1) == K / 2,
                "Bq must have shape (N, K/2); got Aq K=", K, " Bq=(", Bq.size(0), ",", Bq.size(1), ")");
    TORCH_CHECK(scales_A.size(0) == M, "scales_A must have shape (M,); got (", scales_A.size(0), ")");
    TORCH_CHECK(scales_B.size(0) == N && scales_B.size(1) == K / 128,
                "scales_B must have shape (N, K/128); got (", scales_B.size(0), ",", scales_B.size(1), ")");

    auto C = torch::empty({M, N}, scales_A.options());
    w4a8_group_matmul_launcher(
        Aq.data_ptr<int8_t>(),
        Bq.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scales_B.data_ptr<at::Half>()),
        get_bias_ptr(bias, N),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K,
        at::cuda::getCurrentCUDAStream()
    );
    return C;
}

std::vector<torch::Tensor> quantize_sym_int8(const torch::Tensor& A) {
    CHECK_INPUT(A);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    const int M = A.size(0), K = A.size(1);
    // The reduction/quantize pass vectorizes over 8 halfs (128 bits) at a
    // time; K % 8 == 0 keeps every row's vector chunks aligned.
    TORCH_CHECK((K % 8) == 0, "quantize_sym_int8 requires K to be a multiple of 8; got K=", K);

    auto Aq = torch::empty({M, K}, A.options().dtype(torch::kInt8));
    auto scales = torch::empty({M}, A.options());
    quantize_sym_int8_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Aq.data_ptr<int8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        M, K,
        at::cuda::getCurrentCUDAStream()
    );
    return {Aq, scales};
}

std::vector<torch::Tensor> quantize_sym_int4(const torch::Tensor& A) {
    CHECK_INPUT(A);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    const int M = A.size(0), K = A.size(1);
    TORCH_CHECK((K % 8) == 0, "quantize_sym_int4 requires K to be a multiple of 8; got K=", K);

    auto Aq = torch::empty({M, K / 2}, A.options().dtype(torch::kUInt8));
    auto scales = torch::empty({M}, A.options());
    quantize_sym_int4_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Aq.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(scales.data_ptr<at::Half>()),
        M, K,
        at::cuda::getCurrentCUDAStream()
    );
    return {Aq, scales};
}

// // FP16 matmul targeting compute capability 8.9 (Ada Lovelace).
// // Same signature as matmul_16x16; uses cp.async + mma.sync.m16n8k16 instead
// // of WGMMA/TMA. Runs on any sm_80+ device but is tuned for sm_89.
// torch::Tensor matmul_sm89(torch::Tensor A, torch::Tensor B) {
//     CHECK_INPUT(A);
//     CHECK_INPUT(B);
//     TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
//     TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
//     TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
//     TORCH_CHECK(A.size(1) == B.size(1),
//                 "Inner dimensions must match: A is (M, K), B is (N, K); "
//                 "got A=(", A.size(0), ",", A.size(1), ") B=(",
//                 B.size(0), ",", B.size(1), ")");
//     int M = A.size(0), K = A.size(1), N = B.size(0);
//     TORCH_CHECK((K % 8) == 0,
//                 "matmul_sm89 requires K to be a multiple of 8 (cp.async 16-byte vectors); got K=", K);

//     auto C = torch::empty({M, N}, A.options());
//     matmul_sm89_launcher(
//         reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//         reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
//         reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
//         M, N, K
//     );
//     return C;
// }

// torch::Tensor cutlass_int8_matmul(torch::Tensor A, torch::Tensor B) {
//     // A: (M, K) row-major int8
//     // B: (N, K) row-major int8  -- this is K x N column-major from CUTLASS's
//     //    perspective (data layout: element [k, n] at offset k + n*K), which
//     //    matches the kernel's ColumnMajor LayoutB with ldb=K.
//     // C: (M, N) row-major int32
//     CHECK_INPUT(A);
//     CHECK_INPUT(B);
//     TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
//     TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
//     TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

//     int M = A.size(0);
//     int K = A.size(1);
//     int N = B.size(0);
//     TORCH_CHECK(B.size(1) == K,
//                 "B must have shape (N, K) so it appears as a K x N column-major matrix; "
//                 "got B shape (", B.size(0), ", ", B.size(1), ") with K=", K);

//     auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
//     torch::Tensor C = torch::empty({M, N}, options);

//     cutlass::Status status = cutlass_int8_matmul_launcher(
//         A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), C.data_ptr<int32_t>(),
//         M, N, K, at::cuda::getCurrentCUDAStream());

//     TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM execution failed.");

//     return C;
// }

// torch::Tensor cublaslt_int8_matmul(torch::Tensor A, torch::Tensor B) {
//     // Same layout convention as cutlass_int8_matmul:
//     //   A : (M, K) int8 row-major
//     //   B : (N, K) int8 row-major  (== K x N column-major with ld=K)
//     //   C : (M, N) int32 row-major  =  A @ B.T
//     CHECK_INPUT(A);
//     CHECK_INPUT(B);
//     TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
//     TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
//     TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

//     int M = A.size(0);
//     int K = A.size(1);
//     int N = B.size(0);
//     TORCH_CHECK(B.size(1) == K,
//                 "B must have shape (N, K); got (", B.size(0), ", ", B.size(1),
//                 ") with K=", K);

//     auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
//     torch::Tensor C = torch::empty({M, N}, options);

//     cublasStatus_t status = cublaslt_int8_matmul_launcher(
//         A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), C.data_ptr<int32_t>(),
//         M, N, K, at::cuda::getCurrentCUDAStream());

//     TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
//                 "cuBLASLt int8 GEMM failed with status=", static_cast<int>(status));
//     return C;
// }

// std::tuple<torch::Tensor, torch::Tensor> per_token_sym_quant(torch::Tensor X) {
//     // X : (B, T, C) float32 or float16, contiguous, CUDA
//     // returns:
//     //   Q     : (B, T, C) int8
//     //   scale : (B, T)    float32
//     CHECK_INPUT(X);
//     TORCH_CHECK(X.dim() == 3, "X must be (B, T, C)");
//     TORCH_CHECK(X.scalar_type() == torch::kFloat32 ||
//                 X.scalar_type() == torch::kFloat16,
//                 "X must be float32 or float16");

//     const int64_t B = X.size(0);
//     const int64_t T = X.size(1);
//     const int64_t C = X.size(2);
//     const int64_t rows = B * T;
//     TORCH_CHECK(rows <= std::numeric_limits<int>::max(),
//                 "rows = B*T exceeds int range");
//     TORCH_CHECK(C    <= std::numeric_limits<int>::max(),
//                 "C exceeds int range");

//     auto Q = torch::empty({B, T, C},
//                           torch::TensorOptions().dtype(torch::kInt8).device(X.device()));
//     auto scale = torch::empty({B, T},
//                               torch::TensorOptions().dtype(torch::kFloat32).device(X.device()));

//     cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     if (X.scalar_type() == torch::kFloat32) {
//         per_token_sym_quant_launcher_f32(
//             X.data_ptr<float>(),
//             Q.data_ptr<int8_t>(),
//             scale.data_ptr<float>(),
//             static_cast<int>(rows), static_cast<int>(C), stream);
//     } else {
//         per_token_sym_quant_launcher_f16(
//             reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
//             Q.data_ptr<int8_t>(),
//             scale.data_ptr<float>(),
//             static_cast<int>(rows), static_cast<int>(C), stream);
//     }
//     return {Q, scale};
// }

// torch::Tensor rescale_int32_rowcol(torch::Tensor C, torch::Tensor sA, torch::Tensor sB,
//                                    c10::optional<torch::ScalarType> out_dtype) {
//     // C : (M, N) int32 row-major, contiguous, CUDA
//     // sA: (M,)   float32
//     // sB: (N,)   float32
//     // returns Y[m,n] = C[m,n] * sA[m] * sB[n] in float16 (default) or float32.
//     CHECK_INPUT(C);
//     CHECK_INPUT(sA);
//     CHECK_INPUT(sB);
//     TORCH_CHECK(C.dtype()  == torch::kInt32,   "C must be int32");
//     TORCH_CHECK(sA.dtype() == torch::kFloat32, "sA must be float32");
//     TORCH_CHECK(sB.dtype() == torch::kFloat32, "sB must be float32");
//     TORCH_CHECK(C.dim() == 2 && sA.dim() == 1 && sB.dim() == 1,
//                 "C must be 2D, sA/sB must be 1D");
//     const int64_t M = C.size(0);
//     const int64_t N = C.size(1);
//     TORCH_CHECK(sA.size(0) == M, "sA length must match C.size(0)");
//     TORCH_CHECK(sB.size(0) == N, "sB length must match C.size(1)");
//     TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
//                 N <= std::numeric_limits<int>::max(),
//                 "M or N exceeds int range");

//     auto dtype = out_dtype.value_or(torch::kFloat16);
//     TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kFloat32,
//                 "out_dtype must be float16 or float32");
//     auto Y = torch::empty({M, N},
//                           torch::TensorOptions().dtype(dtype).device(C.device()));

//     cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     if (dtype == torch::kFloat16) {
//         rescale_int32_rowcol_launcher_f16(
//             C.data_ptr<int32_t>(), sA.data_ptr<float>(), sB.data_ptr<float>(),
//             reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
//             static_cast<int>(M), static_cast<int>(N), stream);
//     } else {
//         rescale_int32_rowcol_launcher_f32(
//             C.data_ptr<int32_t>(), sA.data_ptr<float>(), sB.data_ptr<float>(),
//             Y.data_ptr<float>(),
//             static_cast<int>(M), static_cast<int>(N), stream);
//     }
//     return Y;
// }

// torch::Tensor unpack_int4_to_int8(torch::Tensor Bp, int64_t C) {
//     // Bp : (N, C/2) int8 storage holding packed signed 4-bit values.
//     //      Packing convention: byte b at column j stores
//     //        low  nibble (bits 0..3) -> output column 2*j
//     //        high nibble (bits 4..7) -> output column 2*j + 1
//     //      Each nibble is treated as a signed 4-bit int in [-8, 7] and is
//     //      sign-extended to int8.
//     // C  : the unpacked column count. Must equal 2 * Bp.size(1).
//     // returns: (N, C) int8.
//     CHECK_INPUT(Bp);
//     TORCH_CHECK(Bp.dtype() == torch::kInt8, "Bp must be int8-storage (packed nibbles)");
//     TORCH_CHECK(Bp.dim() == 2, "Bp must be 2D (N, C/2)");
//     const int64_t N = Bp.size(0);
//     const int64_t Cp = Bp.size(1);
//     TORCH_CHECK(C == 2 * Cp, "C must equal 2 * Bp.size(1); got C=", C,
//                 " Bp.size(1)=", Cp);

//     auto B = torch::empty({N, C},
//                           torch::TensorOptions().dtype(torch::kInt8).device(Bp.device()));

//     cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     unpack_int4_to_int8_launcher(
//         reinterpret_cast<const uint8_t*>(Bp.data_ptr<int8_t>()),
//         B.data_ptr<int8_t>(),
//         static_cast<size_t>(N) * static_cast<size_t>(Cp),
//         stream);
//     return B;
// }

// torch::Tensor matmul_int4xint8_cublas(torch::Tensor A, torch::Tensor Bp) {
//     // A  : (M, C)    int8, row-major contiguous
//     // Bp : (N, C/2)  int8 holding packed signed int4 weights
//     // Returns (M, N) int32 = A @ unpack(Bp).T
//     CHECK_INPUT(A);
//     CHECK_INPUT(Bp);
//     TORCH_CHECK(A.dtype() == torch::kInt8 && Bp.dtype() == torch::kInt8,
//                 "A and Bp must be int8");
//     TORCH_CHECK(A.dim() == 2 && Bp.dim() == 2, "A and Bp must be 2D");
//     const int64_t M = A.size(0);
//     const int64_t C = A.size(1);
//     const int64_t N = Bp.size(0);
//     TORCH_CHECK(Bp.size(1) * 2 == C,
//                 "Bp must have shape (N, C/2); got Bp=(", Bp.size(0), ",", Bp.size(1),
//                 ") with A.size(1)=", C);

//     torch::Tensor B = unpack_int4_to_int8(Bp, C);

//     auto Cout = torch::empty({M, N},
//                              torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
//     cublasStatus_t status = cublaslt_int8_matmul_launcher(
//         A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), Cout.data_ptr<int32_t>(),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(C),
//         at::cuda::getCurrentCUDAStream());
//     TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
//                 "cuBLASLt int8 GEMM (int4xint8 path) failed with status=",
//                 static_cast<int>(status));
//     return Cout;
// }


// torch::Tensor matmul_int4_cublas(torch::Tensor Ap, torch::Tensor Bp) {
//     // Native int4 x int4 GEMM via cuBLASLt (CUDA_R_4I IMMA tensor cores).
//     //
//     //   Ap : (M, K/2) int8 storage holding packed signed int4 activations
//     //   Bp : (N, K/2) int8 storage holding packed signed int4 weights
//     // Packing: byte at column j stores
//     //   low  nibble (bits 0..3) -> element 2*j
//     //   high nibble (bits 4..7) -> element 2*j + 1
//     // Returns (M, N) int32 = Ap_unpacked @ Bp_unpacked.T
//     //
//     // cuBLASLt requires K % 32 == 0 for the int4 path.
//     CHECK_INPUT(Ap);
//     CHECK_INPUT(Bp);
//     TORCH_CHECK(Ap.dtype() == torch::kInt8 && Bp.dtype() == torch::kInt8,
//                 "Ap and Bp must be int8 storage (packed int4)");
//     TORCH_CHECK(Ap.dim() == 2 && Bp.dim() == 2, "Ap and Bp must be 2D");

//     const int64_t M  = Ap.size(0);
//     const int64_t Kp = Ap.size(1);
//     const int64_t N  = Bp.size(0);
//     TORCH_CHECK(Bp.size(1) == Kp,
//                 "Bp must have shape (N, K/2) matching Ap's K; got Ap=(",
//                 Ap.size(0), ",", Ap.size(1), ") Bp=(",
//                 Bp.size(0), ",", Bp.size(1), ")");
//     const int64_t K = Kp * 2;
//     TORCH_CHECK((K % 32) == 0,
//                 "K (unpacked) must be a multiple of 32 for the cuBLASLt int4 "
//                 "path; got K=", K);

//     auto Cout = torch::empty({M, N},
//                              torch::TensorOptions().dtype(torch::kInt32).device(Ap.device()));
//     cublasStatus_t status = cublaslt_int4_matmul_launcher(
//         Ap.data_ptr<int8_t>(), Bp.data_ptr<int8_t>(), Cout.data_ptr<int32_t>(),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
//         at::cuda::getCurrentCUDAStream());
//     TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
//                 "cuBLASLt int4 GEMM failed with status=",
//                 static_cast<int>(status));
//     return Cout;
// }


// torch::Tensor matmul_int8_sm89(torch::Tensor A, torch::Tensor B) {
//     // Hand-written int8 x int8 -> int32 GEMM (tensor cores, mma.m16n8k32.s8).
//     //
//     //   A : (M, K) int8 row-major  (activations)
//     //   B : (N, K) int8 row-major  (weights; == K x N column-major)
//     // Returns (M, N) int32 = A @ B.T
//     //
//     // Requires K % 128 == 0 and N % 128 == 0. Any M is supported.
//     CHECK_INPUT(A);
//     CHECK_INPUT(B);
//     TORCH_CHECK(A.dtype() == torch::kInt8 && B.dtype() == torch::kInt8,
//                 "A and B must be int8");
//     TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

//     const int64_t M = A.size(0);
//     const int64_t K = A.size(1);
//     const int64_t N = B.size(0);
//     TORCH_CHECK(B.size(1) == K,
//                 "B must have shape (N, K) matching A's K; got A=(",
//                 A.size(0), ",", A.size(1), ") B=(", B.size(0), ",", B.size(1), ")");
//     TORCH_CHECK((K % 128) == 0,
//                 "K must be a multiple of 128 for matmul_int8_sm89; got K=", K);
//     TORCH_CHECK((N % 128) == 0,
//                 "N must be a multiple of 128 for matmul_int8_sm89; got N=", N);

//     auto Cout = torch::empty({M, N},
//                              torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
//     matmul_int8_sm89_launcher(
//         reinterpret_cast<const uint16_t*>(A.data_ptr<int8_t>()),
//         reinterpret_cast<const uint16_t*>(B.data_ptr<int8_t>()),
//         Cout.data_ptr<int32_t>(),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     return Cout;
// }


// torch::Tensor matmul_int4(torch::Tensor Ap, torch::Tensor Bp) {
//     // Hand-written int4 x int4 -> int32 GEMM (tensor cores, mma.m16n8k64.s4).
//     //
//     //   Ap : (M, K/2) int8 storage holding packed signed int4 activations
//     //   Bp : (N, K/2) int8 storage holding packed signed int4 weights
//     // Packing: byte at column j stores
//     //   low  nibble (bits 0..3) -> element 2*j
//     //   high nibble (bits 4..7) -> element 2*j + 1
//     // Returns (M, N) int32 = Ap_unpacked @ Bp_unpacked.T
//     //
//     // Requires K (unpacked) % 256 == 0 and N % 128 == 0. Any M is supported.
//     CHECK_INPUT(Ap);
//     CHECK_INPUT(Bp);
//     TORCH_CHECK(Ap.dtype() == torch::kInt8 && Bp.dtype() == torch::kInt8,
//                 "Ap and Bp must be int8 storage (packed int4)");
//     TORCH_CHECK(Ap.dim() == 2 && Bp.dim() == 2, "Ap and Bp must be 2D");

//     const int64_t M  = Ap.size(0);
//     const int64_t Kp = Ap.size(1);
//     const int64_t N  = Bp.size(0);
//     TORCH_CHECK(Bp.size(1) == Kp,
//                 "Bp must have shape (N, K/2) matching Ap's K; got Ap=(",
//                 Ap.size(0), ",", Ap.size(1), ") Bp=(",
//                 Bp.size(0), ",", Bp.size(1), ")");
//     const int64_t K = Kp * 2;
//     TORCH_CHECK((K % 256) == 0,
//                 "K (unpacked) must be a multiple of 256 for matmul_int4; got K=", K);
//     TORCH_CHECK((N % 128) == 0,
//                 "N must be a multiple of 128 for matmul_int4; got N=", N);

//     auto Cout = torch::empty({M, N},
//                              torch::TensorOptions().dtype(torch::kInt32).device(Ap.device()));
//     matmul_int4_sm89_launcher(
//         reinterpret_cast<const uint16_t*>(Ap.data_ptr<int8_t>()),
//         reinterpret_cast<const uint16_t*>(Bp.data_ptr<int8_t>()),
//         Cout.data_ptr<int32_t>(),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     return Cout;
// }


// torch::Tensor matmul_wxax(torch::Tensor A, torch::Tensor Wp, torch::Tensor sW, int64_t bits) {
//     // Fused f16 x grouped-quantized-weight GEMM.
//     //   A  : (M, K)            f16  activations (quantized per-token on the fly)
//     //   Wp : (N, K*bits/16) int16-storage packed signed bits-bit weights, OR the
//     //        equivalent int8 storage: int4 -> (N, K/2) int8, int8 -> (N, K) int8.
//     //        Interpreted as uint16 words: (N, K*bits/16).
//     //   sW : (N, K/128)        f16  per-(channel, group) weight scales.
//     // Returns Y (M, N) f16 = dequant(quant(A) @ Wp^T).
//     //
//     // Requires bits in {4, 8}, N % 128 == 0, K % 128 == 0, and K % 256 == 0 for
//     // int4 (the K-tile is 256 int4). Any M is supported.
//     CHECK_INPUT(A);
//     CHECK_INPUT(Wp);
//     CHECK_INPUT(sW);
//     TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
//     TORCH_CHECK(Wp.dtype() == torch::kInt8, "Wp must be int8 storage (packed weights)");
//     TORCH_CHECK(sW.dtype() == torch::kFloat16, "sW must be float16");
//     TORCH_CHECK(A.dim() == 2 && Wp.dim() == 2 && sW.dim() == 2, "A, Wp, sW must be 2D");
//     TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");

//     const int64_t M = A.size(0);
//     const int64_t K = A.size(1);
//     const int64_t N = Wp.size(0);
//     const int64_t bytes_per_row = Wp.size(1);          // int4: K/2, int8: K
//     const int64_t expect_bytes = (bits == 4) ? (K / 2) : K;
//     TORCH_CHECK(bytes_per_row == expect_bytes,
//                 "Wp must have shape (N, ", expect_bytes, ") for bits=", bits,
//                 "; got (", Wp.size(0), ", ", Wp.size(1), ") with K=", K);
//     TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);
//     TORCH_CHECK((K % 128) == 0, "K must be a multiple of 128; got K=", K);
//     if (bits == 4)
//         TORCH_CHECK((K % 256) == 0, "K must be a multiple of 256 for bits=4; got K=", K);
//     TORCH_CHECK(sW.size(0) == N && sW.size(1) == K / 128,
//                 "sW must have shape (N, K/128) = (", N, ", ", K / 128,
//                 "); got (", sW.size(0), ", ", sW.size(1), ")");

//     auto Y  = torch::empty({M, N}, A.options());
//     auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
//     auto i16 = torch::TensorOptions().dtype(torch::kInt16).device(A.device());
//     auto sA  = torch::empty({M}, f32);
//     const int64_t K_words = K * bits / 16;          // int4: K/4, int8: K/2
//     auto qA  = torch::empty({M, K_words}, i16);      // packed quantized activations

//     matmul_wxax_sm89_launcher(
//         reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//         reinterpret_cast<const uint16_t*>(Wp.data_ptr<int8_t>()),
//         reinterpret_cast<const __half*>(sW.data_ptr<at::Half>()),
//         reinterpret_cast<uint16_t*>(qA.data_ptr<int16_t>()),
//         sA.data_ptr<float>(),
//         reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
//         static_cast<int>(bits), at::cuda::getCurrentCUDAStream());
//     return Y;
// }

// torch::Tensor matmul_f16xint4_fn(torch::Tensor A, torch::Tensor Bp,
//                                  torch::Tensor sB, torch::Tensor zB) {
//     // f16 activation x grouped-int4 weight -> f16 (dequant-to-f16, f16 GEMM).
//     //   A  : (M, K)     f16 activations, row-major
//     //   Bp : (N, K/2)   uint8 packed UNSIGNED int4 weights (low nibble = even k)
//     //   sB : (N, K/128) f16 group scale
//     //   zB : (N, K/128) f16 group offset
//     //        w[n,k] = sB[n,g] * Bp_nibble[n,k] + zB[n,g],  g = k / 128
//     // Returns C (M, N) f16 = A @ dequant(Bp)^T.  Requires K%128==0, N%128==0.
//     CHECK_INPUT(A);
//     CHECK_INPUT(Bp);
//     CHECK_INPUT(sB);
//     CHECK_INPUT(zB);
//     TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
//     TORCH_CHECK(Bp.dtype() == torch::kUInt8, "Bp must be uint8 (packed unsigned nibbles)");
//     TORCH_CHECK(sB.dtype() == torch::kFloat16 && zB.dtype() == torch::kFloat16,
//                 "sB and zB must be float16");
//     TORCH_CHECK(A.dim() == 2 && Bp.dim() == 2 && sB.dim() == 2 && zB.dim() == 2,
//                 "A, Bp, sB, zB must be 2D");

//     const int64_t M = A.size(0);
//     const int64_t K = A.size(1);
//     const int64_t N = Bp.size(0);
//     constexpr int64_t GROUP = 128;
//     TORCH_CHECK(Bp.size(1) == K / 2,
//                 "Bp must have shape (N, K/2); got (", Bp.size(0), ",", Bp.size(1),
//                 ") with K=", K);
//     TORCH_CHECK((K % GROUP) == 0, "K must be a multiple of 128; got K=", K);
//     TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);
//     TORCH_CHECK(sB.size(0) == N && sB.size(1) == K / GROUP,
//                 "sB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");
//     TORCH_CHECK(zB.size(0) == N && zB.size(1) == K / GROUP,
//                 "zB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");

//     auto C = torch::empty({M, N}, A.options());
//     matmul_f16xint4(
//         reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//         Bp.data_ptr<uint8_t>(),
//         reinterpret_cast<__half*>(sB.data_ptr<at::Half>()),
//         reinterpret_cast<__half*>(zB.data_ptr<at::Half>()),
//         reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     return C;
// }

// torch::Tensor matmul_w4a4(torch::Tensor A, torch::Tensor Wp, torch::Tensor sB) {
//     // Per-channel-symmetric W4A4: quantize A to int4 per-token on the fly, run
//     // the pure int4 tensor-core GEMM, and rescale by sA[m]*sB[n].
//     //   A  : (M, K)   f16 activations
//     //   Wp : (N, K/2) int8 packed signed int4 weights (per-channel symmetric)
//     //   sB : (N,)     f32 per-channel weight scale
//     // Returns Y (M, N) f16.  Requires K % 256 == 0 and N % 128 == 0.
//     //
//     // Unlike matmul_wxax (grouped), there is no per-group dequant in the K-loop:
//     // all of K accumulates in int32 and a single sA[m]*sB[n] rescale runs in the
//     // epilogue, so the GEMM is the fast pure-int4 kernel.
//     CHECK_INPUT(A);
//     CHECK_INPUT(Wp);
//     CHECK_INPUT(sB);
//     TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
//     TORCH_CHECK(Wp.dtype() == torch::kInt8, "Wp must be int8 (packed signed int4)");
//     TORCH_CHECK(sB.dtype() == torch::kFloat32, "sB must be float32");
//     TORCH_CHECK(A.dim() == 2 && Wp.dim() == 2 && sB.dim() == 1, "A,Wp 2D; sB 1D");

//     const int64_t M = A.size(0);
//     const int64_t K = A.size(1);
//     const int64_t N = Wp.size(0);
//     TORCH_CHECK(Wp.size(1) == K / 2, "Wp must be (N, K/2); got (", Wp.size(0), ",",
//                 Wp.size(1), ") with K=", K);
//     TORCH_CHECK(sB.size(0) == N, "sB must be (N,) = (", N, ")");
//     TORCH_CHECK((K % 256) == 0, "K must be a multiple of 256; got K=", K);
//     TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);

//     auto stream = at::cuda::getCurrentCUDAStream();
//     auto dev = A.device();
//     auto i16 = torch::TensorOptions().dtype(torch::kInt16).device(dev);
//     auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

//     // 1) quantize+pack A to int4 per-token (qA words + per-token scale sA).
//     auto qA = torch::empty({M, K / 4}, i16);
//     auto sA = torch::empty({M}, f32);
//     per_token_quant_int4_launcher(
//         reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//         reinterpret_cast<uint16_t*>(qA.data_ptr<int16_t>()),
//         sA.data_ptr<float>(), static_cast<int>(M), static_cast<int>(K), stream);

//     // 2) int4 GEMM with the per-channel rescale fused into the epilogue:
//     //    Y[m,n] = sA[m]*sB[n]*(qA @ qW^T), f16 -- no int32 M*N round-trip.
//     auto Y = torch::empty({M, N}, A.options());
//     matmul_int4_rescale_sm89_launcher(
//         reinterpret_cast<const uint16_t*>(qA.data_ptr<int16_t>()),
//         reinterpret_cast<const uint16_t*>(Wp.data_ptr<int8_t>()),
//         sA.data_ptr<float>(), sB.data_ptr<float>(),
//         reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     return Y;
// }

// torch::Tensor matmul_f16xint2_fn(torch::Tensor A, torch::Tensor Bp,
//                                  torch::Tensor sB, torch::Tensor zB) {
//     // W2A16: f16 activation x grouped-asymmetric int2 weight -> f16.
//     //   A  : (M, K)     f16 activations, row-major
//     //   Bp : (N, K/4)   uint8 packed UNSIGNED int2 (element e at bits [2e..2e+1])
//     //   sB : (N, K/128) f16 group scale
//     //   zB : (N, K/128) f16 group offset    w[n,k] = sB[n,g]*q + zB[n,g], g=k/128
//     // Returns C (M, N) f16.  Requires K % 128 == 0, N % 128 == 0.
//     CHECK_INPUT(A);
//     CHECK_INPUT(Bp);
//     CHECK_INPUT(sB);
//     CHECK_INPUT(zB);
//     TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
//     TORCH_CHECK(Bp.dtype() == torch::kUInt8, "Bp must be uint8 (packed unsigned int2)");
//     TORCH_CHECK(sB.dtype() == torch::kFloat16 && zB.dtype() == torch::kFloat16,
//                 "sB and zB must be float16");
//     TORCH_CHECK(A.dim() == 2 && Bp.dim() == 2 && sB.dim() == 2 && zB.dim() == 2,
//                 "A, Bp, sB, zB must be 2D");

//     const int64_t M = A.size(0);
//     const int64_t K = A.size(1);
//     const int64_t N = Bp.size(0);
//     constexpr int64_t GROUP = 128;
//     TORCH_CHECK(Bp.size(1) == K / 4,
//                 "Bp must have shape (N, K/4); got (", Bp.size(0), ",", Bp.size(1),
//                 ") with K=", K);
//     TORCH_CHECK((K % GROUP) == 0, "K must be a multiple of 128; got K=", K);
//     TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);
//     TORCH_CHECK(sB.size(0) == N && sB.size(1) == K / GROUP,
//                 "sB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");
//     TORCH_CHECK(zB.size(0) == N && zB.size(1) == K / GROUP,
//                 "zB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");

//     auto C = torch::empty({M, N}, A.options());
//     matmul_f16xint2(
//         reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//         Bp.data_ptr<uint8_t>(),
//         reinterpret_cast<__half*>(sB.data_ptr<at::Half>()),
//         reinterpret_cast<__half*>(zB.data_ptr<at::Half>()),
//         reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     return C;
// }

// torch::Tensor matmul_w4ax(torch::Tensor A, torch::Tensor Wp, torch::Tensor sB) {
//     // Unified W4Ax over ONE per-channel-symmetric int4 weight. Dispatches by M:
//     //   M <= 64 : W4A16 -- f16 activations, signed-symmetric dequant + f16 GEMM
//     //   M >  64 : W4A4  -- quantize A to int4 on the fly + int4 GEMM + rescale
//     // Both arms consume the SAME weights: Wp (N,K/2) int8 signed packed int4,
//     // sB (N,) f32 per-channel scale. Returns Y (M,N) f16.  Requires K%256==0,
//     // N%128==0.
//     CHECK_INPUT(A);
//     CHECK_INPUT(Wp);
//     CHECK_INPUT(sB);
//     TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
//     TORCH_CHECK(Wp.dtype() == torch::kInt8, "Wp must be int8 (packed signed int4)");
//     TORCH_CHECK(sB.dtype() == torch::kFloat32, "sB must be float32");
//     TORCH_CHECK(A.dim() == 2 && Wp.dim() == 2 && sB.dim() == 1, "A,Wp 2D; sB 1D");

//     const int64_t M = A.size(0);
//     const int64_t K = A.size(1);
//     const int64_t N = Wp.size(0);
//     TORCH_CHECK(Wp.size(1) == K / 2, "Wp must be (N, K/2); got (", Wp.size(0), ",",
//                 Wp.size(1), ") with K=", K);
//     TORCH_CHECK(sB.size(0) == N, "sB must be (N,) = (", N, ")");
//     TORCH_CHECK((K % 256) == 0, "K must be a multiple of 256; got K=", K);
//     TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);

//     constexpr int64_t W4AX_M_THRESHOLD = 64;
//     auto Y = torch::empty({M, N}, A.options());

//     if (M <= W4AX_M_THRESHOLD) {
//         // W4A16: f16 activations, weights dequantized to f16 in registers.
//         matmul_w4a16_sym_launcher(
//             reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//             reinterpret_cast<const uint8_t*>(Wp.data_ptr<int8_t>()),
//             sB.data_ptr<float>(),
//             reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
//             static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     } else {
//         // W4A4: quantize A to int4 per-token, int4 GEMM + fused per-channel rescale.
//         auto stream = at::cuda::getCurrentCUDAStream();
//         auto i16 = torch::TensorOptions().dtype(torch::kInt16).device(A.device());
//         auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
//         auto qA = torch::empty({M, K / 4}, i16);
//         auto sA = torch::empty({M}, f32);
//         per_token_quant_int4_launcher(
//             reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
//             reinterpret_cast<uint16_t*>(qA.data_ptr<int16_t>()),
//             sA.data_ptr<float>(), static_cast<int>(M), static_cast<int>(K), stream);
//         matmul_int4_rescale_sm89_launcher(
//             reinterpret_cast<const uint16_t*>(qA.data_ptr<int16_t>()),
//             reinterpret_cast<const uint16_t*>(Wp.data_ptr<int8_t>()),
//             sA.data_ptr<float>(), sB.data_ptr<float>(),
//             reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
//             static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     }
//     return Y;
// }

// // ---------------------------------------------------------------------------
// // CPU (AVX2/FMA) quantized matmuls: shared validation + dispatch, reused by
// // all 16 w{4,2}a{16,8}_cpu[_asym][_group[_asym]]_matmul Python entry points
// // below (each is a thin pybind lambda that fixes wbits/symmetric/groupwise
// // and calls straight into one of these two).
// #define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
// #define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

// static const float* get_bias_ptr_cpu(const c10::optional<torch::Tensor>& bias_opt, int64_t N) {
//     if (!bias_opt.has_value()) return nullptr;
//     const auto& bias = bias_opt.value();
//     CHECK_INPUT_CPU(bias);
//     TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
//     TORCH_CHECK(bias.dim() == 1 && bias.size(0) == N, "bias must have shape (N,); got ", bias.sizes());
//     return bias.data_ptr<float>();
// }

// // Validates scales (and, for asymmetric, zeros) against (wbits, groupwise)
// // and returns the raw zeros pointer (nullptr if symmetric).
// static const float* validate_weight_scales_cpu(bool symmetric, bool groupwise, int64_t N, int64_t K,
//                                                 const torch::Tensor& scales,
//                                                 const c10::optional<torch::Tensor>& zeros_opt) {
//     CHECK_INPUT_CPU(scales);
//     TORCH_CHECK(scales.dtype() == torch::kFloat32, "scales must be float32");
//     TORCH_CHECK(symmetric == !zeros_opt.has_value(),
//                 symmetric ? "zeros must not be passed for symmetric quantization"
//                           : "zeros is required for asymmetric quantization");
//     if (groupwise) {
//         TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128 (the fixed group size); got K=", K);
//         TORCH_CHECK(scales.dim() == 2 && scales.size(0) == N && scales.size(1) == K / 128,
//                     "scales must have shape (N, K/128); got ", scales.sizes());
//     } else {
//         TORCH_CHECK(scales.dim() == 1 && scales.size(0) == N,
//                     "scales must have shape (N,); got ", scales.sizes());
//     }
//     if (!zeros_opt.has_value()) return nullptr;
//     const auto& zeros = zeros_opt.value();
//     CHECK_INPUT_CPU(zeros);
//     TORCH_CHECK(zeros.dtype() == torch::kFloat32, "zeros must be float32");
//     TORCH_CHECK(zeros.sizes() == scales.sizes(), "zeros must have the same shape as scales");
//     return zeros.data_ptr<float>();
// }

// static torch::Tensor cpu_wxa16_matmul_impl(int wbits, bool symmetric, bool groupwise,
//                                             const torch::Tensor& A, const torch::Tensor& Bq,
//                                             const torch::Tensor& scales,
//                                             const c10::optional<torch::Tensor>& zeros,
//                                             const c10::optional<torch::Tensor>& bias) {
//     CHECK_INPUT_CPU(A);
//     CHECK_INPUT_CPU(Bq);
//     TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
//     TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8");
//     TORCH_CHECK(A.dim() == 2 && Bq.dim() == 2, "A and Bq must be 2D");

//     const int64_t M = A.size(0), K = A.size(1), N = Bq.size(0);
//     const int64_t codes_per_byte = 8 / wbits;
//     TORCH_CHECK(K % codes_per_byte == 0, "K must be a multiple of ", codes_per_byte,
//                 " for ", wbits, "-bit packing; got K=", K);
//     TORCH_CHECK(Bq.size(1) == K / codes_per_byte,
//                 "Bq must have shape (N, K*", wbits, "/8); got Bq=(", Bq.size(0), ",", Bq.size(1), ")");
//     const float* zeros_ptr = validate_weight_scales_cpu(symmetric, groupwise, N, K, scales, zeros);
//     const float* bias_ptr = get_bias_ptr_cpu(bias, N);

//     auto C = torch::empty({M, N}, A.options());
//     wxa16_cpu_matmul_launcher(wbits, symmetric, groupwise,
//         A.data_ptr<float>(), Bq.data_ptr<uint8_t>(), scales.data_ptr<float>(), zeros_ptr,
//         bias_ptr, C.data_ptr<float>(),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     return C;
// }

// static torch::Tensor cpu_wxa8_matmul_impl(int wbits, bool symmetric, bool groupwise,
//                                            const torch::Tensor& Aq, const torch::Tensor& scales_A,
//                                            const torch::Tensor& Bq, const torch::Tensor& scales_B,
//                                            const c10::optional<torch::Tensor>& zeros,
//                                            const c10::optional<torch::Tensor>& bias) {
//     CHECK_INPUT_CPU(Aq);
//     CHECK_INPUT_CPU(scales_A);
//     CHECK_INPUT_CPU(Bq);
//     TORCH_CHECK(Aq.dtype() == torch::kInt8, "Aq must be int8");
//     TORCH_CHECK(scales_A.dtype() == torch::kFloat32, "scales_A must be float32");
//     TORCH_CHECK(Bq.dtype() == torch::kUInt8, "Bq must be uint8");
//     TORCH_CHECK(Aq.dim() == 2 && Bq.dim() == 2, "Aq and Bq must be 2D");
//     TORCH_CHECK(scales_A.dim() == 1 && scales_A.size(0) == Aq.size(0),
//                 "scales_A must have shape (M,); got ", scales_A.sizes());

//     const int64_t M = Aq.size(0), K = Aq.size(1), N = Bq.size(0);
//     const int64_t codes_per_byte = 8 / wbits;
//     TORCH_CHECK(K % codes_per_byte == 0, "K must be a multiple of ", codes_per_byte,
//                 " for ", wbits, "-bit packing; got K=", K);
//     TORCH_CHECK(Bq.size(1) == K / codes_per_byte,
//                 "Bq must have shape (N, K*", wbits, "/8); got Bq=(", Bq.size(0), ",", Bq.size(1), ")");
//     const float* zeros_ptr = validate_weight_scales_cpu(symmetric, groupwise, N, K, scales_B, zeros);
//     const float* bias_ptr = get_bias_ptr_cpu(bias, N);

//     auto C = torch::empty({M, N}, scales_A.options());
//     wxa8_cpu_matmul_launcher(wbits, symmetric, groupwise,
//         Aq.data_ptr<int8_t>(), scales_A.data_ptr<float>(),
//         Bq.data_ptr<uint8_t>(), scales_B.data_ptr<float>(), zeros_ptr,
//         bias_ptr, C.data_ptr<float>(),
//         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
//     return C;
// }

// std::vector<torch::Tensor> quantize_sym_int8_cpu(const torch::Tensor& A) {
//     CHECK_INPUT_CPU(A);
//     TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
//     TORCH_CHECK(A.dim() == 2, "A must be 2D");
//     const int64_t M = A.size(0), K = A.size(1);

//     auto Aq = torch::empty({M, K}, A.options().dtype(torch::kInt8));
//     auto scales = torch::empty({M}, A.options());
//     quantize_sym_int8_cpu_launcher(A.data_ptr<float>(), Aq.data_ptr<int8_t>(), scales.data_ptr<float>(),
//                                     static_cast<int>(M), static_cast<int>(K));
//     return {Aq, scales};
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("matmul_16x16", &matmul_16x16, "FP16 matmul (CUDA)");
    // m.def("matmul_sm89", &matmul_sm89,
    //       "FP16 matmul tuned for sm_89 (Ada Lovelace). cp.async + mma.sync.m16n8k16. "
    //       "A (M,K) fp16, B (N,K) fp16 -> C (M,N) fp16. Requires K % 8 == 0.",
    //       py::arg("A"), py::arg("B"));
    // m.def("matmul_int8", &cutlass_int8_matmul, "CUTLASS Int8 Matrix Multiplication");
    // m.def("matmul_int8_sm89", &matmul_int8_sm89,
    //       "Hand-written int8 x int8 -> int32 tensor-core GEMM (mma.m16n8k32.s8). "
    //       "A (M, K) int8, B (N, K) int8; returns (M, N) int32 = A @ B.T. "
    //       "Requires K % 128 == 0 and N % 128 == 0.",
    //       py::arg("A"), py::arg("B"));
    // m.def("matmul_int8_cublas", &cublaslt_int8_matmul, "cuBLASLt Int8 Matrix Multiplication");
    // m.def("matmul_int4_cublas", &matmul_int4_cublas,
    //       "Native int4 x int4 GEMM via cuBLASLt. "
    //       "Ap (M, K/2) int8 storage, Bp (N, K/2) int8 storage (packed signed nibbles); "
    //       "returns (M, N) int32. Requires K % 32 == 0.",
    //       py::arg("Ap"), py::arg("Bp"));
    // m.def("per_token_sym_quant", &per_token_sym_quant,
    //       "Per-token symmetric int8 quantization. Input (B,T,C) float32/16, "
    //       "returns (Q int8 (B,T,C), scale float32 (B,T)).");
    // m.def("rescale_int32_rowcol", &rescale_int32_rowcol,
    //       "Y[m,n] = C[m,n] * sA[m] * sB[n].  C int32 (M,N), sA float32 (M,), "
    //       "sB float32 (N,). out_dtype defaults to float16.",
    //       py::arg("C"), py::arg("sA"), py::arg("sB"),
    //       py::arg("out_dtype") = c10::optional<torch::ScalarType>(torch::kFloat16));
    // m.def("unpack_int4_to_int8", &unpack_int4_to_int8,
    //       "Unpack signed int4 (low nibble = even col, high nibble = odd col) "
    //       "in a packed (N, C/2) int8 tensor to a (N, C) int8 tensor.",
    //       py::arg("Bp"), py::arg("C"));
    // m.def("matmul_int4xint8_cublas", &matmul_int4xint8_cublas,
    //       "A (M,C) int8 @ unpack(Bp (N,C/2) int4).T -> (M,N) int32 via cuBLASLt.",
    //       py::arg("A"), py::arg("Bp"));
    // m.def("matmul_int4", &matmul_int4,
    //       "Hand-written int4 x int4 -> int32 tensor-core GEMM (mma.m16n8k64.s4). "
    //       "Ap (M, K/2) int8 storage, Bp (N, K/2) int8 storage (packed signed nibbles); "
    //       "returns (M, N) int32. Requires K % 256 == 0 and N % 128 == 0.",
    //       py::arg("Ap"), py::arg("Bp"));
    // m.def("matmul_f16xint4", &matmul_f16xint4_fn,
    //       "f16 activation x grouped-int4 weight -> f16. Int4 weights are "
    //       "dequantized to f16 (w = sB*q + zB, unsigned nibble q, groupsize 128) "
    //       "and the matmul runs on f16 tensor cores. A (M,K) f16, Bp (N,K/2) uint8 "
    //       "packed unsigned nibbles, sB/zB (N,K/128) f16. Returns C (M,N) f16. "
    //       "Requires K % 128 == 0 and N % 128 == 0.",
    //       py::arg("A"), py::arg("Bp"), py::arg("sB"), py::arg("zB"));
    // m.def("matmul_f16xint2", &matmul_f16xint2_fn,
    //       "W2A16: f16 activation x grouped-asymmetric int2 weight -> f16. Int2 "
    //       "weights dequantized to f16 (w = sB*q + zB, unsigned q in [0,3], group "
    //       "128) + f16 tensor-core GEMM. A (M,K) f16, Bp (N,K/4) uint8 packed "
    //       "unsigned 2-bit, sB/zB (N,K/128) f16. Requires K%128==0, N%128==0.",
    //       py::arg("A"), py::arg("Bp"), py::arg("sB"), py::arg("zB"));
    // m.def("matmul_w4ax", &matmul_w4ax,
    //       "Unified W4Ax over one per-channel-symmetric int4 weight: W4A16 (f16 "
    //       "activations) for M<=64, W4A4 (int4 activations) for M>64. A (M,K) f16, "
    //       "Wp (N,K/2) int8 signed packed int4, sB (N,) f32 per-channel scale. "
    //       "Returns Y (M,N) f16. Requires K % 256 == 0 and N % 128 == 0.",
    //       py::arg("A"), py::arg("Wp"), py::arg("sB"));
    // m.def("matmul_w4a4", &matmul_w4a4,
    //       "Per-channel-symmetric W4A4: quantize A to int4 per-token on the fly, "
    //       "pure int4 tensor-core GEMM, single sA[m]*sB[n] rescale. A (M,K) f16, "
    //       "Wp (N,K/2) int8 packed signed int4, sB (N,) f32 per-channel scale. "
    //       "Returns Y (M,N) f16. Requires K % 256 == 0 and N % 128 == 0.",
    //       py::arg("A"), py::arg("Wp"), py::arg("sB"));
    // m.def("matmul_wxax", &matmul_wxax,
    //       "Fused f16-activation x grouped-per-channel quantized-weight GEMM. "
    //       "A (M,K) f16, Wp (N, K*bits/16) int8-storage packed signed weights, "
    //       "sW (N, K/128) f16 group scales; bits in {4,8}. Activations are "
    //       "quantized per-token on the fly. Returns Y (M,N) f16. "
    //       "Requires N % 128 == 0, K % 128 == 0 (K % 256 == 0 for bits=4).",
    //       py::arg("A"), py::arg("Wp"), py::arg("sW"), py::arg("bits"));
    m.def("mymatmul", &mymatmul,
          "Data-parallel fp16 GEMM: A (M,K) f16, B (N,K) f16 -> C (M,N) f16 = "
          "A @ B.T. No stream-K, no workspace/locks -- each output tile's full K "
          "reduction is owned start-to-finish by a single threadblock via a "
          "grid-stride loop. Tile shape (BLOCK_M, BLOCK_N) is chosen internally "
          "per call based on (M, N, K) to keep all SMs busy and avoid wasted "
          "padding compute for small M. Requires K % 8 == 0. Optional bias "
          "(N,) f16 is added inside the kernel's own epilogue.",
          py::arg("A"), py::arg("B"), py::arg("bias") = py::none());
    m.def("w4a16_matmul", &w4a16_matmul,
          "W4A16 GEMM: A (M,K) f16 activations, Bq (N,K/2) uint8 packed "
          "excess-8 (offset-binary) int4 nibbles -- nibble = signed_value + "
          "8, range 0..15, NOT two's complement -- (low nibble = even k, "
          "high nibble = odd k), scales (N,) f16 per-output-channel "
          "symmetric scale. Returns C (M,N) f16 = A @ dequant(Bq).T "
          "(+ bias). Requires K % 32 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    m.def("w2a16_matmul", &w2a16_matmul,
          "W2A16 GEMM: A (M,K) f16 activations, Bq (N,K/4) uint8 packed "
          "excess-2 (offset-binary) int2 codes, 4 per byte -- code = "
          "signed_value + 2, range 0..3, NOT two's complement -- (code j at "
          "bits[2j:2j+2), covering element 4*byte_index+j), scales (N,) f16 "
          "per-output-channel symmetric scale. Returns C (M,N) f16 = "
          "A @ dequant(Bq).T (+ bias). Requires K % 64 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    m.def("w4a16_group_matmul", &w4a16_group_matmul,
          "Groupwise W4A16 GEMM: same excess-8 packed-int4 Bq as "
          "w4a16_matmul, but scales (N, K/128) f16 -- one scale per "
          "(channel, 128-wide K group) instead of one per channel. Returns "
          "C (M,N) f16 = A @ dequant_group(Bq).T (+ bias). Requires "
          "K % 128 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    m.def("w2a16_group_matmul", &w2a16_group_matmul,
          "Groupwise W2A16 GEMM: same excess-2 packed-int2 Bq as "
          "w2a16_matmul, but scales (N, K/128) f16 -- one scale per "
          "(channel, 128-wide K group). Returns C (M,N) f16 = "
          "A @ dequant_group(Bq).T (+ bias). Requires K % 128 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    m.def("w4a16_asym_matmul", &w4a16_asym_matmul,
          "Asymmetric (zero-point) W4A16 GEMM: A (M,K) f16 activations, Bq "
          "(N,K/2) uint8 packed PLAIN UNSIGNED int4 codes (0..15, no "
          "excess-8 encoding), scales/zeros (N,) f16 per-output-channel. "
          "Returns C (M,N) f16 = A @ (scales*code + zeros).T (+ bias). "
          "Requires K % 32 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());
    m.def("w2a16_asym_matmul", &w2a16_asym_matmul,
          "Asymmetric (zero-point) W2A16 GEMM: A (M,K) f16 activations, Bq "
          "(N,K/4) uint8 packed PLAIN UNSIGNED int2 codes (0..3, no "
          "excess-2 encoding), 4 per byte, scales/zeros (N,) f16 "
          "per-output-channel. Returns C (M,N) f16 = A @ (scales*code + "
          "zeros).T (+ bias). Requires K % 64 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());
    m.def("w4a16_group_asym_matmul", &w4a16_group_asym_matmul,
          "Groupwise asymmetric (zero-point) W4A16 GEMM: same PLAIN "
          "UNSIGNED packed-int4 Bq as w4a16_asym_matmul, but scales, zeros "
          "(N, K/128) f16 -- one scale/zero per (channel, 128-wide K "
          "group) instead of one per channel. Returns C (M,N) f16 = "
          "A @ (scales*code + zeros).T (grouped) (+ bias). Requires "
          "K % 128 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());
    m.def("w2a16_group_asym_matmul", &w2a16_group_asym_matmul,
          "Groupwise asymmetric (zero-point) W2A16 GEMM: same PLAIN "
          "UNSIGNED packed-int2 Bq as w2a16_asym_matmul, but scales, zeros "
          "(N, K/128) f16 -- one scale/zero per (channel, 128-wide K "
          "group). Returns C (M,N) f16 = A @ (scales*code + zeros).T "
          "(grouped) (+ bias). Requires K % 128 == 0.",
          py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());
    m.def("w4a4_matmul", &w4a4_matmul,
          "W4A4 GEMM: BOTH operands packed int4, symmetric, run through the "
          "hardware int4xint4 tensor core directly (int32 accumulate, no "
          "fp16 dequant in the K-loop). Aq (M,K/2) uint8, Bq (N,K/2) uint8, "
          "both plain two's-complement signed nibbles (range [-8,7]), 2 per "
          "byte (low nibble = even k, high nibble = odd k) -- NOT excess-8 "
          "encoded. scales_A (M,) f16 per-row scale for A, scales_B (N,) "
          "f16 per-channel scale for B. Returns C (M,N) f16 = "
          "(scales_A[:,None]*scales_B[None,:]) * (dequant(Aq) @ dequant(Bq).T) "
          "(+ bias). Requires K % 32 == 0.",
          py::arg("Aq"), py::arg("Bq"), py::arg("scales_A"), py::arg("scales_B"), py::arg("bias") = py::none());
    m.def("w4a8_matmul", &w4a8_matmul,
          "W4A8 GEMM: int8 activations x packed-int4 weights, run through "
          "the hardware int8 tensor core (B is unpacked register-resident "
          "straight out of the packed cp.async staging buffer, since "
          "there's no int4xint8 MMA instruction). Aq (M,K) int8, plain "
          "two's-complement bytes; Bq (N,K/2) uint8 packed two's-complement "
          "int4 nibbles (NOT excess-8 encoded), 2 per byte (low nibble = "
          "even k, high nibble = odd k). scales_A (M,) f16 per-row scale "
          "for A, scales_B (N,) f16 per-channel scale for B. Returns C "
          "(M,N) f16 = (scales_A[:,None]*scales_B[None,:]) * "
          "(Aq @ dequant(Bq).T) (+ bias). Requires K % 32 == 0.",
          py::arg("Aq"), py::arg("Bq"), py::arg("scales_A"), py::arg("scales_B"), py::arg("bias") = py::none());
    m.def("w4a4_group_matmul", &w4a4_group_matmul,
          "Groupwise W4A4 GEMM: same packed-int4 operand layout as "
          "w4a4_matmul, but scales_B (N, K/128) f16 -- one scale per "
          "(output channel, 128-wide K group) instead of one per channel; "
          "scales_A (M,) f16 per-row (per-token) is unchanged. Returns C "
          "(M,N) f16 (+ bias). Requires K % 128 == 0.",
          py::arg("Aq"), py::arg("Bq"), py::arg("scales_A"), py::arg("scales_B"), py::arg("bias") = py::none());
    m.def("w4a8_group_matmul", &w4a8_group_matmul,
          "Groupwise W4A8 GEMM: same operand layout as w4a8_matmul (Aq "
          "(M,K) int8, Bq (N,K/2) packed int4), but scales_B (N, K/128) f16 "
          "-- one scale per (output channel, 128-wide K group) instead of "
          "one per channel; scales_A (M,) f16 per-row (per-token) is "
          "unchanged. Returns C (M,N) f16 (+ bias). Requires K % 128 == 0.",
          py::arg("Aq"), py::arg("Bq"), py::arg("scales_A"), py::arg("scales_B"), py::arg("bias") = py::none());
    m.def("quantize_sym_int8", &quantize_sym_int8,
          "Per-row (per-token) symmetric fp16 -> int8 activation "
          "quantization: one block per row, a single vectorized pass over "
          "the row cached in shared memory (max-abs reduction via warp "
          "shuffles), then a second pass quantizing from the cache -- one "
          "global read + one global write per row. A (M,K) f16. Returns "
          "(Aq, scales): Aq (M,K) int8 plain two's-complement bytes; "
          "scales (M,) f16. Requires K % 8 == 0.",
          py::arg("A"));
    m.def("quantize_sym_int4", &quantize_sym_int4,
          "Per-row (per-token) symmetric fp16 -> int4 activation "
          "quantization, same single-read/single-write design as "
          "quantize_sym_int8. A (M,K) f16. Returns (Aq, scales): Aq "
          "(M,K/2) uint8 packed two's-complement nibbles (2/byte, low "
          "nibble = even k, high nibble = odd k), NOT excess-8 encoded; "
          "scales (M,) f16. Requires K % 8 == 0.",
          py::arg("A"));
    m.def("get_gpu_metrics", &get_gpu_metrics,
          "Query CUDA device properties and return a dict of GPU metrics "
          "(SM count, tensor cores, smem, L2, registers, etc.). "
          "Pass device=-1 (default) to use the current device.",
          py::arg("device") = -1);

    // ---- CPU (AVX2/FMA) quantized matmuls (asrq/matmulq/csrc/x86/) ----
    // All fp32 tensors (see qmatmul.h's top comment for why); packed codes
    // are two's complement (symmetric) or plain unsigned (asymmetric) --
    // NOT the CUDA kernels' excess-K encoding. groupwise variants use a
    // fixed group size of 128 along K, one scale/zero per (channel, group)
    // instead of one per channel.
    // const char* wxa16_doc =
    //     "CPU (AVX2/FMA) WxA16 GEMM: A (M,K) f32 activations (NOT "
    //     "quantized), Bq packed weight codes, scales [zeros] per-channel or "
    //     "(groupwise) per-(channel,128-wide-K-group) f32. Returns C (M,N) "
    //     "f32 = A @ dequant(Bq).T (+ bias). Requires K a multiple of "
    //     "8/wbits (128 too, for groupwise variants).";
    // const char* wxa8_doc =
    //     "CPU (AVX2/FMA) WxA8 GEMM: Aq (M,K) int8 activations (see "
    //     "quantize_sym_int8_cpu) + scales_A (M,) f32, Bq packed weight "
    //     "codes, scales_B [zeros_B] per-channel or (groupwise) f32. Returns "
    //     "C (M,N) f32 = (scales_A[:,None]*...) * (Aq @ dequant(Bq).T) "
    //     "(+ bias). Requires K a multiple of 8/wbits (128 too, for "
    //     "groupwise variants).";

    // m.def("w4a16_cpu_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                               const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(4, true, false, A, Bq, scales, c10::nullopt, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    // m.def("w4a16_cpu_asym_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                                    const torch::Tensor& zeros, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(4, false, false, A, Bq, scales, zeros, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());
    // m.def("w4a16_cpu_group_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                                     const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(4, true, true, A, Bq, scales, c10::nullopt, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    // m.def("w4a16_cpu_group_asym_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                                          const torch::Tensor& zeros, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(4, false, true, A, Bq, scales, zeros, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());

    // m.def("w2a16_cpu_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                               const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(2, true, false, A, Bq, scales, c10::nullopt, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    // m.def("w2a16_cpu_asym_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                                    const torch::Tensor& zeros, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(2, false, false, A, Bq, scales, zeros, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());
    // m.def("w2a16_cpu_group_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                                     const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(2, true, true, A, Bq, scales, c10::nullopt, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("bias") = py::none());
    // m.def("w2a16_cpu_group_asym_matmul", [](const torch::Tensor& A, const torch::Tensor& Bq, const torch::Tensor& scales,
    //                                          const torch::Tensor& zeros, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa16_matmul_impl(2, false, true, A, Bq, scales, zeros, bias);
    // }, wxa16_doc, py::arg("A"), py::arg("Bq"), py::arg("scales"), py::arg("zeros"), py::arg("bias") = py::none());

    // m.def("w4a8_cpu_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                              const torch::Tensor& scales_B, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(4, true, false, Aq, scales_A, Bq, scales_B, c10::nullopt, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("bias") = py::none());
    // m.def("w4a8_cpu_asym_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                                   const torch::Tensor& scales_B, const torch::Tensor& zeros_B,
    //                                   const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(4, false, false, Aq, scales_A, Bq, scales_B, zeros_B, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("zeros_B"), py::arg("bias") = py::none());
    // m.def("w4a8_cpu_group_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                                    const torch::Tensor& scales_B, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(4, true, true, Aq, scales_A, Bq, scales_B, c10::nullopt, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("bias") = py::none());
    // m.def("w4a8_cpu_group_asym_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                                         const torch::Tensor& scales_B, const torch::Tensor& zeros_B,
    //                                         const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(4, false, true, Aq, scales_A, Bq, scales_B, zeros_B, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("zeros_B"), py::arg("bias") = py::none());

    // m.def("w2a8_cpu_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                              const torch::Tensor& scales_B, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(2, true, false, Aq, scales_A, Bq, scales_B, c10::nullopt, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("bias") = py::none());
    // m.def("w2a8_cpu_asym_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                                   const torch::Tensor& scales_B, const torch::Tensor& zeros_B,
    //                                   const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(2, false, false, Aq, scales_A, Bq, scales_B, zeros_B, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("zeros_B"), py::arg("bias") = py::none());
    // m.def("w2a8_cpu_group_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                                    const torch::Tensor& scales_B, const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(2, true, true, Aq, scales_A, Bq, scales_B, c10::nullopt, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("bias") = py::none());
    // m.def("w2a8_cpu_group_asym_matmul", [](const torch::Tensor& Aq, const torch::Tensor& scales_A, const torch::Tensor& Bq,
    //                                         const torch::Tensor& scales_B, const torch::Tensor& zeros_B,
    //                                         const c10::optional<torch::Tensor>& bias) {
    //     return cpu_wxa8_matmul_impl(2, false, true, Aq, scales_A, Bq, scales_B, zeros_B, bias);
    // }, wxa8_doc, py::arg("Aq"), py::arg("scales_A"), py::arg("Bq"), py::arg("scales_B"), py::arg("zeros_B"), py::arg("bias") = py::none());

    // m.def("quantize_sym_int8_cpu", &quantize_sym_int8_cpu,
    //       "Per-row (per-token) symmetric fp32 -> int8 activation "
    //       "quantization on CPU (AVX2), for the W*A8_cpu kernels above. "
    //       "A (M,K) f32. Returns (Aq, scales): Aq (M,K) int8; scales (M,) f32.",
    //       py::arg("A"));
}
