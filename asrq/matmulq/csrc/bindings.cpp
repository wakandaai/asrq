#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>
#include "cutlass/cutlass.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)   \
  CHECK_CUDA(x);         \
  CHECK_CONTIGUOUS(x)

// Forward declarations from .cu files
// matmul_sm89: tuned for sm_89 (Ada). Used as the default fp16 matmul.
void matmul_sm89_launcher (const __half* A, const __half* B, __half* C, int M, int N, int K);
void mymatmul_launcher(const __half* A, const __half* B, __half* C, int* workspace, int M, int N, int K);

// matmul_int4_sm89: hand-written int4 x int4 -> int32 tensor-core GEMM.
// A, B are packed int4 (2/byte) reinterpreted as uint16 words; K is the
// unpacked int4 contraction length.
void matmul_int4_sm89_launcher(const uint16_t* A, const uint16_t* B, int32_t* C,
                               int M, int N, int K);
// Fused int4 GEMM + per-channel rescale: Y = sA[m]*sB[n]*(A @ B^T), f16 out.
void matmul_int4_rescale_sm89_launcher(const uint16_t* A, const uint16_t* B,
                                       const float* sA, const float* sB, __half* Y,
                                       int M, int N, int K);
// matmul_int8_sm89: hand-written int8 x int8 -> int32 tensor-core GEMM.
// A, B are int8 reinterpreted as uint16 words; K is the int8 contraction length.
void matmul_int8_sm89_launcher(const uint16_t* A, const uint16_t* B, int32_t* C,
                               int M, int N, int K);
// matmul_wxax_sm89: fused f16-activation x grouped-per-channel quantized-weight
// GEMM. A f16 (M,K); W packed BITS-bit words (N, K*BITS/16); sW (N, K/128) f16.
// qA (M, K*BITS/16) uint16 and sA (M,) f32 are scratch filled internally (packed
// quantized activations + per-token scale). Y f16 (M,N). bits is 4 or 8.
void matmul_wxax_sm89_launcher(const __half* A, const uint16_t* W, const __half* sW,
                               uint16_t* qA, float* sA, __half* Y,
                               int M, int N, int K, int bits, cudaStream_t stream);
// matmul_f16xint4: f16 activation x grouped-int4 weight -> f16. The int4 weights
// are dequantized to f16 (w = sB*q + zB, unsigned nibble q, groupsize 128) and
// the contraction runs as an f16 tensor-core GEMM. A f16 (M,K); B packed uint8
// (N, K/2); sB,zB f16 (N, K/128); C f16 (M,N). Requires K%128==0, N%128==0.
void matmul_f16xint4(const __half* A, const uint8_t* B, __half* sB, __half* zB,
                     __half* C, int M, int N, int K);
// matmul_w4a16_sym: W4A16 with per-channel symmetric (signed int4) weights.
// A f16 (M,K); B signed packed int4 (N,K/2); sB f32 (N,) per-channel; C f16.
void matmul_w4a16_sym_launcher(const __half* A, const uint8_t* B, const float* sB,
                               __half* C, int M, int N, int K);
// matmul_f16xint2: W2A16, asymmetric grouped (group=128). A f16 (M,K); B packed
// unsigned int2 (N,K/4); sB,zB f16 (N,K/128); C f16. w = sB*q + zB, q in [0,3].
void matmul_f16xint2(const __half* A, const uint8_t* B, __half* sB, __half* zB,
                     __half* C, int M, int N, int K);
cutlass::Status cutlass_int8_matmul_launcher(const int8_t* A, const int8_t* B, int32_t* C,
                                              int M, int N, int K, cudaStream_t stream);
// cublasStatus_t is forward-declarable as an enum; just declare via cublas header.
#include <cublas_v2.h>
cublasStatus_t cublaslt_int8_matmul_launcher(const int8_t* A, const int8_t* B, int32_t* C,
                                             int M, int N, int K, cudaStream_t stream);
cublasStatus_t cublaslt_int4_matmul_launcher(const int8_t* A_packed, const int8_t* B_packed,
                                             int32_t* C, int M, int N, int K,
                                             cudaStream_t stream);
void per_token_sym_quant_launcher_f32(const float* X, int8_t* Q, float* scale,
                                      int rows, int C, cudaStream_t stream);
// per_token int4 quantize+pack (from matmul_wxax_sm89.cu): A f16 (M,K) ->
// qA (M, K/4) uint16 words (matmul_int4 layout) + sA (M,) f32 per-token scale.
void per_token_quant_int4_launcher(const __half* A, uint16_t* qA, float* sA,
                                   int M, int K, cudaStream_t stream);
void per_token_sym_quant_launcher_f16(const __half* X, int8_t* Q, float* scale,
                                      int rows, int C, cudaStream_t stream);
void rescale_int32_rowcol_launcher_f16(const int32_t* C, const float* sA, const float* sB,
                                       __half* Y, int M, int N, cudaStream_t stream);
void rescale_int32_rowcol_launcher_f32(const int32_t* C, const float* sA, const float* sB,
                                       float* Y, int M, int N, cudaStream_t stream);
void unpack_int4_to_int8_launcher(const uint8_t* Bp, int8_t* B,
                                  size_t n_packed_bytes, cudaStream_t stream);

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

torch::Tensor matmul_16x16(torch::Tensor A, torch::Tensor B) {
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
                "matmul_16x16 requires K to be a multiple of 8 (cp.async 16-byte vectors); got K=", K);
    auto C = torch::empty({M, N}, A.options());

    // matmul_16x16 is now an alias for the sm_89-tuned kernel; the legacy
    // Hopper launcher (matmul_16x16_launcher) was removed.
    matmul_sm89_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K
    );
    return C;
}

torch::Tensor mymatmul(torch::Tensor A, torch::Tensor B, torch::Tensor workspace) {
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
                "matmul_16x16 requires K to be a multiple of 8 (cp.async 16-byte vectors); got K=", K);
    auto C = torch::empty({M, N}, A.options());

    // matmul_16x16 is now an alias for the sm_89-tuned kernel; the legacy
    // Hopper launcher (matmul_16x16_launcher) was removed.
    mymatmul_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        reinterpret_cast<int*>(workspace.data_ptr<int32_t>()),
        M, N, K
    );
    return C;
}

// FP16 matmul targeting compute capability 8.9 (Ada Lovelace).
// Same signature as matmul_16x16; uses cp.async + mma.sync.m16n8k16 instead
// of WGMMA/TMA. Runs on any sm_80+ device but is tuned for sm_89.
torch::Tensor matmul_sm89(torch::Tensor A, torch::Tensor B) {
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
                "matmul_sm89 requires K to be a multiple of 8 (cp.async 16-byte vectors); got K=", K);

    auto C = torch::empty({M, N}, A.options());
    matmul_sm89_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K
    );
    return C;
}

torch::Tensor cutlass_int8_matmul(torch::Tensor A, torch::Tensor B) {
    // A: (M, K) row-major int8
    // B: (N, K) row-major int8  -- this is K x N column-major from CUTLASS's
    //    perspective (data layout: element [k, n] at offset k + n*K), which
    //    matches the kernel's ColumnMajor LayoutB with ldb=K.
    // C: (M, N) row-major int32
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    TORCH_CHECK(B.size(1) == K,
                "B must have shape (N, K) so it appears as a K x N column-major matrix; "
                "got B shape (", B.size(0), ", ", B.size(1), ") with K=", K);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cutlass::Status status = cutlass_int8_matmul_launcher(
        A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), C.data_ptr<int32_t>(),
        M, N, K, at::cuda::getCurrentCUDAStream());

    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM execution failed.");

    return C;
}

torch::Tensor cublaslt_int8_matmul(torch::Tensor A, torch::Tensor B) {
    // Same layout convention as cutlass_int8_matmul:
    //   A : (M, K) int8 row-major
    //   B : (N, K) int8 row-major  (== K x N column-major with ld=K)
    //   C : (M, N) int32 row-major  =  A @ B.T
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dtype() == torch::kInt8, "A must be int8");
    TORCH_CHECK(B.dtype() == torch::kInt8, "B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    TORCH_CHECK(B.size(1) == K,
                "B must have shape (N, K); got (", B.size(0), ", ", B.size(1),
                ") with K=", K);

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    cublasStatus_t status = cublaslt_int8_matmul_launcher(
        A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), C.data_ptr<int32_t>(),
        M, N, K, at::cuda::getCurrentCUDAStream());

    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cuBLASLt int8 GEMM failed with status=", static_cast<int>(status));
    return C;
}

std::tuple<torch::Tensor, torch::Tensor> per_token_sym_quant(torch::Tensor X) {
    // X : (B, T, C) float32 or float16, contiguous, CUDA
    // returns:
    //   Q     : (B, T, C) int8
    //   scale : (B, T)    float32
    CHECK_INPUT(X);
    TORCH_CHECK(X.dim() == 3, "X must be (B, T, C)");
    TORCH_CHECK(X.scalar_type() == torch::kFloat32 ||
                X.scalar_type() == torch::kFloat16,
                "X must be float32 or float16");

    const int64_t B = X.size(0);
    const int64_t T = X.size(1);
    const int64_t C = X.size(2);
    const int64_t rows = B * T;
    TORCH_CHECK(rows <= std::numeric_limits<int>::max(),
                "rows = B*T exceeds int range");
    TORCH_CHECK(C    <= std::numeric_limits<int>::max(),
                "C exceeds int range");

    auto Q = torch::empty({B, T, C},
                          torch::TensorOptions().dtype(torch::kInt8).device(X.device()));
    auto scale = torch::empty({B, T},
                              torch::TensorOptions().dtype(torch::kFloat32).device(X.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (X.scalar_type() == torch::kFloat32) {
        per_token_sym_quant_launcher_f32(
            X.data_ptr<float>(),
            Q.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            static_cast<int>(rows), static_cast<int>(C), stream);
    } else {
        per_token_sym_quant_launcher_f16(
            reinterpret_cast<const __half*>(X.data_ptr<at::Half>()),
            Q.data_ptr<int8_t>(),
            scale.data_ptr<float>(),
            static_cast<int>(rows), static_cast<int>(C), stream);
    }
    return {Q, scale};
}

torch::Tensor rescale_int32_rowcol(torch::Tensor C, torch::Tensor sA, torch::Tensor sB,
                                   c10::optional<torch::ScalarType> out_dtype) {
    // C : (M, N) int32 row-major, contiguous, CUDA
    // sA: (M,)   float32
    // sB: (N,)   float32
    // returns Y[m,n] = C[m,n] * sA[m] * sB[n] in float16 (default) or float32.
    CHECK_INPUT(C);
    CHECK_INPUT(sA);
    CHECK_INPUT(sB);
    TORCH_CHECK(C.dtype()  == torch::kInt32,   "C must be int32");
    TORCH_CHECK(sA.dtype() == torch::kFloat32, "sA must be float32");
    TORCH_CHECK(sB.dtype() == torch::kFloat32, "sB must be float32");
    TORCH_CHECK(C.dim() == 2 && sA.dim() == 1 && sB.dim() == 1,
                "C must be 2D, sA/sB must be 1D");
    const int64_t M = C.size(0);
    const int64_t N = C.size(1);
    TORCH_CHECK(sA.size(0) == M, "sA length must match C.size(0)");
    TORCH_CHECK(sB.size(0) == N, "sB length must match C.size(1)");
    TORCH_CHECK(M <= std::numeric_limits<int>::max() &&
                N <= std::numeric_limits<int>::max(),
                "M or N exceeds int range");

    auto dtype = out_dtype.value_or(torch::kFloat16);
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kFloat32,
                "out_dtype must be float16 or float32");
    auto Y = torch::empty({M, N},
                          torch::TensorOptions().dtype(dtype).device(C.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (dtype == torch::kFloat16) {
        rescale_int32_rowcol_launcher_f16(
            C.data_ptr<int32_t>(), sA.data_ptr<float>(), sB.data_ptr<float>(),
            reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
            static_cast<int>(M), static_cast<int>(N), stream);
    } else {
        rescale_int32_rowcol_launcher_f32(
            C.data_ptr<int32_t>(), sA.data_ptr<float>(), sB.data_ptr<float>(),
            Y.data_ptr<float>(),
            static_cast<int>(M), static_cast<int>(N), stream);
    }
    return Y;
}

torch::Tensor unpack_int4_to_int8(torch::Tensor Bp, int64_t C) {
    // Bp : (N, C/2) int8 storage holding packed signed 4-bit values.
    //      Packing convention: byte b at column j stores
    //        low  nibble (bits 0..3) -> output column 2*j
    //        high nibble (bits 4..7) -> output column 2*j + 1
    //      Each nibble is treated as a signed 4-bit int in [-8, 7] and is
    //      sign-extended to int8.
    // C  : the unpacked column count. Must equal 2 * Bp.size(1).
    // returns: (N, C) int8.
    CHECK_INPUT(Bp);
    TORCH_CHECK(Bp.dtype() == torch::kInt8, "Bp must be int8-storage (packed nibbles)");
    TORCH_CHECK(Bp.dim() == 2, "Bp must be 2D (N, C/2)");
    const int64_t N = Bp.size(0);
    const int64_t Cp = Bp.size(1);
    TORCH_CHECK(C == 2 * Cp, "C must equal 2 * Bp.size(1); got C=", C,
                " Bp.size(1)=", Cp);

    auto B = torch::empty({N, C},
                          torch::TensorOptions().dtype(torch::kInt8).device(Bp.device()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    unpack_int4_to_int8_launcher(
        reinterpret_cast<const uint8_t*>(Bp.data_ptr<int8_t>()),
        B.data_ptr<int8_t>(),
        static_cast<size_t>(N) * static_cast<size_t>(Cp),
        stream);
    return B;
}

torch::Tensor matmul_int4xint8_cublas(torch::Tensor A, torch::Tensor Bp) {
    // A  : (M, C)    int8, row-major contiguous
    // Bp : (N, C/2)  int8 holding packed signed int4 weights
    // Returns (M, N) int32 = A @ unpack(Bp).T
    CHECK_INPUT(A);
    CHECK_INPUT(Bp);
    TORCH_CHECK(A.dtype() == torch::kInt8 && Bp.dtype() == torch::kInt8,
                "A and Bp must be int8");
    TORCH_CHECK(A.dim() == 2 && Bp.dim() == 2, "A and Bp must be 2D");
    const int64_t M = A.size(0);
    const int64_t C = A.size(1);
    const int64_t N = Bp.size(0);
    TORCH_CHECK(Bp.size(1) * 2 == C,
                "Bp must have shape (N, C/2); got Bp=(", Bp.size(0), ",", Bp.size(1),
                ") with A.size(1)=", C);

    torch::Tensor B = unpack_int4_to_int8(Bp, C);

    auto Cout = torch::empty({M, N},
                             torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    cublasStatus_t status = cublaslt_int8_matmul_launcher(
        A.data_ptr<int8_t>(), B.data_ptr<int8_t>(), Cout.data_ptr<int32_t>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(C),
        at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cuBLASLt int8 GEMM (int4xint8 path) failed with status=",
                static_cast<int>(status));
    return Cout;
}


torch::Tensor matmul_int4_cublas(torch::Tensor Ap, torch::Tensor Bp) {
    // Native int4 x int4 GEMM via cuBLASLt (CUDA_R_4I IMMA tensor cores).
    //
    //   Ap : (M, K/2) int8 storage holding packed signed int4 activations
    //   Bp : (N, K/2) int8 storage holding packed signed int4 weights
    // Packing: byte at column j stores
    //   low  nibble (bits 0..3) -> element 2*j
    //   high nibble (bits 4..7) -> element 2*j + 1
    // Returns (M, N) int32 = Ap_unpacked @ Bp_unpacked.T
    //
    // cuBLASLt requires K % 32 == 0 for the int4 path.
    CHECK_INPUT(Ap);
    CHECK_INPUT(Bp);
    TORCH_CHECK(Ap.dtype() == torch::kInt8 && Bp.dtype() == torch::kInt8,
                "Ap and Bp must be int8 storage (packed int4)");
    TORCH_CHECK(Ap.dim() == 2 && Bp.dim() == 2, "Ap and Bp must be 2D");

    const int64_t M  = Ap.size(0);
    const int64_t Kp = Ap.size(1);
    const int64_t N  = Bp.size(0);
    TORCH_CHECK(Bp.size(1) == Kp,
                "Bp must have shape (N, K/2) matching Ap's K; got Ap=(",
                Ap.size(0), ",", Ap.size(1), ") Bp=(",
                Bp.size(0), ",", Bp.size(1), ")");
    const int64_t K = Kp * 2;
    TORCH_CHECK((K % 32) == 0,
                "K (unpacked) must be a multiple of 32 for the cuBLASLt int4 "
                "path; got K=", K);

    auto Cout = torch::empty({M, N},
                             torch::TensorOptions().dtype(torch::kInt32).device(Ap.device()));
    cublasStatus_t status = cublaslt_int4_matmul_launcher(
        Ap.data_ptr<int8_t>(), Bp.data_ptr<int8_t>(), Cout.data_ptr<int32_t>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cuBLASLt int4 GEMM failed with status=",
                static_cast<int>(status));
    return Cout;
}


torch::Tensor matmul_int8_sm89(torch::Tensor A, torch::Tensor B) {
    // Hand-written int8 x int8 -> int32 GEMM (tensor cores, mma.m16n8k32.s8).
    //
    //   A : (M, K) int8 row-major  (activations)
    //   B : (N, K) int8 row-major  (weights; == K x N column-major)
    // Returns (M, N) int32 = A @ B.T
    //
    // Requires K % 128 == 0 and N % 128 == 0. Any M is supported.
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    TORCH_CHECK(A.dtype() == torch::kInt8 && B.dtype() == torch::kInt8,
                "A and B must be int8");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(0);
    TORCH_CHECK(B.size(1) == K,
                "B must have shape (N, K) matching A's K; got A=(",
                A.size(0), ",", A.size(1), ") B=(", B.size(0), ",", B.size(1), ")");
    TORCH_CHECK((K % 128) == 0,
                "K must be a multiple of 128 for matmul_int8_sm89; got K=", K);
    TORCH_CHECK((N % 128) == 0,
                "N must be a multiple of 128 for matmul_int8_sm89; got N=", N);

    auto Cout = torch::empty({M, N},
                             torch::TensorOptions().dtype(torch::kInt32).device(A.device()));
    matmul_int8_sm89_launcher(
        reinterpret_cast<const uint16_t*>(A.data_ptr<int8_t>()),
        reinterpret_cast<const uint16_t*>(B.data_ptr<int8_t>()),
        Cout.data_ptr<int32_t>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    return Cout;
}


torch::Tensor matmul_int4(torch::Tensor Ap, torch::Tensor Bp) {
    // Hand-written int4 x int4 -> int32 GEMM (tensor cores, mma.m16n8k64.s4).
    //
    //   Ap : (M, K/2) int8 storage holding packed signed int4 activations
    //   Bp : (N, K/2) int8 storage holding packed signed int4 weights
    // Packing: byte at column j stores
    //   low  nibble (bits 0..3) -> element 2*j
    //   high nibble (bits 4..7) -> element 2*j + 1
    // Returns (M, N) int32 = Ap_unpacked @ Bp_unpacked.T
    //
    // Requires K (unpacked) % 256 == 0 and N % 128 == 0. Any M is supported.
    CHECK_INPUT(Ap);
    CHECK_INPUT(Bp);
    TORCH_CHECK(Ap.dtype() == torch::kInt8 && Bp.dtype() == torch::kInt8,
                "Ap and Bp must be int8 storage (packed int4)");
    TORCH_CHECK(Ap.dim() == 2 && Bp.dim() == 2, "Ap and Bp must be 2D");

    const int64_t M  = Ap.size(0);
    const int64_t Kp = Ap.size(1);
    const int64_t N  = Bp.size(0);
    TORCH_CHECK(Bp.size(1) == Kp,
                "Bp must have shape (N, K/2) matching Ap's K; got Ap=(",
                Ap.size(0), ",", Ap.size(1), ") Bp=(",
                Bp.size(0), ",", Bp.size(1), ")");
    const int64_t K = Kp * 2;
    TORCH_CHECK((K % 256) == 0,
                "K (unpacked) must be a multiple of 256 for matmul_int4; got K=", K);
    TORCH_CHECK((N % 128) == 0,
                "N must be a multiple of 128 for matmul_int4; got N=", N);

    auto Cout = torch::empty({M, N},
                             torch::TensorOptions().dtype(torch::kInt32).device(Ap.device()));
    matmul_int4_sm89_launcher(
        reinterpret_cast<const uint16_t*>(Ap.data_ptr<int8_t>()),
        reinterpret_cast<const uint16_t*>(Bp.data_ptr<int8_t>()),
        Cout.data_ptr<int32_t>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    return Cout;
}


torch::Tensor matmul_wxax(torch::Tensor A, torch::Tensor Wp, torch::Tensor sW, int64_t bits) {
    // Fused f16 x grouped-quantized-weight GEMM.
    //   A  : (M, K)            f16  activations (quantized per-token on the fly)
    //   Wp : (N, K*bits/16) int16-storage packed signed bits-bit weights, OR the
    //        equivalent int8 storage: int4 -> (N, K/2) int8, int8 -> (N, K) int8.
    //        Interpreted as uint16 words: (N, K*bits/16).
    //   sW : (N, K/128)        f16  per-(channel, group) weight scales.
    // Returns Y (M, N) f16 = dequant(quant(A) @ Wp^T).
    //
    // Requires bits in {4, 8}, N % 128 == 0, K % 128 == 0, and K % 256 == 0 for
    // int4 (the K-tile is 256 int4). Any M is supported.
    CHECK_INPUT(A);
    CHECK_INPUT(Wp);
    CHECK_INPUT(sW);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Wp.dtype() == torch::kInt8, "Wp must be int8 storage (packed weights)");
    TORCH_CHECK(sW.dtype() == torch::kFloat16, "sW must be float16");
    TORCH_CHECK(A.dim() == 2 && Wp.dim() == 2 && sW.dim() == 2, "A, Wp, sW must be 2D");
    TORCH_CHECK(bits == 4 || bits == 8, "bits must be 4 or 8");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = Wp.size(0);
    const int64_t bytes_per_row = Wp.size(1);          // int4: K/2, int8: K
    const int64_t expect_bytes = (bits == 4) ? (K / 2) : K;
    TORCH_CHECK(bytes_per_row == expect_bytes,
                "Wp must have shape (N, ", expect_bytes, ") for bits=", bits,
                "; got (", Wp.size(0), ", ", Wp.size(1), ") with K=", K);
    TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);
    TORCH_CHECK((K % 128) == 0, "K must be a multiple of 128; got K=", K);
    if (bits == 4)
        TORCH_CHECK((K % 256) == 0, "K must be a multiple of 256 for bits=4; got K=", K);
    TORCH_CHECK(sW.size(0) == N && sW.size(1) == K / 128,
                "sW must have shape (N, K/128) = (", N, ", ", K / 128,
                "); got (", sW.size(0), ", ", sW.size(1), ")");

    auto Y  = torch::empty({M, N}, A.options());
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    auto i16 = torch::TensorOptions().dtype(torch::kInt16).device(A.device());
    auto sA  = torch::empty({M}, f32);
    const int64_t K_words = K * bits / 16;          // int4: K/4, int8: K/2
    auto qA  = torch::empty({M, K_words}, i16);      // packed quantized activations

    matmul_wxax_sm89_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const uint16_t*>(Wp.data_ptr<int8_t>()),
        reinterpret_cast<const __half*>(sW.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(qA.data_ptr<int16_t>()),
        sA.data_ptr<float>(),
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        static_cast<int>(bits), at::cuda::getCurrentCUDAStream());
    return Y;
}

torch::Tensor matmul_f16xint4_fn(torch::Tensor A, torch::Tensor Bp,
                                 torch::Tensor sB, torch::Tensor zB) {
    // f16 activation x grouped-int4 weight -> f16 (dequant-to-f16, f16 GEMM).
    //   A  : (M, K)     f16 activations, row-major
    //   Bp : (N, K/2)   uint8 packed UNSIGNED int4 weights (low nibble = even k)
    //   sB : (N, K/128) f16 group scale
    //   zB : (N, K/128) f16 group offset
    //        w[n,k] = sB[n,g] * Bp_nibble[n,k] + zB[n,g],  g = k / 128
    // Returns C (M, N) f16 = A @ dequant(Bp)^T.  Requires K%128==0, N%128==0.
    CHECK_INPUT(A);
    CHECK_INPUT(Bp);
    CHECK_INPUT(sB);
    CHECK_INPUT(zB);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bp.dtype() == torch::kUInt8, "Bp must be uint8 (packed unsigned nibbles)");
    TORCH_CHECK(sB.dtype() == torch::kFloat16 && zB.dtype() == torch::kFloat16,
                "sB and zB must be float16");
    TORCH_CHECK(A.dim() == 2 && Bp.dim() == 2 && sB.dim() == 2 && zB.dim() == 2,
                "A, Bp, sB, zB must be 2D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = Bp.size(0);
    constexpr int64_t GROUP = 128;
    TORCH_CHECK(Bp.size(1) == K / 2,
                "Bp must have shape (N, K/2); got (", Bp.size(0), ",", Bp.size(1),
                ") with K=", K);
    TORCH_CHECK((K % GROUP) == 0, "K must be a multiple of 128; got K=", K);
    TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);
    TORCH_CHECK(sB.size(0) == N && sB.size(1) == K / GROUP,
                "sB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");
    TORCH_CHECK(zB.size(0) == N && zB.size(1) == K / GROUP,
                "zB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");

    auto C = torch::empty({M, N}, A.options());
    matmul_f16xint4(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bp.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(sB.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(zB.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    return C;
}

torch::Tensor matmul_w4a4(torch::Tensor A, torch::Tensor Wp, torch::Tensor sB) {
    // Per-channel-symmetric W4A4: quantize A to int4 per-token on the fly, run
    // the pure int4 tensor-core GEMM, and rescale by sA[m]*sB[n].
    //   A  : (M, K)   f16 activations
    //   Wp : (N, K/2) int8 packed signed int4 weights (per-channel symmetric)
    //   sB : (N,)     f32 per-channel weight scale
    // Returns Y (M, N) f16.  Requires K % 256 == 0 and N % 128 == 0.
    //
    // Unlike matmul_wxax (grouped), there is no per-group dequant in the K-loop:
    // all of K accumulates in int32 and a single sA[m]*sB[n] rescale runs in the
    // epilogue, so the GEMM is the fast pure-int4 kernel.
    CHECK_INPUT(A);
    CHECK_INPUT(Wp);
    CHECK_INPUT(sB);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Wp.dtype() == torch::kInt8, "Wp must be int8 (packed signed int4)");
    TORCH_CHECK(sB.dtype() == torch::kFloat32, "sB must be float32");
    TORCH_CHECK(A.dim() == 2 && Wp.dim() == 2 && sB.dim() == 1, "A,Wp 2D; sB 1D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = Wp.size(0);
    TORCH_CHECK(Wp.size(1) == K / 2, "Wp must be (N, K/2); got (", Wp.size(0), ",",
                Wp.size(1), ") with K=", K);
    TORCH_CHECK(sB.size(0) == N, "sB must be (N,) = (", N, ")");
    TORCH_CHECK((K % 256) == 0, "K must be a multiple of 256; got K=", K);
    TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);

    auto stream = at::cuda::getCurrentCUDAStream();
    auto dev = A.device();
    auto i16 = torch::TensorOptions().dtype(torch::kInt16).device(dev);
    auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

    // 1) quantize+pack A to int4 per-token (qA words + per-token scale sA).
    auto qA = torch::empty({M, K / 4}, i16);
    auto sA = torch::empty({M}, f32);
    per_token_quant_int4_launcher(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<uint16_t*>(qA.data_ptr<int16_t>()),
        sA.data_ptr<float>(), static_cast<int>(M), static_cast<int>(K), stream);

    // 2) int4 GEMM with the per-channel rescale fused into the epilogue:
    //    Y[m,n] = sA[m]*sB[n]*(qA @ qW^T), f16 -- no int32 M*N round-trip.
    auto Y = torch::empty({M, N}, A.options());
    matmul_int4_rescale_sm89_launcher(
        reinterpret_cast<const uint16_t*>(qA.data_ptr<int16_t>()),
        reinterpret_cast<const uint16_t*>(Wp.data_ptr<int8_t>()),
        sA.data_ptr<float>(), sB.data_ptr<float>(),
        reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    return Y;
}

torch::Tensor matmul_f16xint2_fn(torch::Tensor A, torch::Tensor Bp,
                                 torch::Tensor sB, torch::Tensor zB) {
    // W2A16: f16 activation x grouped-asymmetric int2 weight -> f16.
    //   A  : (M, K)     f16 activations, row-major
    //   Bp : (N, K/4)   uint8 packed UNSIGNED int2 (element e at bits [2e..2e+1])
    //   sB : (N, K/128) f16 group scale
    //   zB : (N, K/128) f16 group offset    w[n,k] = sB[n,g]*q + zB[n,g], g=k/128
    // Returns C (M, N) f16.  Requires K % 128 == 0, N % 128 == 0.
    CHECK_INPUT(A);
    CHECK_INPUT(Bp);
    CHECK_INPUT(sB);
    CHECK_INPUT(zB);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Bp.dtype() == torch::kUInt8, "Bp must be uint8 (packed unsigned int2)");
    TORCH_CHECK(sB.dtype() == torch::kFloat16 && zB.dtype() == torch::kFloat16,
                "sB and zB must be float16");
    TORCH_CHECK(A.dim() == 2 && Bp.dim() == 2 && sB.dim() == 2 && zB.dim() == 2,
                "A, Bp, sB, zB must be 2D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = Bp.size(0);
    constexpr int64_t GROUP = 128;
    TORCH_CHECK(Bp.size(1) == K / 4,
                "Bp must have shape (N, K/4); got (", Bp.size(0), ",", Bp.size(1),
                ") with K=", K);
    TORCH_CHECK((K % GROUP) == 0, "K must be a multiple of 128; got K=", K);
    TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);
    TORCH_CHECK(sB.size(0) == N && sB.size(1) == K / GROUP,
                "sB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");
    TORCH_CHECK(zB.size(0) == N && zB.size(1) == K / GROUP,
                "zB must have shape (N, K/128) = (", N, ", ", K / GROUP, ")");

    auto C = torch::empty({M, N}, A.options());
    matmul_f16xint2(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        Bp.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(sB.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(zB.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    return C;
}

torch::Tensor matmul_w4ax(torch::Tensor A, torch::Tensor Wp, torch::Tensor sB) {
    // Unified W4Ax over ONE per-channel-symmetric int4 weight. Dispatches by M:
    //   M <= 64 : W4A16 -- f16 activations, signed-symmetric dequant + f16 GEMM
    //   M >  64 : W4A4  -- quantize A to int4 on the fly + int4 GEMM + rescale
    // Both arms consume the SAME weights: Wp (N,K/2) int8 signed packed int4,
    // sB (N,) f32 per-channel scale. Returns Y (M,N) f16.  Requires K%256==0,
    // N%128==0.
    CHECK_INPUT(A);
    CHECK_INPUT(Wp);
    CHECK_INPUT(sB);
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(Wp.dtype() == torch::kInt8, "Wp must be int8 (packed signed int4)");
    TORCH_CHECK(sB.dtype() == torch::kFloat32, "sB must be float32");
    TORCH_CHECK(A.dim() == 2 && Wp.dim() == 2 && sB.dim() == 1, "A,Wp 2D; sB 1D");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = Wp.size(0);
    TORCH_CHECK(Wp.size(1) == K / 2, "Wp must be (N, K/2); got (", Wp.size(0), ",",
                Wp.size(1), ") with K=", K);
    TORCH_CHECK(sB.size(0) == N, "sB must be (N,) = (", N, ")");
    TORCH_CHECK((K % 256) == 0, "K must be a multiple of 256; got K=", K);
    TORCH_CHECK((N % 128) == 0, "N must be a multiple of 128; got N=", N);

    constexpr int64_t W4AX_M_THRESHOLD = 64;
    auto Y = torch::empty({M, N}, A.options());

    if (M <= W4AX_M_THRESHOLD) {
        // W4A16: f16 activations, weights dequantized to f16 in registers.
        matmul_w4a16_sym_launcher(
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const uint8_t*>(Wp.data_ptr<int8_t>()),
            sB.data_ptr<float>(),
            reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    } else {
        // W4A4: quantize A to int4 per-token, int4 GEMM + fused per-channel rescale.
        auto stream = at::cuda::getCurrentCUDAStream();
        auto i16 = torch::TensorOptions().dtype(torch::kInt16).device(A.device());
        auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
        auto qA = torch::empty({M, K / 4}, i16);
        auto sA = torch::empty({M}, f32);
        per_token_quant_int4_launcher(
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<uint16_t*>(qA.data_ptr<int16_t>()),
            sA.data_ptr<float>(), static_cast<int>(M), static_cast<int>(K), stream);
        matmul_int4_rescale_sm89_launcher(
            reinterpret_cast<const uint16_t*>(qA.data_ptr<int16_t>()),
            reinterpret_cast<const uint16_t*>(Wp.data_ptr<int8_t>()),
            sA.data_ptr<float>(), sB.data_ptr<float>(),
            reinterpret_cast<__half*>(Y.data_ptr<at::Half>()),
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    }
    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_16x16", &matmul_16x16, "FP16 matmul (CUDA)");
    m.def("matmul_sm89", &matmul_sm89,
          "FP16 matmul tuned for sm_89 (Ada Lovelace). cp.async + mma.sync.m16n8k16. "
          "A (M,K) fp16, B (N,K) fp16 -> C (M,N) fp16. Requires K % 8 == 0.",
          py::arg("A"), py::arg("B"));
    m.def("matmul_int8", &cutlass_int8_matmul, "CUTLASS Int8 Matrix Multiplication");
    m.def("matmul_int8_sm89", &matmul_int8_sm89,
          "Hand-written int8 x int8 -> int32 tensor-core GEMM (mma.m16n8k32.s8). "
          "A (M, K) int8, B (N, K) int8; returns (M, N) int32 = A @ B.T. "
          "Requires K % 128 == 0 and N % 128 == 0.",
          py::arg("A"), py::arg("B"));
    m.def("matmul_int8_cublas", &cublaslt_int8_matmul, "cuBLASLt Int8 Matrix Multiplication");
    m.def("matmul_int4_cublas", &matmul_int4_cublas,
          "Native int4 x int4 GEMM via cuBLASLt. "
          "Ap (M, K/2) int8 storage, Bp (N, K/2) int8 storage (packed signed nibbles); "
          "returns (M, N) int32. Requires K % 32 == 0.",
          py::arg("Ap"), py::arg("Bp"));
    m.def("per_token_sym_quant", &per_token_sym_quant,
          "Per-token symmetric int8 quantization. Input (B,T,C) float32/16, "
          "returns (Q int8 (B,T,C), scale float32 (B,T)).");
    m.def("rescale_int32_rowcol", &rescale_int32_rowcol,
          "Y[m,n] = C[m,n] * sA[m] * sB[n].  C int32 (M,N), sA float32 (M,), "
          "sB float32 (N,). out_dtype defaults to float16.",
          py::arg("C"), py::arg("sA"), py::arg("sB"),
          py::arg("out_dtype") = c10::optional<torch::ScalarType>(torch::kFloat16));
    m.def("unpack_int4_to_int8", &unpack_int4_to_int8,
          "Unpack signed int4 (low nibble = even col, high nibble = odd col) "
          "in a packed (N, C/2) int8 tensor to a (N, C) int8 tensor.",
          py::arg("Bp"), py::arg("C"));
    m.def("matmul_int4xint8_cublas", &matmul_int4xint8_cublas,
          "A (M,C) int8 @ unpack(Bp (N,C/2) int4).T -> (M,N) int32 via cuBLASLt.",
          py::arg("A"), py::arg("Bp"));
    m.def("matmul_int4", &matmul_int4,
          "Hand-written int4 x int4 -> int32 tensor-core GEMM (mma.m16n8k64.s4). "
          "Ap (M, K/2) int8 storage, Bp (N, K/2) int8 storage (packed signed nibbles); "
          "returns (M, N) int32. Requires K % 256 == 0 and N % 128 == 0.",
          py::arg("Ap"), py::arg("Bp"));
    m.def("matmul_f16xint4", &matmul_f16xint4_fn,
          "f16 activation x grouped-int4 weight -> f16. Int4 weights are "
          "dequantized to f16 (w = sB*q + zB, unsigned nibble q, groupsize 128) "
          "and the matmul runs on f16 tensor cores. A (M,K) f16, Bp (N,K/2) uint8 "
          "packed unsigned nibbles, sB/zB (N,K/128) f16. Returns C (M,N) f16. "
          "Requires K % 128 == 0 and N % 128 == 0.",
          py::arg("A"), py::arg("Bp"), py::arg("sB"), py::arg("zB"));
    m.def("matmul_f16xint2", &matmul_f16xint2_fn,
          "W2A16: f16 activation x grouped-asymmetric int2 weight -> f16. Int2 "
          "weights dequantized to f16 (w = sB*q + zB, unsigned q in [0,3], group "
          "128) + f16 tensor-core GEMM. A (M,K) f16, Bp (N,K/4) uint8 packed "
          "unsigned 2-bit, sB/zB (N,K/128) f16. Requires K%128==0, N%128==0.",
          py::arg("A"), py::arg("Bp"), py::arg("sB"), py::arg("zB"));
    m.def("matmul_w4ax", &matmul_w4ax,
          "Unified W4Ax over one per-channel-symmetric int4 weight: W4A16 (f16 "
          "activations) for M<=64, W4A4 (int4 activations) for M>64. A (M,K) f16, "
          "Wp (N,K/2) int8 signed packed int4, sB (N,) f32 per-channel scale. "
          "Returns Y (M,N) f16. Requires K % 256 == 0 and N % 128 == 0.",
          py::arg("A"), py::arg("Wp"), py::arg("sB"));
    m.def("matmul_w4a4", &matmul_w4a4,
          "Per-channel-symmetric W4A4: quantize A to int4 per-token on the fly, "
          "pure int4 tensor-core GEMM, single sA[m]*sB[n] rescale. A (M,K) f16, "
          "Wp (N,K/2) int8 packed signed int4, sB (N,) f32 per-channel scale. "
          "Returns Y (M,N) f16. Requires K % 256 == 0 and N % 128 == 0.",
          py::arg("A"), py::arg("Wp"), py::arg("sB"));
    m.def("matmul_wxax", &matmul_wxax,
          "Fused f16-activation x grouped-per-channel quantized-weight GEMM. "
          "A (M,K) f16, Wp (N, K*bits/16) int8-storage packed signed weights, "
          "sW (N, K/128) f16 group scales; bits in {4,8}. Activations are "
          "quantized per-token on the fly. Returns Y (M,N) f16. "
          "Requires N % 128 == 0, K % 128 == 0 (K % 256 == 0 for bits=4).",
          py::arg("A"), py::arg("Wp"), py::arg("sW"), py::arg("bits"));
    m.def("mymatmul", &mymatmul,
          "Fused f16-activation x grouped-per-channel quantized-weight GEMM. "
          "A (M,K) f16, Wp (N, K*bits/16) int8-storage packed signed weights, "
          "sW (N, K/128) f16 group scales; bits in {4,8}. Activations are "
          "quantized per-token on the fly. Returns Y (M,N) f16. "
          "Requires N % 128 == 0, K % 128 == 0 (K % 256 == 0 for bits=4).",
          py::arg("A"), py::arg("B"), py::arg("workspace"));
    m.def("get_gpu_metrics", &get_gpu_metrics,
          "Query CUDA device properties and return a dict of GPU metrics "
          "(SM count, tensor cores, smem, L2, registers, etc.). "
          "Pass device=-1 (default) to use the current device.",
          py::arg("device") = -1);
}
