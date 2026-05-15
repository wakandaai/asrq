import torch
from torch.utils.cpp_extension import load
import os

# Important: Provide the absolute path to your CUTLASS include directory here
# Example: CUTLASS_INCLUDE_DIR = "/home/user/cutlass/include"
CUTLASS_INCLUDE_DIR = os.environ.get("CUTLASS_INCLUDE_DIR", "cutlass/include")

print("Compiling CUDA kernel with PyTorch (this might take a minute)...")
int4_ext = load(
    name="int4_ext",
    sources=["asrq/kernel/gemm.cu", "asrq/kernel/bindings.cpp"],
    extra_include_paths=[CUTLASS_INCLUDE_DIR],
    extra_cflags=["-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "-arch=sm_90",          # Target H100 (Hopper)
        "--expt-relaxed-constexpr",   # required by CUTLASS 3.x / CuTe
        "--expt-extended-lambda",     # required by CuTe closures
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    verbose=True
)
print("Compilation finished!\n")

def test_matmul():
    # Tensor shapes
    M, N, K = 128, 128, 128

    # 1. Create dummy input data on the GPU
    # A is (M, K) standard FP32 activations
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    
    # W_int4_T is (N, K/2) packed INT4 weights, already transposed
    W_int4_T = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device="cuda")
    
    # Scale for the weights (N)
    scale_W = torch.rand(N, dtype=torch.float32, device="cuda") * 0.1

    # 2. Run the custom INT4 extension
    C_custom = int4_ext.matmul_int4(A, W_int4_T, scale_W) # type: ignore

    print(f"Custom Kernel Output Shape: {C_custom.shape}")
    print(f"Sample Custom Output:\n{C_custom[:2, :4]}\n")
    
    print("Extension executed successfully without crashing!")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this on a machine with an NVIDIA GPU.")
    else:
        test_matmul()