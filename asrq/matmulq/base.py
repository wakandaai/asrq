import torch
from torch.utils.cpp_extension import load
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(CUR_DIR, "csrc", "bindings.cpp"),
    os.path.join(CUR_DIR, "csrc", "cuda", "matmul.cu"),
    os.path.join(CUR_DIR, "csrc", "x86", "matmul.cpp"),
]
INCLUDES = [
    os.path.join(CUR_DIR, "csrc", "cuda"),
    os.path.join(CUR_DIR, "csrc", "x86"),
]

# CUDA SM available.  Note: wgmma + TMA require the architecture-specific
# "sm_90a" target (the plain "sm_90" target does NOT enable these features).
cuda_sm = torch.cuda.get_device_capability()
if cuda_sm[0] >= 9:
    # PyTorch's cpp_extension reads TORCH_CUDA_ARCH_LIST to build -gencode
    # flags; "9.0a" maps to sm_90a / compute_90a.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    arch_flag = "sm_90a"
else:
    arch_flag = f"sm_{cuda_sm[0]}{cuda_sm[1]}"
EXTRA_CUDA_CFLAGS = [
    "-O3",
    f"-arch={arch_flag}",
]
EXTRA_CFLAGS = ["-std=c++17"]
# libcuda is the CUDA driver library, required for cuTensorMapEncodeTiled.
EXTRA_LDFLAGS = ["-lcuda"]

print("Compiling CUDA kernel with PyTorch (this might take a minute)...")
mmq = load(
    name="asrq_module",
    sources=SOURCES,
    extra_include_paths=INCLUDES,
    extra_cflags=EXTRA_CFLAGS,
    extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
    extra_ldflags=EXTRA_LDFLAGS,
)


