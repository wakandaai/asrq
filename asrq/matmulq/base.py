# NOTE: env setup MUST happen before importing torch.utils.cpp_extension,
# because cpp_extension caches CUDA_HOME as a module-level constant at
# import time. Importing it later (or after we mutate os.environ) won't help.
import os
import sys
import shutil

# NeMo/Megatron imports earlier in the asrq package can clobber PATH and hide
# the conda env's `ninja` binary; also some clusters (PSC Bridges-2) put CUDA
# at a non-standard location instead of /usr/local/cuda. Fix both up front.
def _ensure_on_path(d: str) -> None:
    if not d:
        return
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if d not in parts:
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

_ensure_on_path(os.path.dirname(sys.executable))  # conda env bin (ninja, etc.)

def _find_cuda_home():
    for var in ("CUDA_HOME", "CUDA_PATH"):
        cand = os.environ.get(var)
        if cand and os.path.isdir(cand):
            return cand
    # Prefer the standard /usr/local/cuda location (and its version-pinned
    # siblings) over whatever `nvcc` happens to resolve to on PATH: this box
    # also has /usr/bin/nvcc (CUDA 11.5) ahead of any CUDA install's own
    # bin/ on PATH, and nvcc 11.5 has a known host-compiler-compatibility bug
    # with newer libstdc++ (it mis-parses certain lambda-closure patterns in
    # device code as a bogus "parameter packs not expanded" error deep in
    # <bits/std_function.h>) -- checking these fixed locations first avoids
    # silently picking up that broken toolchain.
    for cand in ("/usr/local/cuda", "/opt/packages/cuda/v12.6.1",
                 "/opt/cuda", "/usr/lib/cuda"):
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "bin", "nvcc")):
            return cand
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    return None

_cuda_home = _find_cuda_home()
if _cuda_home is None:
    raise RuntimeError(
        "Cannot locate CUDA toolkit. Set CUDA_HOME to a directory whose "
        "`bin/nvcc` and `include/cuda_fp16.h` exist before importing asrq."
    )
os.environ["CUDA_HOME"] = _cuda_home
os.environ["CUDA_PATH"] = _cuda_home
_ensure_on_path(os.path.join(_cuda_home, "bin"))
print(f"Using CUDA_HOME = {_cuda_home}")

# Now safe to import torch's cpp_extension; it will pick up the right CUDA.
import torch
from torch.utils.cpp_extension import load
import torch.utils.cpp_extension as _cpp_ext

# Override the module-level CUDA_HOME constant: NeMo/transformers may have
# imported cpp_extension transitively before our env fix-up, in which case
# the constant is already cached as None or "/usr/local/cuda".
_cpp_ext.CUDA_HOME = _cuda_home
if hasattr(_cpp_ext, "_CUDA_HOME"):
    _cpp_ext._CUDA_HOME = _cuda_home
print(f"torch.utils.cpp_extension.CUDA_HOME = {_cpp_ext.CUDA_HOME}")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES = [
    os.path.join(CUR_DIR, "csrc", "bindings.cpp"),
    # matmul.cu was emptied during a revert; the sm_89-tuned kernel now
    # backs both matmul_16x16 and matmul_sm89.
    # os.path.join(CUR_DIR, "csrc", "cuda", "matmul_sm89.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "matmul_int4_sm89.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "matmul_int8_sm89.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "matmul_wxax_sm89.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "matmulq.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "int8_cutlass.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "int8_cublas.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "int4_cublas.cu"),
    # os.path.join(CUR_DIR, "csrc", "cuda", "quant.cu"),
    # asrq.cu was split into one file per matmul kernel so each can be
    # edited without touching the others: asrq_16x16.cu (fp16xfp16),
    # asrq_4x16.cu (int4xfp16, sym+asym), asrq_2x16.cu (int2xfp16, sym+asym),
    # asrq_4x4.cu (int4xint4), asrq_4x8.cu (int4xint8). Shared
    # tile-scheduling/loader/MMA helpers live in common.h, included by all.
    os.path.join(CUR_DIR, "csrc", "cuda", "asrq_16x16.cu"),
    os.path.join(CUR_DIR, "csrc", "cuda", "asrq_4x16.cu"),
    os.path.join(CUR_DIR, "csrc", "cuda", "asrq_2x16.cu"),
    os.path.join(CUR_DIR, "csrc", "cuda", "asrq_4x4.cu"),
    os.path.join(CUR_DIR, "csrc", "cuda", "asrq_4x8.cu"),
    os.path.join(CUR_DIR, "csrc", "cuda", "asrq_sym_quant.cu"),
    # os.path.join(CUR_DIR, "csrc", "x86", "matmul.cpp"),
    # os.path.join(CUR_DIR, "csrc", "x86", "qmatmul.cpp"),
]
INCLUDES = [
    os.path.join(CUR_DIR, "csrc", "cuda"),
    os.path.join(CUR_DIR, "csrc", "x86"),
    # os.path.join(CUR_DIR, "..", "..", "cutlass", "include"),
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
    "--expt-relaxed-constexpr",
]
EXTRA_CFLAGS = ["-std=c++17"]#, "-O3", "-mavx2", "-mfma", "-fopenmp"]
# libcuda is the CUDA driver library, required for cuTensorMapEncodeTiled.
EXTRA_LDFLAGS = ["-lcuda", "-lcublasLt"]#, "-fopenmp"]

print("Compiling CUDA kernel with PyTorch (this might take a minute)...")

mmq = load(
    name="asrq_module",
    sources=SOURCES,
    extra_include_paths=INCLUDES,
    extra_cflags=EXTRA_CFLAGS,
    extra_cuda_cflags=EXTRA_CUDA_CFLAGS,
    extra_ldflags=EXTRA_LDFLAGS,
    
)

# Sanity check: both kernels should be exposed by the extension.
for _fn in ("mymatmul", "w4a16_matmul", "w2a16_matmul", "w4a16_asym_matmul", "w2a16_asym_matmul", "w4a4_matmul", "w4a8_matmul",
            "w4a4_group_matmul", "w4a8_group_matmul",
            "quantize_sym_int8", "quantize_sym_int4", "w4a16_group_matmul", "w2a16_group_matmul",
            "w4a16_group_asym_matmul", "w2a16_group_asym_matmul",
            # "w4a16_cpu_matmul", "w4a16_cpu_asym_matmul", "w4a16_cpu_group_matmul", "w4a16_cpu_group_asym_matmul",
            # "w2a16_cpu_matmul", "w2a16_cpu_asym_matmul", "w2a16_cpu_group_matmul", "w2a16_cpu_group_asym_matmul",
            # "w4a8_cpu_matmul", "w4a8_cpu_asym_matmul", "w4a8_cpu_group_matmul", "w4a8_cpu_group_asym_matmul",
            # "w2a8_cpu_matmul", "w2a8_cpu_asym_matmul", "w2a8_cpu_group_matmul", "w2a8_cpu_group_asym_matmul",
            # "quantize_sym_int8_cpu",
            ):
    assert hasattr(mmq, _fn), f"asrq_module is missing '{_fn}' binding"


