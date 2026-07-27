# pyright: reportMissingImports=false

import torch
import torch.nn as nn

from asrq.matmulq.base import mmq

# Every (wbits, abits) pair this module knows how to run, and the matmul
# kernel that backs it (see asrq/matmulq/csrc/cuda/): w4a16/w2a16 dequantize
# a packed weight straight into fp16 registers for an f16 tensor core (A
# stays fp16, no activation quantization needed); w4a4/w4a8 quantize A
# on-the-fly (via quantize_sym_int4/quantize_sym_int8, asrq_sym_quant.cu)
# and run both operands through an integer tensor core.
_SUPPORTED_BITS = {(4, 16), (2, 16), (4, 4), (4, 8)}
_GROUP_SIZE = 128  # the only group size the CUDA kernels implement


def _quantize_w4(W, group_size, symmetric):
    """Quantize a (N, K) fp16 weight to excess-8 packed int4, either
    per-channel (group_size=None, one scale/[zero] per row) or groupwise
    (group_size=128, one scale/[zero] per (row, 128-wide K group)) --
    matching w4a16_speed_test.py / w4a16_group_speed_test.py's reference
    quantizers. Returns (packed, scale) if symmetric, (packed, scale, zero)
    if not: packed is (N, K/2) uint8; scale/zero are (N,) or (N, K/128) f16.
    """
    N, K = W.shape
    gs = group_size if group_size is not None else K
    Wg = W.view(N, K // gs, gs).float()
    if symmetric:
        scale = (Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0)
        code = torch.round(Wg / scale).clamp(-8, 7)
        nibble = (code + 8).to(torch.uint8).view(N, K)  # excess-8: -8..7 -> 0..15
        zero = None
    else:
        w_min = Wg.amin(dim=2, keepdim=True)
        w_max = Wg.amax(dim=2, keepdim=True)
        scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
        zero_point = torch.round(-w_min / scale).clamp(0, 15)
        nibble = torch.round(Wg / scale + zero_point).clamp(0, 15).to(torch.uint8).view(N, K)
        zero = (-zero_point * scale).half()
    n_even = nibble[:, 0::2] & 0xF
    n_odd = nibble[:, 1::2] & 0xF
    packed = (n_even | (n_odd << 4)).to(torch.uint8).contiguous()

    scale = scale.half()
    dims_to_drop = (2, 1) if group_size is None else (2,)
    for d in dims_to_drop:
        scale = scale.squeeze(d)
        if zero is not None:
            zero = zero.squeeze(d)
    return (packed, scale.contiguous()) if zero is None else (packed, scale.contiguous(), zero.contiguous())


def _quantize_w2(W, group_size, symmetric):
    """Same as _quantize_w4 but excess-2 packed int2, 4 codes/byte --
    matching w2a16_speed_test.py / w2a16_group_speed_test.py's reference
    quantizers."""
    N, K = W.shape
    gs = group_size if group_size is not None else K
    Wg = W.view(N, K // gs, gs).float()
    if symmetric:
        scale = (Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 2.0)
        code = torch.round(Wg / scale).clamp(-2, 1)
        raw = (code + 2).to(torch.uint8).view(N, K)  # excess-2: -2..1 -> 0..3
        zero = None
    else:
        w_min = Wg.amin(dim=2, keepdim=True)
        w_max = Wg.amax(dim=2, keepdim=True)
        scale = ((w_max - w_min) / 3.0).clamp(min=1e-8)
        zero_point = torch.round(-w_min / scale).clamp(0, 3)
        raw = torch.round(Wg / scale + zero_point).clamp(0, 3).to(torch.uint8).view(N, K)
        zero = (-zero_point * scale).half()
    c0 = raw[:, 0::4] & 0x3
    c1 = raw[:, 1::4] & 0x3
    c2 = raw[:, 2::4] & 0x3
    c3 = raw[:, 3::4] & 0x3
    packed = (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).to(torch.uint8).contiguous()

    scale = scale.half()
    dims_to_drop = (2, 1) if group_size is None else (2,)
    for d in dims_to_drop:
        scale = scale.squeeze(d)
        if zero is not None:
            zero = zero.squeeze(d)
    return (packed, scale.contiguous()) if zero is None else (packed, scale.contiguous(), zero.contiguous())


def _quantize_w4_twos_complement(W, group_size=None):
    """Symmetric int4, plain two's complement (not excess-8) -- the
    weight-side packing matmul_kernel_w4a4/w4a8[_group] expect, since both
    feed an integer tensor core directly with no fp16 LOP3 dequant to serve.
    Either per-channel (group_size=None, one scale per row, matching
    w4a4_speed_test.py's quantize_int4_symmetric) or groupwise
    (group_size=128, one scale per (row, 128-wide K group), matching
    w4a4_group_speed_test.py / w4a8_group_speed_test.py's
    quantize_w4_group_symmetric). Returns (packed, scale): packed (N, K/2)
    uint8; scale (N,) or (N, K/128) f16.
    """
    N, K = W.shape
    gs = group_size if group_size is not None else K
    Wg = W.view(N, K // gs, gs).float()
    scale = (Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0)
    q = torch.round(Wg / scale).clamp(-8, 7).to(torch.int32)
    nibble = (q & 0xF).to(torch.uint8).view(N, K)
    n_even = nibble[:, 0::2] & 0xF
    n_odd = nibble[:, 1::2] & 0xF
    packed = (n_even | (n_odd << 4)).to(torch.uint8).contiguous()

    scale = scale.half().squeeze(2)
    if group_size is None:
        scale = scale.squeeze(1)
    return packed, scale.contiguous()


class ASRQLinear(nn.Linear):
    """Drop-in replacement for nn.Linear backed by the custom CUDA GEMM
    kernels in asrq.matmulq: weights quantized to 2 or 4 bits, activations
    kept in fp16 or quantized to 4/8 bits, chosen via (wbits, abits):

        (4, 16) -> w4a16_matmul / w4a16_group_matmul / their _asym variants
        (2, 16) -> w2a16_matmul / w2a16_group_matmul / their _asym variants
        (4, 4)  -> quantize_sym_int4(x) + w4a4_matmul / w4a4_group_matmul
        (4, 8)  -> quantize_sym_int8(x) + w4a8_matmul / w4a8_group_matmul

    group_size selects per-channel (-1, the default: one weight scale per
    output channel) vs groupwise (128, the only group size the CUDA kernels
    implement: one weight scale per (channel, 128-wide K group)) weight
    quantization, for every (wbits, abits) combination above -- W4A4/W4A8's
    activation quantization (per-token, via quantize_sym_int4/int8) is
    unaffected either way, only the weight-side scale granularity changes.

    symmetric selects symmetric ([-max_abs, max_abs], no zero-point) vs
    asymmetric ([w_min, w_max], zero-point) weight quantization -- only
    meaningful for (4,16)/(2,16); W4A4/W4A8 weights are always symmetric
    two's complement (there's no asymmetric int4xint4/int4xint8 kernel).

    Input to forward() is fp16 with an arbitrary number of leading batch
    dims (..., in_features); output is fp16 (..., out_features), matching
    plain nn.Linear. The module must be quantized before use -- either
    build it directly and call quantize_() once self.weight holds real
    values, or use the from_linear() classmethod, which does both in one
    step.
    """

    def __init__(self, in_features, out_features, bias=True, wbits=4, abits=4, group_size=-1, symmetric=True):
        super().__init__(in_features, out_features, bias=bias)
        self.wbits = wbits
        self.abits = abits
        self.group_size = group_size
        self.symmetric = symmetric
        self._validate_config()

        self.quantized = False
        self.register_buffer("qweight", None, persistent=False)
        self.register_buffer("weight_scales", None, persistent=False)
        self.register_buffer("weight_zeros", None, persistent=False)

    def _validate_config(self):
        if (self.wbits, self.abits) not in _SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported (wbits={self.wbits}, abits={self.abits}); "
                f"supported combinations are {sorted(_SUPPORTED_BITS)}"
            )
        if self.group_size not in (-1, _GROUP_SIZE):
            raise ValueError(f"group_size must be -1 (per-channel) or {_GROUP_SIZE}; got {self.group_size}")
        # cp.async alignment requirements the underlying kernels enforce
        # (see each *_matmul binding's own TORCH_CHECK in bindings.cpp).
        align = 64 if self.wbits == 2 else 32
        if self.in_features % align != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be a multiple of {align} for wbits={self.wbits}"
            )
        if self.group_size == _GROUP_SIZE and self.in_features % _GROUP_SIZE != 0:
            raise ValueError(f"in_features ({self.in_features}) must be a multiple of group_size ({_GROUP_SIZE})")

    @classmethod
    def from_linear(cls, linear: nn.Linear, wbits=4, abits=4, group_size=-1, symmetric=True) -> "ASRQLinear":
        """Build an ASRQLinear from an existing nn.Linear: copies its
        weight/bias, quantizes the weight immediately, and returns the
        ready-to-use module."""
        module = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            wbits=wbits, abits=abits, group_size=group_size, symmetric=symmetric,
        )
        module.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        module.quantize_()
        return module

    @torch.no_grad()
    def quantize_(self):
        """Quantize self.weight in place into the packed buffers the CUDA
        kernels expect, then drop the float weight Parameter -- it's dead
        weight (no pun intended) once quantized, and freeing it halves the
        module's footprint for the (4,16)/(2,16) cases where the packed
        weight is a fraction of the original fp16 size."""
        W = self.weight.data.cuda().half()
        group_size = _GROUP_SIZE if self.group_size == _GROUP_SIZE else None

        if self.abits == 16:
            quantize_fn = _quantize_w4 if self.wbits == 4 else _quantize_w2
            result = quantize_fn(W, group_size, self.symmetric)
            if self.symmetric:
                self.qweight, self.weight_scales = result
                self.weight_zeros = None
            else:
                self.qweight, self.weight_scales, self.weight_zeros = result
        else:
            self.qweight, self.weight_scales = _quantize_w4_twos_complement(W, group_size)
            self.weight_zeros = None

        if self.bias is not None:
            self.bias.data = self.bias.data.cuda().half()
        self.register_parameter("weight", None)
        self.quantized = True

    def _matmul(self, x2d: torch.Tensor) -> torch.Tensor:
        # bias is passed straight into the kernel call and added inside its
        # own epilogue (see get_bias_ptr in bindings.cpp), instead of a
        # separate elementwise add here afterward -- the latter costs a full
        # extra memory-bound pass over the (M,N) output and, unlike cuBLAS's
        # fused bias epilogue, has no free lunch to lean on.
        if self.abits == 16:
            if self.wbits == 4:
                if self.group_size == -1:
                    fn = mmq.w4a16_matmul if self.symmetric else mmq.w4a16_asym_matmul
                else:
                    fn = mmq.w4a16_group_matmul if self.symmetric else mmq.w4a16_group_asym_matmul
            else:
                if self.group_size == -1:
                    fn = mmq.w2a16_matmul if self.symmetric else mmq.w2a16_asym_matmul
                else:
                    fn = mmq.w2a16_group_matmul if self.symmetric else mmq.w2a16_group_asym_matmul
            args = (x2d, self.qweight, self.weight_scales) if self.symmetric \
                else (x2d, self.qweight, self.weight_scales, self.weight_zeros)
            return fn(*args, bias=self.bias)
        elif self.abits == 4:
            xq, x_scales = mmq.quantize_sym_int4(x2d)
            fn = mmq.w4a4_matmul if self.group_size == -1 else mmq.w4a4_group_matmul
            return fn(xq, self.qweight, x_scales, self.weight_scales, bias=self.bias)
        else:  # abits == 8
            xq, x_scales = mmq.quantize_sym_int8(x2d)
            fn = mmq.w4a8_matmul if self.group_size == -1 else mmq.w4a8_group_matmul
            return fn(xq, self.qweight, x_scales, self.weight_scales, bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            raise RuntimeError("ASRQLinear must be quantized (call .quantize_() or build via .from_linear()) before use")

        batch_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).half().contiguous()
        out = self._matmul(x2d)
        return out.reshape(*batch_shape, self.out_features)

    def extra_repr(self) -> str:
        mode = "sym" if self.symmetric else "asym"
        group = "per-channel" if self.group_size == -1 else f"group={self.group_size}"
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, w{self.wbits}a{self.abits} ({mode}, {group})")
