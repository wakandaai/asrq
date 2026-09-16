"""ASRQLinear must compute what the fake-quantized layer it replaces computes.

Evaluation measures WER with fake quantization; ASRQLinear runs the same quantization through
humming's low-bit kernels to measure speed. The two are only comparable if they agree, so every
test compares an ASRQLinear against the fake-quantized reference: weights fake-quantized by the
pipeline's own RTN quantizer, activations by the evaluation's fake quantizer. Humming quantizes
activations itself, with its own float16 arithmetic, so agreement is to kernel rounding -- far
below the quantization error being reproduced, which each test checks as well.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from asrq.core.linear import ASRQLinear, replace_with_asrq_linear
from asrq.quantizers.activation import fake_quantize_activations
from asrq.quantizers.base import QuantConfig
from asrq.quantizers.rtn import RTNQuantizer
from asrq.transforms.rotation.utils import OnlineHadamard, random_hadamard_signs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="humming needs CUDA")
DEVICE = "cuda"


def _fake_quantized(layer, bits=4, group_size=128):
    config = QuantConfig.from_dictconfig(OmegaConf.create(
        {"name": "rtn", "bits": bits, "group_size": group_size, "symmetric": True,
         "exclude_modules": []}
    ))
    RTNQuantizer(layer, "layer", config)()
    return layer


def _linear(in_features=1280, out_features=5120, seed=0):
    torch.manual_seed(seed)
    layer = nn.Linear(in_features, out_features).to(DEVICE).half()
    with torch.no_grad():
        layer.weight.normal_(0, 0.02)
        layer.bias.normal_(0, 0.02)
    return _fake_quantized(layer)


def _inputs(tokens=128, width=1280, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(tokens, width, device=DEVICE, dtype=torch.float16)
    x[:, 3] *= 15
    return x


def _relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@pytest.mark.parametrize("bits, group", [(16, 0), (8, 0), (8, 128), (4, 0), (4, 128)])
def test_matches_the_fake_quantized_layer(bits, group):
    layer, x = _linear(), _inputs()
    real = ASRQLinear.from_linear(layer, 4, 128, bits, group)
    xq = x if bits >= 16 else fake_quantize_activations(x.float(), bits, group or -1, True)
    fake = F.linear(xq.float(), layer.weight.float(), layer.bias.float())
    fp = F.linear(x.float(), layer.weight.float(), layer.bias.float())
    with torch.no_grad():
        out = real(x)
    assert _relative(out, fake) < 5e-3
    if bits < 16:
        assert _relative(fake, fp) > 20 * _relative(out, fake)


def test_keeps_leading_dimensions_and_returns_the_input_dtype():
    layer = _linear()
    real = ASRQLinear.from_linear(layer, 4, 128, 4, 128)
    x = _inputs(tokens=2 * 3 * 50).reshape(2, 3, 50, 1280).to(torch.bfloat16)
    with torch.no_grad():
        out = real(x)
        flat = real(x.reshape(-1, 1280))
    assert out.shape == (2, 3, 50, 5120) and out.dtype == torch.bfloat16
    assert torch.equal(out.reshape(-1, 5120), flat)


def test_survives_the_model_wide_bfloat16_cast_evaluation_does():
    real = ASRQLinear.from_linear(_linear(), 4, 128, 4, 0)
    x = _inputs().to(torch.bfloat16)
    with torch.no_grad():
        before = real(x)
        real.to(torch.bfloat16)
        after = real(x)
    floating = [t for t in list(real.parameters()) + list(real.buffers()) if t.is_floating_point()]
    assert floating and all(t.dtype == torch.float16 for t in floating)
    assert torch.equal(after, before)


def test_a_pointwise_conv_matches_the_equivalent_linear():
    layer = _linear(1024, 1024)
    conv = nn.Conv1d(1024, 1024, 1).to(DEVICE).half()
    with torch.no_grad():
        conv.weight.copy_(layer.weight.unsqueeze(-1))
        conv.bias.copy_(layer.bias)
    as_linear = ASRQLinear.from_linear(layer, 4, 128, 4, 128)
    as_conv = ASRQLinear.from_linear(conv, 4, 128, 4, 128)
    x = torch.randn(2, 1024, 40, device=DEVICE, dtype=torch.float16)
    with torch.no_grad():
        from_conv = as_conv(x)
        from_linear = as_linear(x.transpose(1, 2)).transpose(1, 2)
    assert from_conv.shape == (2, 1024, 40)
    assert torch.equal(from_conv, from_linear)


def test_the_fused_online_hadamard_with_signs_matches_the_separate_module():
    """fc2 behind a folded rotation: the fused path must apply the same X @ D @ H."""
    layer = _linear(5120, 1280)
    block = 128
    signs = random_hadamard_signs(5120, DEVICE)
    hadamard = OnlineHadamard(block, signs=signs)
    x = _inputs(width=5120).abs()
    fused = ASRQLinear.from_linear(layer, 4, 128, 4, 128, hadamard_block_size=block,
                                   hadamard_signs=signs)
    with torch.no_grad():
        rotated = hadamard(x)
        reference = F.linear(fake_quantize_activations(rotated.float(), 4, 128, True),
                             layer.weight.float(), layer.bias.float())
        fp = F.linear(rotated.float(), layer.weight.float(), layer.bias.float())
        out = fused(x)
    assert _relative(out, reference) < 0.1 * _relative(reference, fp)


@pytest.mark.parametrize("tokens", [1, 16, 64, 12000])
@pytest.mark.parametrize("bits", [8, 4])
def test_repeated_calls_with_quantized_activations_are_bit_identical(tokens, bits):
    """humming's default Stream-K schedules round fc2's output differently between identical calls
    at these token counts; the batch-invariant mode quantized-activation layers run must not."""
    layer = _linear(5120, 1280)
    signs = random_hadamard_signs(5120, DEVICE)
    real = ASRQLinear.from_linear(layer, 4, 128, bits, 128, hadamard_block_size=128, hadamard_signs=signs)
    x = _inputs(tokens=tokens, width=5120).abs()
    with torch.no_grad():
        outputs = [real(x) for _ in range(8)]
    assert all(torch.equal(out, outputs[0]) for out in outputs[1:])


def test_only_quantized_activation_layers_use_the_batch_invariant_mode():
    """Batch-invariant schedules slowed fp16-activation layers at decoding sizes, so those keep the default."""
    assert ASRQLinear(1280, 1280, activation_bits=4).compute_config == ASRQLinear.BATCH_INVARIANT
    assert ASRQLinear(1280, 1280, activation_bits=8, activation_group_size=128).compute_config == ASRQLinear.BATCH_INVARIANT
    assert ASRQLinear(1280, 1280, weight_bits=2, activation_bits=16).compute_config is None


def test_replace_moves_the_online_hadamard_into_fc2_and_unwraps_the_activation():
    torch.manual_seed(0)
    block = 128
    signs = random_hadamard_signs(512, DEVICE)
    model = nn.Module()
    model.fc1 = nn.Linear(256, 512).to(DEVICE).half()
    model.act = nn.Sequential(nn.GELU(), OnlineHadamard(block, signs=signs)).to(DEVICE)
    model.fc2 = _fake_quantized(nn.Linear(512, 256).to(DEVICE).half())

    replaced = replace_with_asrq_linear(
        model, {"fc1": (4, 16, 0), "fc2": (4, 4, 128)}, weight_group_size=128,
        online_hadamards={"fc2": "act"},
    )
    assert set(replaced) == {"fc1", "fc2"}
    assert isinstance(model.act, nn.GELU)
    assert model.fc2.hadamard_block_size == block
    assert torch.equal(model.fc2.hadamard_signs.float(), signs.float())
    assert model.fc1.hadamard_block_size is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"activation_bits": 4, "activation_group_size": 32}, "at least 64"),
        ({"activation_bits": 4, "activation_group_size": 100}, "must divide"),
        ({"activation_bits": 2}, "4, 8 or 16"),
        ({"weight_group_size": 100}, "does not divide"),
        ({"hadamard_block_size": 100}, "does not divide"),
    ],
)
def test_rejects_settings_humming_cannot_run(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ASRQLinear(1280, 1280, **kwargs)


def test_rejects_a_convolution_that_is_not_pointwise():
    with pytest.raises(ValueError, match="pointwise"):
        ASRQLinear.from_linear(nn.Conv1d(64, 64, 3).to(DEVICE).half())
