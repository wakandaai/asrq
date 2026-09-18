"""Closed-form norm tweaking: the per-channel norm scale that best undoes the quantized layers' weight error."""

import pytest
import torch
import torch.nn as nn

from asrq.quantizers.gptq import GPTQQuantizer
from asrq.quantizers.norm_tweak import apply_norm_tweaks, capture_norm_tweaks, norm_tweak_scale, scale_norm_output
from asrq.transforms.rotation.utils import _rmsnorm


def _problem(width=8, tokens=200, layers=(6, 5), seed=0):
    torch.manual_seed(seed)
    X = torch.randn(tokens, width, dtype=torch.float64) * torch.rand(width, dtype=torch.float64).add(0.5)
    weights = [torch.randn(out, width, dtype=torch.float64) for out in layers]
    quantized = [W + 0.3 * torch.randn_like(W) for W in weights]
    return X, weights, quantized


def test_the_closed_form_is_the_least_squares_solution():
    X, weights, quantized = _problem()
    s, ratio = norm_tweak_scale(X.T @ X, weights, quantized, ridge=0.0)
    target = torch.cat([(X @ W.T).flatten() for W in weights])
    design = torch.cat([
        torch.stack([(X[:, j:j + 1] @ Q[:, j:j + 1].T).flatten() for j in range(X.shape[1])], dim=1) for Q in quantized
    ])
    expected = torch.linalg.lstsq(design, target.unsqueeze(1)).solution.squeeze(1)
    assert torch.allclose(s, expected, atol=1e-8)
    assert 0 < ratio < 1


def test_no_quantization_error_keeps_the_norm_and_a_column_scale_is_undone():
    X, weights, _ = _problem()
    s, ratio = norm_tweak_scale(X.T @ X, weights, weights, ridge=0.01)
    assert torch.allclose(s, torch.ones_like(s)) and ratio == 1.0
    c = torch.rand(X.shape[1], dtype=torch.float64) + 0.5
    s, ratio = norm_tweak_scale(X.T @ X, weights, [W * c for W in weights], ridge=0.0)
    assert torch.allclose(s, 1 / c) and ratio < 1e-20


@pytest.mark.parametrize("kind", ["layernorm", "scale-free"])
def test_scaling_a_norm_scales_its_output(kind):
    torch.manual_seed(0)
    width = 16
    layer_norm = nn.LayerNorm(width)
    nn.init.normal_(layer_norm.weight, 1.0, 0.1)
    nn.init.normal_(layer_norm.bias, 0.0, 0.1)
    norm = layer_norm if kind == "layernorm" else _rmsnorm(nn.RMSNorm(width), [nn.Linear(width, 4)])
    x = torch.randn(3, width)
    s = torch.rand(width, dtype=torch.float64) + 0.5
    before = norm(x)
    scale_norm_output(norm, s)
    assert torch.allclose(norm(x), before * s.float(), atol=1e-6)


def test_a_tweak_after_gptq_lowers_the_quantized_layers_output_error():
    torch.manual_seed(0)
    width = 64
    block = nn.ModuleDict({"norm": nn.LayerNorm(width), "q": nn.Linear(width, 48), "k": nn.Linear(width, 32)})

    class Config:
        name, bits, group_size, block_size, percdamp, symmetric, exclude_modules = "gptq", 2, 16, 16, 0.01, False, []

    x = torch.randn(4, 50, width) * torch.linspace(0.2, 3.0, width)
    normed = block["norm"](x)
    full = {name: block[name](normed).detach() for name in ("q", "k")}
    quantizers = {name: GPTQQuantizer(block[name], name, Config()) for name in ("q", "k")}
    for name, quantizer in quantizers.items():
        quantizer.add_batch((normed.detach(), None))
    modules = dict(block.named_modules())
    captured = capture_norm_tweaks({"norm": ["q", "k"]}, quantizers, modules)
    for quantizer in quantizers.values():
        quantizer()
    with torch.no_grad():
        untweaked = sum(float((block[n](block["norm"](x)) - full[n]).pow(2).sum()) for n in ("q", "k"))
        (s, ratio), = apply_norm_tweaks(captured, modules, ridge=0.0).values()
        tweaked = sum(float((block[n](block["norm"](x)) - full[n]).pow(2).sum()) for n in ("q", "k"))
    assert ratio < 1 and tweaked < untweaked
    assert abs(tweaked / untweaked - ratio) < 1e-3

