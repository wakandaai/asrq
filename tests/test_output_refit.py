"""Refitting a block's output linear layer, in closed form, to the full-precision block's output."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from asrq.core.model import ModelQ
from asrq.quantizers.output_refit import OutputRefit


def test_the_refit_recovers_a_linear_map_from_its_statistics():
    torch.manual_seed(0)
    layer = nn.Linear(16, 12).double()
    W, b = torch.randn(12, 16, dtype=torch.float64), torch.randn(12, dtype=torch.float64)
    refit = OutputRefit(layer)
    for _ in range(4):
        z = torch.randn(50, 16, dtype=torch.float64)
        refit.add(z, z @ W.T + b)
    before, after = refit.solve(ridge=0.0)
    assert torch.allclose(layer.weight, W, atol=1e-8) and torch.allclose(layer.bias, b, atol=1e-8)
    assert before > 0.1 and after < 1e-12


def test_a_large_ridge_keeps_the_current_weights():
    torch.manual_seed(0)
    layer = nn.Linear(8, 8).double()
    start = layer.weight.detach().clone()
    refit = OutputRefit(layer)
    z = torch.randn(100, 8, dtype=torch.float64)
    refit.add(z, torch.randn(100, 8, dtype=torch.float64))
    refit.solve(ridge=1e6)
    assert (layer.weight - start).abs().max() < 1e-3


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.ff = nn.Linear(width, width)
        self.norm_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width))

    def forward(self, x):
        return self.norm_out(x + torch.relu(self.ff(x)))


def _model_and_samples(width=32):
    torch.manual_seed(0)
    model = nn.Module()
    model.layers = nn.ModuleList([Block(width)])
    samples = [torch.randn(1, 40, width) for _ in range(6)]
    return model, samples


def _modelq(model, **config):
    modelq = SimpleNamespace(model=model, quant_cfg=SimpleNamespace(block_output_refit_ridge=0.0, **config))
    modelq.block_output_layer = lambda name: ModelQ.block_output_layer(modelq, name)
    return modelq


def test_refitting_a_block_lowers_its_output_error_by_the_reported_amount():
    model, samples = _model_and_samples()
    block = model.layers[0]
    with torch.no_grad():
        targets = [block(x) for x in samples]
        block.ff.weight.add_(0.3 * torch.randn_like(block.ff.weight))

    def error():
        with torch.no_grad():
            return sum(float((block(x) - t).pow(2).sum()) for x, t in zip(samples, targets)) / sum(
                float(t.pow(2).sum()) for t in targets)

    modelq = _modelq(model)
    assert modelq.block_output_layer("layers.0") == "layers.0.norm_out.1"
    before = error()
    ModelQ.refit_block_output(modelq, "layers.0", {"layers.0.ff": object()}, lambda i: block(samples[i]), targets)
    after = error()
    assert after < 0.8 * before


def test_a_quantized_output_layer_is_not_refit():
    model, samples = _model_and_samples()
    block = model.layers[0]
    start = block.norm_out[1].weight.detach().clone()
    targets = [torch.zeros(1, 40, 32) for _ in samples]
    ModelQ.refit_block_output(_modelq(model), "layers.0", {"layers.0.norm_out.1": object()}, lambda i: block(samples[i]), targets)
    assert torch.equal(block.norm_out[1].weight, start)
    assert _modelq(model).block_output_layer("layers") is None


class TupleBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return (x + self.fc(x), "cache")


def test_an_inserted_output_linear_starts_as_the_identity_and_is_found_and_not_quantized():
    from asrq.quantizers.output_refit import insert_block_output_linear

    torch.manual_seed(0)
    model = nn.Module()
    model.layers = nn.ModuleList([Block(16), TupleBlock(16)])
    x = torch.randn(2, 5, 16)
    before = (model.layers[0](x), model.layers[1](x)[0])
    for block in model.layers:
        linear = insert_block_output_linear(block, 16)
        assert insert_block_output_linear(block, 16) is linear
    after_tensor, after_tuple = model.layers[0](x), model.layers[1](x)
    assert torch.equal(after_tensor, before[0]) and torch.equal(after_tuple[0], before[1]) and after_tuple[1] == "cache"
    modelq = _modelq(model, exclude_modules=[])
    assert modelq.block_output_layer("layers.0") == "layers.0.norm_out.1"
    assert modelq.block_output_layer("layers.1") == "layers.1.output_linear"
    assert not ModelQ.should_quantize_module(modelq, "layers.1.output_linear", model.layers[1].output_linear)
    assert ModelQ.should_quantize_module(modelq, "layers.1.fc", model.layers[1].fc)
