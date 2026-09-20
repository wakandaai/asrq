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


def test_a_quantized_output_layer_is_refit_and_then_quantized():
    model, samples = _model_and_samples()
    block = model.layers[0]
    layer = block.norm_out[1]
    start = layer.weight.detach().clone()

    class Quantizer:
        def __init__(self):
            self.H = torch.zeros(32, 32, dtype=torch.float64)
            self.nsamples = 7
            self.tokens = 0
            self.weight_when_called = None

        def add_batch(self, batch):
            self.tokens += batch[0].reshape(-1, batch[0].shape[-1]).shape[0]

        def __call__(self):
            self.weight_when_called = layer.weight.detach().clone()
            with torch.no_grad():
                layer.weight.copy_(torch.round(layer.weight * 4) / 4)
            return ("scales", "zeros")

    quantizer = Quantizer()
    modelq = _modelq(model, block_output_refit=True)
    modelq.qparams = {}
    modelq.capture_refit_targets = lambda name: ModelQ.capture_refit_targets(modelq, name)
    targets = [torch.randn(1, 40, 32) for _ in samples]
    name = "layers.0.norm_out.1"
    assert ModelQ.refit_quantizes_output_layer(modelq, "layers.0", {name: quantizer}) == {name}
    ModelQ.refit_block_output(modelq, "layers.0", {name: quantizer}, lambda i: block(samples[i]), targets)
    assert quantizer.nsamples == 0 or quantizer.tokens > 0
    assert not torch.equal(quantizer.weight_when_called, start), "the layer was refit before it was quantized"
    assert torch.equal(layer.weight, torch.round(quantizer.weight_when_called * 4) / 4)
    assert modelq.qparams[name] == ("scales", "zeros")


class TupleBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return (x + self.fc(x), "cache")
