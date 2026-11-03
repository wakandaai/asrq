"""Refitting quantized layers, in closed form, to the full-precision output: a block's output linear layer
(ModelQ.apply_weight_recovery) and every quantized layer before it is quantized (ModelQ.layer_recovery)."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from asrq.core.model import ModelQ
from asrq.quantizers.weight_recovery import WeightRecovery


def test_the_refit_recovers_a_linear_map_from_its_statistics():
    torch.manual_seed(0)
    layer = nn.Linear(16, 12).double()
    W, b = torch.randn(12, 16, dtype=torch.float64), torch.randn(12, dtype=torch.float64)
    refit = WeightRecovery(layer)
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
    refit = WeightRecovery(layer)
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
    modelq = SimpleNamespace(model=model, quant_cfg=SimpleNamespace(weight_recovery_ridge=0.0, **config))
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
    ModelQ.apply_weight_recovery(modelq, "layers.0", {"layers.0.ff": object()}, lambda i: block(samples[i]), targets)
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
    modelq = _modelq(model, weight_recovery=True)
    modelq.qparams = {}
    modelq.capture_weight_recovery = lambda name: ModelQ.capture_weight_recovery(modelq, name)
    targets = [torch.randn(1, 40, 32) for _ in samples]
    name = "layers.0.norm_out.1"
    assert ModelQ.weight_recovery_quantizes(modelq, "layers.0", {name: quantizer}) == {name}
    ModelQ.apply_weight_recovery(modelq, "layers.0", {name: quantizer}, lambda i: block(samples[i]), targets)
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


def test_the_refit_recovers_a_pointwise_conv_from_its_statistics():
    """A pointwise Conv1d is a linear layer whose activations are (B, C, T); the fit reads it as one."""
    torch.manual_seed(0)
    layer = nn.Conv1d(16, 12, kernel_size=1).double()
    W = torch.randn(12, 16, dtype=torch.float64)
    b = torch.randn(12, dtype=torch.float64)
    refit = WeightRecovery(layer)
    for _ in range(4):
        x = torch.randn(2, 16, 25, dtype=torch.float64)
        refit.add(x, torch.einsum("bct,oc->bot", x, W) + b[None, :, None])
    before, after = refit.solve(ridge=0.0)
    assert layer.weight.shape == (12, 16, 1)
    assert torch.allclose(layer.weight[:, :, 0], W, atol=1e-8) and torch.allclose(layer.bias, b, atol=1e-8)
    assert before > 0.1 and after < 1e-12


class FakeQuantizer:
    """A quantizer with the surface layer_recovery uses: a module, a Hessian, add_batch and a call that
    fake-quantizes the weight onto a coarse grid."""

    def __init__(self, module, levels=2):
        super().__init__()
        self.module = module
        width = module.weight.shape[1]
        self.H = torch.zeros(width, width)
        self.nsamples = 11
        self.tokens = 0
        self.levels = levels
        self.weight_when_called = None

    def add_batch(self, batch):
        self.tokens += batch[0].reshape(-1, batch[0].shape[-1]).shape[0]

    def __call__(self):
        self.weight_when_called = self.module.weight.detach().clone()
        with torch.no_grad():
            self.module.weight.copy_(torch.round(self.module.weight * self.levels) / self.levels)
        return ("scales", "zeros")


def _recovery(block, quantizers, samples, ridge=0.0, enabled=True):
    modelq = SimpleNamespace(quant_cfg=SimpleNamespace(
        layer_weight_recovery=enabled, layer_weight_recovery_ridge=ridge,
    ))
    recover = ModelQ.layer_recovery(
        modelq, block, quantizers, lambda b, i: b(samples[i]), len(samples),
    )
    return modelq, recover


def _block_quantizers(block):
    return {"layers.0.ff": FakeQuantizer(block.ff), "layers.0.norm_out.1": FakeQuantizer(block.norm_out[1])}


def test_no_recovery_when_it_is_off():
    model, samples = _model_and_samples()
    block = model.layers[0]
    assert _recovery(block, _block_quantizers(block), samples, enabled=False)[1] is None


def test_a_layer_no_quantized_layer_feeds_is_left_alone():
    """ff reads the block's input, so its input is the same in both runs however the block is quantized and
    the fit would return the weights it already has."""
    model, samples = _model_and_samples()
    block = model.layers[0]
    quantizers = _block_quantizers(block)
    _, recover = _recovery(block, quantizers, samples)
    with torch.no_grad():
        block.norm_out[1].weight.add_(0.3 * torch.randn_like(block.norm_out[1].weight))
    start = block.ff.weight.detach().clone()
    recover("layers.0.ff")
    assert torch.equal(block.ff.weight, start)
    assert quantizers["layers.0.ff"].nsamples == 11, "the Hessian was refreshed for a layer with nothing to fit"


def test_recovering_a_downstream_layer_lowers_the_block_output_error():
    model, samples = _model_and_samples()
    block = model.layers[0]
    with torch.no_grad():
        targets = [block(x) for x in samples]
    quantizers = _block_quantizers(block)
    _, recover = _recovery(block, quantizers, samples)
    with torch.no_grad():
        block.ff.weight.add_(0.3 * torch.randn_like(block.ff.weight))

    def error():
        with torch.no_grad():
            return sum(float((block(x) - t).pow(2).sum()) for x, t in zip(samples, targets)) / sum(
                float(t.pow(2).sum()) for t in targets)

    before = error()
    recover("layers.0.norm_out.1")
    assert error() < 0.8 * before
    quantizer = quantizers["layers.0.norm_out.1"]
    assert quantizer.nsamples == 0 and quantizer.tokens == sum(x.shape[1] for x in samples), (
        "the Hessian was not collected again from the inputs the layer will be quantized on"
    )


def test_quantize_layers_recovers_each_layer_before_quantizing_it():
    model, samples = _model_and_samples()
    block = model.layers[0]
    quantizers = _block_quantizers(block)
    modelq, recover = _recovery(block, quantizers, samples)
    modelq.qparams = {}
    start = block.norm_out[1].weight.detach().clone()
    quantized = list(ModelQ.quantize_layers(modelq, quantizers, [], None, recover))
    assert quantized == list(quantizers)
    refit = quantizers["layers.0.norm_out.1"].weight_when_called
    assert not torch.equal(refit, start), "norm_out.1 was quantized without being refit first"
    assert torch.equal(block.norm_out[1].weight, torch.round(refit * 2) / 2)


class OrderedBlock(nn.Module):
    """A block whose submodules are defined in the opposite order to the one it runs them in, as a Conformer
    block defines its convolution before its attention and runs it after."""

    def __init__(self, width):
        super().__init__()
        self.second = nn.Linear(width, width)
        self.first = nn.Linear(width, width)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


def _order(block, samples, enabled=True):
    modelq = SimpleNamespace(quant_cfg=SimpleNamespace(layer_weight_recovery=enabled))
    quantizers = {name: FakeQuantizer(module) for name, module in block.named_children()}
    return list(ModelQ.in_execution_order(modelq, quantizers, lambda i: block(samples[i])))


def test_layers_are_reordered_by_when_they_run():
    torch.manual_seed(0)
    block = OrderedBlock(8)
    samples = [torch.randn(1, 5, 8)]
    assert list(block.named_children())[0][0] == "second", "the definition order this test relies on changed"
    assert _order(block, samples) == ["first", "second"]


def test_the_definition_order_is_kept_when_the_recovery_is_off():
    torch.manual_seed(0)
    block = OrderedBlock(8)
    assert _order(block, [torch.randn(1, 5, 8)], enabled=False) == ["second", "first"]


def test_a_layer_the_pass_never_calls_keeps_its_place_at_the_end():
    torch.manual_seed(0)
    block = OrderedBlock(8)
    block.unused = nn.Linear(8, 8)
    assert _order(block, [torch.randn(1, 5, 8)]) == ["first", "second", "unused"]
