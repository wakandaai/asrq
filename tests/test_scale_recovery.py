"""Scale recovery: the per-input-channel scale fitted inside GPTQ and folded into whatever feeds the layer."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from asrq.core.model import ModelQ
from asrq.quantizers.gptq import GPTQQuantizer
from asrq.quantizers.gptq_solver import gptq_factors, gptq_quantize, gptq_quantize_columns
from asrq.quantizers.scale_recovery import (
    ScaleTarget,
    capture_scale_recovery,
    quantize_scale_group,
    scale_norm_output,
)
from asrq.transforms.rotation.utils import _ScaleFreeRMSNorm


@pytest.mark.parametrize("kind", ["layernorm", "scale-free"])
def test_scaling_a_norm_scales_its_output(kind):
    torch.manual_seed(0)
    width = 16
    layer_norm = nn.LayerNorm(width)
    nn.init.normal_(layer_norm.weight, 1.0, 0.1)
    nn.init.normal_(layer_norm.bias, 0.0, 0.1)
    norm = layer_norm if kind == "layernorm" else _ScaleFreeRMSNorm(nn.RMSNorm(width), [nn.Linear(width, 4)])
    x = torch.randn(3, width)
    s = torch.rand(width, dtype=torch.float64) + 0.5
    before = norm(x)
    scale_norm_output(norm, s)
    assert torch.allclose(norm(x), before * s.float(), atol=1e-6)


def _gptq_problem(width=64, tokens=400, rows=(48, 32), seed=0):
    torch.manual_seed(seed)
    mixing = torch.eye(width, dtype=torch.float64) + 0.3 * torch.randn(width, width, dtype=torch.float64)
    X = torch.randn(tokens, width, dtype=torch.float64) @ mixing * torch.linspace(0.2, 3.0, width, dtype=torch.float64)
    weights = [torch.randn(out, width, dtype=torch.float64) for out in rows]
    return X, weights, gptq_factors(X.T @ X, 0.01)


def test_gptq_columns_without_a_scale_is_gptq_and_stacked_rows_are_each_layers_gptq():
    X, weights, factors = _gptq_problem()
    grid = (factors, 2, 16, False, 16)
    Q, s = gptq_quantize_columns(weights[0], *grid)
    assert torch.equal(Q, gptq_quantize(weights[0], *grid)) and torch.equal(s, torch.ones_like(s))
    stacked, _ = gptq_quantize_columns(torch.cat(weights), *grid)
    assert torch.allclose(stacked, torch.cat([gptq_quantize(W, *grid) for W in weights]), atol=1e-12)


def test_an_inloop_scale_lowers_the_output_error_below_gptq_and_a_large_ridge_keeps_it_at_one():
    X, weights, factors = _gptq_problem()
    W = weights[0]

    def output_error(Q, s):
        return float((X @ W.T - X @ (Q * s).T).pow(2).sum())

    plain, ones = gptq_quantize_columns(W, factors, 2, 16, False, 16)
    Q, s = gptq_quantize_columns(W, factors, 2, 16, False, 16, scale_ridge=0.0)
    assert output_error(Q, s) < output_error(plain, ones)
    _, held = gptq_quantize_columns(W, factors, 2, 16, False, 16, scale_ridge=1e12)
    assert torch.allclose(held, torch.ones_like(held), atol=1e-6)


@pytest.mark.parametrize("method", ["lockstep", "average"])
def test_an_absorbers_layers_are_quantized_together_and_the_tweak_lowers_their_output_error(method):
    torch.manual_seed(0)
    width = 64
    block = nn.ModuleDict({
        "norm": nn.LayerNorm(width), "q": nn.Linear(width, 48), "k": nn.Linear(width, 32), "o": nn.Linear(width, 16),
    })

    class Config:
        name, bits, group_size, block_size, percdamp, symmetric, exclude_modules = "gptq", 2, 16, 16, 0.01, False, []

    x = torch.randn(4, 50, width) * torch.linspace(0.2, 3.0, width)
    normed = block["norm"](x).detach()
    full = {name: block[name](normed).detach() for name in ("q", "k")}
    quantizers = {name: GPTQQuantizer(block[name], name, Config()) for name in ("q", "o", "k")}
    for quantizer in quantizers.values():
        quantizer.add_batch((normed, None))
    captured = capture_scale_recovery([ScaleTarget("norm", ["q", "k"])], quantizers)
    model = SimpleNamespace(qparams={}, model=block, quant_cfg=SimpleNamespace(
        scale_recovery_ridge=0.0, scale_recovery_method=method,
    ))
    model.apply_scale_recovery = lambda groups: ModelQ.apply_scale_recovery(model, groups)
    names = list(ModelQ.quantize_layers(model, quantizers, captured))
    assert sorted(names) == ["k", "o", "q"] and set(model.qparams) == {"k", "o", "q"}
    assert not any(hasattr(quantizer, "H") for quantizer in quantizers.values())
    # quantize_layers has folded the scale already, so undoing it gives the unscaled error to compare with
    (group,) = captured
    s, ratio = group.scale, group.ratio
    with torch.no_grad():
        tweaked = sum(float((block[n](block["norm"](x)) - full[n]).pow(2).sum()) for n in ("q", "k"))
        block["norm"].weight.div_(s)
        block["norm"].bias.div_(s)
        untweaked = sum(float((block[n](block["norm"](x)) - full[n]).pow(2).sum()) for n in ("q", "k"))
    assert ratio < 1 and tweaked < untweaked
    assert abs(tweaked / untweaked - ratio) < 1e-3


def test_the_average_is_lockstep_for_one_layer_and_agrees_with_it_on_the_first_column():
    X, weights, factors = _gptq_problem()

    class Quantizer:
        def __init__(self, W):
            self.W, self.H = W.clone(), X.T @ X
            self.quant_config = SimpleNamespace(percdamp=0.01, bits=2, group_size=16, symmetric=False, block_size=16)

        def weight_2d(self):
            return self.W

        def set_weight_2d(self, W):
            self.W = W

        def find_quant_params(self, W):
            return (W[:, :1], W[:, :1])

    one = [quantize_scale_group([Quantizer(weights[0])], 0.01, method)[0] for method in ("lockstep", "average")]
    assert torch.allclose(one[0], one[1], atol=1e-12)
    lockstep, average = (quantize_scale_group([Quantizer(W) for W in weights], 0.0, m)[0] for m in ("lockstep", "average"))
    first = factors.perm[0]
    assert torch.allclose(lockstep[first], average[first], atol=1e-12) and not torch.allclose(lockstep, average)


def _conformer(layers=2, rotated=False):
    blocks = nn.ModuleList()
    for _ in range(layers):
        block = nn.Module()
        for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2"):
            setattr(block, norm, nn.LayerNorm(8))
        block.norm_out = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 8)) if rotated else nn.LayerNorm(8)
        blocks.append(block)
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.layers = blocks
    return model


def test_conformer_norms_are_recovered_only_when_no_rotation_left_a_layer_to_refit():
    from asrq.models.nemo.parakeet_ctc import conformer_scale_recovery_targets

    targets = conformer_scale_recovery_targets(_conformer(), "encoder.layers", 2)
    assert len(targets) == 8
    by_norm = {target.norm: target.layers for target in targets}
    assert by_norm["encoder.layers.1.norm_self_att"] == [f"encoder.layers.1.self_attn.linear_{n}" for n in "qkv"]
    assert by_norm["encoder.layers.0.norm_conv"] == ["encoder.layers.0.conv.pointwise_conv1"]
    assert conformer_scale_recovery_targets(_conformer(rotated=True), "encoder.layers", 2) == []


class RecordingQuantizer:
    """A quantizer that only records when it was called."""

    def __init__(self, name, events):
        self.name, self.events = name, events

    def __call__(self):
        self.events.append(f"quantize {self.name}")
        return ("scales", "zeros")


def test_a_groups_scale_is_folded_before_the_next_layer_is_quantized(monkeypatch):
    """layer_recovery reads a layer's drift by running the block, so a scale fitted earlier in the block has to
    be in its norm by then rather than left for the end of the block."""
    import asrq.core.model as model_module
    from asrq.quantizers.scale_recovery import ScaleGroup

    events = []
    quantizers = {name: RecordingQuantizer(name, events) for name in ("q", "fc")}
    group = ScaleGroup(ScaleTarget("norm", ["q"]))
    monkeypatch.setattr(model_module, "quantize_scale_group", lambda *_args, **_kwargs: (
        torch.ones(4), 0.5, [("scales", "zeros")],
    ))
    model = SimpleNamespace(qparams={}, quant_cfg=SimpleNamespace(
        scale_recovery_ridge=0.0, scale_recovery_method="lockstep",
    ))
    model.apply_scale_recovery = lambda groups: events.append(f"fold {groups[0].target.norm}")
    names = list(ModelQ.quantize_layers(
        model, quantizers, [group], recover=lambda name: events.append(f"recover {name}"),
    ))
    assert names == ["q", "fc"]
    assert events == ["recover q", "fold norm", "recover fc", "quantize fc"]


def test_scale_recovery_is_off_when_the_activations_are_quantized(capsys):
    """It fits the scale for the weights alone: the activation quantizer would then round the scaled channels,
    and a rotation's spread of the outliers over those channels is what makes them roundable."""
    from _pipeline import config
    from asrq.experiment import prepare_experiment_config

    cfg = config("whisper", ["quantizer.scale_recovery=true", "activation_bits=4", "transform=scaling"])
    prepare_experiment_config(cfg, learn_rotation=False)
    assert cfg.quantizer.scale_recovery is False
    assert "scale_recovery is off" in capsys.readouterr().out

    weight_only = config("whisper", ["quantizer.scale_recovery=true", "activation_bits=16", "transform=scaling"])
    prepare_experiment_config(weight_only, learn_rotation=False)
    assert weight_only.quantizer.scale_recovery is True
