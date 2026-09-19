"""Saving a quantized model and loading it instead of quantizing again (ModelQ.save_quantized / load_quantized, and
exp.py's fingerprinted quantized_path)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from _pipeline import config
from asrq.core.model import ModelQ
from asrq.experiment import quantized_model_fingerprint, quantized_model_path
from asrq.quantizers.output_refit import insert_block_output_linear
from asrq.transforms.rotation.utils import _rmsnorm


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = _rmsnorm(nn.RMSNorm(width), [nn.Linear(width, width)])
        self.fc = nn.Linear(width, width)

    def forward(self, x):
        return x + self.fc(self.norm(x))


def _model(seed=0, width=16):
    torch.manual_seed(seed)
    model = nn.Module()
    model.layers = nn.ModuleList([Block(width), Block(width)])
    return model


def _modelq(model):
    modelq = SimpleNamespace(model=model, QUANTIZED_FORMAT=ModelQ.QUANTIZED_FORMAT)
    return modelq


def _quantized(width=16):
    """A model as quantization leaves it: an inserted, refit output Linear and a tweaked scale-free norm."""
    model = _model(width=width)
    linear = insert_block_output_linear(model.layers[1], width)
    with torch.no_grad():
        linear.weight.add_(0.1 * torch.randn_like(linear.weight))
    model.layers[0].norm.register_buffer("weight", torch.rand(width) + 0.5)
    return model


def test_a_saved_model_loads_into_a_fresh_one_with_its_added_structure(tmp_path):
    path = str(tmp_path / "q" / "model.pt")
    quantized = _quantized()
    ModelQ.save_quantized(_modelq(quantized), path, "abc")
    fresh = _model(seed=1)
    ModelQ.load_quantized(_modelq(fresh), path, "abc")
    assert isinstance(fresh.layers[1].output_linear, nn.Linear) and "weight" in fresh.layers[0].norm._buffers
    for (name, expected), (_, got) in zip(quantized.state_dict().items(), fresh.state_dict().items()):
        assert got.dtype == expected.dtype and torch.equal(got, expected), name
    x = torch.randn(3, 16)
    expected = x
    for block in quantized.layers:
        expected = block(expected)
    got = x
    for block in fresh.layers:
        got = block(got)
    assert torch.equal(got, expected)


def test_a_mismatched_fingerprint_is_refused(tmp_path):
    path = str(tmp_path / "model.pt")
    ModelQ.save_quantized(_modelq(_quantized()), path, "abc")
    with pytest.raises(ValueError, match="different settings"):
        ModelQ.load_quantized(_modelq(_model()), path, "xyz")


def _cfg(*overrides, rotation=None):
    cfg = config("whisper", list(overrides))
    OmegaConf.set_struct(cfg, False)
    if rotation is not None:
        cfg.transform.path = rotation
    return cfg


def test_the_fingerprint_follows_the_weights_settings_only(tmp_path):
    rotation = tmp_path / "rotation.pt"
    rotation.write_bytes(b"one")
    base = quantized_model_fingerprint(_cfg(rotation=str(rotation)))
    assert base == quantized_model_fingerprint(_cfg("eval_batches=2", "inference=humming", rotation=str(rotation)))
    assert base != quantized_model_fingerprint(_cfg("quantizer.bits=2", rotation=str(rotation)))
    assert base != quantized_model_fingerprint(_cfg("quantizer.norm_tweak=true", rotation=str(rotation)))
    assert base != quantized_model_fingerprint(_cfg("calibration.num_samples=16", rotation=str(rotation)))
    rotation.write_bytes(b"two")
    assert base != quantized_model_fingerprint(_cfg(rotation=str(rotation)))


def test_quantized_path_auto_explicit_and_disabled():
    cfg = _cfg()
    assert quantized_model_path(cfg)[0] is None
    cfg.quantized_path = "auto"
    path, fingerprint = quantized_model_path(cfg)
    assert path == f"outputs/quantized/openai-whisper-large-v3-{fingerprint}.pt"
    cfg.quantized_path = "/tmp/elsewhere.pt"
    assert quantized_model_path(cfg)[0] == "/tmp/elsewhere.pt"
