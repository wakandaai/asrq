"""Saving a quantized model and loading it instead of quantizing again (ModelQ.save_quantized / load_quantized, and
exp.py's fingerprinted quantized_path)."""

import datetime
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from _pipeline import config
from asrq.core.model import ModelQ
from asrq.experiment import experiment_run_name, quantized_model_fingerprint, quantized_model_path
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
    """A model as quantization leaves it: rounded weights and a norm tweaking gave a scale-free norm a scale."""
    model = _model(width=width)
    with torch.no_grad():
        for block in model.layers:
            block.fc.weight.copy_(block.fc.weight.mul(4).round().div(4))
    model.layers[0].norm.register_buffer("weight", torch.rand(width) + 0.5)
    return model


def test_a_saved_model_loads_into_a_fresh_one_with_its_added_structure(tmp_path):
    path = str(tmp_path / "q" / "model.pt")
    quantized = _quantized()
    ModelQ.save_quantized(_modelq(quantized), path, "abc")
    fresh = _model(seed=1)
    ModelQ.load_quantized(_modelq(fresh), path, "abc")
    assert "weight" in fresh.layers[0].norm._buffers
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
    assert base != quantized_model_fingerprint(_cfg("quantizer.scale_recovery=true", rotation=str(rotation)))
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


def test_the_run_directory_name_carries_the_settings_and_the_time():
    name = experiment_run_name(_cfg("quantizer.bits=2", "quantizer.symmetric=False", "activation_bits=4", "method=demo"))
    settings, stamp = name.rsplit("_", 1)
    assert settings == "openai-whisper-large-v3_demo_gptq_w2g128asym_a4_rotation"
    day, month, year, second, minute, hour = stamp.split("-")
    now = datetime.datetime.now()
    assert (int(day), int(month), int(year), int(hour)) == (now.day, now.month, now.year % 100, now.hour)
    assert 0 <= int(second) < 60 and 0 <= int(minute) < 60
