"""ULBQ: the k-means quantizer, on the layer types the models actually contain."""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from asrq.quantizers.ulbq import ULBQConfig, ULBQQuantizer


def _config(bits=4):
    return ULBQConfig(OmegaConf.create(
        {"name": "ulbq", "bits": bits, "block_size": 16, "percdamp": 0.01, "exclude_modules": []}
    ))


def _quantize(layer, feed):
    quantizer = ULBQQuantizer(layer, "layer", _config())
    quantizer.add_batch((feed, None))
    quantizer()
    return layer.weight.data


def test_a_pointwise_conv_is_quantized_like_a_linear():
    """A pointwise Conv1d's weight is (out, in, 1), which the column loop cannot index as a 2D slice; the
    quantizer goes through weight_2d, as GPTQ and RTN do."""
    torch.manual_seed(0)
    width = 32
    linear = nn.Linear(width, width, bias=False)
    conv = nn.Conv1d(width, width, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(linear.weight.unsqueeze(-1))
    tokens = torch.randn(2, 20, width)
    linear_weight = _quantize(linear, tokens).clone()
    conv_weight = _quantize(conv, tokens.transpose(1, 2)).clone()
    assert conv_weight.shape == (width, width, 1)
    assert torch.allclose(conv_weight.squeeze(-1), linear_weight, atol=1e-6)


def test_quantization_changes_the_weights_but_keeps_their_scale():
    torch.manual_seed(0)
    width = 32
    layer = nn.Linear(width, width, bias=False)
    before = layer.weight.data.clone()
    after = _quantize(layer, torch.randn(2, 20, width))
    assert not torch.equal(after, before)
    assert after.abs().max() <= before.abs().max() * 1.5
    assert torch.unique(after).numel() < before.numel(), "a codebook has fewer values than the weight has entries"
