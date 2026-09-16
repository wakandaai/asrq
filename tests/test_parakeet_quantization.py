"""Which Parakeet encoder layers GPTQ/RTN quantize.

The Conformer's pointwise convolutions are rotated and activation-quantized like the feed-forward
linears, so their weights must be quantized too; the identity Linears a rotation inserts after each
block's norm_out stand in for a LayerNorm and must not be.
"""

import torch.nn as nn
from omegaconf import OmegaConf

from asrq.models.nemo.parakeet_ctc import ParakeetCTCQ
from asrq.quantizers.base import QuantConfig


def _modelq(exclude=(), quantize_block_output_linear=None):
    cfg = {"name": "rtn", "bits": 4, "group_size": 128, "symmetric": True, "exclude_modules": list(exclude)}
    if quantize_block_output_linear is not None:
        cfg["quantize_block_output_linear"] = quantize_block_output_linear
    modelq = object.__new__(ParakeetCTCQ)
    modelq.quant_cfg = QuantConfig.from_dictconfig(OmegaConf.create(cfg))
    return modelq


def test_linears_and_pointwise_convs_are_quantized():
    modelq = _modelq()
    assert modelq.should_quantize_module("encoder.layers.3.feed_forward1.linear2", nn.Linear(4096, 1024))
    assert modelq.should_quantize_module("encoder.layers.3.self_attn.linear_pos", nn.Linear(1024, 1024))
    assert modelq.should_quantize_module("encoder.layers.3.conv.pointwise_conv1", nn.Conv1d(1024, 1024, 1))
    assert modelq.should_quantize_module("encoder.layers.3.conv.pointwise_conv2", nn.Conv1d(1024, 1024, 1))


def test_depthwise_convs_norms_and_inserted_block_output_linears_are_not():
    modelq = _modelq()
    depthwise = nn.Conv1d(1024, 1024, 9, groups=1024)
    assert not modelq.should_quantize_module("encoder.layers.3.conv.depthwise_conv.conv", depthwise)
    assert not modelq.should_quantize_module("encoder.layers.3.norm_conv", nn.LayerNorm(1024))
    assert not modelq.should_quantize_module("encoder.layers.3.norm_out.1", nn.Linear(1024, 1024))


def test_excluded_modules_are_respected():
    modelq = _modelq(exclude=["encoder.layers.0.conv.pointwise_conv1"])
    assert not modelq.should_quantize_module("encoder.layers.0.conv.pointwise_conv1", nn.Conv1d(1024, 1024, 1))


def test_the_inserted_block_output_linears_are_quantized_only_when_configured():
    linear = nn.Linear(1024, 1024)
    assert not _modelq().should_quantize_module("encoder.layers.3.norm_out.1", linear)
    assert not _modelq(quantize_block_output_linear=False).should_quantize_module("encoder.layers.3.norm_out.1", linear)
    assert _modelq(quantize_block_output_linear=True).should_quantize_module("encoder.layers.3.norm_out.1", linear)
    excluded = _modelq(exclude=["encoder.layers.3.norm_out.1"], quantize_block_output_linear=True)
    assert not excluded.should_quantize_module("encoder.layers.3.norm_out.1", linear)
