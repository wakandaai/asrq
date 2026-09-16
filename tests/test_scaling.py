"""The scaling (SmoothQuant-style) transform: exact on each model's structure, and seen by the quantizers.

scale_model moves a per-channel scale out of each layer's input and into its weight. With arbitrary
scales the rewrite must leave the model's output unchanged: tiny Whisper and Canary-Qwen models with the
real attribute names run the real layer mappings (Canary-Qwen's Conformer encoder is Parakeet's, so it
covers the Parakeet mapping's module kinds too). Where an activation feeds the layer, the reciprocal is an
InputScale after that activation, like the rotation's online Hadamard, so whatever observes the layer's
input -- the activation quantizer and the GPTQ input hook -- sees the scaled tensor.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM, WhisperConfig, WhisperForConditionalGeneration

from asrq.quantizers.activation import attach_activation_quantization
from asrq.transforms.rotation.whisper_utils import get_whisper_activation_roles
from asrq.transforms.scaling.base import InputScale, obtain_scales, scale_model
from asrq.transforms.scaling.whisper_utils import get_whisper_layers_to_scale

TOLERANCE = 1e-9
HEAD_DIM = 16


def _random_scales(model, layers_to_scale, head_dim):
    """A positive scale per entry, sized to its layers' input channels (head_dim for attention outputs)."""
    scales = {}
    for names, _ in layers_to_scale:
        name = names[0]
        width = head_dim if any(k in name for k in ("linear_out", "o_proj", "out_proj")) else model.get_submodule(name).weight.shape[1]
        scales[name] = torch.randn(width, dtype=torch.float64).mul(0.5).exp()
    return scales


def _scale(model, layers_to_scale, head_dim, tmp_path):
    scales = _random_scales(model, layers_to_scale, head_dim)
    path = tmp_path / "scales.pt"
    torch.save(scales, path)
    scale_model(SimpleNamespace(model=model, processor=None), None, layers_to_scale, head_dim, str(path))
    return scales


def _perturb_norms(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) or type(module).__name__ == "Qwen3RMSNorm":
                module.weight.normal_(1.0, 0.1)
            if isinstance(module, nn.LayerNorm):
                module.bias.normal_(0.0, 0.1)


def _scaled_names(layers_to_scale):
    return {name for names, _ in layers_to_scale for name in names}


@pytest.fixture
def whisper():
    torch.manual_seed(0)
    config = WhisperConfig(
        vocab_size=100, num_mel_bins=16, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
        decoder_attention_heads=4, d_model=64, encoder_ffn_dim=128, decoder_ffn_dim=128,
        max_source_positions=20, max_target_positions=16, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        decoder_start_token_id=1,
    )
    model = WhisperForConditionalGeneration(config).double().eval()
    _perturb_norms(model)
    inputs = {"input_features": torch.randn(2, 16, 40, dtype=torch.float64), "decoder_input_ids": torch.randint(3, 100, (2, 8))}
    return model, inputs


@pytest.fixture
def canary_qwen():
    pytest.importorskip("nemo")
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

    torch.manual_seed(0)
    model = nn.Module()
    model.perception = nn.Module()
    model.perception.encoder = ConformerEncoder(
        feat_in=80, n_layers=2, d_model=64, n_heads=4, ff_expansion_factor=2, conv_kernel_size=9,
        subsampling_factor=4, subsampling_conv_channels=16,
    )
    model.llm = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=HEAD_DIM,
    ))
    model = model.double().eval()
    _perturb_norms(model)
    features, lengths, ids = torch.randn(2, 80, 120, dtype=torch.float64), torch.tensor([120, 100]), torch.randint(0, 100, (2, 12))

    def run():
        encoded, _ = model.perception.encoder(audio_signal=features, length=lengths)
        return encoded, model.llm(input_ids=ids, use_cache=False).logits

    return model, run


def test_scaled_whisper_reproduces_the_original_logits(whisper, tmp_path):
    model, inputs = whisper
    with torch.no_grad():
        before = model(**inputs).logits
        _scale(model, get_whisper_layers_to_scale(model), HEAD_DIM, tmp_path)
        after = model(**inputs).logits
    layer = model.model.encoder.layers[0]
    assert type(layer.fc2) is nn.Linear
    assert isinstance(layer.activation_fn, nn.Sequential) and isinstance(layer.activation_fn[-1], InputScale)
    assert (after - before).abs().max() < TOLERANCE


def test_scaled_canary_qwen_reproduces_the_original_outputs(canary_qwen, tmp_path):
    from asrq.transforms.scaling.canary_qwen_utils import get_canary_qwen_layers_to_scale

    model, run = canary_qwen
    with torch.no_grad():
        encoded, logits = run()
        _scale(model, get_canary_qwen_layers_to_scale(model), HEAD_DIM, tmp_path)
        scaled_encoded, scaled_logits = run()
    layer = model.perception.encoder.layers[0]
    assert type(layer.conv.pointwise_conv2) is nn.Conv1d and layer.conv.activation[-1].channels_first
    assert layer.feed_forward1.activation[-1] is not layer.feed_forward2.activation[-1]
    assert type(model.llm.model.layers[0].mlp.act_fn) is not nn.Sequential
    assert (scaled_encoded - encoded).abs().max() < TOLERANCE
    assert (scaled_logits - logits).abs().max() < TOLERANCE


def test_scaling_targets_every_activation_quantized_layer(whisper, canary_qwen):
    from asrq.transforms.rotation.canary_qwen_utils import get_canary_qwen_activation_roles
    from asrq.transforms.scaling.canary_qwen_utils import get_canary_qwen_layers_to_scale

    whisper_model, _ = whisper
    assert _scaled_names(get_whisper_layers_to_scale(whisper_model)) == set(get_whisper_activation_roles(whisper_model))
    canary_model, _ = canary_qwen
    assert _scaled_names(get_canary_qwen_layers_to_scale(canary_model)) == set(get_canary_qwen_activation_roles(canary_model))


def test_quantizer_hooks_see_the_scaled_fc2_input(whisper, tmp_path):
    model, inputs = whisper
    name = "model.encoder.layers.1.fc2"
    unscaled = []
    handle = model.get_submodule(name).register_forward_pre_hook(lambda _m, args: unscaled.append(args[0]))
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    scales = _scale(model, get_whisper_layers_to_scale(model), HEAD_DIM, tmp_path)

    quantizer_inputs, hook_inputs = [], []
    attach_activation_quantization(model, {name: lambda x: quantizer_inputs.append(x) or x})
    model.get_submodule(name).register_forward_hook(lambda _m, args, _out: hook_inputs.append(args[0]))
    with torch.no_grad():
        model(**inputs)
    expected = unscaled[0] / scales[name]
    assert (quantizer_inputs[0] - expected).abs().max() < TOLERANCE
    assert (hook_inputs[0] - expected).abs().max() < TOLERANCE


def test_input_scale_divides_the_channel_axis():
    scale = torch.rand(8) + 0.5
    x = torch.randn(3, 8, 5)
    assert torch.allclose(InputScale(scale, channels_first=True)(x), x / scale.view(1, -1, 1))
    assert torch.allclose(InputScale(scale)(x.transpose(1, 2)), x.transpose(1, 2) / scale)


def test_searched_scales_keep_whisper_exact(whisper, tmp_path):
    model, inputs = whisper
    model = model.float()
    inputs = {**inputs, "input_features": inputs["input_features"].float()}
    layers_to_scale = get_whisper_layers_to_scale(model)
    modelQ = SimpleNamespace(
        model=model, processor=None, calibration_samples=[(None, None)] * 2,
        activation_quantization_roles=lambda: get_whisper_activation_roles(model),
    )
    path = tmp_path / "searched.pt"
    with torch.no_grad():
        before = model(**inputs).logits
        obtain_scales(
            modelQ, layers_to_scale, str(path), lambda _x, _text, _modelQ: model(**inputs), HEAD_DIM,
            wbit=4, abit=8, weight_group_size=32, activation_group_size=32, activation_groupwise_roles=["attn_out", "fc2"],
        )
        scales = torch.load(path)
        assert set(scales) == {names[0] for names, _ in layers_to_scale}
        assert all((scale > 0).all() for scale in scales.values())
        assert scales["model.encoder.layers.0.self_attn.out_proj"].numel() == HEAD_DIM
        scale_model(SimpleNamespace(model=model, processor=None), None, layers_to_scale, HEAD_DIM, str(path))
        after = model(**inputs).logits
    assert (after - before).abs().max() < 1e-3
