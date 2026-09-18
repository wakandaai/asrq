"""Canary-Qwen rotation mapping, end to end on a tiny model of the same structure.

Canary-Qwen (NeMo SALM) is a Conformer speech encoder, a projection, and a Qwen3 LLM whose tied token
embedding lives outside the LLM. The tiny model below has exactly those attribute names, so the real
canary_qwen_utils mappings drive learn_rotations and apply_rotations on it. Everything the Whisper and
Parakeet tests do not reach is exercised here: two residual streams with their own R1, Qwen RMSNorm
folding, a rotate-only entry hook, grouped-query attention under R2, and an online Hadamard at a SwiGLU
down projection's own input. The rewrites are exact, so the rotated model must reproduce the original
logits whatever the weights.
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("nemo")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the online Hadamard needs humming's CUDA kernel")

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from asrq.core.linear import ASRQLinear, replace_with_asrq_linear  # noqa: E402
from asrq.transforms.rotation import canary_qwen_utils as cq  # noqa: E402
from asrq.transforms.rotation import utils  # noqa: E402

DEVICE = "cuda"
BLOCK = 16
TOLERANCE = 1e-4


class TinyCanaryQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.perception = nn.Module()
        self.perception.encoder = ConformerEncoder(
            feat_in=80, n_layers=2, d_model=64, n_heads=4, ff_expansion_factor=2, conv_kernel_size=9,
            subsampling_factor=4, subsampling_conv_channels=16,
        )
        self.perception.proj = nn.Linear(64, 64)
        self.llm = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
        ))
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

    def forward(self, features, lengths, ids):
        encoded, _ = self.perception.encoder(audio_signal=features, length=lengths)
        audio = self.perception.proj(encoded.transpose(1, 2))
        text = self.embed_tokens(ids)
        embeds = torch.cat([text[:, :3], audio, text[:, 3:]], dim=1)
        return self.llm(inputs_embeds=embeds, use_cache=False).logits


def _build():
    torch.manual_seed(0)
    model = TinyCanaryQwen().to(DEVICE).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.LayerNorm) or type(module).__name__ == "Qwen3RMSNorm":
                module.weight.normal_(1.0, 0.1)
            if isinstance(module, nn.LayerNorm):
                module.bias.normal_(0.0, 0.1)
    return model


def _batches(count=2):
    torch.manual_seed(1)
    return [{
        "features": torch.randn(2, 80, 120, device=DEVICE),
        "lengths": torch.tensor([120, 100], device=DEVICE),
        "ids": torch.randint(0, 100, (2, 12), device=DEVICE),
    } for _ in range(count)]


def _logits(model, batch):
    logits = model(batch["features"], batch["lengths"], batch["ids"])
    return logits, torch.ones(logits.shape[:2], dtype=torch.bool, device=DEVICE)


@pytest.fixture(scope="module")
def learned(tmp_path_factory):
    path = tmp_path_factory.mktemp("canary") / "rotations.pt"
    model = _build()
    layers = cq.get_canary_qwen_rotation_layers(model)
    utils.learn_rotations(
        model, str(path), hidden_size={cq.ENCODER: 64, cq.LLM: 64}, learning_rate=0.5,
        compute_logits=_logits, train_loader=_batches(), epochs=1, hadamard_block_size=BLOCK,
        tolerance=TOLERANCE, activation_bits=4, activation_group_size=BLOCK,
        groupwise_roles=["attn_out", "fc2"], activation_roles=cq.get_canary_qwen_activation_roles(model),
        **layers,
    )
    return path


def test_the_search_saves_one_r1_per_stream_and_r2s_for_both(learned):
    checkpoint = torch.load(learned, weights_only=False)
    assert {k: tuple(v.shape) for k, v in checkpoint["R1"].items()} == {cq.ENCODER: (64, 64), cq.LLM: (64, 64)}
    assert set(checkpoint["R2s"]) == {
        "perception.encoder.layers.0.self_attn", "perception.encoder.layers.1.self_attn",
        "llm.model.layers.0.self_attn", "llm.model.layers.1.self_attn",
    }
    assert {name: tuple(r.shape) for name, r in checkpoint["R2s"].items() if name.startswith("llm")} == {
        "llm.model.layers.0.self_attn": (16, 16), "llm.model.layers.1.self_attn": (16, 16),
    }
    assert checkpoint["hadamard_sign_seed"] > 0


def test_the_applied_rotation_reproduces_the_original_logits(learned):
    batch = _batches(1)[0]
    reference = _build()
    with torch.no_grad():
        expected, _ = _logits(reference, batch)
    rotated = _build()
    utils.apply_rotations(rotated, str(learned), hadamard_block_size=BLOCK, **cq.get_canary_qwen_rotation_layers(rotated))
    with torch.no_grad():
        got, _ = _logits(rotated, batch)
    assert (got - expected).abs().max() / expected.abs().max() < TOLERANCE

    entries = {n: m for n, m in rotated.named_modules() if isinstance(m, utils.ResidualStreamRotation)}
    assert set(entries) == {
        "perception.encoder.layers.0.input_rotation", "perception.encoder.layers.1.norm_out.input_unrotation",
        "llm.model.layers.0.input_rotation", "llm.model.norm.input_unrotation",
    }
    assert entries["perception.encoder.layers.0.input_rotation"].center
    assert not entries["llm.model.layers.0.input_rotation"].center
    assert all(isinstance(layer.mlp.down_proj.input_hadamard, utils.OnlineHadamard) for layer in rotated.llm.model.layers)
    assert type(rotated.llm.model.layers[0].input_layernorm).__name__ == "_rmsnorm"
    assert rotated.llm.model.layers[0].input_layernorm.upcast


def test_the_output_head_stays_tied_and_unrotated(learned):
    rotated = _build()
    original_head = rotated.llm.lm_head.weight.detach().clone()
    utils.apply_rotations(rotated, str(learned), hadamard_block_size=BLOCK, **cq.get_canary_qwen_rotation_layers(rotated))
    assert rotated.llm.lm_head.weight.data_ptr() == rotated.embed_tokens.weight.data_ptr()
    assert torch.equal(rotated.llm.lm_head.weight, original_head)


def test_a_half_precision_rotated_llm_matches_its_float32_self(learned):
    """The folded Qwen RMSNorms normalize in float32, as Qwen3RMSNorm does."""
    batch = _batches(1)[0]
    rotated = _build()
    utils.apply_rotations(rotated, str(learned), hadamard_block_size=BLOCK, **cq.get_canary_qwen_rotation_layers(rotated))
    with torch.no_grad():
        full, _ = _logits(rotated, batch)
        half, _ = _logits(rotated.to(torch.float16), {**batch, "features": batch["features"].half()})
    assert (half.float() - full).abs().max() / full.abs().max() < 1e-2


def test_humming_takes_the_down_projection_hadamard_into_its_asrq_linear(learned):
    batch = _batches(1)[0]
    rotated = _build()
    utils.apply_rotations(rotated, str(learned), hadamard_block_size=BLOCK, **cq.get_canary_qwen_rotation_layers(rotated))
    rotated = rotated.half()
    with torch.no_grad():
        expected, _ = _logits(rotated, {**batch, "features": batch["features"].half()})
    names = [f"llm.model.layers.{i}.mlp.down_proj" for i in range(2)]
    replaced = replace_with_asrq_linear(rotated, {n: (8, 16, 0) for n in names}, BLOCK, {n: None for n in names})
    assert all(isinstance(m, ASRQLinear) and m.hadamard_block_size == BLOCK for m in replaced.values())
    assert all(m.hadamard_sign_seed > 0 and m.hadamard_signs is None for m in replaced.values())
    with torch.no_grad():
        got, _ = _logits(rotated, {**batch, "features": batch["features"].half()})
    assert (got.float() - expected.float()).abs().max() / expected.float().abs().max() < 5e-2
