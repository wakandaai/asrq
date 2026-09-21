"""End-to-end tests for the Whisper rotation pipeline.

test_rotation_utils.py checks each fold against an explicit reference in isolation. This file
checks the parts that only fail when combined: the norm-to-neighbour mapping, the
residual-stream hooks, and the two entry points that drive them. A fold can be individually
correct and still produce a wrong model if a norm is mapped to the wrong neighbour or a stack
is missing its centering hook, and nothing but a whole-model forward pass catches that.

A small randomly-initialised WhisperForConditionalGeneration is used rather than a checkpoint:
the rewrites are exact reparameterisations, so they hold for any weights, and the norms are
given a non-trivial scale and shift so that folding them is not vacuous.

CUDA is required because the online Hadamard runs through humming's kernel.
"""

import copy
import importlib.util
import sys
import types as _types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROTATION = Path(__file__).resolve().parents[1] / "asrq" / "transforms" / "rotation"

pytest.importorskip("transformers")
pytest.importorskip("humming")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the online Hadamard needs humming's CUDA kernel"
)

from transformers import WhisperConfig, WhisperForConditionalGeneration  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Same stubbing as test_rotation_utils: asrq/__init__.py does not import cleanly yet. Drop this
# once `import asrq` works.
for _pkg in ("asrq", "asrq.transforms", "asrq.transforms.rotation", "asrq.quantizers"):
    _module = sys.modules.setdefault(_pkg, _types.ModuleType(_pkg))
    if not getattr(_module, "__file__", None):
        _module.__path__ = [str(_ROTATION.parents[2].joinpath(*_pkg.split(".")))]
_load("asrq.quantizers.activation", _ROTATION.parents[1] / "quantizers" / "activation.py")
_load("asrq.transforms.rotation.hadamard_utils", _ROTATION / "hadamard_utils.py")
rounding = _load("asrq.quantizers.weight_rounding", _ROTATION.parents[1] / "quantizers" / "weight_rounding.py")
solver = _load("asrq.quantizers.gptq_solver", _ROTATION.parents[1] / "quantizers" / "gptq_solver.py")
cayley = _load("asrq.transforms.rotation.cayley_sgd", _ROTATION / "cayley_sgd.py")
hadamard_search = _load("asrq.transforms.rotation.hadamard_search", _ROTATION / "hadamard_search.py")
activation = sys.modules["asrq.quantizers.activation"]
utils = _load("asrq.transforms.rotation.utils", _ROTATION / "utils.py")

whisper_utils = _load("asrq.transforms.rotation.whisper_utils", _ROTATION / "whisper_utils.py")

DEVICE = "cuda"
DTYPE = torch.float32
D_MODEL = 64
# a power of two, at most 512, dividing the feed-forward width
HADAMARD_BLOCK_SIZE = 64
# activation group size for tests that quantize fc2 group-wise; its online Hadamard has to match
ACTIVATION_GROUP = 16
# float32 through a whole encoder-decoder stack, so this cannot be near fp64 tolerances; a
# broken fold moves the loss by 1e-3 or more, which this still catches by two orders of magnitude
TOLERANCE = 1e-5


def _build_model(seed=0, d_model=D_MODEL):
    torch.manual_seed(seed)
    config = WhisperConfig(
        vocab_size=200, num_mel_bins=8, d_model=d_model, encoder_layers=2, decoder_layers=2,
        encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=4 * d_model,
        decoder_ffn_dim=4 * d_model, max_source_positions=50, max_target_positions=32,
        decoder_start_token_id=1, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = WhisperForConditionalGeneration(config).to(device=DEVICE, dtype=DTYPE).eval()
    # a default LayerNorm has scale 1 and shift 0, which would make folding them a no-op
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight, mean=1.0, std=0.1)
            nn.init.normal_(module.bias, std=0.1)
    return model


def _batches(count=3, seed=1):
    torch.manual_seed(seed)
    batches = []
    for _ in range(count):
        tokens = torch.randint(3, 200, (2, 13))
        batches.append(
            {
                "input_features": torch.randn(2, 8, 100, dtype=DTYPE),
                "decoder_input_ids": tokens[:, :-1],
                "labels": tokens[:, 1:],
            }
        )
    return batches


@pytest.fixture
def model():
    return _build_model()


@pytest.fixture
def batches():
    return _batches()


def _cross_entropy(model, batch):
    """Teacher-forced cross-entropy from whisper_logits_fn, for tests that need a scalar."""
    logits, mask = whisper_utils.whisper_logits_fn(model, batch)
    return F.cross_entropy(logits[mask], batch["labels"].to(DEVICE)[mask])


def _logits(model, batch):
    with torch.no_grad():
        return model(**{k: v.to(DEVICE) for k, v in batch.items()}).logits


def test_norm_folding_with_centering_preserves_the_output(model, batches):
    """The fold is only equivalent once the hooks supply the centering it cannot fold back."""
    before = _logits(model, batches[0])

    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    whisper_utils.attach_whisper_rotation_hooks(model, torch.eye(D_MODEL, device=DEVICE))

    after = _logits(model, batches[0])
    assert (after - before).abs().max() / before.abs().max() < TOLERANCE


def test_norm_folding_alone_is_not_equivalent(model, batches):
    """Without the centering hooks the first norm of each stack loses its mean subtraction.

    Guards the claim the previous test rests on: if this ever passed, the hooks would be dead
    code and an empty previous list would be silently hiding a mapping error.
    """
    before = _logits(model, batches[0])
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    after = _logits(model, batches[0])
    assert (after - before).abs().max() / before.abs().max() > 1e-3


def test_learn_rotations_verifies_and_saves(model, batches, tmp_path):
    """The whole search runs, both internal checks pass, and the rotations stay orthogonal."""
    path = tmp_path / "rotations.pt"
    R1, R2s = utils.learn_rotations(
        model, str(path),
        hidden_size=D_MODEL,
        learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn,
        train_loader=batches,
        epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE,
        tolerance=TOLERANCE,
        **whisper_utils.get_whisper_rotation_layers(model),
    )

    assert R1.shape == (D_MODEL, D_MODEL)
    assert len(R2s) == len(whisper_utils.get_whisper_layers_to_rotate(model))

    identity = torch.eye(D_MODEL, device=R1.device, dtype=R1.dtype)
    assert (R1.data.T @ R1.data - identity).abs().max() < 1e-5

    saved = torch.load(path)
    assert saved["hadamard_block_size"] == HADAMARD_BLOCK_SIZE
    assert torch.equal(saved["R1"], R1.data.cpu())


def test_apply_rotations_reproduces_the_unrotated_output(batches, tmp_path):
    """A model with the rotation folded into its weights computes the original function."""
    path = tmp_path / "rotations.pt"
    searched = _build_model()
    utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
        **whisper_utils.get_whisper_rotation_layers(searched),
    )

    fresh = _build_model()
    before = _logits(fresh, batches[0])
    handles = utils.apply_rotations(
        fresh, str(path),
        hadamard_block_size=HADAMARD_BLOCK_SIZE,
        **whisper_utils.get_whisper_rotation_layers(fresh),
    )
    after = _logits(fresh, batches[0])

    assert len(handles) == 3
    assert (after - before).abs().max() / before.abs().max() < TOLERANCE
    assert torch.equal(after.argmax(-1), before.argmax(-1))


def test_apply_rotations_rejects_a_mismatched_hadamard_block_size(batches, tmp_path):
    """The absorbed and the online Hadamard have to be the same matrix."""
    path = tmp_path / "rotations.pt"
    searched = _build_model()
    utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
        **whisper_utils.get_whisper_rotation_layers(searched),
    )

    fresh = _build_model()
    with pytest.raises(ValueError, match="hadamard_block_size"):
        utils.apply_rotations(
            fresh, str(path),
            hadamard_block_size=HADAMARD_BLOCK_SIZE // 2,
            **whisper_utils.get_whisper_rotation_layers(fresh),
        )


def test_the_verification_catches_a_wrong_rotation_fold(model, batches, tmp_path, monkeypatch):
    """A V projection folded with O's formula is caught, not silently accepted.

    This is the bug the fold functions actually had: the two formulas differ only in which
    operand goes on which side, so it raises nothing on a square layer and merely returns the
    wrong answer.
    """
    monkeypatch.setattr(utils, "patch_v_with_rotation", utils.patch_o_with_rotation)
    with pytest.raises(RuntimeError, match="rotation patching changed the model's output"):
        utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
            **whisper_utils.get_whisper_rotation_layers(model),
        )


def test_the_verification_catches_a_dropped_next_layer(model, batches, tmp_path):
    """A norm mapped to fewer next layers than read it leaves that layer unshifted."""
    structure = whisper_utils.get_whisper_rotation_layers(model)
    structure["norm_layers"] = [
        (norm, previous, nxt[:-1] if len(nxt) > 1 else nxt)
        for norm, previous, nxt in structure["norm_layers"]
    ]
    with pytest.raises(RuntimeError, match="norm folding changed the model's output"):
        utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE, **structure,
        )


def test_the_loss_is_invariant_to_the_rotation(model, batches):
    """With no quantizer in the loss, the objective does not depend on R1 or R2 at all.

    The rewrite is an exact reparameterisation, which is what every other test here asserts --
    and that is exactly why the search needs activation quantization: without it the loss is
    flat along the manifold. test_activation_quantization_makes_the_loss_depend_on_the_rotation
    is the counterpart with quantization switched on.
    """
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    groups = whisper_utils.get_whisper_layers_to_rotate(model)

    # one mutable rotation set, so it can be swapped without re-patching the layers
    R1 = nn.Parameter(torch.eye(D_MODEL, device=DEVICE))
    R2s = {
        name: nn.Parameter(torch.eye(head_dim, device=DEVICE))
        for name, head_dim, _ in groups
    }
    whisper_utils.attach_whisper_rotation_hooks(model, R1)
    utils.patch_rotations(model, groups, R1, R2s, {}, HADAMARD_BLOCK_SIZE)

    losses = []
    for seed in range(4):
        torch.manual_seed(seed)
        with torch.no_grad():
            R1.copy_(torch.linalg.qr(torch.randn(D_MODEL, D_MODEL, device=DEVICE))[0])
            for name, head_dim, _ in groups:
                R2s[name].copy_(
                    torch.linalg.qr(torch.randn(head_dim, head_dim, device=DEVICE))[0]
                )
            losses.append(float(_cross_entropy(model, batches[0])))

    spread = max(losses) - min(losses)
    assert spread / abs(losses[0]) < TOLERANCE, (
        f"the loss varied by {spread:.2e} across rotations, so something in the rewrite is "
        f"not rotation-equivariant"
    )


def test_a_head_dimension_r2_is_used_per_attention_module(model):
    """R2 is head_dim wide, not d_model: the mapping has to report the head dimension."""
    groups = whisper_utils.get_whisper_layers_to_rotate(model)
    expected = D_MODEL // model.config.encoder_attention_heads
    assert {head_dim for _, head_dim, _ in groups} == {expected}
    assert expected != D_MODEL, "a single-head model would make this test vacuous"


def test_deepcopy_of_a_patched_model_keeps_its_rotation(model, batches, tmp_path):
    """The patched forwards are bound methods, so a copy uses the copy's weights.

    A bare closure would shadow forward on the instance just as well but keep pointing at the
    original layer, so the copy would silently ignore its own weights.
    """
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    groups = whisper_utils.get_whisper_layers_to_rotate(model)
    R1 = nn.Parameter(torch.eye(D_MODEL, device=DEVICE))
    R2s = {
        name: nn.Parameter(torch.eye(head_dim, device=DEVICE))
        for name, head_dim, _ in groups
    }
    whisper_utils.attach_whisper_rotation_hooks(model, R1)
    utils.patch_rotations(model, groups, R1, R2s, {}, HADAMARD_BLOCK_SIZE)

    duplicate = copy.deepcopy(model)
    with torch.no_grad():
        duplicate.model.encoder.layers[0].self_attn.v_proj.weight.mul_(2.0)

    assert not torch.allclose(_logits(duplicate, batches[0]), _logits(model, batches[0]))


@pytest.mark.parametrize("channels_first", [False, True])
def test_online_hadamard_applies_the_matrix_absorbed_into_fc2(channels_first):
    """OnlineHadamard is X @ H for the same H online_hadamard_matrix materialises."""
    torch.manual_seed(0)
    width = 4 * HADAMARD_BLOCK_SIZE
    x = torch.randn(3, 20, width, device=DEVICE, dtype=DTYPE)
    H = utils.online_hadamard_matrix(width, HADAMARD_BLOCK_SIZE, DEVICE)
    module = utils.OnlineHadamard(HADAMARD_BLOCK_SIZE, channels_first=channels_first)

    with torch.no_grad():
        if channels_first:
            got = module(x.transpose(1, 2)).transpose(1, 2)
        else:
            got = module(x)
    assert (got - x @ H).abs().max() < 1e-5


def _rotated_whisper(batches, tmp_path):
    searched = _build_model()
    path = tmp_path / "rotations.pt"
    utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
        **whisper_utils.get_whisper_rotation_layers(searched),
    )
    fresh = _build_model()
    utils.apply_rotations(
        fresh, str(path),
        hadamard_block_size=HADAMARD_BLOCK_SIZE,
        **whisper_utils.get_whisper_rotation_layers(fresh),
    )
    return fresh


def test_apply_rotations_leaves_fc2_unpatched_behind_an_online_hadamard(batches, tmp_path):
    """Every FC2 is an ordinary Linear, and its activation ends in the online Hadamard."""
    model = _rotated_whisper(batches, tmp_path)
    for activation_name, fc2_name in whisper_utils.get_whisper_online_hadamard_layers(model):
        fc2 = utils.get_module(model, fc2_name)
        activation = utils.get_module(model, activation_name)
        assert type(fc2) is nn.Linear
        assert "forward" not in vars(fc2), f"{fc2_name} still has an instance-level forward"
        assert isinstance(activation, nn.Sequential)
        assert isinstance(activation[-1], utils.OnlineHadamard)


def test_a_forward_hook_on_fc2_sees_the_hadamard_basis(batches, tmp_path):
    """The input a Hessian-collecting hook records is the tensor the folded weight multiplies.

    The weight holds R1.T @ W @ H, so the input it multiplies is X @ H. This is the reason the
    Hadamard is a module ahead of FC2 rather than a patch of FC2's forward.
    """
    model = _rotated_whisper(batches, tmp_path)
    activation_name, fc2_name = whisper_utils.get_whisper_online_hadamard_layers(model)[0]
    activation = utils.get_module(model, activation_name)
    fc2 = utils.get_module(model, fc2_name)

    seen = {}
    activation[0].register_forward_hook(lambda m, a, out: seen.__setitem__("act", out))
    fc2.register_forward_hook(lambda m, a, out: seen.__setitem__("fc2_in", a[0]))
    _logits(model, batches[0])

    H = utils.online_hadamard_matrix(fc2.in_features, HADAMARD_BLOCK_SIZE, DEVICE)
    D = torch.diag(activation[-1].signs)
    assert (seen["fc2_in"] - seen["act"] @ D @ H).abs().max() < 1e-4
    assert (seen["fc2_in"] - seen["act"]).abs().max() > 1e-2


def test_a_type_4_layer_without_an_online_hadamard_is_rejected(model, batches, tmp_path):
    """A FC2 absorbing H with nothing applying it online would be silently wrong."""
    structure = whisper_utils.get_whisper_rotation_layers(model)
    structure["online_hadamard_layers"] = structure["online_hadamard_layers"][1:]
    with pytest.raises(ValueError, match="online_hadamard_layers"):
        utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE, **structure,
        )


def test_inserting_the_online_hadamard_twice_is_a_no_op(model):
    activation_name, fc2_name = whisper_utils.get_whisper_online_hadamard_layers(model)[0]
    first = utils.insert_online_hadamard_after_activation(
        model, activation_name, fc2_name, HADAMARD_BLOCK_SIZE
    )
    second = utils.insert_online_hadamard_after_activation(
        model, activation_name, fc2_name, HADAMARD_BLOCK_SIZE
    )
    activation = utils.get_module(model, activation_name)
    assert first == second
    assert len(activation) == 2 and not isinstance(activation[0], nn.Sequential)


def _parakeet():
    pytest.importorskip("nemo")
    from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder

    parakeet_utils = _load(
        "asrq.transforms.rotation.parakeet_ctc_utils", _ROTATION / "parakeet_ctc_utils.py"
    )

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ConformerEncoder(
                feat_in=80, n_layers=3, d_model=D_MODEL, n_heads=4, ff_expansion_factor=2,
                conv_kernel_size=31, subsampling_factor=4, subsampling_conv_channels=16,
            )

    def build():
        torch.manual_seed(0)
        model = Stub().to(device=DEVICE, dtype=DTYPE).eval()
        for module in model.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.normal_(module.weight, mean=1.0, std=0.1)
                nn.init.normal_(module.bias, std=0.1)
        return model

    return parakeet_utils, build


def test_parakeet_apply_rotations_with_a_shared_feed_forward_activation(tmp_path):
    """Parakeet's feed-forwards share one Swish; each still gets its own online Hadamard."""
    parakeet_utils, build = _parakeet()
    torch.manual_seed(1)
    batches = [
        {"x": torch.randn(2, 80, 200, device=DEVICE, dtype=DTYPE),
         "length": torch.tensor([200, 180], device=DEVICE)}
        for _ in range(2)
    ]

    def compute_logits(model, batch):
        """The encoder output stands in for CTC log-probabilities; any per-frame vector works."""
        out, lengths = model.encoder(audio_signal=batch["x"], length=batch["length"])
        frames = torch.arange(out.shape[-1], device=DEVICE)
        return out.transpose(1, 2), frames.unsqueeze(0) < lengths.unsqueeze(1)

    def encode(model):
        with torch.no_grad():
            return model.encoder(audio_signal=batches[0]["x"], length=batches[0]["length"])[0]

    reference = build()
    shared = {id(layer.feed_forward1.activation) for layer in reference.encoder.layers}
    assert len(shared) == 1, "the premise of this test is a shared activation"
    expected = encode(reference)

    path = tmp_path / "rotations.pt"
    searched = build()
    utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=1e-3, compute_logits=compute_logits,
        train_loader=batches, epochs=1, hadamard_block_size=HADAMARD_BLOCK_SIZE,
        tolerance=TOLERANCE, **parakeet_utils.get_parakeet_rotation_layers(searched),
    )

    fresh = build()
    utils.apply_rotations(
        fresh, str(path),
        hadamard_block_size=HADAMARD_BLOCK_SIZE,
        **parakeet_utils.get_parakeet_rotation_layers(fresh),
    )

    got = encode(fresh)
    assert (got - expected).abs().max() / expected.abs().max() < TOLERANCE

    hadamards = []
    for layer in fresh.encoder.layers:
        for ff in (layer.feed_forward1, layer.feed_forward2):
            assert isinstance(ff.activation, nn.Sequential)
            assert not isinstance(ff.activation[0], nn.Sequential)
            hadamards.append(ff.activation[1])
        assert layer.conv.activation[1].channels_first
    assert len({id(h) for h in hadamards}) == len(hadamards)


@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("group_size", [16, -1])
def test_fake_quantize_activations_matches_a_direct_reference(symmetric, group_size):
    """Group-wise quantize-dequantize, written the slow obvious way, one group at a time."""
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64, device=DEVICE) * torch.linspace(0.1, 10, 64, device=DEVICE)
    bits = 4
    got = utils.fake_quantize_activations(x, bits, group_size, symmetric)

    group = 64 if group_size == -1 else group_size
    expected = torch.empty_like(x)
    for index in torch.cartesian_prod(torch.arange(2), torch.arange(5)):
        b, t = index.tolist()
        for start in range(0, 64, group):
            g = x[b, t, start:start + group]
            if symmetric:
                q_max = 2 ** (bits - 1) - 1
                scale = max(float(g.abs().max()) / q_max, 1e-8)
                q = torch.clamp(torch.round(g / scale), -q_max - 1, q_max) * scale
            else:
                q_max = 2**bits - 1
                scale = max(float(g.max() - g.min()) / q_max, 1e-8)
                zero = round(-float(g.min()) / scale)
                q = (torch.clamp(torch.round(g / scale) + zero, 0, q_max) - zero) * scale
            expected[b, t, start:start + group] = q
    assert (got - expected).abs().max() < 1e-5
    assert len(torch.unique(got[0, 0, :group])) <= 2**bits


def test_fake_quantize_activations_passes_the_gradient_straight_through():
    x = torch.randn(3, 32, device=DEVICE, requires_grad=True)
    upstream = torch.randn(3, 32, device=DEVICE)
    (utils.fake_quantize_activations(x, 4, 16, True) * upstream).sum().backward()
    assert torch.equal(x.grad, upstream)


def test_fake_quantize_activations_rejects_a_group_size_that_does_not_divide():
    with pytest.raises(ValueError, match="not divisible"):
        utils.fake_quantize_activations(torch.randn(2, 60, device=DEVICE), 8, 16, True)


def test_the_conv_patch_quantizes_the_channel_axis(monkeypatch):
    """A pointwise Conv1d reads (batch, channels, time); groups must run along channels.

    cuDNN convolutions default to TF32 while cuBLAS matmuls do not, which alone puts the conv
    and the Linear 7e-4 apart; it is switched off so the comparison can be exact. Quantizing the
    time axis instead is checked to be far outside that.
    """
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(0)
    conv = nn.Conv1d(64, 32, 1).to(DEVICE)
    linear = nn.Linear(64, 32).to(DEVICE)
    with torch.no_grad():
        linear.weight.copy_(conv.weight.squeeze(-1))
        linear.bias.copy_(conv.bias)
    R1 = torch.linalg.qr(torch.randn(64, 64, device=DEVICE))[0]
    quantize = utils.make_activation_quantizer(4, 16, False)
    utils.patch_q_k_fc1_with_rotation(conv, R1, quantize)
    utils.patch_q_k_fc1_with_rotation(linear, R1, quantize)

    x = torch.randn(2, 64, 10, device=DEVICE)
    with torch.no_grad():
        from_conv = conv(x).transpose(1, 2)
        from_linear = linear(x.transpose(1, 2))
        across_time = F.linear(
            utils.fake_quantize_activations(x, 4, 10, False).transpose(1, 2),
            conv.weight.squeeze(-1) @ R1,
            conv.bias,
        )
    assert (from_conv - from_linear).abs().max() < 1e-5
    assert (from_conv - across_time).abs().max() > 1e-2


def test_activation_quantization_makes_the_loss_depend_on_the_rotation(model, batches):
    """With quantized inputs the loss varies across rotations and has a tangent gradient.

    The mirror of test_the_loss_is_invariant_to_the_rotation: the same patched model, the same
    rotations, but with the inputs of the rotated layers fake-quantized. This is the property
    the search relies on.
    """
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    groups = whisper_utils.get_whisper_layers_to_rotate(model)
    R1 = nn.Parameter(torch.eye(D_MODEL, device=DEVICE))
    R2s = {
        name: nn.Parameter(torch.eye(head_dim, device=DEVICE))
        for name, head_dim, _ in groups
    }
    whisper_utils.attach_whisper_rotation_hooks(model, R1)
    utils.patch_rotations(
        model, groups, R1, R2s, {}, HADAMARD_BLOCK_SIZE,
        {name: utils.make_activation_quantizer(4, 16, True)
         for name in whisper_utils.get_whisper_activation_roles(model)},
    )

    losses = []
    for seed in range(4):
        torch.manual_seed(seed)
        with torch.no_grad():
            R1.copy_(torch.linalg.qr(torch.randn(D_MODEL, D_MODEL, device=DEVICE))[0])
            for name, head_dim, _ in groups:
                R2s[name].copy_(
                    torch.linalg.qr(torch.randn(head_dim, head_dim, device=DEVICE))[0]
                )
            losses.append(float(_cross_entropy(model, batches[0])))
    assert (max(losses) - min(losses)) / abs(losses[0]) > 100 * TOLERANCE

    _cross_entropy(model, batches[0]).backward()
    G, Q = R1.grad.double(), R1.data.double()
    tangent = G @ Q.T - Q @ G.T
    assert tangent.abs().max() / G.abs().max() > 1e-3


def test_learn_rotations_with_activation_quantization_moves_the_rotation(
    model, batches, tmp_path
):
    """The search changes R1, stays on the manifold, and records its quantization.

    A randomly initialised model's output distribution is close to uniform, so its KL to full
    precision is tiny at 4 bits (2e-4) and so is the gradient. 2-bit activations and a large
    learning rate give this toy model a signal comparable to a trained one's at 4 bits.
    """
    init = utils.random_hadamard_matrix
    seen_inits = []

    def recording_init(size, device, seed=None):
        matrix = init(size, device, seed)
        seen_inits.append(matrix.clone())
        return matrix

    utils.random_hadamard_matrix = recording_init
    try:
        R1, _ = utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=20.0,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=_batches(6), epochs=2,
            hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
            activation_bits=2, activation_group_size=ACTIVATION_GROUP, activation_symmetric=True,
            activation_roles=whisper_utils.get_whisper_activation_roles(model),
            **whisper_utils.get_whisper_rotation_layers(model),
        )
    finally:
        utils.random_hadamard_matrix = init

    start = seen_inits[0].to(device=R1.device, dtype=R1.dtype)
    identity = torch.eye(D_MODEL, device=R1.device)
    assert (R1.data - start).abs().max() > 1e-3
    assert (R1.data.T @ R1.data - identity).abs().max() < 1e-4

    saved = torch.load(tmp_path / "rotations.pt")
    assert saved["activation_quantization"] == {
        "bits": 2, "group_size": 16, "symmetric": True, "groupwise_roles": None,
    }
    assert saved["objective"] == "kl"
    assert saved["activation_quantization"]["groupwise_roles"] is None


def test_activation_quantizer_switches_off_inside_disabled():
    x = torch.randn(2, 5, 32, device=DEVICE)
    quantizer = utils.ActivationQuantizer(4, 16, True)
    quantized = quantizer(x)
    assert not torch.equal(quantized, x)
    with quantizer.disabled():
        assert torch.equal(quantizer(x), x)
    assert torch.equal(quantizer(x), quantized)


def test_masked_kl_divergence_matches_a_direct_reference():
    torch.manual_seed(0)
    student = torch.randn(2, 6, 50, device=DEVICE, dtype=torch.float16)
    teacher = torch.randn(2, 6, 50, device=DEVICE, dtype=torch.float16)
    mask = torch.rand(2, 6, device=DEVICE) > 0.4

    p = F.softmax(teacher.double(), -1)[mask]
    expected = (p * (p.log() - F.log_softmax(student.double(), -1)[mask])).sum(-1).mean()
    got = utils.masked_kl_divergence(student, teacher, mask)
    assert abs(float(got) - float(expected)) < 1e-4


def test_the_logit_check_is_not_vacuous_for_a_distillation_objective(model, batches):
    """The check compares against the original logits, so a broken model cannot pass it.

    A KL divergence between the model and itself is zero whatever the model computes, which is
    why the verification compares logits rather than the search objective.
    """
    with torch.no_grad():
        reference, _ = whisper_utils.whisper_logits_fn(model, batches[0])
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    with pytest.raises(RuntimeError, match="norm folding changed the model's output"):
        utils.check_logits_unchanged(
            model, whisper_utils.whisper_logits_fn, batches[0], reference, "norm folding",
            TOLERANCE,
        )


def test_the_search_objective_starts_at_zero_without_quantization(model, batches):
    """With quantization off the teacher and student coincide, so the KL is exactly zero."""
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    groups = whisper_utils.get_whisper_layers_to_rotate(model)
    R1 = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL, device=DEVICE))[0]
    R2s = {
        name: torch.linalg.qr(torch.randn(head_dim, head_dim, device=DEVICE))[0]
        for name, head_dim, _ in groups
    }
    quantizer = utils.ActivationQuantizer(4, 16, True)
    whisper_utils.attach_whisper_rotation_hooks(model, R1)
    utils.patch_rotations(
        model, groups, R1, R2s, {}, HADAMARD_BLOCK_SIZE,
        {name: quantizer for name in whisper_utils.get_whisper_activation_roles(model)},
    )

    with torch.no_grad():
        with quantizer.disabled():
            teacher, mask = whisper_utils.whisper_logits_fn(model, batches[0])
            same = utils.masked_kl_divergence(
                whisper_utils.whisper_logits_fn(model, batches[0])[0], teacher, mask
            )
        quantized = utils.masked_kl_divergence(
            whisper_utils.whisper_logits_fn(model, batches[0])[0], teacher, mask
        )
    assert float(same) < 1e-6
    assert float(quantized) > 1e-4


@pytest.fixture(scope="module")
def large_v3_tokenizer():
    from transformers import WhisperTokenizer

    try:
        return WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    except OSError:
        pytest.skip("the whisper-large-v3 tokenizer is not available offline")


def test_decoder_targets_start_from_the_prefix_generate_forces(large_v3_tokenizer):
    """The decoder input is exactly generate()'s forced prefix followed by the text."""
    tokenizer = copy.deepcopy(large_v3_tokenizer)
    decoder_input_ids, labels = whisper_utils.build_whisper_decoder_targets(
        tokenizer, "hello world"
    )

    tokenizer.set_prefix_tokens(language="english", task="transcribe", predict_timestamps=False)
    forced = tokenizer.prefix_tokens
    text = tokenizer.encode("hello world", add_special_tokens=False)

    assert decoder_input_ids.tolist() == forced + text
    assert labels.tolist() == [-100] * (len(forced) - 1) + text + [tokenizer.eos_token_id]
    assert decoder_input_ids.count_nonzero() == len(decoder_input_ids)


def test_decoder_targets_never_double_the_start_token(large_v3_tokenizer):
    start = large_v3_tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    decoder_input_ids, labels = whisper_utils.build_whisper_decoder_targets(
        large_v3_tokenizer, "hello world"
    )
    assert decoder_input_ids.tolist().count(start) == 1
    assert start not in labels.tolist()
    assert -100 not in decoder_input_ids.tolist()


def test_collate_pads_decoder_inputs_and_masks_padded_labels(large_v3_tokenizer):
    pad = large_v3_tokenizer.pad_token_id
    items = [
        (torch.zeros(8, 100), *whisper_utils.build_whisper_decoder_targets(large_v3_tokenizer, t))
        for t in ("hello", "hello world, this is longer")
    ]
    batch = whisper_utils.whisper_collate_fn(items, pad_token_id=pad)
    short_len = items[0][2].size(0)

    assert batch["decoder_input_ids"].shape == batch["labels"].shape
    assert (batch["labels"][0, short_len:] == -100).all()
    assert (batch["decoder_input_ids"][0, short_len:] == pad).all()
    assert torch.equal(batch["labels"][0, :short_len], items[0][2])


def test_the_logits_mask_covers_only_the_text_and_end_of_text(model):
    """whisper_logits_fn counts exactly the labelled positions, not the prefix or padding."""
    torch.manual_seed(0)
    tokens = torch.randint(3, 200, (2, 13))
    labels = tokens[:, 1:].clone()
    labels[:, :3] = -100
    labels[1, 9:] = -100
    batch = {
        "input_features": torch.randn(2, 8, 100, dtype=DTYPE),
        "decoder_input_ids": tokens[:, :-1],
        "labels": labels,
    }
    with torch.no_grad():
        logits, mask = whisper_utils.whisper_logits_fn(model, batch)
        direct = model(
            input_features=batch["input_features"].to(DEVICE),
            decoder_input_ids=batch["decoder_input_ids"].to(DEVICE),
        ).logits
    assert torch.equal(mask.cpu(), labels != -100)
    assert torch.equal(logits, direct)


def test_the_ce_objective_trains_on_the_training_loss_not_the_kl(
    model, batches, tmp_path, monkeypatch
):
    """objective="ce" moves the rotation through compute_loss and never builds a teacher."""

    def no_teacher(*args, **kwargs):
        raise AssertionError("the ce objective must not compute a KL divergence")

    monkeypatch.setattr(utils, "masked_kl_divergence", no_teacher)
    losses = []

    def recording_loss(student, batch):
        loss = whisper_utils.whisper_loss_fn(student, batch)
        losses.append(float(loss.detach()))
        return loss

    R1, _ = utils.learn_rotations(
        model, str(tmp_path / "rotations.pt"),
        hidden_size=D_MODEL, learning_rate=0.5,
        compute_logits=whisper_utils.whisper_logits_fn, compute_loss=recording_loss,
        objective="ce", train_loader=_batches(6), epochs=2,
        hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
        activation_bits=4, activation_group_size=ACTIVATION_GROUP, activation_symmetric=True,
        activation_roles=whisper_utils.get_whisper_activation_roles(model),
        **whisper_utils.get_whisper_rotation_layers(model),
    )

    identity = torch.eye(D_MODEL, device=R1.device)
    assert len(losses) == 2 + 6 * 2
    assert (R1.data.T @ R1.data - identity).abs().max() < 1e-4
    assert torch.load(tmp_path / "rotations.pt")["objective"] == "ce"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"objective": "mse"}, "objective must be 'kl' or 'ce'"),
        ({"objective": "ce"}, "objective='ce' needs compute_loss"),
    ],
)
def test_a_bad_objective_is_rejected_before_the_model_is_touched(
    model, batches, tmp_path, overrides, message
):
    with pytest.raises(ValueError, match=message):
        utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
            **overrides, **whisper_utils.get_whisper_rotation_layers(model),
        )
    assert any(isinstance(m, nn.LayerNorm) for m in model.modules())


def test_whisper_loss_fn_is_cross_entropy_over_the_counted_positions(model, batches):
    batch = batches[0]
    with torch.no_grad():
        loss = whisper_utils.whisper_loss_fn(model, batch)
        logits, mask = whisper_utils.whisper_logits_fn(model, batch)
    labels = batch["labels"].to(DEVICE)
    log_probs = F.log_softmax(logits.double(), -1)
    expected = -log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)[mask].mean()
    assert abs(float(loss) - float(expected)) < 1e-5


def test_the_equivalence_checks_run_without_tf32_and_restore_it(
    model, batches, tmp_path, monkeypatch
):
    """TF32 rounding alone moved real Parakeet's logits by 1.8e-3, as much as a broken fold."""
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    seen = []
    check = utils.check_logits_unchanged

    def recording_check(*args, **kwargs):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return check(*args, **kwargs)

    monkeypatch.setattr(utils, "check_logits_unchanged", recording_check)
    utils.learn_rotations(
        model, str(tmp_path / "rotations.pt"),
        hidden_size=D_MODEL, learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
        **whisper_utils.get_whisper_rotation_layers(model),
    )
    assert seen == [(False, False), (False, False)]
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32


def test_quantizers_follow_each_layer_role():
    roles = {"a.q": "q", "a.k": "k", "a.out": "attn_out", "a.fc1": "fc1", "a.fc2": "fc2",
             "a.unknown": None}
    quantizers = utils.build_activation_quantizers(roles, 4, 128, True, ["attn_out", "fc2"])
    assert {n: q.group_size for n, q in quantizers.items()} == {
        "a.q": -1, "a.k": -1, "a.out": 128, "a.fc1": -1, "a.fc2": 128, "a.unknown": -1,
    }
    assert quantizers["a.q"] is quantizers["a.fc1"] and quantizers["a.out"] is quantizers["a.fc2"]

    everything = utils.build_activation_quantizers(roles, 4, 128, True, None)
    assert everything["a.q"].group_size == 128 and everything["a.unknown"].group_size == -1

    nothing = utils.build_activation_quantizers(roles, 4, 128, True, [])
    assert {q.group_size for q in nothing.values()} == {-1}

    with pytest.raises(ValueError, match="unknown activation roles"):
        utils.build_activation_quantizers(roles, 4, 128, True, ["mlp"])


def test_quantizers_disabled_switches_off_every_quantizer():
    roles = {"q": "q", "fc2": "fc2"}
    quantizers = utils.build_activation_quantizers(roles, 4, 16, True, ["fc2"])
    x = torch.randn(2, 3, 32, device=DEVICE)
    with utils.quantizers_disabled(quantizers):
        assert all(torch.equal(q(x), x) for q in quantizers.values())
    assert all(q.enabled for q in quantizers.values())


def test_whisper_roles_cover_every_attention_projection_and_both_feed_forward_layers(model):
    roles = whisper_utils.get_whisper_activation_roles(model)
    modules = dict(model.named_modules())
    assert all(name in modules for name in roles)
    n_enc, n_dec = model.config.encoder_layers, model.config.decoder_layers
    counts = {role: list(roles.values()).count(role) for role in set(roles.values())}
    assert counts == {"q": n_enc + 2 * n_dec, "k": n_enc + 2 * n_dec, "v": n_enc + 2 * n_dec,
                      "attn_out": n_enc + 2 * n_dec, "fc1": n_enc + n_dec, "fc2": n_enc + n_dec}


def test_the_search_quantizes_each_layer_by_role(model, batches):
    """Every rotated layer is quantized by the quantizer named for it, fc2 behind its Hadamard."""
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    groups = whisper_utils.get_whisper_layers_to_rotate(model)
    R1 = torch.eye(D_MODEL, device=DEVICE)
    R2s = {name: torch.eye(head_dim, device=DEVICE) for name, head_dim, _ in groups}
    whisper_utils.attach_whisper_rotation_hooks(model, R1)
    roles = whisper_utils.get_whisper_activation_roles(model)

    seen = {}

    def recording(role):
        def quantize(x):
            seen.setdefault(role, set()).add(x.shape[-1])
            return x
        return quantize

    utils.patch_rotations(model, groups, R1, R2s, {}, HADAMARD_BLOCK_SIZE,
                          {name: recording(role) for name, role in roles.items()})
    _logits(model, batches[0])
    assert seen == {"q": {D_MODEL}, "k": {D_MODEL}, "v": {D_MODEL}, "attn_out": {D_MODEL},
                    "fc1": {D_MODEL}, "fc2": {4 * D_MODEL}}


def test_the_search_records_which_roles_were_groupwise(model, batches, tmp_path):
    utils.learn_rotations(
        model, str(tmp_path / "rotations.pt"),
        hidden_size=D_MODEL, learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
        activation_bits=4, activation_group_size=ACTIVATION_GROUP,
        groupwise_roles=["fc2", "attn_out"],
        activation_roles=whisper_utils.get_whisper_activation_roles(model),
        **whisper_utils.get_whisper_rotation_layers(model),
    )
    saved = torch.load(tmp_path / "rotations.pt")["activation_quantization"]
    assert saved["groupwise_roles"] == ["attn_out", "fc2"] and saved["group_size"] == 16


def test_activation_quantization_without_roles_is_rejected(model, batches, tmp_path):
    with pytest.raises(ValueError, match="needs activation_roles"):
        utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, activation_bits=4,
            **whisper_utils.get_whisper_rotation_layers(model),
        )


@pytest.mark.parametrize("groupwise_roles", [None, [], ["attn_out", "fc2"]])
def test_evaluation_quantizes_the_same_tensors_as_the_search(batches, tmp_path, groupwise_roles):
    """Evaluation quantizes the same tensor at every layer as the search did.

    The search quantizes inside patched forwards with a dense online Hadamard; evaluation folds
    the rotation, runs humming's Hadamard as a module and quantizes through pre-hooks. The
    capturing quantizers record their input and pass it through unquantized. With quantization
    active the paths still agree to float32 rounding, but a value sitting exactly on a
    quantization step can round the other way and shift every later layer's input; passing
    through isolates what is being tested -- where each quantizer sits and which basis it sees --
    from that. Both paths build their quantizers with the same function and settings.
    """
    path = tmp_path / "rotations.pt"
    searched = _build_model()
    roles = whisper_utils.get_whisper_activation_roles(searched)
    R1, R2s = utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=0.5,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
        activation_bits=4, activation_group_size=ACTIVATION_GROUP, groupwise_roles=groupwise_roles,
        activation_roles=roles, **whisper_utils.get_whisper_rotation_layers(searched),
    )
    def capturing(store, name):
        def quantize(x):
            store[name] = x.detach().float().clone()
            return x
        return quantize

    search_inputs, eval_inputs = {}, {}
    utils.patch_rotations(
        searched, whisper_utils.get_whisper_layers_to_rotate(searched), R1, R2s, {},
        ACTIVATION_GROUP, {name: capturing(search_inputs, name) for name in roles},
        {4 * D_MODEL: utils.seeded_hadamard_signs(torch.load(path)["hadamard_sign_seed"], 4 * D_MODEL, DEVICE)},
    )
    _logits(searched, batches[0])

    evaluated = _build_model()
    utils.apply_rotations(evaluated, str(path), hadamard_block_size=ACTIVATION_GROUP,
                          **whisper_utils.get_whisper_rotation_layers(evaluated))
    handles = activation.attach_activation_quantization(
        evaluated, {name: capturing(eval_inputs, name) for name in roles}
    )
    _logits(evaluated, batches[0])

    assert len(handles) == len(roles)
    assert set(search_inputs) == set(eval_inputs) == set(roles)
    for name in roles:
        a, b = search_inputs[name], eval_inputs[name]
        assert a.shape == b.shape, name
        assert (a - b).abs().max() / a.abs().max() < 1e-5, name


def test_evaluation_hooks_quantize_a_pointwise_conv_on_its_channel_axis(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv1d(64, 32, 1), nn.Linear(32, 32)).to(DEVICE)
    reference = nn.Linear(64, 32).to(DEVICE)
    with torch.no_grad():
        reference.weight.copy_(model[0].weight.squeeze(-1))
        reference.bias.copy_(model[0].bias)
    quantize = utils.make_activation_quantizer(4, 16, False)
    activation.attach_activation_quantization(model, {"0": quantize})

    x = torch.randn(2, 64, 10, device=DEVICE)
    with torch.no_grad():
        got = model[0](x).transpose(1, 2)
        expected = reference(quantize(x.transpose(1, 2)))
    assert (got - expected).abs().max() < 1e-5


def test_evaluation_quantizes_fc2_in_the_hadamard_basis(batches, tmp_path):
    """fc2's input hook sees the online Hadamard's output: the basis its weight was folded for."""
    model = _rotated_whisper(batches, tmp_path)
    roles = whisper_utils.get_whisper_activation_roles(model)
    fc2_name = next(name for name, role in roles.items() if role == "fc2")
    fc2 = utils.get_module(model, fc2_name)
    activation_fn = utils.get_module(model, fc2_name.rsplit(".", 1)[0] + ".activation_fn")

    seen = {}
    activation_fn[0].register_forward_hook(lambda m, a, out: seen.__setitem__("act", out))

    def recording(x):
        seen["quantizer_input"] = x
        return x

    activation.attach_activation_quantization(model, {fc2_name: recording})
    _logits(model, batches[0])

    H = utils.online_hadamard_matrix(fc2.in_features, HADAMARD_BLOCK_SIZE, DEVICE)
    D = torch.diag(activation_fn[-1].signs)
    assert (seen["quantizer_input"] - seen["act"] @ D @ H).abs().max() < 1e-4


def test_block_random_hadamard_is_orthogonal_and_block_diagonal():
    R = utils.block_random_hadamard_matrix(64, 16, DEVICE)
    mask = utils.block_diagonal_mask(64, 16, DEVICE)
    assert (R.T @ R - torch.eye(64, device=DEVICE, dtype=R.dtype)).abs().max() < 1e-12
    assert torch.count_nonzero(R[~mask]) == 0
    assert torch.count_nonzero(R[mask]) == int(mask.sum())
    with pytest.raises(ValueError, match="does not divide"):
        utils.block_random_hadamard_matrix(64, 24, DEVICE)


def test_sgdg_keeps_a_block_diagonal_matrix_block_diagonal_even_through_qr_retraction(monkeypatch):
    """Every quantity in the update is built from the parameter and its gradient."""
    mask = utils.block_diagonal_mask(64, 16, DEVICE)
    R = nn.Parameter(utils.block_random_hadamard_matrix(64, 16, DEVICE).float())
    optimizer = cayley.SGDG([R], lr=5.0, stiefel=True)
    for retract in (False, True):
        monkeypatch.setattr(cayley.random, "randint", lambda a, b, r=retract: 1 if r else 2)
        R.grad = torch.randn(64, 64, device=DEVICE).masked_fill_(~mask, 0.0)
        before = R.data.clone()
        optimizer.step()
        assert not torch.equal(R.data, before)
        assert torch.count_nonzero(R.data[~mask]) == 0
        assert (R.data.T @ R.data - torch.eye(64, device=DEVICE)).abs().max() < 1e-4


def test_learn_rotations_with_a_block_r1_stays_block_diagonal(model, tmp_path):
    captured = []
    original = utils.block_random_hadamard_matrix

    def recording(size, block_size, device):
        matrix = original(size, block_size, device)
        captured.append(matrix.clone())
        return matrix

    utils.block_random_hadamard_matrix = recording
    try:
        R1, _ = utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=20.0,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=_batches(6), epochs=2,
            hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
            activation_bits=2, activation_group_size=ACTIVATION_GROUP, r1_block_size=16,
            activation_roles=whisper_utils.get_whisper_activation_roles(model),
            **whisper_utils.get_whisper_rotation_layers(model),
        )
    finally:
        utils.block_random_hadamard_matrix = original

    mask = utils.block_diagonal_mask(D_MODEL, 16, DEVICE)
    identity = torch.eye(D_MODEL, device=DEVICE)
    assert torch.count_nonzero(R1.data[~mask]) == 0
    assert (R1.data - captured[0].to(R1.device, R1.dtype)).abs().max() > 1e-3
    assert (R1.data.T @ R1.data - identity).abs().max() < 1e-4
    assert torch.load(tmp_path / "rotations.pt")["r1_block_size"] == 16


def test_a_block_size_that_does_not_divide_the_width_is_rejected_first(model, batches, tmp_path):
    with pytest.raises(ValueError, match="does not divide hidden_size"):
        utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"),
            hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, r1_block_size=24,
            **whisper_utils.get_whisper_rotation_layers(model),
        )
    assert any(isinstance(m, nn.LayerNorm) for m in model.modules())


@pytest.mark.parametrize("channels_first", [False, True])
def test_online_hadamard_with_signs_is_the_randomized_hadamard(channels_first):
    torch.manual_seed(0)
    width = 4 * HADAMARD_BLOCK_SIZE
    x = torch.randn(3, 20, width, device=DEVICE)
    signs = utils.random_hadamard_signs(width, DEVICE)
    H = utils.online_hadamard_matrix(width, HADAMARD_BLOCK_SIZE, DEVICE)
    module = utils.OnlineHadamard(HADAMARD_BLOCK_SIZE, channels_first=channels_first, signs=signs)
    with torch.no_grad():
        got = module(x.transpose(1, 2)).transpose(1, 2) if channels_first else module(x)
    assert (got - x @ torch.diag(signs) @ H).abs().max() < 1e-5
    assert set(signs.unique().tolist()) == {-1.0, 1.0}


def test_random_signs_keep_a_non_zero_mean_from_becoming_one_spike_per_block():
    """The failure on whisper-large-v3: a GELU output's mean turned into a spike per block."""
    torch.manual_seed(0)
    width = 4 * HADAMARD_BLOCK_SIZE
    x = 1.0 + 0.5 * torch.randn(64, width, device=DEVICE)

    def group_crest(t):
        g = t.reshape(t.shape[0], -1, HADAMARD_BLOCK_SIZE)
        return float((g.abs().amax(-1) / g.pow(2).mean(-1).sqrt()).median())

    plain = utils.OnlineHadamard(HADAMARD_BLOCK_SIZE)
    randomized = utils.OnlineHadamard(
        HADAMARD_BLOCK_SIZE, signs=utils.random_hadamard_signs(width, DEVICE)
    )
    with torch.no_grad():
        assert group_crest(plain(x)) > 6.0
        assert group_crest(randomized(x)) < 3.5


def test_the_search_saves_a_sign_seed_and_apply_uses_it(batches, tmp_path):
    model = _rotated_whisper(batches, tmp_path)
    checkpoint = torch.load(tmp_path / "rotations.pt")
    seed = checkpoint["hadamard_sign_seed"]
    assert seed > 0 and "hadamard_signs" not in checkpoint
    expected = utils.seeded_hadamard_signs(seed, 4 * D_MODEL, DEVICE)
    hadamards = [m for m in model.modules() if isinstance(m, utils.OnlineHadamard)]
    assert hadamards and all(h.sign_seed == seed and torch.equal(h.signs, expected) for h in hadamards)


def test_a_checkpoint_with_stored_sign_vectors_still_applies_exactly(batches, tmp_path):
    """Checkpoints saved before sign seeds hold one sign vector per fc2 width; they fold and multiply as before."""
    _rotated_whisper(batches, tmp_path)
    path = tmp_path / "rotations.pt"
    checkpoint = torch.load(path)
    signs = utils.seeded_hadamard_signs(checkpoint.pop("hadamard_sign_seed"), 4 * D_MODEL, "cpu")
    checkpoint["hadamard_signs"] = {4 * D_MODEL: signs}
    torch.save(checkpoint, path)

    fresh = _build_model()
    before = _logits(fresh, batches[0])
    utils.apply_rotations(fresh, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                          **whisper_utils.get_whisper_rotation_layers(fresh))
    after = _logits(fresh, batches[0])
    hadamards = [m for m in fresh.modules() if isinstance(m, utils.OnlineHadamard)]
    assert hadamards and all(h.sign_seed == 0 and torch.equal(h.signs.cpu(), signs) for h in hadamards)
    assert (after - before).abs().max() / before.abs().max() < TOLERANCE


def test_a_checkpoint_without_signs_applies_the_plain_hadamard(batches, tmp_path):
    """Rotations searched before random signs existed still apply exactly as they were searched."""
    path = tmp_path / "rotations.pt"
    searched = _build_model()
    utils.learn_rotations(
        searched, str(path), hidden_size=D_MODEL, learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE, hadamard_random_signs=False,
        **whisper_utils.get_whisper_rotation_layers(searched),
    )
    checkpoint = torch.load(path)
    assert checkpoint["hadamard_sign_seed"] is None
    del checkpoint["hadamard_sign_seed"]
    torch.save(checkpoint, path)

    fresh = _build_model()
    before = _logits(fresh, batches[0])
    utils.apply_rotations(fresh, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                          **whisper_utils.get_whisper_rotation_layers(fresh))
    after = _logits(fresh, batches[0])
    assert all(m.signs is None for m in fresh.modules() if isinstance(m, utils.OnlineHadamard))
    assert (after - before).abs().max() / before.abs().max() < TOLERANCE


def _learn_r1_only(model, path, batches, **extra):
    return utils.learn_rotations(
        model, str(path), hidden_size=D_MODEL, learning_rate=0.5,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
        activation_bits=4, activation_group_size=16, groupwise_roles=["attn_out", "fc2"],
        activation_roles=whisper_utils.get_whisper_activation_roles(model),
        learn_r2=False, fc2_online_hadamard=False,
        **whisper_utils.get_whisper_rotation_layers(model), **extra,
    )


def test_r1_only_search_optimises_nothing_but_r1(model, batches, tmp_path, monkeypatch):
    optimised = []
    original = utils.SGDG

    def recording(params, **kwargs):
        params = list(params)
        optimised.extend(params)
        return original(params, **kwargs)

    monkeypatch.setattr(utils, "SGDG", recording)
    R1, R2s = _learn_r1_only(model, tmp_path / "rotations.pt", batches)

    assert len(optimised) == 1 and optimised[0] is R1
    assert R2s == {}
    saved = torch.load(tmp_path / "rotations.pt")
    assert saved["learn_r2"] is False and saved["fc2_online_hadamard"] is False
    assert saved["R2s"] == {} and saved["hadamard_sign_seed"] is None


def test_r1_only_apply_leaves_out_proj_and_fc2_inputs_unrotated(batches, tmp_path):
    """Only the residual stream is rotated: out_proj and fc2 read exactly the unrotated tensors."""
    path = tmp_path / "rotations.pt"
    _learn_r1_only(_build_model(), path, batches)

    def capture_inputs(model):
        roles = whisper_utils.get_whisper_activation_roles(model)
        seen = {}

        def recording(name):
            def record(x):
                seen[name] = x.detach().float().clone()
                return x
            return record

        handles = activation.attach_activation_quantization(
            model, {name: recording(name) for name in roles}
        )
        logits = _logits(model, batches[0])
        for handle in handles:
            handle.remove()
        return roles, seen, logits

    roles, reference_inputs, reference_logits = capture_inputs(_build_model())

    rotated = _build_model()
    utils.apply_rotations(rotated, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                          **whisper_utils.get_whisper_rotation_layers(rotated))
    _, rotated_inputs, rotated_logits = capture_inputs(rotated)

    assert not any(isinstance(m, utils.OnlineHadamard) for m in rotated.modules())
    change = (rotated_logits - reference_logits).abs().max() / reference_logits.abs().max()
    assert change < TOLERANCE
    for name, role in roles.items():
        a, b = reference_inputs[name], rotated_inputs[name]
        difference = float((a - b).abs().max() / a.abs().max())
        if role in ("attn_out", "fc2"):
            assert difference < 1e-4, (name, difference)
        else:
            assert difference > 1e-2, (name, difference)


def test_r1_only_evaluation_quantizes_the_same_tensors_as_the_search(batches, tmp_path):
    path = tmp_path / "rotations.pt"
    searched = _build_model()
    roles = whisper_utils.get_whisper_activation_roles(searched)
    R1, _ = _learn_r1_only(searched, path, batches)

    def capturing(store, name):
        def record(x):
            store[name] = x.detach().float().clone()
            return x
        return record

    search_inputs, eval_inputs = {}, {}
    utils.patch_rotations(
        searched, whisper_utils.get_whisper_layers_to_rotate(searched), R1, None, {},
        HADAMARD_BLOCK_SIZE, {name: capturing(search_inputs, name) for name in roles},
        use_r2=False, fc2_online_hadamard=False,
    )
    _logits(searched, batches[0])

    evaluated = _build_model()
    utils.apply_rotations(evaluated, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                          **whisper_utils.get_whisper_rotation_layers(evaluated))
    activation.attach_activation_quantization(
        evaluated, {name: capturing(eval_inputs, name) for name in roles}
    )
    _logits(evaluated, batches[0])

    assert set(search_inputs) == set(eval_inputs) == set(roles)
    for name in roles:
        a, b = search_inputs[name], eval_inputs[name]
        assert (a - b).abs().max() / a.abs().max() < 1e-5, name


def test_r1_only_works_on_a_conformer_with_pointwise_convs(tmp_path):
    parakeet_utils, build = _parakeet()
    torch.manual_seed(1)
    batches = [{"x": torch.randn(2, 80, 200, device=DEVICE, dtype=DTYPE),
                "length": torch.tensor([200, 180], device=DEVICE)} for _ in range(2)]

    def compute_logits(model, batch):
        out, lengths = model.encoder(audio_signal=batch["x"], length=batch["length"])
        frames = torch.arange(out.shape[-1], device=DEVICE)
        return out.transpose(1, 2), frames.unsqueeze(0) < lengths.unsqueeze(1)

    def encode(model):
        with torch.no_grad():
            return model.encoder(audio_signal=batches[0]["x"], length=batches[0]["length"])[0]

    expected = encode(build())
    searched = build()
    path = tmp_path / "rotations.pt"
    utils.learn_rotations(
        searched, str(path), hidden_size=D_MODEL, learning_rate=0.5, compute_logits=compute_logits,
        train_loader=batches, epochs=1, hadamard_block_size=HADAMARD_BLOCK_SIZE,
        tolerance=TOLERANCE, activation_bits=4, activation_group_size=16,
        groupwise_roles=["attn_out", "fc2"],
        activation_roles=parakeet_utils.get_parakeet_activation_roles(searched),
        learn_r2=False, fc2_online_hadamard=False,
        **parakeet_utils.get_parakeet_rotation_layers(searched),
    )
    fresh = build()
    utils.apply_rotations(fresh, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                          **parakeet_utils.get_parakeet_rotation_layers(fresh))
    got = encode(fresh)
    assert not any(isinstance(m, utils.OnlineHadamard) for m in fresh.modules())
    assert (got - expected).abs().max() / expected.abs().max() < TOLERANCE


def test_the_block_output_linears_get_the_block_out_role_only_when_asked():
    parakeet_utils, build = _parakeet()
    model = build()
    assert "block_out" not in parakeet_utils.get_parakeet_activation_roles(model, True).values()
    parakeet_utils.get_parakeet_rotation_layers(model)
    roles = parakeet_utils.get_parakeet_activation_roles(model, quantize_block_output_linear=True)
    inserted = {name for name, role in roles.items() if role == "block_out"}
    assert inserted == {"encoder.layers.0.norm_out.1", "encoder.layers.1.norm_out.1"}
    assert "block_out" not in parakeet_utils.get_parakeet_activation_roles(model).values()


def test_a_search_that_quantizes_the_block_output_linears_still_folds_exactly(tmp_path):
    """With block_out quantized, the search attaches a quantizer to the rotation-type-5 patch; the
    folded model must still compute the original function once quantization is off."""
    parakeet_utils, build = _parakeet()
    torch.manual_seed(1)
    batches = [{"x": torch.randn(2, 80, 200, device=DEVICE, dtype=DTYPE),
                "length": torch.tensor([200, 180], device=DEVICE)} for _ in range(2)]

    def compute_logits(model, batch):
        out, lengths = model.encoder(audio_signal=batch["x"], length=batch["length"])
        frames = torch.arange(out.shape[-1], device=DEVICE)
        return out.transpose(1, 2), frames.unsqueeze(0) < lengths.unsqueeze(1)

    def encode(model):
        with torch.no_grad():
            return model.encoder(audio_signal=batches[0]["x"], length=batches[0]["length"])[0]

    expected = encode(build())
    searched = build()
    layers = parakeet_utils.get_parakeet_rotation_layers(searched)
    roles = parakeet_utils.get_parakeet_activation_roles(searched, quantize_block_output_linear=True)
    quantized_inputs = {}
    original = utils.patch_rotations

    def recording_patch(model, layers_to_rotate, R1, R2s, hadamards, hadamard_block_size,
                        quantize_inputs=None, *args, **kwargs):
        quantized_inputs.update(quantize_inputs or {})
        return original(model, layers_to_rotate, R1, R2s, hadamards, hadamard_block_size,
                        quantize_inputs, *args, **kwargs)

    utils.patch_rotations = recording_patch
    try:
        utils.learn_rotations(
            searched, str(tmp_path / "rotations.pt"), hidden_size=D_MODEL, learning_rate=0.5,
            compute_logits=compute_logits, train_loader=batches, epochs=1,
            hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE, activation_bits=4,
            activation_group_size=ACTIVATION_GROUP, groupwise_roles=["attn_out", "fc2"], activation_roles=roles,
            **layers,
        )
    finally:
        utils.patch_rotations = original
    assert {"encoder.layers.0.norm_out.1", "encoder.layers.1.norm_out.1"} <= set(quantized_inputs)

    fresh = build()
    utils.apply_rotations(fresh, str(tmp_path / "rotations.pt"), hadamard_block_size=ACTIVATION_GROUP,
                          **parakeet_utils.get_parakeet_rotation_layers(fresh))
    got = encode(fresh)
    assert (got - expected).abs().max() / expected.abs().max() < TOLERANCE


def test_a_checkpoint_from_before_the_switches_still_folds_r2_and_the_hadamard(batches, tmp_path):
    model = _rotated_whisper(batches, tmp_path)
    checkpoint = torch.load(tmp_path / "rotations.pt")
    assert checkpoint["learn_r2"] and checkpoint["fc2_online_hadamard"]
    del checkpoint["learn_r2"], checkpoint["fc2_online_hadamard"]
    torch.save(checkpoint, tmp_path / "old.pt")

    fresh = _build_model()
    before = _logits(fresh, batches[0])
    utils.apply_rotations(fresh, str(tmp_path / "old.pt"), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                          **whisper_utils.get_whisper_rotation_layers(fresh))
    assert sum(isinstance(m, utils.OnlineHadamard) for m in fresh.modules()) == \
        sum(isinstance(m, utils.OnlineHadamard) for m in model.modules()) > 0
    assert (_logits(fresh, batches[0]) - before).abs().max() / before.abs().max() < TOLERANCE


@pytest.mark.parametrize(
    "block, group, roles, hadamard, raises",
    [
        (128, 128, ["attn_out", "fc2"], True, False),
        (128, 64, ["attn_out", "fc2"], True, True),
        (128, 64, None, True, True),
        (128, 64, ["q", "k", "v"], True, False),
        (128, 64, ["attn_out", "fc2"], False, False),
    ],
)
def test_the_fc2_hadamard_block_must_match_its_quantization_groups(block, group, roles, hadamard,
                                                                    raises):
    if raises:
        with pytest.raises(ValueError, match="hadamard_block_size equal to activation_group_size"):
            utils.check_hadamard_matches_groups(block, group, roles, hadamard)
    else:
        utils.check_hadamard_matches_groups(block, group, roles, hadamard)


def test_learn_rotations_rejects_a_mismatched_hadamard_block_before_touching_the_model(
    model, batches, tmp_path
):
    with pytest.raises(ValueError, match="hadamard_block_size equal to activation_group_size"):
        utils.learn_rotations(
            model, str(tmp_path / "rotations.pt"), hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, activation_bits=4,
            activation_group_size=HADAMARD_BLOCK_SIZE // 2, groupwise_roles=["fc2"],
            activation_roles=whisper_utils.get_whisper_activation_roles(model),
            **whisper_utils.get_whisper_rotation_layers(model),
        )
    assert any(isinstance(m, nn.LayerNorm) for m in model.modules())


def _applied(batches, tmp_path):
    """A rotation learned on a fresh model, the checkpoint path, and a second fresh model with it
    applied."""
    path = tmp_path / "rotations.pt"
    utils.learn_rotations(
        _build_model(), str(path), hidden_size=D_MODEL, learning_rate=0.5,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
        **whisper_utils.get_whisper_rotation_layers(_build_model()),
    )
    model = _build_model()
    handles = utils.apply_rotations(model, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                                    **whisper_utils.get_whisper_rotation_layers(model))
    return path, model, handles


def test_applied_rotations_are_modules_whose_rotation_is_in_the_state_dict(batches, tmp_path):
    _, model, handles = _applied(batches, tmp_path)
    rotations = {n: m for n, m in model.named_modules()
                 if isinstance(m, utils.ResidualStreamRotation)}
    assert set(rotations) == {
        "model.encoder.layers.0.input_rotation",
        "model.decoder.layers.0.input_rotation",
        "model.decoder.layer_norm.input_unrotation",
    }
    assert len(handles) == 3
    keys = model.state_dict().keys()
    assert all(f"{name}.rotation" in keys for name in rotations)

    model.half()
    assert all(m.rotation.dtype == torch.float32 for m in rotations.values())
    assert all(m.rotation.device.type == "cuda" for m in rotations.values())


def test_a_rotated_model_survives_torch_save_and_load_without_reapplying(batches, tmp_path):
    _, model, _ = _applied(batches, tmp_path)
    expected = _logits(model, batches[0])

    torch.save(model, tmp_path / "model.pt")
    loaded = torch.load(tmp_path / "model.pt", weights_only=False)
    assert torch.equal(_logits(loaded, batches[0]), expected)
    assert torch.equal(_logits(copy.deepcopy(model), batches[0]), expected)


def test_a_rotated_state_dict_loads_into_a_model_prepared_with_the_same_rotation(batches, tmp_path):
    path, model, _ = _applied(batches, tmp_path)
    expected = _logits(model, batches[0])
    state = {k: v.clone() for k, v in model.state_dict().items()}

    fresh = _build_model()
    utils.apply_rotations(fresh, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE,
                          **whisper_utils.get_whisper_rotation_layers(fresh))
    with torch.no_grad():
        for tensor in fresh.state_dict().values():
            if tensor.is_floating_point():
                tensor.add_(torch.randn_like(tensor) * 1e-3)
    fresh.load_state_dict(state)
    assert torch.equal(_logits(fresh, batches[0]), expected)


def test_applying_or_learning_without_saying_which_hooks_to_attach_is_an_error(batches, tmp_path):
    """Leaving out the residual-stream hooks silently gives a wrong Whisper model, so the argument
    has no default; None must be passed on purpose."""
    path, _, _ = _applied(batches, tmp_path)
    fresh = _build_model()
    layers = whisper_utils.get_whisper_rotation_layers(fresh)
    del layers["attach_rotation_hooks"]
    with pytest.raises(TypeError, match="attach_rotation_hooks"):
        utils.apply_rotations(fresh, str(path), hadamard_block_size=HADAMARD_BLOCK_SIZE, **layers)
    with pytest.raises(TypeError, match="attach_rotation_hooks"):
        utils.learn_rotations(
            fresh, str(tmp_path / "unused.pt"), hidden_size=D_MODEL, learning_rate=1e-3,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, **layers,
        )


def test_removing_a_persistent_rotation_detaches_its_module_and_hook(model):
    layer = model.model.encoder.layers[0]
    R1 = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL, device=DEVICE))[0]
    handle = utils.add_residual_stream_entry_hook(layer, R1, persistent=True)
    assert isinstance(layer.input_rotation, utils.ResidualStreamRotation)
    assert len(layer._forward_pre_hooks) == 1
    with pytest.raises(ValueError, match="already has a input_rotation"):
        utils.add_residual_stream_entry_hook(layer, R1, persistent=True)
    handle.remove()
    assert not hasattr(layer, "input_rotation") and len(layer._forward_pre_hooks) == 0


def test_the_search_leaves_no_rotation_modules_behind(model, batches, tmp_path):
    utils.learn_rotations(
        model, str(tmp_path / "rotations.pt"), hidden_size=D_MODEL, learning_rate=1e-3,
        compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
        hadamard_block_size=HADAMARD_BLOCK_SIZE, tolerance=TOLERANCE,
        **whisper_utils.get_whisper_rotation_layers(model),
    )
    assert not any(isinstance(m, utils.ResidualStreamRotation) for m in model.modules())


def _evolution_settings(**overrides):
    settings = dict(generations=4, offspring=4, survivors=(2, 1), stage_samples=(2, 4, 6), flips=2, seed=3)
    return hadamard_search.EvolutionConfig(**{**settings, **overrides})


@pytest.mark.parametrize("symmetric", [True, False])
def test_the_evolutionary_search_saves_a_signed_hadamard_r1_that_applies_exactly(batches, tmp_path, symmetric):
    """R1 = diag(s1) @ H @ diag(s2) from the saved signs, a non-increasing KL, R2s left random, exact apply.

    The default mutate="auto" leaves s2 at its initial signs under symmetric quantization, which it does
    not affect, and mutates it under asymmetric quantization.
    """
    path = tmp_path / "evolved.pt"
    searched = _build_model()
    evolution = _evolution_settings(generations=8)
    R1, R2s = utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=0.0, compute_logits=whisper_utils.whisper_logits_fn,
        train_loader=batches, epochs=1, hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
        activation_bits=2, activation_group_size=ACTIVATION_GROUP, activation_symmetric=symmetric,
        activation_roles=whisper_utils.get_whisper_activation_roles(searched),
        search="evolution", evolution=evolution,
        **whisper_utils.get_whisper_rotation_layers(searched),
    )

    saved = torch.load(path)
    assert saved["search"]["name"] == "evolution"
    assert saved["search"]["settings"]["offspring"] == 4
    s1, s2 = saved["search"]["R1_signs"][None]
    expected = hadamard_search.signed_hadamard(hadamard_search.hadamard_basis(D_MODEL, None, "cpu"), s1, s2)
    assert (saved["R1"].double() - expected).abs().max() < 1e-6
    trajectory = [record["fitness"] for record in saved["search"]["history"]]
    assert len(trajectory) == 9 and all(b <= a for a, b in zip(trajectory, trajectory[1:]))
    initial_s1, initial_s2 = hadamard_search.random_sign_vectors(
        {None: D_MODEL}, torch.Generator().manual_seed(evolution.seed), "cpu"
    )[None]
    assert saved["search"]["settings"]["mutate"] == ("s1" if symmetric else "s1_s2")
    # Mutations only flip s1 under symmetric quantization, so s2 survives unless a random child was accepted,
    # which replaces both vectors.
    accepted_random = any(record["accepted"] and record.get("best_child_random")
                          for record in saved["search"]["history"][1:])
    if symmetric and not accepted_random:
        assert torch.equal(s2, initial_s2)
    assert not torch.equal(s1, initial_s1)
    for R2 in saved["R2s"].values():
        assert (R2.T @ R2 - torch.eye(R2.shape[0])).abs().max() < 1e-5

    fresh = _build_model()
    before = _logits(fresh, batches[0])
    utils.apply_rotations(fresh, str(path), hadamard_block_size=ACTIVATION_GROUP,
                          **whisper_utils.get_whisper_rotation_layers(fresh))
    after = _logits(fresh, batches[0])
    assert (after - before).abs().max() / before.abs().max() < TOLERANCE


def test_the_evolutionary_search_needs_the_kl_objective(model, batches, tmp_path):
    with pytest.raises(ValueError, match="objective='kl'"):
        utils.learn_rotations(
            model, str(tmp_path / "r.pt"), hidden_size=D_MODEL, learning_rate=0.0,
            compute_logits=whisper_utils.whisper_logits_fn, train_loader=batches, epochs=1,
            hadamard_block_size=HADAMARD_BLOCK_SIZE, objective="ce", compute_loss=_cross_entropy,
            search="evolution", **whisper_utils.get_whisper_rotation_layers(model),
        )


@pytest.mark.parametrize("symmetric", [True, False])
def test_with_symmetric_quantization_s2_does_not_change_the_kl(model, batches, symmetric):
    """x @ diag(s1) @ H @ diag(s2): s2 flips rotated coordinates, which a symmetric quantizer commutes with.

    The flips are random: flipping a Walsh pattern of s1 (every other position) permutes H's columns, which a
    per-token quantizer is blind to as well.
    """
    utils.fold_norms(model, whisper_utils.get_whisper_norm_layers(model))
    groups = whisper_utils.get_whisper_layers_to_rotate(model)
    H = hadamard_search.hadamard_basis(D_MODEL, None, DEVICE)
    R1 = torch.zeros(D_MODEL, D_MODEL, device=DEVICE)
    R2s = {name: utils.random_hadamard_matrix(head_dim, DEVICE).float() for name, head_dim, _ in groups}
    whisper_utils.attach_whisper_rotation_hooks(model, R1)
    quantizers = activation.build_activation_quantizers(
        whisper_utils.get_whisper_activation_roles(model), 4, ACTIVATION_GROUP, symmetric, ["attn_out", "fc2"],
    )
    utils.patch_rotations(model, groups, R1, R2s, {}, ACTIVATION_GROUP, quantizers,
                          {4 * D_MODEL: utils.random_hadamard_signs(4 * D_MODEL, DEVICE)})

    def kl(s1, s2):
        R1.copy_(hadamard_search.signed_hadamard(H, s1, s2))
        with torch.no_grad():
            with activation.quantizers_disabled(quantizers):
                teacher, mask = whisper_utils.whisper_logits_fn(model, batches[0])
            student, _ = whisper_utils.whisper_logits_fn(model, batches[0])
        return float(utils.masked_kl_divergence(student, teacher, mask))

    generator = torch.Generator().manual_seed(0)
    (s1, s2), = hadamard_search.random_sign_vectors({None: D_MODEL}, generator, DEVICE).values()
    half = torch.where(torch.randperm(D_MODEL, generator=generator) < D_MODEL // 2, -1.0, 1.0).to(DEVICE)
    base = kl(s1, s2)
    s2_flipped = kl(s1, s2 * half)
    s1_flipped = kl(s1 * half, s2)
    assert abs(s1_flipped - base) > 1e-2 * base
    if symmetric:
        assert abs(s2_flipped - base) < 1e-4 * base
    else:
        assert abs(s2_flipped - base) > 1e-3 * base


def _weight_only_quantization(symmetric=True, method="rtn"):
    return {"method": method, "bits": 2, "group_size": ACTIVATION_GROUP, "symmetric": symmetric, "layers": None,
            "percdamp": 0.01, "block_size": ACTIVATION_GROUP}


@pytest.mark.parametrize("symmetric", [True, False])
def test_the_weight_only_search_scores_the_folded_model_with_rounded_weights(batches, tmp_path, symmetric):
    """The saved fitness is the KL of apply_rotations' model with RTN weights, recomputed independently.

    The search restores and refolds the weights for every candidate; if a restore missed a layer or a bias,
    the model it scored would differ from the one the checkpoint applies to.
    """
    path = tmp_path / "weight_only.pt"
    searched = _build_model()
    evolution = _evolution_settings(generations=6)
    utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=0.0, compute_logits=whisper_utils.whisper_logits_fn,
        train_loader=batches, epochs=1, hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
        search="evolution", evolution=evolution, weight_quantization=_weight_only_quantization(symmetric),
        **whisper_utils.get_whisper_rotation_layers(searched),
    )
    saved = torch.load(path)
    assert saved["search"]["quantization"] == "weights"
    assert saved["search"]["weight_quantization"] == {
        "method": "rtn", "bits": 2, "group_size": ACTIVATION_GROUP, "symmetric": symmetric,
    }
    assert saved["activation_quantization"]["bits"] is None
    assert saved["search"]["settings"]["mutate"] == "s1_s2"
    trajectory = [record["fitness"] for record in saved["search"]["history"]]
    assert all(b <= a for a, b in zip(trajectory, trajectory[1:]))

    original = _build_model()
    references = [_logits(original, batch) for batch in batches]
    rotated = _build_model()
    layers = whisper_utils.get_whisper_rotation_layers(rotated)
    utils.apply_rotations(rotated, str(path), hadamard_block_size=ACTIVATION_GROUP, **layers)
    modules = dict(rotated.named_modules())
    with torch.no_grad():
        for group in layers["layers_to_rotate"]:
            for name, _ in group[2]:
                weight = utils._weight_2d(modules[name])
                weight.copy_(rounding.round_weight(weight, 2, ACTIVATION_GROUP, symmetric)[0])
        kls = []
        for batch, reference in zip(batches, references):
            logits, mask = whisper_utils.whisper_logits_fn(rotated, batch)
            kls.append(float(utils.masked_kl_divergence(logits, reference, mask)))
    assert abs(sum(kls) / len(kls) - trajectory[-1]) < 1e-3 * trajectory[-1]
    assert trajectory[0] > 0


def test_a_weight_only_search_needs_the_evolutionary_search_and_no_activation_bits(model, batches, tmp_path):
    common = dict(
        hidden_size=D_MODEL, learning_rate=0.0, compute_logits=whisper_utils.whisper_logits_fn,
        train_loader=batches, epochs=1, hadamard_block_size=ACTIVATION_GROUP,
        weight_quantization=_weight_only_quantization(), **whisper_utils.get_whisper_rotation_layers(model),
    )
    with pytest.raises(ValueError, match="search='evolution'"):
        utils.learn_rotations(model, str(tmp_path / "a.pt"), **common)
    with pytest.raises(ValueError, match="activation_bits=None"):
        utils.learn_rotations(
            model, str(tmp_path / "b.pt"), search="evolution", activation_bits=4, activation_group_size=ACTIVATION_GROUP,
            activation_roles=whisper_utils.get_whisper_activation_roles(model), **common,
        )


@pytest.mark.parametrize("symmetric", [True, False])
def test_the_gptq_weight_only_search_scores_gptq_on_the_rotated_hessians(batches, tmp_path, symmetric):
    """The saved fitness is recomputed with the GPTQ quantizer on Hessians collected in the rotated model.

    The search collects each Hessian once with R1 = I and rotates it as R1.T @ H @ R1 for the layers reading
    the residual stream, keeping the other layers' factors fixed. Collecting them afresh from the applied
    rotation must give the same quantized model, up to float rounding in the Hessians.
    """
    path = tmp_path / "gptq.pt"
    searched = _build_model()
    utils.learn_rotations(
        searched, str(path),
        hidden_size=D_MODEL, learning_rate=0.0, compute_logits=whisper_utils.whisper_logits_fn,
        train_loader=batches, epochs=1, hadamard_block_size=ACTIVATION_GROUP, tolerance=TOLERANCE,
        search="evolution", evolution=_evolution_settings(generations=4),
        weight_quantization=_weight_only_quantization(symmetric, method="gptq"),
        **whisper_utils.get_whisper_rotation_layers(searched),
    )
    saved = torch.load(path)
    assert saved["search"]["weight_quantization"]["method"] == "gptq"
    trajectory = [record["fitness"] for record in saved["search"]["history"]]
    assert all(b <= a for a, b in zip(trajectory, trajectory[1:]))

    original = _build_model()
    references = [_logits(original, batch) for batch in batches]
    rotated = _build_model()
    layers = whisper_utils.get_whisper_rotation_layers(rotated)
    utils.apply_rotations(rotated, str(path), hadamard_block_size=ACTIVATION_GROUP, **layers)
    modules = dict(rotated.named_modules())
    names = [name for group in layers["layers_to_rotate"] for name, _ in group[2]]
    hessians = {name: [torch.zeros(utils._weight_2d(modules[name]).shape[1],) * 0, 0] for name in names}
    for name in names:
        width = utils._weight_2d(modules[name]).shape[1]
        hessians[name][0] = torch.zeros(width, width, device=DEVICE)

    def hook(_module, inputs, _output, name):
        x = inputs[0].reshape(-1, inputs[0].shape[-1])
        hessians[name][0], hessians[name][1] = solver.add_to_hessian(hessians[name][0], hessians[name][1], x)

    handles = [modules[name].register_forward_hook(lambda m, i, o, name=name: hook(m, i, o, name)) for name in names]
    with torch.no_grad():
        for batch in batches:
            whisper_utils.whisper_logits_fn(rotated, batch)
        for handle in handles:
            handle.remove()
        for name in names:
            weight = utils._weight_2d(modules[name])
            factors = solver.gptq_factors(hessians[name][0], 0.01)
            weight.copy_(solver.gptq_quantize(weight, factors, 2, ACTIVATION_GROUP, symmetric, ACTIVATION_GROUP))
        kls = []
        for batch, reference in zip(batches, references):
            logits, mask = whisper_utils.whisper_logits_fn(rotated, batch)
            kls.append(float(utils.masked_kl_divergence(logits, reference, mask)))
    assert abs(sum(kls) / len(kls) - trajectory[-1]) < 0.02 * trajectory[-1]


@pytest.mark.parametrize("channels_first", [False, True])
def test_a_seeded_online_hadamard_equals_multiplying_by_its_signs(channels_first):
    """humming generates the seeded signs inside its kernel: the same output as multiplying first, bit for bit."""
    width, seed = 4 * D_MODEL, 12345
    signs = utils.seeded_hadamard_signs(seed, width, DEVICE)
    x = torch.randn(3, width, 7, device=DEVICE) if channels_first else torch.randn(5, width, device=DEVICE)
    seeded = utils.OnlineHadamard(HADAMARD_BLOCK_SIZE, channels_first=channels_first, signs=signs, sign_seed=seed)
    multiplied = utils.OnlineHadamard(HADAMARD_BLOCK_SIZE, channels_first=channels_first, signs=signs)
    assert torch.equal(seeded(x), multiplied(x))


def test_seeded_signs_are_balanced_and_differ_between_seeds():
    first, second = (utils.seeded_hadamard_signs(seed, 4096, "cpu") for seed in (1, 2))
    assert set(first.unique().tolist()) == {-1.0, 1.0}
    assert abs(float(first.mean())) < 0.05 and float((first != second).float().mean()) > 0.4
    assert torch.equal(utils.seeded_hadamard_signs(1, 64, "cpu"), first[:64])
    with pytest.raises(ValueError):
        utils.seeded_hadamard_signs(0, 8, "cpu")
