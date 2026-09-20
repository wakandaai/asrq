"""Correctness tests for asrq.transforms.rotation.utils.

Every transform in that module is an exact algebraic rewrite, so each test compares the
transformed layer against an explicit reference that applies the same operation the slow,
obvious way. Everything runs in float64 so the tolerances can be tight enough to catch a
transposed or swapped operand rather than just a shape error.

The module is loaded by path because asrq/__init__.py does not currently import cleanly; swap
this for `from asrq.transforms.rotation import utils` once the package import chain is fixed.
"""

import copy
import importlib.util
import sys
import types as _types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_ROTATION = Path(__file__).resolve().parents[1] / "asrq" / "transforms" / "rotation"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# utils imports its siblings by absolute path, which would pull in the package __init__ chain.
# That chain does not currently import cleanly, so stub the parent packages and register the
# siblings directly. Drop this once `import asrq` works and load utils normally.
for _pkg in ("asrq", "asrq.transforms", "asrq.transforms.rotation", "asrq.quantizers"):
    sys.modules.setdefault(_pkg, _types.ModuleType(_pkg))
_load("asrq.quantizers.activation", _ROTATION.parents[1] / "quantizers" / "activation.py")
_load("asrq.transforms.rotation.hadamard_utils", _ROTATION / "hadamard_utils.py")
_load("asrq.quantizers.weight_rounding", _ROTATION.parents[1] / "quantizers" / "weight_rounding.py")
_load("asrq.quantizers.gptq_solver", _ROTATION.parents[1] / "quantizers" / "gptq_solver.py")
_load("asrq.transforms.rotation.cayley_sgd", _ROTATION / "cayley_sgd.py")
hadamard_search = _load("asrq.transforms.rotation.hadamard_search", _ROTATION / "hadamard_search.py")
utils = _load("rotation_utils", _ROTATION / "utils.py")

DTYPE = torch.float64
ATOL = 1e-9
LAYER_KINDS = ["linear", "conv"]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def rot(n):
    """A random orthogonal matrix, standing in for a learned rotation."""
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=DTYPE))
    return q


def make_layer(kind, d_in, d_out, bias=True):
    if kind == "linear":
        return nn.Linear(d_in, d_out, bias=bias).to(DTYPE)
    return nn.Conv1d(d_in, d_out, kernel_size=1, bias=bias).to(DTYPE)


def make_input(kind, d_in, batch=4, length=7):
    """Linear consumes (N, C); Conv1d consumes (N, C, L)."""
    if kind == "linear":
        return torch.randn(batch, d_in, dtype=DTYPE)
    return torch.randn(batch, d_in, length, dtype=DTYPE)


def rotate_features(kind, t, R, transpose=False):
    """Apply t @ R (or t @ R.T) along whichever axis holds the features."""
    M = R.T if transpose else R
    if kind == "linear":
        return t @ M
    return (t.transpose(1, 2) @ M).transpose(1, 2)


# --------------------------------------------------------------------------------------
# _weight_2d
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_weight_2d_returns_out_in_matrix(kind):
    layer = make_layer(kind, 8, 5)
    assert utils._weight_2d(layer).shape == (5, 8)


@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_weight_2d_is_a_view_and_preserves_parameter_shape(kind):
    layer = make_layer(kind, 8, 5)
    original_shape = layer.weight.shape

    w = utils._weight_2d(layer)
    assert w.data_ptr() == layer.weight.data_ptr()

    w.mul_(2.0)
    assert layer.weight.shape == original_shape
    assert isinstance(layer.weight, nn.Parameter)


def test_weight_2d_rejects_non_pointwise_conv():
    with pytest.raises(ValueError, match="pointwise"):
        utils._weight_2d(nn.Conv1d(8, 8, kernel_size=3))


def test_weight_2d_rejects_grouped_conv():
    with pytest.raises(ValueError, match="pointwise"):
        utils._weight_2d(nn.Conv1d(8, 8, kernel_size=1, groups=8))


# --------------------------------------------------------------------------------------
# _RMSNorm / _rmsnorm  (LayerNorm -> RMSNorm -> pure normalization)
# --------------------------------------------------------------------------------------


def run_norm_block(kind_in, kind_out, previous_layer, norm, next_layer, x):
    """previous -> norm -> next layer, transposing as each layer kind requires.

    The norm always sees (N, ..., C); a Conv1d neighbour works in (N, C, L).
    """
    if kind_in == "conv":
        h = previous_layer(x)
    else:
        h = previous_layer(x.transpose(1, 2)).transpose(1, 2)
    h = norm(h.transpose(1, 2))
    return next_layer(h.transpose(1, 2)) if kind_out == "conv" else next_layer(h).transpose(1, 2)


def make_layer_norm(dim):
    ln = nn.LayerNorm(dim).to(DTYPE)
    nn.init.normal_(ln.weight, mean=1.0, std=0.1)
    nn.init.normal_(ln.bias, std=0.1)
    return ln


@pytest.mark.parametrize("kind_out", LAYER_KINDS)
@pytest.mark.parametrize("kind_in", LAYER_KINDS)
def test_rmsnorm_conversion_preserves_layer_norm_output(kind_in, kind_out):
    """_RMSNorm folds centering into the previous layer, the shift into the next one."""
    dim = 8
    ln = make_layer_norm(dim)
    previous_layer, next_layer = make_layer(kind_in, 5, dim), make_layer(kind_out, dim, 3)
    ref_ln, ref_previous_layer, ref_next_layer = (
        copy.deepcopy(m) for m in (ln, previous_layer, next_layer)
    )

    x = torch.randn(4, 5, 7, dtype=DTYPE)
    expected = run_norm_block(kind_in, kind_out, ref_previous_layer, ref_ln, ref_next_layer, x)

    norm = utils._RMSNorm(ln, [previous_layer], [next_layer])
    got = run_norm_block(kind_in, kind_out, previous_layer, norm, next_layer, x)

    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("kind_out", LAYER_KINDS)
@pytest.mark.parametrize("kind_in", LAYER_KINDS)
def test_full_chain_to_scale_free_normalization(kind_in, kind_out):
    """LayerNorm -> _RMSNorm -> _rmsnorm leaves a norm with no learnable state at all."""
    dim = 8
    ln = make_layer_norm(dim)
    previous_layer, next_layer = make_layer(kind_in, 5, dim), make_layer(kind_out, dim, 3)
    ref_ln, ref_previous_layer, ref_next_layer = (
        copy.deepcopy(m) for m in (ln, previous_layer, next_layer)
    )

    x = torch.randn(4, 5, 7, dtype=DTYPE)
    expected = run_norm_block(kind_in, kind_out, ref_previous_layer, ref_ln, ref_next_layer, x)

    norm = utils._rmsnorm(utils._RMSNorm(ln, [previous_layer], [next_layer]), [next_layer])
    got = run_norm_block(kind_in, kind_out, previous_layer, norm, next_layer, x)

    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)
    assert list(norm.state_dict()) == []


def test_rmsnorm_accepts_torch_rmsnorm():
    """_rmsnorm also folds the scale of a stock nn.RMSNorm."""
    dim = 8
    rms = nn.RMSNorm(dim, eps=1e-6).to(DTYPE)
    nn.init.normal_(rms.weight, mean=1.0, std=0.1)
    next_layer = make_layer("linear", dim, 3)
    ref_rms, ref_next_layer = copy.deepcopy(rms), copy.deepcopy(next_layer)

    x = torch.randn(4, dim, dtype=DTYPE)
    expected = ref_next_layer(ref_rms(x))
    got = next_layer(utils._rmsnorm(rms, [next_layer])(x))

    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


def test_rmsnorm_conversion_with_several_next_layers():
    """One norm feeding Q, K and V: the shift and scale fold into each independently."""
    dim = 8
    ln = make_layer_norm(dim)
    previous_layer = make_layer("linear", 5, dim)
    next_layers = [make_layer("linear", dim, 3) for _ in range(3)]
    ref_ln, ref_previous = copy.deepcopy(ln), copy.deepcopy(previous_layer)
    ref_next = [copy.deepcopy(m) for m in next_layers]

    x = torch.randn(4, 5, dtype=DTYPE)
    expected = [m(ref_ln(ref_previous(x))) for m in ref_next]

    norm = utils._rmsnorm(utils._RMSNorm(ln, [previous_layer], next_layers), next_layers)
    got = [m(norm(previous_layer(x))) for m in next_layers]

    for g, e in zip(got, expected):
        torch.testing.assert_close(g, e, atol=ATOL, rtol=0)


def test_rmsnorm_conversion_with_several_previous_layers():
    """The norm reads a sum of layers: centering the sum is centering each contribution."""
    dim = 8
    ln = make_layer_norm(dim)
    previous_layers = [make_layer("linear", 5, dim) for _ in range(3)]
    next_layer = make_layer("linear", dim, 3)
    ref_ln, ref_next = copy.deepcopy(ln), copy.deepcopy(next_layer)
    ref_previous = [copy.deepcopy(m) for m in previous_layers]

    x = torch.randn(4, 5, dtype=DTYPE)
    expected = ref_next(ref_ln(sum(m(x) for m in ref_previous)))

    norm = utils._rmsnorm(utils._RMSNorm(ln, previous_layers, [next_layer]), [next_layer])
    got = next_layer(norm(sum(m(x) for m in previous_layers)))

    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


def test_rmsnorm_conversion_with_no_next_layers_keeps_the_shift():
    """Nowhere to fold the shift, so the norm keeps it and stays output-equivalent."""
    dim = 8
    ln = make_layer_norm(dim)
    previous_layer = make_layer("linear", 5, dim)
    ref_ln, ref_previous = copy.deepcopy(ln), copy.deepcopy(previous_layer)

    x = torch.randn(4, 5, dtype=DTYPE)
    expected = ref_ln(ref_previous(x))

    norm = utils._RMSNorm(ln, [previous_layer], [])
    torch.testing.assert_close(norm(previous_layer(x)), expected, atol=ATOL, rtol=0)
    assert norm.shift is not None


def test_rmsnorm_conversion_with_no_previous_layers_leaves_them_alone():
    """Nowhere to fold the centering, so neighbours are untouched and the caller centres."""
    dim = 8
    ln = make_layer_norm(dim)
    previous_layer = make_layer("linear", 5, dim)
    before = previous_layer.weight.detach().clone()

    norm = utils._RMSNorm(ln, [], [make_layer("linear", dim, 3)])

    torch.testing.assert_close(previous_layer.weight, before, atol=0, rtol=0)
    assert norm.shift is None


def test_scale_fold_requires_next_layers():
    """_rmsnorm cannot drop a scale that has nowhere to go."""
    dim = 8
    norm = utils._RMSNorm(
        make_layer_norm(dim), [make_layer("linear", 5, dim)], [make_layer("linear", dim, 3)]
    )
    with pytest.raises(ValueError, match="nowhere to fold"):
        utils._rmsnorm(norm, [])


def test_RMSNorm_shift_fold_uses_the_unscaled_next_layer_weight():
    """Pins the ordering constraint: the shift is beta @ W2.T for the pre-scale W2."""
    dim = 8
    ln = make_layer_norm(dim)
    previous_layer, next_layer = make_layer("linear", 5, dim), make_layer("linear", dim, 3)
    expected_bias = next_layer.bias.detach() + ln.bias.detach() @ next_layer.weight.detach().T

    utils._RMSNorm(ln, [previous_layer], [next_layer])

    torch.testing.assert_close(next_layer.bias, expected_bias, atol=ATOL, rtol=0)


class NestedStub(nn.Module):
    """A ModuleList of layers, the shape real encoders use."""

    def __init__(self, dim, n=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.ModuleDict({"norm_out": make_layer_norm(dim)}) for _ in range(n)]
        )


@pytest.mark.parametrize("wrapped", [False, True])
def test_replace_module_handles_attributes_and_container_indices(wrapped):
    """After insert_linear_after_norm the norm lives at an index, not an attribute."""
    model = NestedStub(8).to(DTYPE)
    name = "layers.1.norm_out"
    if wrapped:
        name, linear_name = utils.insert_linear_after_norm(model, name)
        assert name.split(".")[-1].isdigit()

    utils.replace_module(model, name, nn.Identity())

    modules = dict(model.named_modules())
    assert isinstance(modules[name], nn.Identity)
    if wrapped:
        # the sibling Linear keeps its place, so the Sequential still runs in order
        assert isinstance(modules[linear_name], nn.Linear)
        assert list(model.layers[1].norm_out._modules) == ["0", "1"]
    # the other layers are untouched
    assert isinstance(modules["layers.0.norm_out"], nn.LayerNorm)


class TwoBlockStub(nn.Module):
    """The shape that matters: a block ending in a norm, feeding the next block's first norm."""

    def __init__(self, dim):
        super().__init__()
        self.ff2 = nn.Linear(dim, dim)
        self.norm_out = make_layer_norm(dim)
        self.norm_ff1 = make_layer_norm(dim)
        self.ff1 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.ff1(self.norm_ff1(self.norm_out(self.ff2(x))))


def test_insert_linear_after_norm_is_initially_the_identity():
    """The model's output must not change until something is folded into the new Linear."""
    model = TwoBlockStub(8).to(DTYPE)
    reference = copy.deepcopy(model)
    x = torch.randn(4, 8, dtype=DTYPE)

    norm_name, linear_name = utils.insert_linear_after_norm(model, "norm_out")

    assert (norm_name, linear_name) == ("norm_out.0", "norm_out.1")
    modules = dict(model.named_modules())
    assert isinstance(modules[norm_name], nn.LayerNorm)
    assert isinstance(modules[linear_name], nn.Linear)
    torch.testing.assert_close(model(x), reference(x), atol=ATOL, rtol=0)


def test_inserted_linear_absorbs_the_next_norms_centering():
    """The point of the insertion: the next norm now has a weight to fold its centering into."""
    dim = 8
    model = TwoBlockStub(dim).to(DTYPE)
    reference = copy.deepcopy(model)
    x = torch.randn(4, dim, dtype=DTYPE)
    expected = reference(x)

    norm_name, linear_name = utils.insert_linear_after_norm(model, "norm_out")
    inserted = dict(model.named_modules())[linear_name]

    # norm_out first, so its shift folds using the inserted Linear's pre-centering weight
    converted = utils._rmsnorm(
        utils._RMSNorm(model.norm_out[0], [model.ff2], [inserted]), [inserted]
    )
    model.norm_out[0] = converted
    model.norm_ff1 = utils._rmsnorm(
        utils._RMSNorm(model.norm_ff1, [inserted], [model.ff1]), [model.ff1]
    )

    torch.testing.assert_close(model(x), expected, atol=ATOL, rtol=0)
    assert not torch.allclose(inserted.weight, torch.eye(dim, dtype=DTYPE))


class ResidualStack(nn.Module):
    """Two pre-norm residual blocks then a final norm, the shape the centering has to survive."""

    def __init__(self, dim):
        super().__init__()
        self.b0_norm = make_layer_norm(dim)
        self.b0_in, self.b0_out = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.b1_norm = make_layer_norm(dim)
        self.b1_in, self.b1_out = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.final_norm = make_layer_norm(dim)

    def forward(self, x):
        x = x + self.b0_out(torch.relu(self.b0_in(self.b0_norm(x))))
        x = x + self.b1_out(torch.relu(self.b1_in(self.b1_norm(x))))
        return self.final_norm(x)


def test_centering_matches_multiplying_by_M():
    """x - mean(x) is x @ M, just cheaper."""
    dim = 16
    x = torch.randn(4, 7, dim, dtype=DTYPE)
    R1 = rot(dim)
    M = torch.eye(dim, dtype=DTYPE) - torch.full((dim, dim), 1.0 / dim, dtype=DTYPE)

    got = utils.center_and_rotate_residual_stream(x, R1)

    torch.testing.assert_close(got, (x @ M) @ R1, atol=ATOL, rtol=0)


def test_entry_and_exit_hooks_cancel_the_rotation():
    dim = 16
    x = torch.randn(4, dim, dtype=DTYPE)
    R1 = rot(dim)

    rotated = utils.center_and_rotate_residual_stream(x, R1)
    restored = utils.unrotate_residual_stream(rotated, R1)

    # the rotation is undone; the centering deliberately is not
    torch.testing.assert_close(restored, x - x.mean(-1, keepdim=True), atol=ATOL, rtol=0)


def test_entry_centering_covers_a_first_norm_with_no_previous_layer():
    """The residual-stream design: centering once on entry keeps every later norm equivalent.

    The first block's norm has previous=[] because the stream is not any one layer's output.
    Every later contribution is centered by the fold into the layer that wrote it, and a sum of
    centered vectors stays centered, so the whole stack reproduces the original.
    """
    dim = 16
    model = ResidualStack(dim).to(DTYPE)
    reference = copy.deepcopy(model)
    x = torch.randn(4, dim, dtype=DTYPE)
    expected = reference(x)

    utils.replace_module(
        model,
        "b0_norm",
        utils._rmsnorm(utils._RMSNorm(model.b0_norm, [], [model.b0_in]), [model.b0_in]),
    )
    utils.replace_module(
        model,
        "b1_norm",
        utils._rmsnorm(
            utils._RMSNorm(model.b1_norm, [model.b0_out], [model.b1_in]), [model.b1_in]
        ),
    )
    utils.replace_module(
        model, "final_norm", utils._RMSNorm(model.final_norm, [model.b1_out], [])
    )

    centered = x - x.mean(-1, keepdim=True)
    torch.testing.assert_close(model(centered), expected, atol=ATOL, rtol=0)


def test_rmsnorm_is_only_rotation_equivariant_without_scale_or_shift():
    """Why a norm that keeps its scale cannot sit inside the rotated region.

    Scaling elementwise is multiplying by diag(gamma), which does not commute with R1. A norm
    with no next layer to absorb its scale must therefore have the stream rotated back on the way
    into it, which is what add_unrotate_input_hook is for.
    """
    dim = 8
    x = torch.randn(4, dim, dtype=DTYPE)
    R1 = rot(dim)
    gamma = torch.randn(dim, dtype=DTYPE).abs() + 0.5
    beta = torch.randn(dim, dtype=DTYPE)

    def normed(t):
        return t / t.pow(2).mean(-1, keepdim=True).sqrt()

    # scale-free: equivariant
    torch.testing.assert_close(normed(x @ R1), normed(x) @ R1, atol=ATOL, rtol=0)
    # with a scale or a shift: not equivariant
    assert not torch.allclose(normed(x @ R1) * gamma, (normed(x) * gamma) @ R1, atol=1e-6)
    assert not torch.allclose(normed(x @ R1) + beta, (normed(x) + beta) @ R1, atol=1e-6)
    # and the reason: rotating diag(gamma) gives a matrix, not a vector
    torch.testing.assert_close(
        (normed(x) * gamma) @ R1, normed(x) @ (torch.diag(gamma) @ R1), atol=ATOL, rtol=0
    )


@pytest.mark.parametrize("keyword", [None, "x"])
def test_unrotate_input_hook_leaves_the_rotated_basis_before_a_module(keyword):
    dim = 16
    R1 = rot(dim)
    x = torch.randn(4, dim, dtype=DTYPE)

    class Identity(nn.Module):
        def forward(self, x=None, other=None):
            return x

    module = Identity()
    utils.add_unrotate_input_hook(module, R1, keyword=keyword)

    got = module(x=x) if keyword else module(x)

    torch.testing.assert_close(got, utils.unrotate_residual_stream(x, R1), atol=ATOL, rtol=0)


def test_entry_hook_then_unrotate_hook_restores_the_centered_stream():
    """The pair used by every model: center+rotate on entry, unrotate before the last norm."""
    dim = 16
    R1 = rot(dim)
    x = torch.randn(4, dim, dtype=DTYPE)

    class Passthrough(nn.Module):
        def forward(self, x):
            return x

    entry, last = Passthrough(), Passthrough()
    utils.add_residual_stream_entry_hook(entry, R1)
    utils.add_unrotate_input_hook(last, R1)

    torch.testing.assert_close(
        last(entry(x)), x - x.mean(-1, keepdim=True), atol=ATOL, rtol=0
    )


# --------------------------------------------------------------------------------------
# rotation folds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("d_in,d_out", [(8, 8), (8, 5)])
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_fold_q_k_fc1(kind, d_in, d_out):
    """R1 on the input: W <- W @ R1, bias untouched."""
    layer = make_layer(kind, d_in, d_out)
    ref = copy.deepcopy(layer)
    R1 = rot(d_in)
    x = make_input(kind, d_in)

    expected = ref(rotate_features(kind, x, R1, transpose=True))
    original_bias = layer.bias.detach().clone()
    utils.fold_rotation_into_Q_K_FC1(layer, R1)

    torch.testing.assert_close(layer(x), expected, atol=ATOL, rtol=0)
    torch.testing.assert_close(layer.bias, original_bias, atol=0, rtol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_in,d_out", [(8, 8), (8, 5)])
def test_fold_v(d_in, d_out, bias):
    """R1 on the input, R2 on the output: W <- R2.T @ W @ R1, b <- b @ R2."""
    layer = make_layer("linear", d_in, d_out, bias=bias)
    ref = copy.deepcopy(layer)
    R1, R2 = rot(d_in), rot(d_out)
    x = make_input("linear", d_in)

    expected = ref(x @ R1.T) @ R2
    utils.fold_rotation_into_V(layer, R1, R2)

    torch.testing.assert_close(layer(x), expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_in,d_out", [(8, 8), (8, 5)])
def test_fold_o(d_in, d_out, bias):
    """R2 on the input, R1 on the output: W <- R1.T @ W @ R2, b <- b @ R1."""
    layer = make_layer("linear", d_in, d_out, bias=bias)
    ref = copy.deepcopy(layer)
    R1, R2 = rot(d_out), rot(d_in)
    x = make_input("linear", d_in)

    expected = ref(x @ R2.T) @ R1
    utils.fold_rotation_into_O(layer, R1, R2)

    torch.testing.assert_close(layer(x), expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_in,d_out", [(8, 8), (5, 8)])
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_fold_fc2(kind, d_in, d_out, bias):
    """R1 on the output: W <- R1.T @ W, b <- b @ R1."""
    layer = make_layer(kind, d_in, d_out, bias=bias)
    ref = copy.deepcopy(layer)
    R1 = rot(d_out)
    x = make_input(kind, d_in)

    expected = rotate_features(kind, ref(x), R1)
    utils.fold_rotation_into_FC2(layer, R1)

    torch.testing.assert_close(layer(x), expected, atol=ATOL, rtol=0)


def fold_v_o_pair(v, o, R1, R2):
    utils.fold_rotation_into_V(v, R1, R2)
    utils.fold_rotation_into_O(o, R1, R2)
    return lambda x: o(v(x))


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize(
    "fold,n_rotations",
    [
        (utils.fold_rotation_into_Q_K_FC1, 1),
        (utils.fold_rotation_into_FC2, 1),
    ],
    ids=["q_k_fc1", "fc2"],
)
def test_fold_preserves_weight_dtype_and_shape(fold, n_rotations, kind, weight_dtype):
    """A float64 rotation must not promote the layer it is folded into.

    The folds write through Tensor.copy_, which casts into existing storage; switching any of
    them to `weight.data = ...` would silently change the layer's dtype instead.
    """
    dim = 8
    layer = make_layer(kind, dim, dim).to(weight_dtype)
    weight_shape, bias_shape = layer.weight.shape, layer.bias.shape

    fold(layer, *[rot(dim) for _ in range(n_rotations)])

    assert layer.weight.dtype == weight_dtype
    assert layer.bias.dtype == weight_dtype
    assert layer.weight.shape == weight_shape
    assert layer.bias.shape == bias_shape
    assert isinstance(layer.weight, nn.Parameter)
    assert isinstance(layer.bias, nn.Parameter)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "fold", [utils.fold_rotation_into_V, utils.fold_rotation_into_O], ids=["v", "o"]
)
def test_two_sided_fold_preserves_weight_dtype(fold, weight_dtype):
    dim = 8
    layer = make_layer("linear", dim, dim).to(weight_dtype)

    fold(layer, rot(dim), rot(dim))

    assert layer.weight.dtype == weight_dtype
    assert layer.bias.dtype == weight_dtype


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16])
def test_rmsnorm_conversion_preserves_neighbour_dtypes(weight_dtype):
    """_RMSNorm assigns to .data rather than using copy_, so this pins the dtypes."""
    dim = 8
    ln = nn.LayerNorm(dim).to(weight_dtype)
    previous_layer = make_layer("linear", 5, dim).to(weight_dtype)
    next_layer = make_layer("linear", dim, 3).to(weight_dtype)

    utils._RMSNorm(ln, [previous_layer], [next_layer])

    assert previous_layer.weight.dtype == weight_dtype
    assert previous_layer.bias.dtype == weight_dtype
    assert next_layer.bias.dtype == weight_dtype


def head_blockdiag(R2, num_heads):
    """The matrix _rotate_heads applies implicitly, built explicitly for comparison."""
    return torch.block_diag(*([R2] * num_heads))


@pytest.mark.parametrize("d_model,num_heads", [(64, 4), (64, 8), (128, 4)])
def test_fold_v_applies_R2_per_head(d_model, num_heads):
    """R2 is head_dim-sized and acts on each head, i.e. as blockdiag(R2, ...) on the output."""
    head_dim = d_model // num_heads
    layer = make_layer("linear", d_model, d_model)
    ref = copy.deepcopy(layer)
    R1, R2 = rot(d_model), rot(head_dim)
    x = make_input("linear", d_model)

    expected = ref(x @ R1.T) @ head_blockdiag(R2, num_heads)
    utils.fold_rotation_into_V(layer, R1, R2)

    torch.testing.assert_close(layer(x), expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("d_model,num_heads", [(64, 4), (64, 8), (128, 4)])
def test_fold_o_applies_R2_per_head(d_model, num_heads):
    head_dim = d_model // num_heads
    layer = make_layer("linear", d_model, d_model)
    ref = copy.deepcopy(layer)
    R1, R2 = rot(d_model), rot(head_dim)
    x = make_input("linear", d_model)

    expected = ref(x @ head_blockdiag(R2, num_heads).T) @ R1
    utils.fold_rotation_into_O(layer, R1, R2)

    torch.testing.assert_close(layer(x), expected, atol=ATOL, rtol=0)


@pytest.mark.parametrize("d_model,num_heads", [(64, 4), (128, 8)])
def test_head_wise_R2_cancels_between_v_and_o(d_model, num_heads):
    """The pair's output must not depend on R2 at all, whatever the head count."""
    head_dim = d_model // num_heads
    v, o = make_layer("linear", d_model, d_model), make_layer("linear", d_model, d_model)
    R1 = rot(d_model)
    x = make_input("linear", d_model)
    expected = copy.deepcopy(o)(copy.deepcopy(v)(x @ R1.T)) @ R1

    outputs = []
    for _ in range(2):
        R2 = rot(head_dim)
        vi, oi = copy.deepcopy(v), copy.deepcopy(o)
        utils.fold_rotation_into_V(vi, R1, R2)
        utils.fold_rotation_into_O(oi, R1, R2)
        outputs.append(oi(vi(x)))

    torch.testing.assert_close(outputs[0], expected, atol=ATOL, rtol=0)
    torch.testing.assert_close(outputs[0], outputs[1], atol=ATOL, rtol=0)


@pytest.mark.parametrize("d_model,num_heads", [(64, 4), (64, 8)])
def test_head_wise_patches_match_their_folds(d_model, num_heads):
    head_dim = d_model // num_heads
    R1, R2 = rot(d_model), rot(head_dim)
    x = make_input("linear", d_model)
    for fold, patch in [
        (utils.fold_rotation_into_V, utils.patch_v_with_rotation),
        (utils.fold_rotation_into_O, utils.patch_o_with_rotation),
    ]:
        base = make_layer("linear", d_model, d_model)
        folded, patched = copy.deepcopy(base), copy.deepcopy(base)
        fold(folded, R1, R2)
        patch(patched, R1, R2)
        torch.testing.assert_close(patched(x), folded(x), atol=0, rtol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_model", [8, 64, 128])
def test_rotation_type_5_via_patch_o_with_R2_equal_to_R1(d_model, bias):
    """A layer inside the residual stream: rotated input and rotated output.

    learn_rotations dispatches rotation type 5 to patch_o_with_rotation with R2 set to R1. That
    works because _rotate_heads with a head_dim as wide as the axis is a single block, so the
    head-wise fold degenerates to the full-width one.
    """
    layer = make_layer("linear", d_model, d_model, bias=bias)
    ref = copy.deepcopy(layer)
    R1 = rot(d_model)
    x = make_input("linear", d_model)

    utils.patch_o_with_rotation(layer, R1, R1)

    torch.testing.assert_close(layer(x @ R1), ref(x) @ R1, atol=ATOL, rtol=0)


def test_rotate_heads_with_full_width_R_is_a_plain_matmul():
    d_model = 64
    R1 = rot(d_model)
    w = torch.randn(d_model, d_model, dtype=DTYPE)
    torch.testing.assert_close(utils._rotate_heads(w, R1), w @ R1, atol=ATOL, rtol=0)


def test_rotate_heads_rejects_an_indivisible_axis():
    with pytest.raises(ValueError, match="not divisible by head_dim"):
        utils._rotate_heads(torch.randn(4, 30, dtype=DTYPE), rot(16))


def test_v_and_o_compose_to_the_original_pair_in_the_rotated_basis():
    """o_new(v_new(x)) == o(v(x @ R1.T)) @ R1: only the residual rotation survives."""
    dim = 8
    v, o = make_layer("linear", dim, dim), make_layer("linear", dim, dim)
    ref_v, ref_o = copy.deepcopy(v), copy.deepcopy(o)
    R1, R2 = rot(dim), rot(dim)
    x = make_input("linear", dim)

    expected = ref_o(ref_v(x @ R1.T)) @ R1
    got = fold_v_o_pair(v, o, R1, R2)(x)

    torch.testing.assert_close(got, expected, atol=ATOL, rtol=0)


def test_v_and_o_rotations_cancel_regardless_of_R2():
    """O undoes exactly the R2 that V applies, so the pair's output cannot depend on it."""
    dim = 8
    v, o = make_layer("linear", dim, dim), make_layer("linear", dim, dim)
    R1 = rot(dim)
    x = make_input("linear", dim)

    first = fold_v_o_pair(copy.deepcopy(v), copy.deepcopy(o), R1, rot(dim))(x)
    second = fold_v_o_pair(copy.deepcopy(v), copy.deepcopy(o), R1, rot(dim))(x)

    torch.testing.assert_close(first, second, atol=ATOL, rtol=0)


# --------------------------------------------------------------------------------------
# rotation patches  (forward-time equivalents, for searching R with Cayley SGD)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_in,d_out", [(8, 8), (8, 5)])
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_patch_q_k_fc1_matches_fold(kind, d_in, d_out, bias):
    layer = make_layer(kind, d_in, d_out, bias=bias)
    folded, patched = copy.deepcopy(layer), copy.deepcopy(layer)
    R1 = rot(d_in)
    x = make_input(kind, d_in)

    utils.fold_rotation_into_Q_K_FC1(folded, R1)
    utils.patch_q_k_fc1_with_rotation(patched, R1)

    torch.testing.assert_close(patched(x), folded(x), atol=0, rtol=0)
    torch.testing.assert_close(patched.weight, layer.weight, atol=0, rtol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_in,d_out", [(8, 8), (8, 5)])
def test_patch_v_matches_fold(d_in, d_out, bias):
    layer = make_layer("linear", d_in, d_out, bias=bias)
    folded, patched = copy.deepcopy(layer), copy.deepcopy(layer)
    R1, R2 = rot(d_in), rot(d_out)
    x = make_input("linear", d_in)

    utils.fold_rotation_into_V(folded, R1, R2)
    utils.patch_v_with_rotation(patched, R1, R2)

    torch.testing.assert_close(patched(x), folded(x), atol=0, rtol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_in,d_out", [(8, 8), (8, 5)])
def test_patch_o_matches_fold(d_in, d_out, bias):
    layer = make_layer("linear", d_in, d_out, bias=bias)
    folded, patched = copy.deepcopy(layer), copy.deepcopy(layer)
    R1, R2 = rot(d_out), rot(d_in)
    x = make_input("linear", d_in)

    utils.fold_rotation_into_O(folded, R1, R2)
    utils.patch_o_with_rotation(patched, R1, R2)

    torch.testing.assert_close(patched(x), folded(x), atol=0, rtol=0)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("d_in,d_out", [(8, 8), (5, 8)])
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_patch_fc2_matches_fold(kind, d_in, d_out, bias):
    layer = make_layer(kind, d_in, d_out, bias=bias)
    folded, patched = copy.deepcopy(layer), copy.deepcopy(layer)
    R1 = rot(d_out)
    x = make_input(kind, d_in)

    utils.fold_rotation_into_FC2(folded, R1)
    utils.patch_fc2_with_rotation(patched, R1)

    torch.testing.assert_close(patched(x), folded(x), atol=0, rtol=0)


def test_patch_rejects_non_pointwise_conv():
    with pytest.raises(ValueError, match="pointwise"):
        utils.patch_q_k_fc1_with_rotation(nn.Conv1d(8, 8, kernel_size=3), rot(8))


def patch_whole_block(dim, R1, R2):
    """Patch one transformer block's worth of layers with the shared rotations."""
    layers = {name: make_layer("linear", dim, dim) for name in ("q", "k", "v", "o", "fc1", "fc2")}
    for layer in layers.values():
        layer.weight.requires_grad_(False)
        layer.bias.requires_grad_(False)
    for name in ("q", "k", "fc1"):
        utils.patch_q_k_fc1_with_rotation(layers[name], R1)
    utils.patch_v_with_rotation(layers["v"], R1, R2)
    utils.patch_o_with_rotation(layers["o"], R1, R2)
    utils.patch_fc2_with_rotation(layers["fc2"], R1)
    return layers


def test_shared_rotation_parameters_receive_one_gradient_each():
    """The SpinQuant case: R1 and R2 are single shared parameters across the whole block."""
    dim = 8
    R1, R2 = nn.Parameter(rot(dim)), nn.Parameter(rot(dim))
    layers = patch_whole_block(dim, R1, R2)

    x = make_input("linear", dim)
    sum(layer(x).pow(2).sum() for layer in layers.values()).backward()

    assert R1.grad is not None and R1.grad.shape == (dim, dim)
    assert R2.grad is not None and R2.grad.shape == (dim, dim)
    assert all(layer.weight.grad is None for layer in layers.values())


def test_patching_does_not_add_state_dict_entries():
    """Rotations are held by reference, never registered, so checkpoints stay loadable."""
    dim = 8
    layers = patch_whole_block(dim, nn.Parameter(rot(dim)), nn.Parameter(rot(dim)))
    for layer in layers.values():
        assert sorted(layer.state_dict()) == ["bias", "weight"]
        nn.Linear(dim, dim).to(DTYPE).load_state_dict(layer.state_dict())


def test_patched_layer_tracks_in_place_updates_to_the_rotation():
    """A Cayley step updates R in place; every patched layer must see it."""
    dim = 8
    R1 = nn.Parameter(rot(dim))
    layer = make_layer("linear", dim, dim)
    utils.patch_q_k_fc1_with_rotation(layer, R1)

    x = make_input("linear", dim)
    before = layer(x).clone()
    with torch.no_grad():
        R1.copy_(rot(dim))

    assert not torch.allclose(layer(x), before)


@pytest.mark.parametrize(
    "patch,rotations",
    [
        (lambda layer, r: utils.patch_q_k_fc1_with_rotation(layer, r[0]), 1),
        (lambda layer, r: utils.patch_v_with_rotation(layer, r[0], r[1]), 2),
        (lambda layer, r: utils.patch_o_with_rotation(layer, r[0], r[1]), 2),
        (lambda layer, r: utils.patch_fc2_with_rotation(layer, r[0]), 1),
    ],
    ids=["q_k_fc1", "v", "o", "fc2"],
)
def test_rotation_matmul_runs_in_the_promoted_dtype(patch, rotations):
    """A float64 rotation over a float32 model must not be downcast for the matmul.

    Compared against a float64 reference: casting R to float32 first would round the rotation
    before the product and land a visible distance away.
    """
    dim = 8
    rots = [rot(dim) for _ in range(rotations)]  # float64
    layer = nn.Linear(dim, dim).to(torch.float32)
    exact, rounded = copy.deepcopy(layer).to(DTYPE), copy.deepcopy(layer)

    patch(layer, rots)
    patch(exact, rots)
    patch(rounded, [r.to(torch.float32) for r in rots])

    x = torch.randn(4, dim, dtype=torch.float32)
    got, ref = layer(x), exact(x.to(DTYPE))

    assert got.dtype == torch.float32
    torch.testing.assert_close(got.to(DTYPE), ref, atol=1e-6, rtol=0)
    assert not torch.equal(got, rounded(x))


@pytest.mark.parametrize(
    "fold,patch,n_rotations",
    [
        (utils.fold_rotation_into_Q_K_FC1, utils.patch_q_k_fc1_with_rotation, 1),
        (utils.fold_rotation_into_V, utils.patch_v_with_rotation, 2),
        (utils.fold_rotation_into_O, utils.patch_o_with_rotation, 2),
        (utils.fold_rotation_into_FC2, utils.patch_fc2_with_rotation, 1),
    ],
    ids=["q_k_fc1", "v", "o", "fc2"],
)
def test_fold_reproduces_patch_in_mixed_precision(fold, patch, n_rotations):
    """Baking a searched float64 rotation into a float32 model must not change its output.

    Both paths promote to float64 for the product and round once, so the folded model is
    bit-identical to the patched model that was searched.
    """
    dim = 8
    rots = [rot(dim) for _ in range(n_rotations)]  # float64
    layer = nn.Linear(dim, dim).to(torch.float32)
    folded, patched = copy.deepcopy(layer), copy.deepcopy(layer)

    fold(folded, *rots)
    patch(patched, *rots)

    x = torch.randn(4, dim, dtype=torch.float32)
    torch.testing.assert_close(folded(x), patched(x), atol=0, rtol=0)


def test_gradient_reaches_rotation_through_a_dtype_cast():
    """R may be float32 while the layer is float64; .to() must stay differentiable."""
    dim = 8
    R1 = nn.Parameter(rot(dim).float())
    layer = make_layer("linear", dim, dim)
    layer.weight.requires_grad_(False)
    layer.bias.requires_grad_(False)
    utils.patch_q_k_fc1_with_rotation(layer, R1)

    layer(make_input("linear", dim)).pow(2).sum().backward()

    assert R1.grad is not None
    assert R1.grad.dtype == torch.float32


def test_deepcopy_of_patched_layer_uses_its_own_weights():
    """Bound via types.MethodType, so copy.deepcopy rebinds __self__ to the copy."""
    dim = 8
    R1 = nn.Parameter(rot(dim))
    layer = make_layer("linear", dim, dim)
    utils.patch_q_k_fc1_with_rotation(layer, R1)

    duplicate = copy.deepcopy(layer)
    with torch.no_grad():
        duplicate.weight.mul_(100.0)

    x = make_input("linear", dim)
    assert not torch.allclose(duplicate(x), layer(x))
    assert duplicate.forward.__self__ is duplicate


def test_patching_one_layer_does_not_affect_other_instances():
    """forward is shadowed per instance, not replaced on nn.Linear."""
    dim = 8
    original_forward = nn.Linear.forward
    patched, untouched = make_layer("linear", dim, dim), make_layer("linear", dim, dim)
    utils.patch_q_k_fc1_with_rotation(patched, rot(dim))

    x = make_input("linear", dim)
    torch.testing.assert_close(
        untouched(x),
        torch.nn.functional.linear(x, untouched.weight, untouched.bias),
        atol=0,
        rtol=0,
    )
    assert nn.Linear.forward is original_forward
    assert "forward" not in untouched.__dict__


def _checkpoint(activation=None, search=None, model=None):
    return {"R1": torch.eye(4), "R2s": {}, "activation_quantization": activation, "search": search, "model": model}


def test_settings_matching_the_search_are_accepted():
    check_rotation_settings = utils.check_rotation_settings

    activation = {"bits": 4, "group_size": 128, "symmetric": True, "groupwise_roles": ["attn_out", "fc2"]}
    checkpoint = _checkpoint(activation)
    check_rotation_settings("x.pt", dict(activation), checkpoint=checkpoint)
    check_rotation_settings("x.pt", {**activation, "groupwise_roles": ["fc2", "attn_out"]}, checkpoint=checkpoint)
    weights = {"bits": 2, "group_size": 128, "symmetric": False}
    weight_only = _checkpoint({"bits": None, "group_size": 128, "symmetric": True, "groupwise_roles": None},
                              {"quantization": "weights", "weight_quantization": dict(weights)})
    check_rotation_settings("x.pt", {"bits": 16, "group_size": 128, "symmetric": True, "groupwise_roles": None},
                            weights, checkpoint=weight_only)


@pytest.mark.parametrize("override, message", [
    ({"bits": 8}, "activation bits"),
    ({"symmetric": False}, "activation symmetric"),
    ({"group_size": 64}, "activation group_size"),
    ({"groupwise_roles": ["fc2"]}, "group-wise roles"),
])
def test_a_rotation_searched_under_other_activation_settings_is_refused(override, message):
    check_rotation_settings = utils.check_rotation_settings

    activation = {"bits": 4, "group_size": 128, "symmetric": True, "groupwise_roles": ["attn_out", "fc2"]}
    with pytest.raises(ValueError, match=message):
        check_rotation_settings("x.pt", {**activation, **override}, checkpoint=_checkpoint(activation))


def test_a_weight_only_rotation_is_refused_for_another_weight_grid():
    check_rotation_settings = utils.check_rotation_settings

    checkpoint = _checkpoint({"bits": None, "group_size": 128, "symmetric": True, "groupwise_roles": None},
                             {"quantization": "weights", "weight_quantization": {"bits": 2, "group_size": 128, "symmetric": False}})
    activation = {"bits": 16, "group_size": 128, "symmetric": True, "groupwise_roles": None}
    with pytest.raises(ValueError, match="weight bits"):
        check_rotation_settings("x.pt", activation, {"bits": 4, "group_size": 128, "symmetric": False}, checkpoint=checkpoint)


def test_a_rotation_searched_for_another_model_is_refused():
    check_rotation_settings = utils.check_rotation_settings

    activation = {"bits": None, "group_size": 128, "symmetric": True, "groupwise_roles": None}
    checkpoint = _checkpoint(activation, model="nvidia/parakeet-ctc-1.1b")
    check_rotation_settings("x.pt", dict(activation), checkpoint=checkpoint, model_name="nvidia/parakeet-ctc-1.1b")
    check_rotation_settings("x.pt", dict(activation), checkpoint=_checkpoint(activation), model_name="openai/whisper-large-v3")
    with pytest.raises(ValueError, match="was searched for nvidia/parakeet-ctc-1.1b"):
        check_rotation_settings("x.pt", dict(activation), checkpoint=checkpoint, model_name="openai/whisper-large-v3")
