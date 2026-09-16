"""The (1 + lambda) sign search with multi-stage selection, on synthetic fitness functions.

The model-facing half -- the search inside learn_rotations, and the invariance of the KL to s2 under
symmetric quantization -- is tested in test_rotation_e2e.py.
"""

import pytest
import torch
import torch.nn as nn

from asrq.quantizers.gptq import GPTQQuantizer
from asrq.quantizers.gptq_solver import add_to_hessian, gptq_factors, gptq_factors_batched, gptq_quantize, gptq_quantize_batched
from asrq.quantizers.rtn import RTNQuantizer
from asrq.quantizers.weight_rounding import round_weight
from asrq.transforms.rotation.hadamard_search import (
    EvolutionConfig,
    default_stage_samples,
    evolve_signs,
    hadamard_basis,
    mutate_signs,
    random_sign_vectors,
    signed_hadamard,
    stage_cost,
)


@pytest.mark.parametrize("samples, stages", [(64, (8, 16, 64)), (256, (8, 64, 256)), (768, (16, 64, 768)), (8, (8, 8, 8))])
def test_default_stages_follow_the_calibration_size(samples, stages):
    assert default_stage_samples(samples) == stages


@pytest.mark.parametrize("width, block", [(1280, None), (1024, None), (256, 64)])
def test_a_signed_hadamard_is_orthogonal(width, block):
    signs = random_sign_vectors({None: width}, torch.Generator().manual_seed(0), "cpu")[None]
    R1 = signed_hadamard(hadamard_basis(width, block, "cpu"), *signs)
    assert (R1.T @ R1 - torch.eye(width, dtype=torch.float64)).abs().max() < 1e-10
    if block is not None:
        blocks = torch.arange(width) // block
        assert (R1[blocks.unsqueeze(0) != blocks.unsqueeze(1)] == 0).all()


@pytest.mark.parametrize("mutate", ["s1_s2", "s1"])
def test_a_mutation_flips_exactly_the_requested_positions(mutate):
    generator = torch.Generator().manual_seed(0)
    parent = random_sign_vectors({"encoder": 16, "llm": 32}, generator, "cpu")
    flipped = {"s1": 0, "s2": 0}
    total = 0
    for _ in range(200):
        child = mutate_signs(parent, 3, mutate, generator)
        changes = 0
        for stream in parent:
            for which, name in ((0, "s1"), (1, "s2")):
                changed = int((child[stream][which] != parent[stream][which]).sum())
                flipped[name] += changed
                changes += changed
        assert changes == 3
        total += changes
    assert torch.equal(parent["encoder"][0], random_sign_vectors({"encoder": 16, "llm": 32}, torch.Generator().manual_seed(0), "cpu")["encoder"][0])
    if mutate == "s1":
        assert flipped["s2"] == 0
    else:
        assert 0.3 < flipped["s2"] / total < 0.7


def _target_problem(width=64, seed=0):
    """Fitness: the fraction of s1 positions that differ from a hidden target, plus a per-batch offset shared
    by every candidate scored on those batches."""
    target = random_sign_vectors({None: width}, torch.Generator().manual_seed(seed + 1), "cpu")[None][0]
    calls = []

    def fitness(signs, batches):
        calls.append(len(batches))
        mismatch = float((signs[None][0] != target).float().mean())
        return mismatch + 0.001 * sum(batches) / len(batches)

    return fitness, calls


def test_the_search_improves_and_never_accepts_a_worse_child():
    fitness, calls = _target_problem()
    config = EvolutionConfig(generations=40, offspring=8, survivors=(4, 2), stage_samples=(8, 16, 64), flips=1, mutate="s1")
    parent = random_sign_vectors({None: 64}, torch.Generator().manual_seed(0), "cpu")
    _, final = evolve_signs(parent, fitness, [4] * 16, config, log=lambda _line: None)
    trajectory = [record["fitness"] for record in config.history]
    assert all(later <= earlier for earlier, later in zip(trajectory, trajectory[1:]))
    assert final == trajectory[-1] < trajectory[0] - 0.2
    assert any(record["accepted"] for record in config.history[1:])


def test_each_stage_scores_its_share_of_the_data():
    fitness, calls = _target_problem()
    config = EvolutionConfig(generations=2, offspring=8, survivors=(3, 2), stage_samples=(8, 16, 64), mutate="s1_s2")
    evolve_signs(random_sign_vectors({None: 64}, torch.Generator().manual_seed(0), "cpu"), fitness, [4] * 16, config,
                 log=lambda _line: None)
    assert calls == [16] + [2] * 8 + [4] * 3 + [16] * 2 + [2] * 8 + [4] * 3 + [16] * 2
    assert stage_cost(config, 64) == 8 * 8 + 3 * 16 + 2 * 64


@pytest.mark.parametrize("mutate, symmetric, resolved", [
    ("auto", True, "s1"), ("auto", False, "s1_s2"), ("s1_s2", True, "s1_s2"), ("s1", False, "s1"),
])
def test_auto_mutates_s2_only_for_asymmetric_quantization(mutate, symmetric, resolved):
    config = EvolutionConfig(mutate=mutate)
    config.resolve_mutate(symmetric)
    assert config.mutate == resolved == config.settings()["mutate"]


def test_an_unresolved_auto_is_rejected_by_the_search():
    fitness, _ = _target_problem()
    with pytest.raises(ValueError, match="resolve_mutate"):
        evolve_signs(random_sign_vectors({None: 64}, torch.Generator().manual_seed(0), "cpu"), fitness, [4] * 16,
                     EvolutionConfig(generations=1), log=lambda _line: None)


@pytest.mark.parametrize("settings", [
    {"survivors": (4, 8)}, {"offspring": 2, "survivors": (4, 2)}, {"mutate": "s2"}, {"flips": 0},
    {"stage_samples": (8, 16)},
])
def test_inconsistent_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        EvolutionConfig(**settings)


@pytest.mark.parametrize("symmetric", [True, False])
def test_weight_rounding_sees_s2_on_the_input_side_only(symmetric):
    """W @ R1 and R1.T @ W (the two folds of a residual-stream rotation) rounded as the weight quantizers do.

    diag(s2) negates whole rows of R1.T @ W, mirroring each rounding group, which both grids commute with. It
    negates single entries of W @ R1, which neither does: the asymmetric grid is offset by the group minimum,
    and the symmetric one has an extra negative code on whichever extreme dominates.
    """
    torch.manual_seed(0)
    W = torch.randn(96, 64, dtype=torch.float64)
    H = hadamard_basis(64, None, "cpu")
    (s1, s2), = random_sign_vectors({None: 64}, torch.Generator().manual_seed(0), "cpu").values()
    flipped = s2 * torch.where(torch.rand(64) < 0.5, -1.0, 1.0)

    def error(s2, side):
        R1 = signed_hadamard(H, s1, s2)
        rotated = W @ R1 if side == "input" else R1.T @ W.T
        quantized = round_weight(rotated, 2, 16, symmetric)[0]
        restored = quantized @ R1.T if side == "input" else (R1 @ quantized).T
        return float((restored - W).norm())

    assert abs(error(s2, "output") - error(flipped, "output")) < 1e-9 * error(s2, "output")
    assert abs(error(s2, "input") - error(flipped, "input")) > 1e-4 * error(s2, "input")


def test_round_weight_is_the_rtn_quantizer():
    class Config:
        name, bits, group_size, symmetric, exclude_modules = "rtn", 2, 16, False, []

    torch.manual_seed(0)
    for symmetric in (True, False):
        Config.symmetric = symmetric
        layer = nn.Linear(64, 32)
        expected = round_weight(layer.weight.detach().clone(), 2, 16, symmetric)[0]
        RTNQuantizer(layer, "layer", Config())()
        assert torch.equal(layer.weight.detach(), expected)


@pytest.mark.parametrize("symmetric", [True, False])
def test_the_gptq_quantizer_is_the_solver_on_its_hessian(symmetric):

    class Config:
        name, bits, group_size, block_size, percdamp, exclude_modules = "gptq", 2, 16, 16, 0.01, []

    Config.symmetric = symmetric
    torch.manual_seed(0)
    layer = nn.Linear(64, 48)
    X = torch.randn(4, 30, 64)
    X[..., 5] = 0
    H, _ = add_to_hessian(torch.zeros(64, 64), 0, X.reshape(-1, 64))
    expected = gptq_quantize(layer.weight.detach().clone(), gptq_factors(H, 0.01), 2, 16, symmetric, 16)
    quantizer = GPTQQuantizer(layer, "layer", Config())
    quantizer.add_batch((X, None))
    quantizer()
    assert torch.equal(layer.weight.detach(), expected)


@pytest.mark.parametrize("symmetric", [True, False])
def test_batched_gptq_is_per_layer_gptq(symmetric):
    torch.manual_seed(0)
    hessians, weights = [], torch.randn(4, 48, 32)
    for layer in range(4):
        X = torch.randn(200, 32)
        X[:, layer] = 0
        hessians.append(X.T @ X / 200)
    hessians = torch.stack(hessians)
    batched = gptq_quantize_batched(weights, gptq_factors_batched(hessians, 0.01), 2, 8, symmetric, 8)
    for layer in range(4):
        single = gptq_quantize(weights[layer], gptq_factors(hessians[layer], 0.01), 2, 8, symmetric, 8)
        assert torch.allclose(batched[layer], single, atol=1e-6)
