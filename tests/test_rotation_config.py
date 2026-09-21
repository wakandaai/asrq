"""The rotation transform's config: what each setting resolves to before a search runs."""

import pytest

from _pipeline import config
from asrq.experiment import prepare_experiment_config
from asrq.transforms.rotation.base import RotationTransformConfig


def _rotation_cfg(*overrides):
    cfg = config("whisper", list(overrides))
    prepare_experiment_config(cfg, learn_rotation=True)
    return RotationTransformConfig(cfg.transform)


@pytest.mark.parametrize("search, objective, resolved", [
    ("cayley", "auto", "ce"), ("evolution", "auto", "kl"), ("cayley", "kl", "kl"), ("cayley", "ce", "ce"),
])
def test_auto_picks_the_objective_each_search_uses(search, objective, resolved):
    assert _rotation_cfg(f"transform.search={search}", f"transform.objective={objective}").objective == resolved


def test_the_evolutionary_search_refuses_the_cross_entropy_objective():
    with pytest.raises(ValueError, match="minimises the KL divergence"):
        _rotation_cfg("transform.search=evolution", "transform.objective=ce")


def test_an_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="must be 'kl', 'ce' or 'auto'"):
        _rotation_cfg("transform.objective=mse")


@pytest.mark.parametrize("module, builder", [
    ("whisper_utils", "build_whisper_dataloader"),
    ("parakeet_ctc_utils", "build_parakeet_dataloader"),
    ("canary_qwen_utils", "build_canary_qwen_dataloader"),
])
def test_every_loader_takes_the_arguments_the_rotation_transform_passes(module, builder):
    """The transform passes batch_size and sort_by_length to whichever loader the model uses, and an evolution
    search is the only caller that sets sort_by_length, so a missing parameter only surfaces there."""
    import importlib
    import inspect

    function = getattr(importlib.import_module(f"asrq.transforms.rotation.{module}"), builder)
    parameters = inspect.signature(function).parameters
    assert {"batch_size", "seed", "sort_by_length"} <= set(parameters)
    source = inspect.getsource(function)
    assert "length_sorted_batches" in source and "sort_by_length" in source
