"""The rotation transform's config: what each setting resolves to before a search runs."""

import pytest

from _pipeline import config
from asrq.experiment import prepare_experiment_config
from asrq.transforms.rotation.base import RotationTransformConfig


def _rotation_cfg(*overrides):
    cfg = config("whisper", list(overrides))
    prepare_experiment_config(cfg, learn_rotation=True)
    return RotationTransformConfig(cfg.transform)


@pytest.mark.parametrize("objective, resolved", [("auto", "ce"), ("kl", "kl"), ("ce", "ce")])
def test_auto_resolves_to_the_objective_cayley_sgd_uses(objective, resolved):
    assert _rotation_cfg(f"transform.objective={objective}").objective == resolved


def test_an_unknown_objective_is_rejected():
    with pytest.raises(ValueError, match="must be 'kl', 'ce' or 'auto'"):
        _rotation_cfg("transform.objective=mse")


@pytest.mark.parametrize("module, builder", [
    ("whisper_utils", "build_whisper_dataloader"),
    ("parakeet_ctc_utils", "build_parakeet_dataloader"),
    ("canary_qwen_utils", "build_canary_qwen_dataloader"),
])
def test_every_loader_takes_the_arguments_the_rotation_transform_passes(module, builder):
    """The transform passes batch_size to whichever loader the model uses, through a per-model lambda, so a
    missing parameter only surfaces at search time."""
    import importlib
    import inspect

    function = getattr(importlib.import_module(f"asrq.transforms.rotation.{module}"), builder)
    assert {"batch_size", "seed"} <= set(inspect.signature(function).parameters)
