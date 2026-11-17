"""The rotation transform's config: what each setting resolves to before a rotation is learned."""

import pytest

from _pipeline import config
from asrq.experiment import prepare_experiment_config
from asrq.transforms.rotation.base import RotationTransformConfig


def _rotation_cfg(*overrides):
    cfg = config("whisper", list(overrides))
    prepare_experiment_config(cfg, learn_rotation=True)
    return RotationTransformConfig(cfg.transform)


def test_the_default_config_builds_and_resolves_its_off_values():
    cfg = _rotation_cfg()
    assert cfg.r1_block_size is None and not hasattr(cfg, "objective")
    assert _rotation_cfg("transform.r1_block_size=128").r1_block_size == 128


@pytest.mark.parametrize("module, builder", [
    ("whisper_utils", "build_whisper_dataloader"),
    ("parakeet_ctc_utils", "build_parakeet_dataloader"),
    ("canary_qwen_utils", "build_canary_qwen_dataloader"),
])
def test_every_loader_takes_the_arguments_the_rotation_transform_passes(module, builder):
    """The transform passes batch_size to whichever loader the model uses, through a per-model lambda, so a
    missing parameter only surfaces at learning time."""
    import importlib
    import inspect

    function = getattr(importlib.import_module(f"asrq.transforms.rotation.{module}"), builder)
    assert {"batch_size", "seed"} <= set(inspect.signature(function).parameters)
