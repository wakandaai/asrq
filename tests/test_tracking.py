"""Weights & Biases logging: off by default, and what each stage logs when it is on."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from _pipeline import config
from asrq import tracking


class FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.rows = []
        self.summary = {}
        self.config = SimpleNamespace(update=lambda data, allow_val_change=False: self.rows.append(("config", data)))
        self.finished = False

    def log(self, data, commit=True):
        self.rows.append(("log", data))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    runs = []

    def init(**kwargs):
        runs.append(FakeRun(**kwargs))
        return runs[-1]

    module = types.ModuleType("wandb")
    module.init = init
    module.define_metric = lambda *args, **kwargs: None
    module.Histogram = lambda values: ("histogram", len(values))
    monkeypatch.setitem(sys.modules, "wandb", module)
    return runs


def _cfg(**settings):
    cfg = config("whisper", [])
    OmegaConf.set_struct(cfg, False)
    cfg.wandb = OmegaConf.create({"enabled": True, "project": "p", "entity": None, "group": None, "tags": [],
                                  "mode": "offline", **settings})
    return cfg


def test_logging_is_a_no_op_without_a_run():
    assert not tracking.enabled()
    tracking.log({"a": 1})
    tracking.summary({"b": 2})
    tracking.step_metric("stage", "stage/step")
    assert tracking.histogram(torch.zeros(3)) is None


def test_a_disabled_config_opens_no_run(fake_wandb):
    cfg = _cfg(enabled=False)
    with tracking.start(cfg, "quantize") as run:
        assert run is None and not tracking.enabled()
    assert fake_wandb == []


def test_a_missing_wandb_only_warns(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setattr("builtins.__import__", _raise_for_wandb(__import__))
    with tracking.start(_cfg(), "quantize") as run:
        assert run is None
    assert "wandb is not installed" in capsys.readouterr().out


def _raise_for_wandb(real_import):
    def guarded(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("no wandb")
        return real_import(name, *args, **kwargs)

    return guarded


def test_a_run_carries_the_config_and_closes(fake_wandb):
    cfg = _cfg(tags=["w2"])
    with tracking.start(cfg, "rotation") as run:
        assert tracking.enabled()
        tracking.log({"search/generation": 1, "search/fitness": 0.5})
        tracking.summary({"search/final_fitness": 0.4})
        tracking.config_update({"rotation_file": "x.pt"})
    (fake,) = fake_wandb
    assert fake.kwargs["job_type"] == "rotation" and fake.kwargs["tags"] == ["w2"] and fake.kwargs["mode"] == "offline"
    assert fake.kwargs["config"]["model"]["name"] == cfg.model.name
    assert ("log", {"search/generation": 1, "search/fitness": 0.5}) in fake.rows
    assert fake.summary["search/final_fitness"] == 0.4
    assert ("config", {"rotation_file": "x.pt"}) in fake.rows
    assert fake.finished and not tracking.enabled()


def test_the_evolutionary_search_logs_every_generation(fake_wandb):
    from asrq.transforms.rotation.hadamard_search import EvolutionConfig, evolve_signs, random_sign_vectors

    target = random_sign_vectors({None: 32}, torch.Generator().manual_seed(1), "cpu")[None][0]

    def fitness(signs, batches):
        return float((signs[None][0] != target).float().mean())

    with tracking.start(_cfg(), "rotation"):
        config_ = EvolutionConfig(generations=3, offspring=4, survivors=(2, 1), stage_samples=(4, 4, 8), flips=1,
                                  mutate="s1")
        evolve_signs(random_sign_vectors({None: 32}, torch.Generator().manual_seed(0), "cpu"), fitness, [4] * 2,
                     config_, log=lambda _line: None)
    (fake,) = fake_wandb
    generations = [row for kind, row in fake.rows if kind == "log" and "search/generation" in row]
    assert [row["search/generation"] for row in generations] == [0, 1, 2, 3]
    assert {"search/fitness", "search/best_child", "search/accepted", "search/candidates_seen"} <= set(generations[-1])
    assert fake.summary["search/name"] == "evolution" and "search/final_fitness" in fake.summary


def test_scale_recovery_and_refit_log_per_group_and_block(fake_wandb):
    from asrq.core.model import ModelQ
    from asrq.quantizers.scale_recovery import ScaleGroup, ScaleTarget

    norm = nn.LayerNorm(8)
    modelq = SimpleNamespace(model=nn.ModuleDict({"norm": norm}),
                             quant_cfg=SimpleNamespace(scale_recovery_ridge=0.01))
    group = ScaleGroup(ScaleTarget("norm", ["q"]), scale=torch.full((8,), 0.9), ratio=0.5)
    with tracking.start(_cfg(), "quantize"):
        ModelQ.apply_scale_recovery(modelq, [group])
    (fake,) = fake_wandb
    (row,) = [row for kind, row in fake.rows if kind == "log" and "scale_recovery/group" in row]
    assert row["scale_recovery/error_ratio"] == 0.5 and row["scale_recovery/scale_min"] == pytest.approx(0.9)
    assert row["scale_recovery/scales"] == ("histogram", 8)
