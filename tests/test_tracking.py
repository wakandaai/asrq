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

    def save(self, path, base_path=None, policy=None):
        self.rows.append(("save", path))

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
    module.Table = lambda columns, data: ("table", tuple(columns), len(data))
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


def test_the_experiment_name_groups_and_names_the_run(fake_wandb):
    cfg = _cfg()
    cfg.exp_name = "weight_only_w4_w2"
    cfg.method = "wo_w2"
    with tracking.start(cfg, "quantize"):
        pass
    (fake,) = fake_wandb
    assert fake.kwargs["group"] == "weight_only_w4_w2"
    assert fake.kwargs["name"] == "weight_only_w4_w2_whisper-large-v3_wo_w2"


def test_without_an_experiment_name_the_run_is_named_by_model_and_method(fake_wandb):
    cfg = _cfg()
    cfg.exp_name = None
    cfg.method = "wo_w2"
    with tracking.start(cfg, "quantize"):
        pass
    (fake,) = fake_wandb
    assert fake.kwargs["group"] is None and fake.kwargs["name"] == "whisper-large-v3_wo_w2"


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


def test_a_results_csv_is_logged_as_a_table_and_uploaded(fake_wandb, tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "model,method,quantizer,transform,wbits,abits,dataset,split,wer\n"
        "openai/whisper-large-v3,wa_w4a4,gptq,rotation,4,4,librispeech,test.clean,2.11\n"
        "openai/whisper-large-v3,wa_w4a4,gptq,rotation,4,4,librispeech,test.other,4.53\n"
    )
    with tracking.start(_cfg(), "quantize"):
        tracking.results_csv(str(results))
    (fake,) = fake_wandb
    (logged,) = [row for kind, row in fake.rows if kind == "log" and "eval/results" in row]
    kind, columns, count = logged["eval/results"]
    assert kind == "table" and columns[-1] == "wer" and count == 2
    assert ("save", str(results)) in fake.rows


def test_a_missing_results_file_is_ignored(fake_wandb, tmp_path):
    with tracking.start(_cfg(), "quantize"):
        tracking.results_csv(str(tmp_path / "absent.csv"))
        tracking.save_file(str(tmp_path / "absent.yaml"))
    (fake,) = fake_wandb
    assert not [row for kind, row in fake.rows if kind == "save"]
