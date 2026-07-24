import sys
import types
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from unittest import mock

from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parents[1]
ASRQ_ROOT = ROOT / "asrq"


def _stub_package(name: str, path: Optional[Path] = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _stub_module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


transformers_module = _stub_package("transformers")
transformers_models_module = _stub_package("transformers.models")
transformers_qwen3_module = _stub_package("transformers.models.qwen3")
transformers_whisper_module = _stub_package("transformers.models.whisper")
transformers_qwen3_modeling_module = _stub_module(
    "transformers.models.qwen3.modeling_qwen3",
    Qwen3RMSNorm=type("Qwen3RMSNorm", (), {}),
)
transformers_whisper_modeling_module = _stub_module(
    "transformers.models.whisper.modeling_whisper",
    WhisperForConditionalGeneration=type("WhisperForConditionalGeneration", (), {}),
    BaseModelOutput=type("BaseModelOutput", (), {}),
    BaseModelOutputWithPastAndCrossAttentions=type("BaseModelOutputWithPastAndCrossAttentions", (), {}),
    create_causal_mask=lambda *args, **kwargs: None,
    EncoderDecoderCache=type("EncoderDecoderCache", (), {}),
    DynamicCache=type("DynamicCache", (), {}),
    logger=types.SimpleNamespace(warning_once=lambda *args, **kwargs: None),
)
transformers_module.WhisperProcessor = type("WhisperProcessor", (), {})
transformers_module.WhisperForConditionalGeneration = type("WhisperForConditionalGeneration", (), {})
transformers_module.models = transformers_models_module
transformers_models_module.qwen3 = transformers_qwen3_module
transformers_models_module.whisper = transformers_whisper_module
transformers_qwen3_module.modeling_qwen3 = transformers_qwen3_modeling_module
transformers_whisper_module.modeling_whisper = transformers_whisper_modeling_module
_stub_module("datasets", load_dataset=lambda *args, **kwargs: None)
_stub_module("soundfile", write=lambda *args, **kwargs: None)
_stub_package("matplotlib")
_stub_module(
    "matplotlib.pyplot",
    figure=lambda *args, **kwargs: None,
    plot=lambda *args, **kwargs: None,
    xlabel=lambda *args, **kwargs: None,
    ylabel=lambda *args, **kwargs: None,
    title=lambda *args, **kwargs: None,
    legend=lambda *args, **kwargs: None,
    tight_layout=lambda *args, **kwargs: None,
    savefig=lambda path, *args, **kwargs: Path(path).touch(),
    close=lambda *args, **kwargs: None,
)

_stub_package("asrq", ASRQ_ROOT)
_stub_package("asrq.core", ASRQ_ROOT / "core")
_stub_package("asrq.transforms", ASRQ_ROOT / "transforms")
rotation_pkg = _stub_package("asrq.transforms.rotation", ASRQ_ROOT / "transforms" / "rotation")
rotation_pkg.obtain_rotations_for_whisper = lambda *args, **kwargs: None
rotation_pkg.rotate_whisper_model = lambda *args, **kwargs: None
rotation_pkg.obtain_rotations_for_whisper_search = lambda *args, **kwargs: None
rotation_pkg.obtain_rotations_for_canary_qwen = lambda *args, **kwargs: None
rotation_pkg.rotate_canary_qwen = lambda *args, **kwargs: None
rotation_pkg.obtain_rotations_for_parakeet = lambda *args, **kwargs: None
rotation_pkg.rotate_parakeet = lambda *args, **kwargs: None
rotation_pkg.obtain_rotations_for_parakeet_search = lambda *args, **kwargs: None

RotationTransformConfig = import_module("asrq.transforms.rotation.base").RotationTransformConfig
search_module = import_module("asrq.transforms.rotation.search")
parakeet_module = import_module("asrq.transforms.rotation.parakeet_ctc_utils")
whisper_module = import_module("asrq.transforms.rotation.whisper_utils")
AlternatingSearchParams = search_module.AlternatingSearchParams
GlobalRotationSearchSite = search_module.GlobalRotationSearchSite
RotationSearchParams = search_module.RotationSearchParams
RotationSearchSite = search_module.RotationSearchSite
SignHadamardCandidate = search_module.SignHadamardCandidate
ThreeSignHadamardCandidate = search_module.ThreeSignHadamardCandidate
normalized_hadamard_matrix = search_module.normalized_hadamard_matrix
run_alternating_rotation_search = search_module.run_alternating_rotation_search
save_alternating_search_artifacts = search_module.save_alternating_search_artifacts
_ParakeetGlobalQeSearchAdapter = parakeet_module._ParakeetGlobalQeSearchAdapter
obtain_rotations_for_parakeet_search = parakeet_module.obtain_rotations_for_parakeet_search
_WhisperGlobalQeSearchAdapter = whisper_module._WhisperGlobalQeSearchAdapter
obtain_rotations_for_whisper_search = whisper_module.obtain_rotations_for_whisper_search


class FakeLocalAdapter:
    def __init__(self) -> None:
        self.site = RotationSearchSite(
            site_id="local.site",
            block_id="local.block",
            dimension=8,
            base_h=normalized_hadamard_matrix(8),
            current_candidate=SignHadamardCandidate(
                s0=torch.ones(8, dtype=torch.int8),
                s1=torch.ones(8, dtype=torch.int8),
            ),
        )
        self.commit_calls = 0

    def sites(self):
        return [self.site]

    def refresh_caches(self) -> None:
        return None

    def score_site_candidate(self, site, candidate) -> float:
        return float((candidate.s0 != self.site.current_candidate.s0).sum() + (candidate.s1 != self.site.current_candidate.s1).sum())

    def commit_site_candidate(self, site, candidate) -> None:
        self.commit_calls += 1
        self.site.current_candidate = candidate.clone()

    @property
    def refine_calls(self) -> int:
        return max(0, self.commit_calls - 1)


class FakeGlobalAdapter:
    def __init__(self, scores: list[float]) -> None:
        self.site = RotationSearchSite(
            site_id="global.site",
            block_id="global.block",
            dimension=8,
            base_h=normalized_hadamard_matrix(8),
            current_candidate=SignHadamardCandidate(
                s0=torch.ones(8, dtype=torch.int8),
                s1=torch.ones(8, dtype=torch.int8),
            ),
        )
        self._scores = iter(scores)
        self.commit_calls = 0

    def global_site(self) -> RotationSearchSite:
        return self.site

    def refresh_global_caches(self) -> None:
        return None

    def score_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> float:
        return next(self._scores)

    def commit_global_candidate(self, candidate: ThreeSignHadamardCandidate) -> None:
        self.commit_calls += 1


class FakeParakeetSearchModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = types.SimpleNamespace(
            conv=types.SimpleNamespace(d_model=8),
            self_attn=types.SimpleNamespace(d_k=4),
        )
        self.encoder = types.SimpleNamespace(layers=[layer, layer])
        self.tokenizer = types.SimpleNamespace(pad_id=0)
        self.dummy = torch.nn.Linear(1, 1)

    def to(self, device: str):
        return self


class FakeParakeetGlobalLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(1.0))


class FakeWhisperGlobalLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(1.0))


class FakeWhisperSearchModel:
    def __init__(self) -> None:
        encoder_layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(head_dim=4))
        decoder_layer = types.SimpleNamespace(
            self_attn=types.SimpleNamespace(head_dim=4),
            encoder_attn=types.SimpleNamespace(head_dim=4),
        )
        self.model = types.SimpleNamespace(
            encoder=types.SimpleNamespace(layers=[encoder_layer]),
            decoder=types.SimpleNamespace(layers=[decoder_layer]),
        )
        self.config = types.SimpleNamespace(d_model=8, encoder_layers=1, decoder_layers=1, use_cache=False)
        self.dtype = torch.float32

    def to(self, device: str):
        return self

    def generate(self, input_features, max_new_tokens=128):
        return torch.tensor([[1, 2, 3]])

    def named_modules(self):
        return []


def test_parakeet_search_uses_global_qe_loss_when_search_mode_is_alternating() -> None:
    model = FakeParakeetGlobalLossModel()
    qe_param = torch.nn.Parameter(torch.eye(4, dtype=torch.float32), requires_grad=False)
    calibration_batches = [{"audios": torch.ones(1, 2), "audio_lens": torch.tensor([2])}]
    adapter = _ParakeetGlobalQeSearchAdapter(
        model,
        calibration_batches,
        qe_param,
        weight_bits=5,
        activation_bits=7,
    )
    candidate = ThreeSignHadamardCandidate(
        s0=torch.ones(4, dtype=torch.int8),
        s1=torch.tensor([1, -1, 1, -1], dtype=torch.int8),
        s2=torch.ones(4, dtype=torch.int8),
    )
    fake_quant_calls: list[tuple[bool, int, int]] = []

    def fake_loss_fn(current_model, batch) -> torch.Tensor:
        assert current_model is model
        assert batch is not calibration_batches[0]
        adapter.loss_name = "ctc"  # type: ignore[attr-defined]
        return torch.tensor(2.5)

    with mock.patch.object(parakeet_module, "parakeet_ctc_loss_fn", side_effect=fake_loss_fn), mock.patch.object(
        parakeet_module,
        "set_rotation_fake_quant_state",
        side_effect=lambda current_model, enabled, activation_bits, weight_bits: fake_quant_calls.append(
            (enabled, activation_bits, weight_bits)
        ),
    ):
        score = adapter.score_global_candidate(candidate)

    assert isinstance(score, float)
    assert score == 2.5
    assert adapter.loss_name == "ctc"  # type: ignore[attr-defined]
    assert fake_quant_calls == [(True, 7, 5), (False, 7, 5)]


def test_whisper_global_qe_metric_uses_teacher_forced_ce() -> None:
    model = FakeWhisperGlobalLossModel()
    qe_param = torch.nn.Parameter(torch.eye(8, dtype=torch.float32), requires_grad=False)
    calibration_batches = [{"input_features": torch.ones(1, 2), "labels": torch.ones(1, 2, dtype=torch.long)}]
    adapter = _WhisperGlobalQeSearchAdapter(
        model,
        calibration_batches,
        qe_param,
        weight_bits=5,
        activation_bits=7,
        global_site=GlobalRotationSearchSite(
            site_id="encoder.qe",
            dimension=8,
            base_h=normalized_hadamard_matrix(8),
            current_candidate=ThreeSignHadamardCandidate(
                s0=torch.ones(8, dtype=torch.int8),
                s1=torch.ones(8, dtype=torch.int8),
                s2=torch.ones(8, dtype=torch.int8),
            ),
        ),
    )
    candidate = ThreeSignHadamardCandidate(
        s0=torch.ones(8, dtype=torch.int8),
        s1=torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.int8),
        s2=torch.ones(8, dtype=torch.int8),
    )
    fake_quant_calls: list[tuple[bool, int, int]] = []

    def fake_loss_fn(current_model, batch) -> torch.Tensor:
        assert current_model is model
        assert batch is not calibration_batches[0]
        adapter.loss_name = "teacher_forced_ce"  # type: ignore[attr-defined]
        return torch.tensor(1.75)

    with mock.patch.object(whisper_module, "whisper_loss_fn", side_effect=fake_loss_fn), mock.patch.object(
        whisper_module,
        "set_rotation_fake_quant_state",
        side_effect=lambda current_model, enabled, activation_bits, weight_bits: fake_quant_calls.append(
            (enabled, activation_bits, weight_bits)
        ),
    ):
        score = adapter.score_global_candidate(candidate)

    assert isinstance(score, float)
    assert score == 1.75
    assert adapter.loss_name == "teacher_forced_ce"  # type: ignore[attr-defined]
    assert fake_quant_calls == [(True, 7, 5), (False, 7, 5)]


def test_parakeet_obtain_rotations_dispatches_to_alternating_search() -> None:
    model = FakeParakeetSearchModel()
    q2_params = RotationSearchParams(generations=2, patience=1, population_size=1, elite_count=1, seed=1, verbose=False)
    qe_params = RotationSearchParams(generations=3, patience=1, population_size=1, elite_count=1, seed=2, verbose=False)
    q2_refine_params = RotationSearchParams(generations=4, patience=1, population_size=1, elite_count=1, seed=3, verbose=False)
    alternating_result = search_module.AlternatingRotationSearchResult(
        local_result=search_module.RotationSearchResult(best_candidates={}, histories={}),
        global_history=[search_module.RotationSearchHistory(site_id="qe")],
        best_global_candidate=ThreeSignHadamardCandidate(
            s0=torch.ones(8, dtype=torch.int8),
            s1=torch.ones(8, dtype=torch.int8),
            s2=torch.ones(8, dtype=torch.int8),
        ),
        best_global_score=0.25,
        outer_rounds_completed=1,
    )
    recorded: dict[str, object] = {}
    batch = {
        "audios": torch.ones(1, 8),
        "audio_lens": torch.tensor([8]),
        "tokens": torch.ones(1, 2, dtype=torch.long),
        "token_lens": torch.tensor([2]),
    }

    class FakeDataLoader:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __iter__(self):
            yield batch

    def fake_run_alternating(local_adapter, global_adapter, params):
        recorded["local_adapter"] = local_adapter
        recorded["global_adapter"] = global_adapter
        recorded["params"] = params
        return alternating_result

    with mock.patch.object(parakeet_module, "transcribe", return_value="ok"), mock.patch.object(
        parakeet_module, "prepare_parakeet_ctc_for_rotation"
    ), mock.patch.object(
        parakeet_module, "get_orthogonal_matrix", side_effect=lambda size, **kwargs: torch.eye(size, dtype=torch.float64)
    ), mock.patch.object(
        parakeet_module, "modify_parakeet_ctc_layers_with_rotation_params"
    ), mock.patch.object(
        parakeet_module, "monkey_patch_parakeet_ctc_for_train"
    ), mock.patch.object(
        parakeet_module, "set_rotation_fake_quant_state"
    ), mock.patch.object(
        parakeet_module, "ParakeetCalibrationDataset", return_value=[object()]
    ), mock.patch.object(
        parakeet_module.torch.utils.data, "DataLoader", FakeDataLoader
    ), mock.patch.object(
        parakeet_module, "run_alternating_rotation_search", side_effect=fake_run_alternating
    ), mock.patch.object(
        parakeet_module, "run_rotation_search"
    ) as local_search_mock, mock.patch.object(
        parakeet_module, "save_rotation_search_artifacts", return_value={"history_path": "local.pt", "plot_path": "local.png"}
    ) as save_local_mock, mock.patch.object(
        parakeet_module, "_save_global_rotation_search_artifacts", return_value="global.pt"
    ) as save_global_mock, mock.patch.object(
        parakeet_module.torch, "save"
    ) as torch_save_mock:
        obtain_rotations_for_parakeet_search(
            model,
            text_audio_path="audio.wav",
            calib_samples=1,
            batch_size=1,
            search_params=q2_params,
            save_path="rotations.pt",
            device="cpu",
            search_mode="alternating",
            q2_params=q2_params,
            qe_params=qe_params,
            q2_refine_params=q2_refine_params,
            outer_rounds=2,
            outer_patience=1,
            qe_min_delta=1e-3,
        )

    assert "global_adapter" in recorded
    assert isinstance(recorded["global_adapter"], _ParakeetGlobalQeSearchAdapter)
    assert isinstance(recorded["params"], AlternatingSearchParams)
    assert recorded["params"].q2_params is q2_params
    assert recorded["params"].qe_params is qe_params
    assert recorded["params"].q2_refine_params is q2_refine_params
    assert local_search_mock.call_count == 0
    assert save_local_mock.call_count == 1
    assert save_global_mock.call_count == 1
    assert torch_save_mock.call_count == 1


def test_whisper_obtain_rotations_dispatches_to_alternating_search() -> None:
    model = FakeWhisperSearchModel()

    class FakeProcessor:
        def __call__(self, audio, sampling_rate, return_tensors):
            return types.SimpleNamespace(input_features=torch.ones(1, 4))

        def batch_decode(self, tokens, skip_special_tokens=True):
            return ["ok"]

    processor = FakeProcessor()
    q2_params = RotationSearchParams(generations=2, patience=1, population_size=1, elite_count=1, seed=1, verbose=False)
    qe_params = RotationSearchParams(generations=3, patience=1, population_size=1, elite_count=1, seed=2, verbose=False)
    q2_refine_params = RotationSearchParams(generations=4, patience=1, population_size=1, elite_count=1, seed=3, verbose=False)
    alternating_result = search_module.AlternatingRotationSearchResult(
        local_result=search_module.RotationSearchResult(best_candidates={}, histories={}),
        global_history=[search_module.RotationSearchHistory(site_id="encoder.qe")],
        best_global_candidate=ThreeSignHadamardCandidate(
            s0=torch.ones(8, dtype=torch.int8),
            s1=torch.ones(8, dtype=torch.int8),
            s2=torch.ones(8, dtype=torch.int8),
        ),
        best_global_score=0.25,
        outer_rounds_completed=1,
    )
    recorded: dict[str, object] = {}
    batch = {
        "input_features": torch.ones(1, 4),
        "labels": torch.ones(1, 2, dtype=torch.long),
    }

    class FakeDataLoader:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def __iter__(self):
            yield batch

    def fake_run_alternating(local_adapter, global_adapter, params):
        recorded["local_adapter"] = local_adapter
        recorded["global_adapter"] = global_adapter
        recorded["params"] = params
        return alternating_result

    with mock.patch.object(whisper_module, "prepare_whisper_for_rotation"), mock.patch.object(
        whisper_module, "get_orthogonal_matrix", side_effect=lambda size, **kwargs: torch.eye(size, dtype=torch.float64)
    ), mock.patch.object(
        whisper_module, "modify_whisper_layers_with_rotation_params"
    ), mock.patch.object(
        whisper_module, "monkey_patch_whisper"
    ), mock.patch.object(
        whisper_module, "set_rotation_fake_quant_state"
    ), mock.patch.object(
        whisper_module, "WhisperCalibrationDataset", return_value=[object()]
    ), mock.patch.object(
        whisper_module.torch.utils.data, "DataLoader", FakeDataLoader
    ), mock.patch.object(
        whisper_module, "run_alternating_rotation_search", side_effect=fake_run_alternating
    ), mock.patch.object(
        whisper_module, "run_rotation_search"
    ) as local_search_mock, mock.patch.object(
        whisper_module, "save_rotation_search_artifacts", return_value={"history_path": "local.pt", "plot_path": "local.png"}
    ) as save_local_mock, mock.patch.object(
        whisper_module, "_save_global_rotation_search_artifacts", return_value="global.pt"
    ) as save_global_mock, mock.patch.object(
        whisper_module.torch, "save"
    ) as torch_save_mock:
        obtain_rotations_for_whisper_search(
            model,
            processor,
            test_audio=torch.ones(4).numpy(),
            test_audio_sr=16000,
            calib_samples=1,
            batch_size=1,
            search_params=q2_params,
            save_path="rotations.pt",
            search_mode="alternating",
            q2_params=q2_params,
            qe_search_params=qe_params,
            q2_refine_params=q2_refine_params,
            outer_rounds=2,
            outer_patience=1,
            qe_min_delta=1e-3,
            weight_bits=5,
            activation_bits=7,
        )

    assert "global_adapter" in recorded
    assert isinstance(recorded["global_adapter"], _WhisperGlobalQeSearchAdapter)
    assert isinstance(recorded["params"], AlternatingSearchParams)
    assert recorded["params"].q2_params is q2_params
    assert recorded["params"].qe_params is qe_params
    assert recorded["params"].q2_refine_params is q2_refine_params
    assert local_search_mock.call_count == 0
    assert save_local_mock.call_count == 1
    assert save_global_mock.call_count == 1
    assert torch_save_mock.call_count == 1


def test_alternating_search_saves_local_and_global_histories() -> None:
    with TemporaryDirectory() as temp_dir:
        rotation_path = str(Path(temp_dir) / "rotation.pt")
        result = search_module.AlternatingRotationSearchResult(
            local_result=search_module.RotationSearchResult(best_candidates={}, histories={}),
            global_history=[
                search_module.RotationSearchHistory(
                    site_id="encoder.qe.round_1",
                    generation_indices=[0, 1],
                    best_scores=[1.5, 1.0],
                    committed_score=1.0,
                )
            ],
            best_global_candidate=ThreeSignHadamardCandidate(
                s0=torch.ones(8, dtype=torch.int8),
                s1=torch.ones(8, dtype=torch.int8),
                s2=torch.ones(8, dtype=torch.int8),
            ),
            best_global_score=1.0,
            outer_rounds_completed=1,
        )

        artifacts = save_alternating_search_artifacts(result, rotation_path)

        assert artifacts["local_history_path"].endswith("_search_history.pt")
        assert artifacts["global_history_path"].endswith("_global_search_history.pt")
        assert Path(artifacts["local_history_path"]).exists()
        assert Path(artifacts["global_history_path"]).exists()


def test_parakeet_global_history_artifact_save_writes_file() -> None:
    with TemporaryDirectory() as temp_dir:
        rotation_path = Path(temp_dir) / "rotation.pt"
        rotation_path.touch()

        history_path = parakeet_module._save_global_rotation_search_artifacts(
            [search_module.RotationSearchHistory(site_id="encoder.qe.round_1")],
            str(rotation_path),
        )

        assert history_path.endswith("_global_search_history.pt")
        assert Path(history_path).exists()


class RotationAlternatingSearchConfigTests(unittest.TestCase):
    def test_rotation_transform_config_exposes_alternating_search_knobs(self) -> None:
        cfg = OmegaConf.create(
            {
                "name": "rotation",
                "model_name": "dummy/model",
                "type": "search",
                "search_mode": "alternating",
                "outer_rounds": 3,
                "outer_patience": 1,
                "qe_min_delta": 1e-3,
                "generations": 16,
                "q2_refine_generations": 6,
                "qe_generations": 10,
                "patience": 4,
                "qe_patience": 3,
                "population_size": 8,
                "elite_count": 2,
                "parent_pool_fraction": 0.5,
                "mutate_both_probability": 0.1,
                "large_mutation_probability": 0.1,
                "small_mutation_min": 1,
                "small_mutation_max": 2,
                "medium_mutation_min": 4,
                "medium_mutation_max": 8,
                "large_mutation_fraction": 0.25,
                "num_samples": 128,
                "epochs": 1,
                "learning_rate": 0.01,
                "batch_size": 1,
                "learn_rotation": True,
                "use": True,
                "path": "",
            }
        )

        rotation_cfg = RotationTransformConfig(cfg)

        self.assertEqual(rotation_cfg.search_mode, "alternating")
        self.assertEqual(rotation_cfg.outer_rounds, 3)
        self.assertEqual(rotation_cfg.outer_patience, 1)
        self.assertEqual(rotation_cfg.qe_min_delta, 1e-3)
        self.assertEqual(rotation_cfg.q2_refine_generations, 6)
        self.assertEqual(rotation_cfg.qe_generations, 10)
        self.assertEqual(rotation_cfg.qe_patience, 3)

        qe_params = rotation_cfg.qe_search_params(seed=9)
        q2_refine_params = rotation_cfg.q2_refine_search_params(seed=11)

        self.assertIsInstance(qe_params, RotationSearchParams)
        self.assertIsInstance(q2_refine_params, RotationSearchParams)
        self.assertEqual(qe_params.generations, 10)
        self.assertEqual(qe_params.patience, 3)
        self.assertEqual(qe_params.seed, 9)
        self.assertEqual(q2_refine_params.generations, 6)
        self.assertEqual(q2_refine_params.patience, 4)
        self.assertEqual(q2_refine_params.seed, 11)

    def test_rotation_transform_config_defaults_to_local_q2_search(self) -> None:
        cfg = OmegaConf.create(
            {
                "name": "rotation",
                "model_name": "dummy/model",
                "type": "search",
                "generations": 16,
                "patience": 4,
                "population_size": 8,
                "elite_count": 2,
                "parent_pool_fraction": 0.5,
                "mutate_both_probability": 0.1,
                "large_mutation_probability": 0.1,
                "small_mutation_min": 1,
                "small_mutation_max": 2,
                "medium_mutation_min": 4,
                "medium_mutation_max": 8,
                "large_mutation_fraction": 0.25,
                "num_samples": 128,
                "epochs": 1,
                "learning_rate": 0.01,
                "batch_size": 1,
                "learn_rotation": True,
                "use": True,
                "path": "",
            }
        )

        rotation_cfg = RotationTransformConfig(cfg)

        self.assertEqual(rotation_cfg.search_mode, "local_q2")
        self.assertEqual(rotation_cfg.outer_rounds, 1)
        self.assertEqual(rotation_cfg.outer_patience, 1)
        self.assertEqual(rotation_cfg.qe_min_delta, 0.0)

        params = rotation_cfg.search_params(seed=7)

        self.assertIsInstance(params, RotationSearchParams)
        self.assertEqual(params.generations, 16)
        self.assertEqual(params.patience, 4)
        self.assertEqual(params.seed, 7)

    def test_alternating_search_skips_q2_refine_when_qe_does_not_improve(self) -> None:
        local_adapter = FakeLocalAdapter()
        global_adapter = FakeGlobalAdapter(scores=[1.0, 0.99995, 0.99990])
        params = AlternatingSearchParams(
            q2_params=RotationSearchParams(generations=1, patience=1, population_size=1, elite_count=1, seed=1, verbose=False),
            qe_params=RotationSearchParams(generations=1, patience=1, population_size=1, elite_count=1, seed=2, verbose=False),
            q2_refine_params=RotationSearchParams(generations=1, patience=1, population_size=1, elite_count=1, seed=3, verbose=False),
            outer_rounds=3,
            outer_patience=1,
            qe_min_delta=1e-4,
        )

        result = run_alternating_rotation_search(local_adapter, global_adapter, params)

        self.assertEqual(result.outer_rounds_completed, 1)
        self.assertEqual(local_adapter.refine_calls, 0)


if __name__ == "__main__":
    unittest.main()
