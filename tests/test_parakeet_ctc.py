"""Parakeet-CTC-1.1B through rot-exp.py and exp.py, at small scale.

Runs asrq.experiment's learn_rotation_experiment and run_experiment (the bodies of asrq/rot-exp.py and
asrq/exp.py) on the real model: a rotation is learned and saved, then applied, GPTQ-quantized and
evaluated with fake W4A4, and again with humming ASRQLinear layers, CUDA-graph transcription, float16 and
the inserted block-output Linears quantized. The scaling transform goes through exp.py the same two ways,
at W4A8: scales searched, saved and applied (an InputScale after the activation feeding each linear2 and
pointwise_conv2), GPTQ, evaluation. See tests/_pipeline.py for the scale and skip conditions.
"""

import gc

import pytest
import torch

from _pipeline import EVALUATE_8_UTTERANCES, SCALING_SMALL, SMALL, WER_CEILING, config, requirements, results
from asrq.core.linear import ASRQLinear
from asrq.experiment import learn_rotation_experiment, run_experiment
from asrq.transforms.rotation.utils import OnlineHadamard, ResidualStreamRotation
from asrq.transforms.scaling.base import InputScale

MODEL = "nvidia/parakeet-ctc-1.1b"
pytestmark = requirements(MODEL)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def learned_rotation(tmp_path_factory):
    path = tmp_path_factory.mktemp("rotation") / "parakeet.pt"
    torch.manual_seed(0)
    modelQ = learn_rotation_experiment(config("parakeet", [*SMALL, f"transform.path={path}"]))
    del modelQ
    _free()
    return path


def test_rot_exp_learns_and_saves_r1_r2_and_the_online_hadamard_signs(learned_rotation):
    checkpoint = torch.load(learned_rotation, weights_only=False)
    R1 = checkpoint["R1"].double()
    assert R1.shape == (1024, 1024)
    assert (R1.T @ R1 - torch.eye(1024, dtype=torch.float64)).abs().max() < 1e-4
    assert len(checkpoint["R2s"]) == 42
    assert all(R2.shape == (128, 128) for R2 in checkpoint["R2s"].values())
    assert checkpoint["learn_r2"] and checkpoint["fc2_online_hadamard"]
    assert set(checkpoint["hadamard_signs"]) == {1024, 4096}
    assert checkpoint["activation_quantization"]["bits"] == 4


def test_exp_rotates_quantizes_with_gptq_and_evaluates(learned_rotation, tmp_path):
    modelQ, results_file = run_experiment(
        config("parakeet", [*SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={learned_rotation}"]),
        results_dir=str(tmp_path),
    )
    try:
        modules = list(modelQ.model.modules())
        assert sum(isinstance(m, OnlineHadamard) for m in modules) == 3 * 42
        assert sum(isinstance(m, ResidualStreamRotation) for m in modules) == 2
        quantized = set(modelQ.qparams)
        assert sum(".conv.pointwise_conv" in name for name in quantized) == 2 * 42
        assert not any(name.endswith(".norm_out.1") for name in quantized)
        (row,) = results(results_file)
        assert (row["dataset"], row["split"], row["abits"]) == ("librispeech", "test.clean", "4")
        assert float(row["wer"]) < WER_CEILING
    finally:
        del modelQ
        _free()


def test_exp_with_humming_layers_cuda_graphs_and_quantized_block_output_linears(learned_rotation, tmp_path):
    modelQ, results_file = run_experiment(
        config("parakeet", [
            *SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={learned_rotation}", "inference=humming",
            "model.generate_fn=generate_parakeet_cuda_graphs", "eval_dtype=float16",
            "model.quantize_block_output_linear=true",
        ]),
        results_dir=str(tmp_path),
    )
    try:
        layers = [m for m in modelQ.model.modules() if isinstance(m, ASRQLinear)]
        assert len(layers) == 10 * 42 + 41
        assert sum(layer.conv1d_layout for layer in layers) == 2 * 42
        assert sum(layer.hadamard_block_size is not None for layer in layers) == 3 * 42
        (row,) = results(results_file)
        assert float(row["wer"]) < WER_CEILING
    finally:
        del modelQ
        _free()


@pytest.fixture(scope="module")
def scaled_experiment(tmp_path_factory):
    """exp.py with the scaling transform: scales searched and saved, applied, GPTQ W4, fake A8."""
    directory = tmp_path_factory.mktemp("scaling")
    modelQ, results_file = run_experiment(
        config("parakeet", [*SCALING_SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={directory / 'scales.pt'}"]),
        results_dir=str(directory),
    )
    input_scales = sum(isinstance(m, InputScale) for m in modelQ.model.modules())
    del modelQ
    _free()
    return directory / "scales.pt", input_scales, results(results_file)


def test_exp_searches_applies_scales_quantizes_with_gptq_and_evaluates(scaled_experiment):
    scales, input_scales, rows = scaled_experiment
    assert len(torch.load(scales)) == 8 * 42
    assert input_scales == 3 * 42
    (row,) = rows
    assert (row["transform"], row["abits"]) == ("scaling", "8")
    assert float(row["wer"]) < WER_CEILING


def test_exp_with_scaling_humming_layers_and_cuda_graphs(scaled_experiment, tmp_path):
    scales, _, _ = scaled_experiment
    modelQ, results_file = run_experiment(
        config("parakeet", [
            *SCALING_SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={scales}", "transform.obtain_scales=False",
            "inference=humming", "model.generate_fn=generate_parakeet_cuda_graphs", "eval_dtype=float16",
        ]),
        results_dir=str(tmp_path),
    )
    try:
        modules = list(modelQ.model.modules())
        layers = [m for m in modules if isinstance(m, ASRQLinear)]
        assert len(layers) == 10 * 42
        assert sum(isinstance(m, InputScale) for m in modules) == 3 * 42
        (row,) = results(results_file)
        assert float(row["wer"]) < WER_CEILING
    finally:
        del modelQ
        _free()
