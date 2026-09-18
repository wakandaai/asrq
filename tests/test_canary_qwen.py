"""Canary-Qwen-2.5B through rot-exp.py and exp.py, at small scale.

Runs asrq.experiment's learn_rotation_experiment and run_experiment (the bodies of asrq/rot-exp.py and
asrq/exp.py) on the real model: a rotation with one R1 per residual stream (Conformer encoder, Qwen LLM)
is learned and saved, then applied with the LoRA adapters merged, GPTQ-quantized and evaluated with fake
W4A4, and again with humming ASRQLinear layers. The scaling transform goes through exp.py the same two
ways, at W4A8: scales searched, saved and applied (InputScales in the encoder, down_proj's folded into
up_proj), GPTQ, evaluation. See tests/_pipeline.py for the scale and skip conditions.

The evaluation set holds the first 8 test-clean utterances, its longest (30.6-35.0 s), whose transcripts
can outgrow the 128 tokens evaluation generates. As for Whisper, the quantized pipeline is held to the
unquantized model's WER on the same utterances plus WER_MARGIN; a broken rotation or quantizer scores
near 100.
"""

import gc

import pytest
import torch

import asrq.evaluation.openasr as openasr
from _pipeline import EVALUATE_8_UTTERANCES, SCALING_SMALL, SMALL, config, requirements, results
from asrq.core.linear import ASRQLinear
from asrq.experiment import learn_rotation_experiment, run_experiment
from asrq.models.nemo.canary_qwen import CanaryQwenQ
from asrq.transforms.rotation.utils import OnlineHadamard, ResidualStreamRotation
from asrq.transforms.scaling.base import InputScale

MODEL = "nvidia/canary-qwen-2.5b"
WER_MARGIN = 5.0
pytestmark = requirements(MODEL)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def unquantized_wer():
    """The unquantized model's WER on the same 8 utterances, as the quantized tests evaluate them."""
    model = CanaryQwenQ.load_model()[0].cuda()
    result = openasr.evaluate_model(
        model, batch_size=8, dataset="librispeech", split="test.clean", eval_id="unquantized",
        save_results_manifest=False, save_results_metrics=False, processor=None,
        generate_fn=openasr.generate_canaryqwen, batches_to_eval=1,
    )
    del model
    _free()
    return result["wer"]


@pytest.fixture(scope="module")
def learned_rotation(tmp_path_factory):
    path = tmp_path_factory.mktemp("rotation") / "canary_qwen.pt"
    torch.manual_seed(0)
    modelQ = learn_rotation_experiment(config("canary_qwen", [*SMALL, f"transform.path={path}"]))
    del modelQ
    _free()
    return path


def test_rot_exp_learns_one_r1_per_stream_r2s_and_the_online_hadamard_signs(learned_rotation):
    checkpoint = torch.load(learned_rotation, weights_only=False)
    assert {k: tuple(v.shape) for k, v in checkpoint["R1"].items()} == {"encoder": (1024, 1024), "llm": (2048, 2048)}
    for R1 in checkpoint["R1"].values():
        R1 = R1.double()
        assert (R1.T @ R1 - torch.eye(R1.shape[0], dtype=torch.float64)).abs().max() < 1e-4
    names = set(checkpoint["R2s"])
    assert sum(n.startswith("perception.") for n in names) == 32 and sum(n.startswith("llm.") for n in names) == 28
    assert checkpoint["hadamard_sign_seed"] > 0


def test_exp_rotates_merges_quantizes_with_gptq_and_evaluates(learned_rotation, unquantized_wer, tmp_path):
    modelQ, results_file = run_experiment(
        config("canary_qwen", [*SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={learned_rotation}"]),
        results_dir=str(tmp_path),
    )
    try:
        model = modelQ.model
        assert type(model.llm).__name__ == "Qwen3ForCausalLM"
        modules = dict(model.named_modules())
        assert sum(isinstance(m, OnlineHadamard) for m in modules.values()) == 3 * 32 + 28
        assert sum(isinstance(m, ResidualStreamRotation) for m in modules.values()) == 4
        quantized = set(modelQ.qparams)
        assert sum(n.startswith("perception.encoder.layers.") for n in quantized) == 11 * 32
        assert sum(n.startswith("llm.model.layers.") for n in quantized) == 7 * 28
        (row,) = results(results_file)
        assert (row["dataset"], row["split"], row["abits"]) == ("librispeech", "test.clean", "4")
        assert float(row["wer"]) < unquantized_wer + WER_MARGIN
    finally:
        del modelQ
        _free()


def test_exp_with_humming_layers(learned_rotation, unquantized_wer, tmp_path):
    modelQ, results_file = run_experiment(
        config("canary_qwen", [
            *SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={learned_rotation}", "inference=humming",
            "eval_dtype=float16",
        ]),
        results_dir=str(tmp_path),
    )
    try:
        layers = {n: m for n, m in modelQ.model.named_modules() if isinstance(m, ASRQLinear)}
        assert len(layers) == len(modelQ.activation_quantization_roles())
        assert sum(n.endswith("mlp.down_proj") and m.hadamard_block_size is not None for n, m in layers.items()) == 28
        (row,) = results(results_file)
        assert float(row["wer"]) < unquantized_wer + WER_MARGIN
    finally:
        del modelQ
        _free()


@pytest.fixture(scope="module")
def scaled_experiment(tmp_path_factory):
    """exp.py with the scaling transform: scales searched and saved, applied, GPTQ W4, fake A8."""
    directory = tmp_path_factory.mktemp("scaling")
    modelQ, results_file = run_experiment(
        config("canary_qwen", [*SCALING_SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={directory / 'scales.pt'}"]),
        results_dir=str(directory),
    )
    input_scales = sum(isinstance(m, InputScale) for m in modelQ.model.modules())
    del modelQ
    _free()
    return directory / "scales.pt", input_scales, results(results_file)


def test_exp_searches_applies_scales_quantizes_with_gptq_and_evaluates(scaled_experiment, unquantized_wer):
    scales, input_scales, rows = scaled_experiment
    assert len(torch.load(scales)) == 8 * 32 + 4 * 28
    assert input_scales == 3 * 32
    (row,) = rows
    assert (row["transform"], row["abits"]) == ("scaling", "8")
    assert float(row["wer"]) < unquantized_wer + WER_MARGIN


def test_exp_with_scaling_and_humming_layers(scaled_experiment, unquantized_wer, tmp_path):
    scales, _, _ = scaled_experiment
    modelQ, results_file = run_experiment(
        config("canary_qwen", [
            *SCALING_SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={scales}", "transform.obtain_scales=False",
            "inference=humming", "eval_dtype=float16",
        ]),
        results_dir=str(tmp_path),
    )
    try:
        modules = list(modelQ.model.modules())
        layers = [m for m in modules if isinstance(m, ASRQLinear)]
        assert len(layers) == len(modelQ.activation_quantization_roles())
        assert sum(isinstance(m, InputScale) for m in modules) == 3 * 32
        (row,) = results(results_file)
        assert float(row["wer"]) < unquantized_wer + WER_MARGIN
    finally:
        del modelQ
        _free()
