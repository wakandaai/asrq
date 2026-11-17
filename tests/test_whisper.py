"""Whisper-large-v3 through rot-exp.py and exp.py, at small scale.

Runs asrq.experiment's learn_rotation_experiment and run_experiment (the bodies of asrq/rot-exp.py and
asrq/exp.py) on the real model: a rotation is learned and saved, then applied, GPTQ-quantized and
evaluated with fake W4A4. The scaling transform goes through exp.py the same way, at W4A8: scales searched, saved and applied
(an InputScale after each fc2's activation), GPTQ, evaluation. See tests/_pipeline.py for the scale and
skip conditions.

The evaluation set holds the first 8 test-clean utterances, which are its longest (30.6-35.0 s): short-form
generation drops everything past Whisper's 30 s window, so the unquantized model itself scores ~17.6% WER
on them. The quantized pipeline is therefore held to the unquantized model's WER on the same utterances
plus WER_MARGIN, not to an absolute ceiling; a broken rotation or quantizer scores near 100.
"""

import gc

import pytest
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import asrq.evaluation.openasr as openasr
from _pipeline import EVALUATE_8_UTTERANCES, SCALING_SMALL, SMALL, config, requirements, results
from asrq.experiment import learn_rotation_experiment, run_experiment
from asrq.transforms.rotation.utils import OnlineHadamard, ResidualStreamRotation
from asrq.transforms.scaling.base import InputScale

MODEL = "openai/whisper-large-v3"
WER_MARGIN = 5.0
pytestmark = requirements(MODEL)


def _free():
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def unquantized_wer():
    """The unquantized model's WER on the same 8 utterances, as the quantized tests evaluate them."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="sdpa").cuda()
    result = openasr.evaluate_model(
        model.eval(), batch_size=8, dataset="librispeech", split="test.clean", eval_id="unquantized",
        save_results_manifest=False, save_results_metrics=False, processor=AutoProcessor.from_pretrained(MODEL),
        generate_fn=openasr.generate_whisper, batches_to_eval=1,
    )
    del model
    _free()
    return result["wer"]


@pytest.fixture(scope="module")
def learned_rotation(tmp_path_factory):
    path = tmp_path_factory.mktemp("rotation") / "whisper.pt"
    torch.manual_seed(0)
    modelQ = learn_rotation_experiment(config("whisper", [*SMALL, f"transform.path={path}"]))
    del modelQ
    _free()
    return path


def test_rot_exp_learns_and_saves_r1_r2_and_the_online_hadamard_signs(learned_rotation):
    checkpoint = torch.load(learned_rotation, weights_only=False)
    R1 = checkpoint["R1"].double()
    assert R1.shape == (1280, 1280)
    assert (R1.T @ R1 - torch.eye(1280, dtype=torch.float64)).abs().max() < 1e-4
    names = set(checkpoint["R2s"])
    assert sum(".encoder." in n for n in names) == 32 and sum(".decoder." in n for n in names) == 64
    assert all(R2.shape == (64, 64) for R2 in checkpoint["R2s"].values())
    assert checkpoint["learn_r2"] and checkpoint["fc2_online_hadamard"]
    assert checkpoint["hadamard_sign_seed"] > 0
    assert checkpoint["activation_quantization"]["groupwise_roles"] == ["attn_out", "fc2"]


def test_exp_rotates_quantizes_with_gptq_and_evaluates(learned_rotation, unquantized_wer, tmp_path):
    modelQ, results_file = run_experiment(
        config("whisper", [*SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={learned_rotation}"]),
        results_dir=str(tmp_path),
    )
    try:
        modules = list(modelQ.model.modules())
        assert sum(isinstance(m, OnlineHadamard) for m in modules) == 64
        assert sum(isinstance(m, ResidualStreamRotation) for m in modules) == 3
        (row,) = results(results_file)
        assert (row["dataset"], row["split"], row["abits"]) == ("librispeech", "test.clean", "4")
        assert float(row["wer"]) < unquantized_wer + WER_MARGIN
    finally:
        del modelQ
        _free()


@pytest.fixture(scope="module")
def scaled_experiment(tmp_path_factory):
    """exp.py with the scaling transform: scales searched and saved, applied, GPTQ W4, fake A8."""
    directory = tmp_path_factory.mktemp("scaling")
    modelQ, results_file = run_experiment(
        config("whisper", [*SCALING_SMALL, *EVALUATE_8_UTTERANCES, f"transform.path={directory / 'scales.pt'}"]),
        results_dir=str(directory),
    )
    input_scales = sum(isinstance(m, InputScale) for m in modelQ.model.modules())
    del modelQ
    _free()
    return directory / "scales.pt", input_scales, results(results_file)


def test_exp_searches_applies_scales_quantizes_with_gptq_and_evaluates(scaled_experiment, unquantized_wer):
    scales, input_scales, rows = scaled_experiment
    assert len(torch.load(scales)) == 4 * 32 + 6 * 32 + 1
    assert input_scales == 64
    (row,) = rows
    assert (row["transform"], row["abits"]) == ("scaling", "8")
    assert float(row["wer"]) < unquantized_wer + WER_MARGIN
