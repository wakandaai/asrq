"""Support for the model pipeline tests: configs from tests/configs, the small scale, and skip conditions.

The tests run asrq.experiment's learn_rotation_experiment and run_experiment, the functions behind
asrq/rot-exp.py and asrq/exp.py, on a real model at the smallest scale that still exercises every step:
8 calibration utterances (2 search steps, GPTQ on 8 utterances) and 8 evaluation utterances. Scaling runs
at W4A8: without a rotation, 4-bit activations break the models, and what is tested is the pipeline. They
need a GPU, the model and the Open ASR Leaderboard LibriSpeech split in the Hugging Face cache, and the
local calibration set; otherwise they are skipped. Deselect them with ``pytest -m "not integration"``.
"""

import csv
import os
from pathlib import Path
from typing import List, Sequence

import pytest
import torch
from hydra import compose, initialize_config_dir

CONFIG_DIR = Path(__file__).parent / "configs"
CALIBRATION = Path("outputs/calibration/librispeech_train_clean_360_2048")
HF_HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
DATASET = "hf-audio/open-asr-leaderboard"

SMALL = [
    "activation_bits=4",
    "calibration.num_samples=8",
    "transform.num_samples=8",
    "transform.batch_size=4",
    "transform.epochs=1",
    "transform.learning_rate=0.5",
]
SCALING_SMALL = [
    "transform=scaling",
    "activation_bits=8",
    "calibration.num_samples=8",
]
EVALUATE_8_UTTERANCES = [
    "eval_batches=1",
    "model.eval_batch_size=8",
    "create_audio_files=False",
    "eval_datasets=[{dataset:librispeech,split:test.clean}]",
]
# A broken rotation, fold or quantizer gives a WER near 100; the quantized models score ~2-4.
WER_CEILING = 15.0


def _cached(repo_id: str, kind: str) -> bool:
    return (HF_HUB / f"{kind}--{repo_id.replace('/', '--')}").exists()


def requirements(model_id: str) -> List:
    return [
        pytest.mark.integration,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
        pytest.mark.skipif(not _cached(model_id, "models"), reason=f"{model_id} is not in the Hugging Face cache"),
        pytest.mark.skipif(not _cached(DATASET, "datasets"), reason=f"{DATASET} is not in the Hugging Face cache"),
        pytest.mark.skipif(not CALIBRATION.exists(), reason="local calibration set not built"),
    ]


def config(model: str, overrides: Sequence[str]):
    """The tests/configs experiment config for ``model``, as exp.py and rot-exp.py compose it.

    The transform is rotation unless ``overrides`` selects another one, as SCALING_SMALL does.
    """
    transform = [] if any(o.startswith("transform=") for o in overrides) else ["transform=rotation"]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        return compose(config_name="config", overrides=[
            f"model={model}", "quantizer=gptq", *transform, f"calibration.path={CALIBRATION}", *overrides,
        ])


def results(path: str) -> List[dict]:
    with open(path) as handle:
        return list(csv.DictReader(handle))


def release(*objects) -> None:
    for obj in objects:
        del obj
    torch.cuda.empty_cache()
