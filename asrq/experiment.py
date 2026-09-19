"""The steps behind asrq/exp.py and asrq/rot-exp.py, as functions the scripts and the pipeline tests share.

The scripts only parse the Hydra config and seed; everything else runs here, so a test that calls these
functions exercises exactly what the scripts do.
"""

import datetime
import hashlib
import json
import os
from typing import Callable, Optional, Tuple

from omegaconf import DictConfig, OmegaConf, open_dict

import asrq.evaluation.openasr as openasr
from asrq.calibration.base import CalibConfig
from asrq.core.model import ModelQ
from asrq.evaluation.base import evaluate_openasr
from asrq.quantizers.base import QuantConfig
from asrq.transforms.base import BaseTransform, TransformConfig


def prepare_experiment_config(cfg: DictConfig, learn_rotation: bool) -> None:
    """Copy the settings the quantizer and transform read from elsewhere in the config, in place.

    The model config supplies the modules to leave unquantized and, for Conformer models, whether the
    Linear a rotation inserts after each block's output norm is quantized. The top-level activation
    settings are the ones evaluation uses; the rotation search gets the same ones, so a rotation is
    learned against the quantization it will be evaluated with.

    Args:
        cfg: The composed experiment config (quantizer, model, transform groups and top-level settings).
        learn_rotation: For the rotation transform, whether obtain_transform searches a rotation
            (rot-exp.py) or only a saved one is applied (exp.py). ``transform.search`` chooses how.
    """
    quantize_block_output_linear = cfg.model.get("quantize_block_output_linear", False)
    with open_dict(cfg.quantizer):
        cfg.quantizer.exclude_modules = cfg.model.exclude_modules
        cfg.quantizer.quantize_block_output_linear = quantize_block_output_linear
    with open_dict(cfg.transform):
        cfg.transform.model_name = cfg.model.name
        cfg.transform.quantize_block_output_linear = quantize_block_output_linear
        cfg.transform.wbits = cfg.quantizer.bits
        cfg.transform.abits = cfg.activation_bits
        cfg.transform.activation_group_size = cfg.activation_group_size
        cfg.transform.activation_symmetric = cfg.activation_symmetric
        cfg.transform.activation_groupwise_roles = cfg.activation_groupwise_roles
        cfg.transform.wgroup = cfg.quantizer.get("group_size", None)
        cfg.transform.wsymmetric = cfg.quantizer.get("symmetric", True)
        cfg.transform.wmethod = cfg.quantizer.name
        cfg.transform.wpercdamp = cfg.quantizer.get("percdamp", 0.01)
        cfg.transform.wblock_size = cfg.quantizer.get("block_size", 128)
        if cfg.transform.name == "rotation":
            cfg.transform.learn_rotation = learn_rotation


def quantized_model_fingerprint(cfg: DictConfig) -> str:
    """A hash of every setting that determines the quantized weights, for reusing a saved quantized model.

    Covers the model, the whole quantizer config (norm tweaking and block output refitting included), the calibration
    set and size, the transform config and the contents of the transform's saved file (a rotation checkpoint or
    scales), so a changed setting or a re-run search never loads stale weights. Evaluation and inference settings are
    left out. Call it after the transform is obtained, when its file exists.
    """
    transform = OmegaConf.to_container(cfg.transform, resolve=True)
    transform_file = transform.get("path") or ""
    file_hash = None
    if transform_file and os.path.isfile(transform_file):
        with open(transform_file, "rb") as handle:
            file_hash = hashlib.sha1(handle.read()).hexdigest()
    payload = {
        "format": ModelQ.QUANTIZED_FORMAT,
        "model": cfg.model.name,
        "quantizer": OmegaConf.to_container(cfg.quantizer, resolve=True),
        "calibration": OmegaConf.to_container(cfg.calibration, resolve=True),
        "transform": transform,
        "transform_file": file_hash,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def quantized_model_path(cfg: DictConfig) -> Tuple[Optional[str], str]:
    """Where exp.py saves the quantized model and loads it from, and the settings' fingerprint.

    ``quantized_path: auto`` names the file ``outputs/quantized/<model>-<fingerprint>.pt``; an explicit path is used as
    given and checked against the fingerprint when loaded; null or an empty string disables saving and loading.
    """
    fingerprint = quantized_model_fingerprint(cfg)
    setting = cfg.get("quantized_path", None)
    if not setting:
        return None, fingerprint
    if setting == "auto":
        return os.path.join("outputs", "quantized", f"{cfg.model.name.replace('/', '-')}-{fingerprint}.pt"), fingerprint
    return str(setting), fingerprint


def run_experiment(cfg: DictConfig, results_dir: str = "results/evaluations") -> Tuple[ModelQ, Optional[str]]:
    """exp.py: load the model, apply the transform, quantize, optionally convert to humming, evaluate.

    With ``quantized_path`` set, the quantized model is saved after quantization and, on a later run with the same
    settings, loaded instead of quantizing again; see quantized_model_path.

    Args:
        cfg: The composed experiment config; prepared in place.
        results_dir: Directory for the results CSV and a copy of the config.

    Returns:
        ``(modelQ, results_file)``; results_file is None when ``cfg.evaluate`` is off.
    """
    prepare_experiment_config(cfg, learn_rotation=False)
    modelQ = ModelQ.from_pretrained(
        cfg.model.name, QuantConfig.from_dictconfig(cfg.quantizer), CalibConfig.from_dictconfig(cfg.calibration)
    )

    if cfg.transform.name == "none":
        print("No transform will be applied.")
        modelQ.model.to("cuda")
    else:
        transform = BaseTransform.from_config(TransformConfig.from_dictconfig(cfg.transform))
        modelQ.model.to("cuda")
        transform.obtain_transform(modelQ)
        if cfg.transform.use:
            transform.apply_transform(modelQ)

    quantized_path, fingerprint = quantized_model_path(cfg)
    if quantized_path is not None and os.path.isfile(quantized_path):
        print(f"Loading the quantized model from {quantized_path} instead of quantizing")
        modelQ.load_quantized(quantized_path, fingerprint)
    else:
        modelQ.quantize()
        if quantized_path is not None:
            try:
                modelQ.save_quantized(quantized_path, fingerprint)
                print(f"Saved the quantized model to {quantized_path}")
            except Exception as error:
                print(f"WARNING: could not save the quantized model to {quantized_path}: {error}")
    if cfg.get("inference", "fake") == "humming":
        # real low-bit layers, for measuring speed; evaluation then skips its fake quantization
        replaced = modelQ.to_asrq_linear(cfg)
        print(f"Replaced {len(replaced)} layers with humming ASRQLinear")

    if not cfg.evaluate:
        return modelQ, None
    stem = os.path.join(
        results_dir,
        f"{cfg.model.name.replace('/', '-')}_{cfg.method}_{cfg.quantizer.name}_{cfg.transform.name}_"
        f"{cfg.quantizer.bits}_{cfg.activation_bits}-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    results_file = f"{stem}_results.csv"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{stem}_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    evaluate_openasr(modelQ=modelQ, cfg=cfg, generate_fn=getattr(openasr, cfg.model.generate_fn),
                     evaluation_results_file=results_file, create_audio_files=cfg.create_audio_files)
    return modelQ, results_file


def learn_rotation_experiment(cfg: DictConfig, reseed: Optional[Callable[[], None]] = None) -> ModelQ:
    """rot-exp.py: load the model and learn a rotation, saved to ``cfg.transform.path``.

    Args:
        cfg: The composed experiment config with the rotation transform; prepared in place.
        reseed: Called after the model is loaded and moved to the GPU, so the search's initial rotation
            and dataloader shuffle do not depend on what loading consumed from the random generators.

    Returns:
        The ModelQ, holding the model the search ran on.
    """
    if cfg.transform.name != "rotation":
        raise ValueError(f"rotation learning needs the rotation transform, got '{cfg.transform.name}'")
    prepare_experiment_config(cfg, learn_rotation=True)
    print(OmegaConf.to_yaml(cfg))
    transform = BaseTransform.from_config(TransformConfig.from_dictconfig(cfg.transform))
    modelQ = ModelQ.from_pretrained(
        cfg.model.name, QuantConfig.from_dictconfig(cfg.quantizer), CalibConfig.from_dictconfig(cfg.calibration)
    )
    modelQ.model.to("cuda")
    if reseed is not None:
        reseed()
    transform.obtain_transform(modelQ)
    print("Done with obtaining rotations")
    return modelQ
