"""Weights & Biases logging, off unless ``wandb.enabled`` is set in the config.

Every stage logs through this module, so nothing else imports wandb and nothing changes when it is off: ``log``
and ``summary`` are cheap no-ops until ``start`` opens a run. A run covers one script: rot-exp.py logs a
rotation search, exp.py logs a quantization and its evaluation.

What each stage logs is documented in dev/docs/wandb_logging.md; the keys are namespaced by stage, and the metrics
of a stage that repeats (a generation, a block, a split) carry their own counter as the x axis.
"""

from contextlib import contextmanager
from typing import Any, Dict, Mapping, Optional

_run = None
_defined = set()


def enabled() -> bool:
    """Whether a run is open."""
    return _run is not None


@contextmanager
def start(cfg, job_type: str, name: Optional[str] = None):
    """Open a run for this script, or nothing when logging is off or wandb is missing.

    Args:
        cfg: The composed experiment config; ``cfg.wandb`` holds project, entity, group, tags and mode, and the
            whole config is stored as the run's config.
        job_type: ``"rotation"`` for rot-exp.py, ``"quantize"`` for exp.py.
        name: The run's name; a default is built from the model, method and job type.
    """
    global _run
    settings = cfg.get("wandb", None)
    if not settings or not settings.get("enabled", False) or _run is not None:
        yield None
        return
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb.enabled is set but wandb is not installed; logging is off")
        yield None
        return
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(cfg, resolve=True)
    _run = wandb.init(
        project=settings.get("project", "asrq"),
        entity=settings.get("entity", None),
        group=settings.get("group", None),
        tags=list(settings.get("tags", []) or []),
        mode=settings.get("mode", "online"),
        job_type=job_type,
        name=name or f"{cfg.model.name.split('/')[-1]}-{cfg.get('method', job_type)}",
        config=container,
    )
    try:
        yield _run
    finally:
        _defined.clear()
        run, _run = _run, None
        run.finish()


def step_metric(prefix: str, counter: str) -> None:
    """Make ``counter`` the x axis of every metric under ``prefix``; once per run and key."""
    if _run is None or prefix in _defined:
        return
    import wandb

    _defined.add(prefix)
    wandb.define_metric(counter)
    wandb.define_metric(f"{prefix}/*", step_metric=counter)


def log(data: Mapping[str, Any], commit: bool = True) -> None:
    """Log one row of metrics; a no-op when no run is open."""
    if _run is None:
        return
    _run.log(dict(data), commit=commit)


def summary(data: Mapping[str, Any]) -> None:
    """Set run summary values, which end up in the run's table; a no-op when no run is open."""
    if _run is None:
        return
    for key, value in data.items():
        _run.summary[key] = value


def histogram(values) -> Optional[Any]:
    """A wandb histogram of a tensor, or None when no run is open."""
    if _run is None:
        return None
    import wandb

    return wandb.Histogram(values.detach().float().cpu().numpy())


def config_update(data: Dict[str, Any]) -> None:
    """Add resolved settings the config did not hold, such as a rotation checkpoint's search record."""
    if _run is None:
        return
    _run.config.update(data, allow_val_change=True)
