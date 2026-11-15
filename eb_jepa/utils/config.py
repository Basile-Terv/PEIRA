from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def get_exp_name(cfg) -> str:
    """Short experiment name encoding the key hyperparameters (seed appended separately)."""
    parts = [
        cfg.data.get("dataset") or "unk",
        cfg.model.type,
        cfg.loss.type,
        cfg.optim.get("type", "lars"),
        "proj" if cfg.model.use_projector else "noproj",
        f"bs{cfg.data.batch_size}",
        f"ep{cfg.optim.epochs}",
    ]
    if cfg.model.use_projector:
        parts += [f"ph{cfg.model.proj_hidden_dim}", f"po{cfg.model.proj_output_dim}"]
    if cfg.loss.type == "vicreg":
        parts += [f"std{cfg.loss.std_coeff}", f"cov{cfg.loss.cov_coeff}"]
    elif cfg.loss.type == "bcs":
        parts.append(f"lmbd{cfg.loss.lmbd}")
    return "_".join(str(p) for p in parts)


def load_config(
    config_path: Union[str, Path],
    cli_overrides: Optional[Dict[str, Any]] = None,
    quiet: bool = False,
) -> DictConfig:
    """Load a YAML config with optional dot-notation overrides (e.g., ``optim.lr=0.1``).

    A top-level ``_base_: <relative path>`` key inherits from another YAML first.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    base = cfg.pop("_base_", None)
    if base is not None:
        # ``_base_: <path relative to this file>``: this file overrides its base.
        cfg = OmegaConf.merge(load_config(config_path.parent / base, quiet=True), cfg)
    if not quiet:
        logger.info(f"Loaded config from {config_path}")

    if cli_overrides:
        # Convert dot notation to nested dict
        override_dict = {}
        for key, value in cli_overrides.items():
            if isinstance(value, str):
                # Fire leaves YAML spellings ("false", "null") as strings; parse
                # them like the YAML file would, so "false" is not truthy.
                parsed = OmegaConf.from_dotlist([f"v={value}"])
                value = OmegaConf.to_container(parsed)["v"]
            keys = key.split(".")
            current = override_dict
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = value

        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_dict))
        if not quiet:
            logger.info(f"Applied {len(cli_overrides)} config overrides")

    return cfg


def get_checkpoints_dir() -> Path:
    """Get the base checkpoints directory from EBJEPA_CKPTS env variable."""
    return Path(os.environ.get("EBJEPA_CKPTS", "checkpoints"))


def get_unified_experiment_dir(
    example_name: str,
    sweep_name: str,
    exp_name: str,
    seed: int,
    dataset_name: str = "",
    base_dir: Union[str, Path, None] = None,
    create: bool = True,
) -> Path:
    """Create experiment dir: {base_dir}/{example_name}/{dataset_name}/{sweep_name}/{exp_name}_seed{seed}."""
    if base_dir is None:
        base_dir = get_checkpoints_dir()

    parts = [Path(base_dir), example_name]
    if dataset_name:
        parts.append(dataset_name)
    parts.extend([sweep_name, f"{exp_name}_seed{seed}"])

    exp_dir = Path(*parts).absolute()

    if create:
        exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def get_default_run_name(prefix: str = "dev") -> str:
    """Generate a timestamped run name with the given prefix."""
    return datetime.now().strftime(f"{prefix}_%Y-%m-%d_%H-%M")


def resolve_experiment_folder(
    example_name: str, cfg, folder: Union[str, Path, None] = None
) -> tuple[Path, str]:
    """Resolve experiment folder and name from config, creating the directory.

    Args:
        example_name: Top-level folder name (e.g. "image_jepa").
        cfg: Full config object.
        folder: Explicit folder path; if None, auto-generated.

    Returns:
        (folder, exp_name) tuple.
    """
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        exp_name = folder.name.rsplit("_seed", 1)[0]
        return folder, exp_name

    sweep_name = get_default_run_name("dev")
    exp_name = get_exp_name(cfg)
    folder = get_unified_experiment_dir(
        example_name=example_name,
        sweep_name=sweep_name,
        exp_name=exp_name,
        seed=cfg.meta.seed,
        dataset_name=cfg.data.get("dataset") or "",
    )
    return folder, exp_name


def log_config(cfg: Union[Dict, DictConfig], title: str = "Run Configuration") -> None:
    """Log configuration in a readable format."""
    logger.info("=" * 60)
    logger.info(f"⚙️  {title}:")
    logger.info("=" * 60)

    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    for section, values in cfg.items():
        if isinstance(values, dict):
            for key, value in values.items():
                logger.info(f"  {section}.{key}={value}")
        else:
            logger.info(f"  {section}={values}")
    logger.info("=" * 60)


def log_data_info(
    dataset_name: str,
    num_batches: int,
    batch_size: int,
    train_samples: Optional[int] = None,
    val_samples: Optional[int] = None,
) -> None:
    """Log dataset information.

    ``batch_size`` is the **global** (config-level) batch size.  When running
    multi-GPU, the per-GPU batch size is also displayed.
    """
    ws = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    bs_str = (
        f"{batch_size}" if ws == 1 else f"{batch_size} global ({batch_size // ws}/gpu)"
    )
    parts = [f"📦 Data: {dataset_name} | {num_batches} batches x {bs_str}"]
    if train_samples is not None and val_samples is not None:
        parts.append(f"train={train_samples:,} | val={val_samples:,}")
    logger.info(" | ".join(parts))


def log_model_info(model: nn.Module, param_counts: Dict[str, int]) -> None:
    """Log model structure and parameter counts."""
    logger.info(f"🧠 Model:\n{model}")
    param_str = " | ".join(f"{k}={v:,}" for k, v in param_counts.items())
    logger.info(f"🔢 Parameters: {param_str}")


def log_epoch(
    epoch: int,
    metrics: Dict[str, float],
    total_epochs: Optional[int] = None,
    elapsed_time: Optional[float] = None,
) -> None:
    """Log epoch summary: 📊 [Epoch 001/100] metric1=val1 | metric2=val2 | time=123.4s."""
    if total_epochs:
        prefix = f"[Epoch {epoch:03d}/{total_epochs}]"
    else:
        prefix = f"[Epoch {epoch:03d}]"

    metrics_str = format_metrics(metrics)

    if elapsed_time is not None:
        logger.info(f"📊 {prefix} {metrics_str} | time={elapsed_time:.1f}s")
    else:
        logger.info(f"📊 {prefix} {metrics_str}")


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics dict as 'loss=0.1234 | acc=95.12'."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)
