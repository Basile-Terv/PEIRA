from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import Optimizer

from eb_jepa.utils.checkpoint import is_training_complete
from eb_jepa.utils.distributed import cleanup_distributed
from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def setup_amp(cfg, device: torch.device) -> tuple[torch.dtype, bool, GradScaler]:
    """Configure mixed-precision training from config.

    Returns:
        (dtype, use_amp, scaler) tuple.
    """
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    use_scaler = use_amp and dtype == torch.float16
    scaler = GradScaler(device.type, enabled=use_scaler)
    logger.info(
        f"Using AMP with {dtype=}, scaler={'on' if use_scaler else 'off'}"
        if use_amp
        else "AMP disabled"
    )
    return dtype, use_amp, scaler


def optimizer_step(
    scaler: GradScaler,
    optimizers: Union[Optimizer, List[Optimizer]],
    model: Optional[nn.Module] = None,
    grad_clip: Optional[float] = None,
) -> Optional[float]:
    """Unscale, clip, step, and update scaler in one call.

    Returns the pre-clip gradient norm when *grad_clip* is set, else ``None``.
    """
    if isinstance(optimizers, Optimizer):
        optimizers = [optimizers]
    grad_norm = None
    if grad_clip is not None:
        for opt in optimizers:
            scaler.unscale_(opt)
        if model is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            ).item()
    for opt in optimizers:
        scaler.step(opt)
    scaler.update()
    return grad_norm


def setup_device(device: str = "auto", local_rank: int = 0) -> torch.device:
    """Set up the compute device. Options: 'auto', 'cuda', or 'cpu'.

    ``local_rank`` selects ``cuda:{local_rank}`` (e.g. under ``torchrun``). With the
    default 0, callers that restrict ``CUDA_VISIBLE_DEVICES`` per process keep ``cuda:0``.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device, local_rank) if device == "cuda" else torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    logger.info(f"Using device: {dev}")
    return dev


def setup_seed(seed: int, rank: int = 0) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Base random seed.
        rank: Distributed rank offset for data diversity across processes.
    """
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def exit_if_training_complete(
    folder: Path | str,
    total_epochs: int,
    is_main: bool,
) -> bool:
    """True (after distributed teardown) if a completed checkpoint already exists.

    Stops a requeued or relaunched finished run from restarting at epoch 0.
    """
    if not is_training_complete(folder, total_epochs):
        return False
    if is_main:
        logger.info(
            f"Training already complete at {folder} "
            "(completed checkpoint exists); exiting no-op."
        )
    cleanup_distributed()
    return True

