from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

from eb_jepa.utils.distributed import unwrap_model, unwrap_state_dict
from eb_jepa.utils.logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    scaler: Optional[Any] = None,
    **extra_state,
) -> None:
    """Save a training checkpoint (model, optimizer, scheduler, scaler, extra_state)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    checkpoint.update(extra_state)

    # Atomic write: a crash mid-save leaves the previous checkpoint intact.
    tmp_path = path.parent / (path.name + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint. Returns dict with epoch, step, and extra_state.

    The returned 'epoch' is the epoch to resume training from (0-indexed).
    If no checkpoint exists, returns epoch=0 to start fresh.
    If a checkpoint exists with epoch=N, returns epoch=N+1 to resume from the next epoch.

    Args:
        loss_fn: Optional stateful loss (e.g. ``PEIRALoss`` covariance buffers and
            step counter) restored from ``loss_state_dict`` when present.
    """
    path = Path(path)
    if not path.exists():
        logger.warning(f"Checkpoint not found: {path}")
        return {"epoch": 0, "step": 0, "resumed": False}

    map_location = device if device else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Handle compiled/DDP model state dicts
    state_dict = unwrap_state_dict(checkpoint.get("model_state_dict", {}))

    msg = model.load_state_dict(state_dict)
    logger.info(f"Loaded model state from: {path} with msg: {msg}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        msg = optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Restored optimizer state with msg: {msg}")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        msg = scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Restored scheduler state with msg: {msg}")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        msg = scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info(f"Restored scaler state with msg: {msg}")

    if loss_fn is not None and "loss_state_dict" in checkpoint:
        msg = loss_fn.load_state_dict(checkpoint["loss_state_dict"], strict=False)
        logger.info(f"Restored loss_fn state with msg: {msg}")

    return {
        "epoch": checkpoint.get("epoch", 0) + 1,  # Resume from next epoch
        "step": checkpoint.get("step", 0),
        "resumed": True,
        **{
            k: v
            for k, v in checkpoint.items()
            if k
            not in [
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "scaler_state_dict",
                "loss_state_dict",
                "epoch",
                "step",
            ]
        },
    }


def resume_training(
    folder: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    loss_fn: Optional[nn.Module] = None,
) -> tuple[int, Dict[str, Any]]:
    """Auto-resume from ``folder / latest.pth.tar``.

    No checkpoint = start from epoch 0 (no-op).  Returns ``(start_epoch, ckpt_info)``.
    """
    ckpt_info = load_checkpoint(
        Path(folder) / "latest.pth.tar",
        model,
        optimizer,
        scheduler,
        scaler,
        device=device,
        loss_fn=loss_fn,
    )
    start_epoch = ckpt_info.get("epoch", 0)
    return start_epoch, ckpt_info


def save_training_state(
    folder: Union[str, Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    save_every: int = 0,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    step: int = 0,
    force_versioned_epoch: Optional[int] = None,
    **extra_state,
) -> None:
    """Save ``latest.pth.tar`` always; also ``e-{epoch}.pth.tar`` every *save_every* epochs.

    ``force_versioned_epoch`` additionally writes ``e-{force_versioned_epoch}.pth.tar``
    regardless of ``save_every`` (used for the final checkpoint).
    """
    folder = Path(folder)
    names = ["latest.pth.tar"]
    if save_every > 0 and epoch % save_every == 0 and epoch > 0:
        names.append(f"e-{epoch}.pth.tar")
    if force_versioned_epoch is not None:
        names.append(f"e-{int(force_versioned_epoch)}.pth.tar")
    for name in names:
        save_checkpoint(
            folder / name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=step,
            scaler=scaler,
            **extra_state,
        )


def trainable_state_dict(module: nn.Module) -> dict[str, Any]:
    """Return ``module.state_dict()`` minus parameters with ``requires_grad=False``.

    Buffers (e.g. BatchNorm running stats, sincos position embeddings) are
    preserved unconditionally — they are not parameters and therefore have no
    ``requires_grad`` flag. Frozen parameters are dropped under the convention
    that they will be re-initialized from external sources at model rebuild
    time (e.g. pretrained weights loaded when the model is built).

    Operates on the unwrapped module so DDP/torch.compile prefixes are
    stripped from both the parameter names and the returned state-dict keys
    consistently.
    """
    inner = unwrap_model(module)
    frozen_param_names = {
        name for name, p in inner.named_parameters() if not p.requires_grad
    }
    full_state = inner.state_dict()
    return {k: v for k, v in full_state.items() if k not in frozen_param_names}


def derive_resume_position(
    ckpt_info: dict[str, Any],
    save_every_steps: int,
    steps_per_epoch: int,
    default_start_epoch: int,
) -> tuple[int, int]:
    """Compute ``(start_epoch, start_idx_in_first_epoch)`` for intra-epoch resume.

    When ``save_every_steps <= 0`` or no checkpoint was loaded, returns
    ``(default_start_epoch, 0)`` — preserving the end-of-epoch +1 semantics
    from ``load_checkpoint``. When ``save_every_steps > 0`` and the loaded
    checkpoint was written mid-epoch (``ckpt_info["step"]`` not on an epoch
    boundary), overrides ``start_epoch`` to the actual mid-epoch position so
    the for-epoch loop resumes into the saved epoch rather than the next one.
    Caller skips the first ``start_idx_in_first_epoch`` batches of the
    resumed epoch.
    """
    if save_every_steps <= 0 or not ckpt_info.get("resumed", False):
        return default_start_epoch, 0
    next_step = ckpt_info.get("step", 0) + 1
    derived_idx = next_step % steps_per_epoch
    if derived_idx == 0:
        # End-of-epoch save: existing +1 semantics from load_checkpoint apply.
        return default_start_epoch, 0
    return next_step // steps_per_epoch, derived_idx


def should_save_intra_epoch(global_step: int, save_every_steps: int) -> bool:
    """Gate predicate for the intra-epoch save call site.

    False when ``save_every_steps <= 0`` (feature disabled) or
    ``global_step == 0`` (don't save before any training has happened).
    Otherwise True iff ``global_step`` is a multiple of ``save_every_steps``.
    """
    return (
        save_every_steps > 0 and global_step > 0 and global_step % save_every_steps == 0
    )


def is_training_complete(folder: Union[str, Path], total_epochs: int) -> bool:
    """Recognize completed latest-only runs and historical final snapshots."""
    if folder is None or total_epochs is None or total_epochs <= 0:
        return False
    folder = Path(folder)
    if (folder / f"e-{int(total_epochs)}.pth.tar").exists():
        return True
    latest = folder / "latest.pth.tar"
    if not latest.exists():
        return False
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    return bool(
        checkpoint.get("training_complete", False)
        and checkpoint.get("epoch", -1) + 1 >= total_epochs
    )
