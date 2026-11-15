"""Self-supervised pretraining of a ResNet on ImageNet-1K or CIFAR-10 with PEIRA, joint self-distillation, SIGReg or VICReg.

A frozen-feature linear probe is trained online for monitoring; the paper's numbers come
from ``probe.py`` on the final checkpoint.

Usage:
    python train.py --fname configs/in1k/peira_resnet50.yaml
    python train.py --fname configs/cifar10/peira_resnet18.yaml --meta.seed=42
    torchrun --nproc_per_node=8 train.py --fname configs/in1k/peira_resnet50.yaml --meta.seed=1000
"""

import json
import os
import time
from itertools import islice
from typing import Callable, Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from peira_loss import PEIRALoss
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from data.datasets import ImageDataset, build_imagenet
from data.transforms import (
    DATASET_ABBREV,
    get_dataset_info,
    get_val_transforms,
    make_transforms,
)
from eb_jepa.losses.anticollapse import BCS, VICRegLoss, VICRegReferenceLoss
from eb_jepa.utils.checkpoint import (
    derive_resume_position,
    resume_training,
    save_training_state,
    should_save_intra_epoch,
    trainable_state_dict,
)
from eb_jepa.utils.config import (
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    resolve_experiment_folder,
)
from eb_jepa.utils.distributed import (
    cleanup_distributed,
    local_batch_size,
    make_sampler,
    setup_distributed,
    unwrap_state_dict,
    wrap_ddp,
)
from eb_jepa.utils.logging import get_logger
from eb_jepa.utils.optimizers import build_optimizer
from eb_jepa.utils.rank import EffectiveRankAccumulator
from eb_jepa.utils.training import (
    exit_if_training_complete,
    optimizer_step,
    setup_amp,
    setup_device,
    setup_seed,
)
from metrics import compute_spectral_metrics, evaluate_linear_probe
from models import ImageSSL, LinearProbe, TorchvisionResNet, init_weights

logger = get_logger(__name__)


def make_warmup_cosine_scheduler(
    optimizer, warmup_epochs, max_epochs, warmup_start_lr, base_lr, min_lr
):
    """Warmup + cosine annealing using built-in PyTorch schedulers."""
    warmup = LinearLR(
        optimizer,
        start_factor=max(warmup_start_lr / base_lr, 1e-8) if base_lr > 0 else 1e-8,
        end_factor=1.0,
        total_iters=max(warmup_epochs - 1, 1),
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(max_epochs - warmup_epochs, 1),
        eta_min=min_lr,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    linear_probe,
    scaler,
    device,
    epoch,
    loss_fn,
    use_amp=True,
    dtype=torch.float16,
    tqdm_silent=False,
    rank_acc: Optional[EffectiveRankAccumulator] = None,
    grad_clip: float = 1.0,
    channels_last: bool = False,
    probe_optimizer: torch.optim.Optimizer | None = None,
    save_callback: Callable[[int], None] | None = None,
    save_every_steps: int = 0,
    start_idx: int = 0,
    *,
    predictor_update: Callable[[], None] | None = None,
):
    """Train for one epoch (LR schedule stepped per optimizer step).

    Both views are batched through one forward pass; the online probe reads detached
    backbone features of the first view. ``start_idx`` skips the first batches of a
    resumed epoch.
    """
    model.train()
    linear_probe.train()

    def _to_dev(t):
        # channels_last only makes sense for 4D activations; skip for 1D labels etc.
        if channels_last and t.dim() == 4:
            return t.to(device, non_blocking=True, memory_format=torch.channels_last)
        return t.to(device, non_blocking=True)

    # Dynamic loss accumulator
    loss_totals = {}
    total_linear_loss = 0
    linear_correct = 0
    linear_correct_top5 = 0
    linear_total = 0
    last_grad_norm = None
    num_batches = len(train_loader)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}" + (f" (resumed @ idx={start_idx})" if start_idx else ""),
        disable=tqdm_silent,
        total=num_batches,
        initial=start_idx,
    )
    for batch_idx, ((view1, view2), target) in enumerate(
        islice(pbar, start_idx, num_batches), start=start_idx
    ):
        target = _to_dev(target)

        with autocast(device.type, enabled=use_amp, dtype=dtype):
            both_views = torch.cat([_to_dev(view1), _to_dev(view2)], dim=0)  # [2B, C, H, W]
            all_features, all_z = model(both_views)
            features, _ = all_features.chunk(2)  # [B, D_backbone], probe input
            z1, z2 = all_z.chunk(2)  # [B, D]
            loss_dict = loss_fn(z1, z2)
            loss = loss_dict["loss"]

        with torch.no_grad():
            features_frozen = features.detach().float()

        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs, target)

        _, predicted = linear_outputs.max(1)
        linear_correct_batch = predicted.eq(target).sum().item()
        _, top5_pred = linear_outputs.topk(5, dim=1)
        linear_correct_top5_batch = top5_pred.eq(target.unsqueeze(1)).any(1).sum().item()

        total_loss_batch = loss + linear_loss

        optimizer.zero_grad()
        if probe_optimizer is not None:
            probe_optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        # Step the online probe outside optimizer_step so grad_clip applies to the
        # backbone only. optimizer_step clips the union of every param group of every
        # optimizer it is handed, so folding the probe in would rescale its gradients
        # by the shared clip ratio (effective probe lr = probe_lr x clip_ratio) and
        # depress the online accuracy without touching the learned features.
        if probe_optimizer is not None:
            scaler.step(probe_optimizer)
        scale_before = scaler.get_scale() if predictor_update is not None else None
        last_grad_norm = optimizer_step(scaler, [optimizer], model, grad_clip=grad_clip)
        if predictor_update is not None and scaler.get_scale() >= scale_before:
            # Keep the predictor fixed during encoder backward; skip on AMP overflow.
            predictor_update()

        scheduler.step()

        if rank_acc is not None:
            rank_acc.accumulate("train/backbone_effective_rank", features.detach())
            rank_acc.accumulate("train/projector_effective_rank", z1.detach())

        # Update metrics dynamically based on loss_dict keys
        for key, value in loss_dict.items():
            if key not in loss_totals:
                loss_totals[key] = 0
            loss_totals[key] += value.item()
        total_linear_loss += linear_loss.item()

        # Update linear probe accuracy (pre-computed under autocast)
        linear_total += target.size(0)
        linear_correct += linear_correct_batch
        linear_correct_top5 += linear_correct_top5_batch

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Linear": f"{linear_loss.item():.4f}",
                "Acc": f"{100.0 * linear_correct / linear_total:.2f}%",
            }
        )

        # Intra-epoch checkpoint (latest.pth.tar only).
        _global_step = epoch * num_batches + batch_idx
        if save_callback is not None and should_save_intra_epoch(_global_step, save_every_steps):
            save_callback(_global_step)

    # Build return dict dynamically. `processed` is the number of iterations
    # actually run (== num_batches when start_idx=0, less on a resumed epoch).
    processed = batch_idx + 1 - start_idx
    metrics = {key: total / processed for key, total in loss_totals.items()}
    metrics["linear_loss"] = total_linear_loss / processed
    metrics["linear_acc"] = 100.0 * linear_correct / linear_total
    metrics["linear_acc_top5"] = 100.0 * linear_correct_top5 / linear_total
    if last_grad_norm is not None:
        metrics["grad_norm"] = last_grad_norm
        metrics["grad_norm_clip_ratio"] = min(1.0, grad_clip / (last_grad_norm + 1e-12))

    return metrics


def run(
    fname: str = "configs/in1k/peira_resnet50.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """Pretrain an ImageSSL model on CIFAR-10 or ImageNet-1K.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., optim.epochs=50)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Subcfg aliases — destructure once so the body reads as cfg_X.field
    # instead of cfg.X.field. OmegaConf returns views, so mutating cfg_X
    # mutates cfg.
    cfg_meta = cfg["meta"]
    cfg_data = cfg["data"]
    cfg_model = cfg["model"]
    cfg_optim = cfg["optim"]
    cfg_logging = cfg["logging"]
    cfg_loss = cfg.get("loss", None)

    # Distributed setup
    local_rank, world_size, is_main = setup_distributed()

    # Setup using shared utilities
    device = setup_device(cfg_meta.device, local_rank=int(os.environ.get("LOCAL_RANK", 0)))
    setup_seed(cfg_meta.seed, rank=local_rank)

    exp_dir, _ = resolve_experiment_folder("image_jepa", cfg, folder)

    # Idempotency gate: relaunching a completed run is a no-op.
    if exit_if_training_complete(exp_dir, cfg_optim.epochs, is_main):
        return {"status": "already_completed"}

    dataset_name = cfg_data.get("dataset", "cifar10")
    ds = DATASET_ABBREV.get(dataset_name, dataset_name)
    image_size, num_classes = get_dataset_info(dataset_name)
    logger.info(
        f"Loading {dataset_name} dataset (image_size={image_size}, num_classes={num_classes})..."
    )

    val_transforms = get_val_transforms(dataset_name)

    # Config data_dir takes precedence; fall back to EBJEPA_DSETS env var
    data_dir = cfg_data.get("data_dir") or os.environ.get("EBJEPA_DSETS")
    logger.info(f"Using data directory: {data_dir}")

    if dataset_name == "cifar10":
        base_train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=None)
        val_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=val_transforms)
    elif dataset_name == "imagenet1k":
        base_train_dataset = build_imagenet(data_dir, "train")
        val_dataset = build_imagenet(data_dir, "val", val_transforms)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_dataset = ImageDataset(base_train_dataset, make_transforms(dataset_name))

    per_gpu_bs = local_batch_size(cfg_data.batch_size)
    sampler = make_sampler(train_dataset)
    nw = cfg_data.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    log_data_info(
        dataset_name,
        len(train_loader),
        cfg_data.batch_size,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )

    # Initialize model
    logger.info("Initializing model...")
    if cfg_model.type != "resnet":
        raise ValueError(f"Unknown model type: {cfg_model.type}")
    backbone = TorchvisionResNet(
        arch=cfg_model.get("resnet_arch", "resnet18"), image_size=image_size
    )
    features_dim = backbone.features_dim
    model = ImageSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=cfg_model.proj_hidden_dim,
        proj_output_dim=cfg_model.proj_output_dim,
        proj_final_bias=cfg_model.get("proj_final_bias", True),
    )

    if not cfg_model.use_projector:
        model.projector = nn.Identity()

    init_scale = cfg.get("init", {}).get("scale")
    if init_scale is not None and init_scale != 1.0:
        logger.info(f"Re-initializing model weights with scale={init_scale}")
        model.apply(lambda m: init_weights(m, scale=init_scale))

    channels_last = bool(cfg_model.get("channels_last", False))
    if channels_last:
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in backbone.parameters())
    projector_params = (
        sum(p.numel() for p in model.projector.parameters()) if cfg_model.use_projector else 0
    )
    log_model_info(model, {"encoder": encoder_params, "projector": projector_params})

    # Save and log configuration
    if is_main:
        OmegaConf.save(cfg, exp_dir / "config.yaml")
    log_config(cfg)

    linear_probe = LinearProbe(feature_dim=features_dim, num_classes=num_classes).to(device)

    dtype, use_amp, scaler = setup_amp(cfg, device)

    backbone_group = {
        "params": list(model.parameters()),
        "weight_decay": float(cfg_optim.get("weight_decay", 0.0)),
        "lr": cfg_optim.lr,
    }

    # When ``probe_separate`` is true, the online linear probe gets its own
    # SGD optimizer (fixed LR, no weight decay, no cosine schedule), so it is
    # fully decoupled from the backbone's LARS recipe. This mirrors the
    # paper's offline-probe protocol (SGD + constant-ish LR) and fixes the
    # online probe's tendency to under-read when coupled to the backbone's
    # cosine decay.
    probe_separate = bool(cfg_optim.get("probe_separate", False))
    if probe_separate:
        probe_lr = cfg_optim.get("probe_lr", 0.3)
        probe_momentum = cfg_optim.get("probe_momentum", 0.9)
        probe_optimizer = torch.optim.SGD(
            linear_probe.parameters(),
            lr=probe_lr,
            momentum=probe_momentum,
            weight_decay=0.0,
        )
        param_groups = [backbone_group]
        logger.info(
            f"Probe is separately optimized: SGD lr={probe_lr}, mom={probe_momentum}, wd=0 (no schedule)"
        )
    else:
        probe_optimizer = None
        probe_group = {"params": list(linear_probe.parameters()), "weight_decay": 0.0, "lr": 0.1}
        param_groups = [backbone_group, probe_group]
    optimizer = build_optimizer(cfg_optim, param_groups)

    steps_per_epoch = len(train_loader)
    total_steps = cfg_optim.epochs * steps_per_epoch
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=cfg_optim.warmup_epochs * steps_per_epoch,
        max_epochs=total_steps,
        warmup_start_lr=cfg_optim.warmup_start_lr,
        base_lr=cfg_optim.lr,
        min_lr=cfg_optim.min_lr,
    )

    # Initialize loss function
    if cfg_loss.type == "vicreg":
        loss_fn = VICRegLoss(
            sim_coeff=cfg_loss.get("sim_coeff", 1.0),
            std_coeff=cfg_loss.std_coeff,
            cov_coeff=cfg_loss.cov_coeff,
        )
    elif cfg_loss.type == "vicreg_ref":
        loss_fn = VICRegReferenceLoss(
            sim_coeff=cfg_loss.get("sim_coeff", 25.0),
            std_coeff=cfg_loss.get("std_coeff", 25.0),
            cov_coeff=cfg_loss.get("cov_coeff", 1.0),
        )
    elif cfg_loss.type == "bcs":
        loss_fn = BCS(
            num_slices=int(cfg_loss.get("num_slices", 1024)),
            lmbd=cfg_loss.lmbd,
        )
    elif cfg_loss.type == "peira":
        latent_dim = cfg_model.proj_output_dim if cfg_model.use_projector else features_dim
        loss_fn = PEIRALoss(
            latent_dim=latent_dim,
            T_max=total_steps,
            reg=cfg_loss.reg,
            lr=cfg_loss.lr,
            lr_min=cfg_loss.lr_min,
            predictor_cov=cfg_loss.get("predictor_cov", "ema"),
        )
    elif cfg_loss.type == "self_distillation_joint":
        from peira_loss import JointSelfDistillationLoss

        latent_dim = cfg_model.proj_output_dim if cfg_model.use_projector else features_dim
        loss_fn = JointSelfDistillationLoss(
            latent_dim=latent_dim,
            reg=cfg_loss.reg,
            symmetric=cfg_loss.get("symmetric", False),
            lr=cfg_loss.lr,
            lr_min=cfg_loss.get("lr_min", cfg_loss.lr),
            T_max=total_steps,
        )
    else:
        raise ValueError(f"Unknown loss type: {cfg_loss.type}")

    loss_fn = loss_fn.to(device)

    # Auto-resume from checkpoint (no-op if no checkpoint exists). ``loss_fn`` is
    # threaded through so a stateful loss (PEIRALoss: running cov buffers +
    # inner-SGD step counter) resumes its schedule instead of restarting it;
    # stateless losses (VICReg/SIGReg) are a no-op on both save and load.
    start_epoch, ckpt_info = resume_training(
        exp_dir,
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        loss_fn=loss_fn,
    )
    if "linear_probe_state_dict" in ckpt_info:
        linear_probe.load_state_dict(unwrap_state_dict(ckpt_info["linear_probe_state_dict"]))
    if probe_optimizer is not None and "probe_optimizer_state_dict" in ckpt_info:
        probe_optimizer.load_state_dict(ckpt_info["probe_optimizer_state_dict"])
        logger.info("Restored online-probe optimizer state")

    # Training loop
    logger.info(f"Starting training for {cfg_optim.epochs} epochs...")
    start_time = time.time()
    tqdm_silent = cfg_logging.get("tqdm_silent", False) or not is_main

    # Compile model (before DDP wrapping)
    if torch.cuda.is_available() and cfg_model.get("compile", False):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)

    model = wrap_ddp(
        model,
        device,
        sync_batchnorm=True,
        gradient_as_bucket_view=bool(cfg_model.get("gradient_as_bucket_view", False)),
        static_graph=bool(cfg_model.get("static_graph", False)),
    )
    linear_probe = wrap_ddp(linear_probe, device)

    rank_acc = EffectiveRankAccumulator()
    grad_clip = cfg_optim.get("grad_clip", 1.0)
    # PEIRA on CIFAR-10 clips only from ``clip_epoch`` on: clipping from the start
    # plateaus performance (paper, App. G.1).
    clip_epoch = int(cfg_optim.get("clip_epoch", 0))
    if grad_clip is not None:
        logger.info(f"Gradient clipping enabled: max_norm={grad_clip} from epoch {clip_epoch}")
    spectral_every = int(cfg_logging.get("spectral_every", 0))

    # Intra-epoch save support — overrides the end-of-epoch resume position
    # when a mid-epoch checkpoint was loaded; no-op otherwise.
    save_every_steps = cfg_logging.get("save_every_steps", 0)
    start_epoch, start_idx_in_first_epoch = derive_resume_position(
        ckpt_info,
        save_every_steps,
        steps_per_epoch,
        start_epoch,
    )
    # Mutable holder so end-of-epoch val accuracy survives across iterations
    # and is available to intra-epoch saves (uses the previous epoch's value).
    val_state = {"acc": 0.0, "acc_top5": 0.0}

    def _checkpoint(epoch_: int, global_step_: int, *, end_of_epoch: bool = False) -> None:
        """Save model + auxiliary state. Single source of truth for ckpt_extra."""
        if not is_main:
            return
        is_final = end_of_epoch and (epoch_ + 1) >= int(cfg_optim.epochs)
        # Persist loss state ONLY when non-empty: PEIRALoss carries its running
        # cov buffers + ``_step`` extra-state, while VICReg/SIGReg return an
        # empty state_dict and add nothing to the checkpoint (no-op guard).
        loss_state = loss_fn.state_dict()
        loss_extra = {"loss_state_dict": loss_state} if loss_state else {}
        # Without it, a requeue drops the online probe's SGD momentum buffer.
        if probe_optimizer is not None:
            loss_extra["probe_optimizer_state_dict"] = probe_optimizer.state_dict()
        save_training_state(
            exp_dir,
            model,
            optimizer,
            epoch_,
            save_every=int(cfg_logging.get("save_every", 0)) if end_of_epoch else 0,
            scheduler=scheduler,
            scaler=scaler,
            step=global_step_,
            force_versioned_epoch=(
                int(cfg_optim.epochs)
                if is_final and int(cfg_logging.get("save_every", 0)) > 0
                else None
            ),
            training_complete=is_final,
            linear_probe_state_dict=trainable_state_dict(linear_probe),
            linear_val_acc=val_state["acc"],
            linear_val_acc_top5=val_state["acc_top5"],
            **loss_extra,
        )

    for epoch in range(start_epoch, cfg_optim.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        skip_n = start_idx_in_first_epoch if epoch == start_epoch else 0
        start_idx_in_first_epoch = 0  # only skip on the first resumed epoch

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            linear_probe,
            scaler,
            device,
            epoch,
            loss_fn,
            use_amp,
            dtype,
            tqdm_silent,
            rank_acc=rank_acc,
            grad_clip=grad_clip if epoch >= clip_epoch else None,
            channels_last=channels_last,
            probe_optimizer=probe_optimizer,
            save_callback=lambda gs: _checkpoint(epoch, gs),
            save_every_steps=save_every_steps,
            start_idx=skip_n,
            predictor_update=(
                loss_fn.step_predictor if cfg_loss.type == "self_distillation_joint" else None
            ),
        )

        # Evaluate linear probe on validation set
        val_acc, val_acc_top5, val_loss = evaluate_linear_probe(
            model, linear_probe, val_loader, device, use_amp
        )
        val_state["acc"] = val_acc
        val_state["acc_top5"] = val_acc_top5

        # Log metrics - dynamically add train_ prefix to all train_metrics keys
        log_dict = {"epoch": epoch}
        for key, value in train_metrics.items():
            if "acc" in key:
                log_dict[f"{ds}_train_{key}"] = value
            else:
                log_dict[f"train_{key}"] = value
        log_dict["val_loss"] = val_loss
        log_dict[f"{ds}_val_acc"] = val_acc
        log_dict[f"{ds}_val_acc_top5"] = val_acc_top5
        log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]
        log_dict.update(rank_acc.compute())
        if is_main:
            with open(exp_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(log_dict, default=float) + "\n")

        # Spectrum and alignment of the projector covariances (CIFAR-10 diagnostics figure).
        if is_main and spectral_every > 0 and epoch % spectral_every == 0:
            with torch.no_grad():
                spectral = compute_spectral_metrics(model, train_loader, device)
            with open(exp_dir / "spectral.jsonl", "a") as f:
                row = {"epoch": epoch, **{k: v.tolist() for k, v in spectral.items()}}
                f.write(json.dumps(row) + "\n")

        # Log progress
        if is_main and epoch % cfg_logging.log_every == 0:
            elapsed = time.time() - start_time
            log_epoch(
                epoch,
                {
                    "loss": train_metrics["loss"],
                    f"{ds}_val_acc": val_acc,
                    f"{ds}_val_acc_top5": val_acc_top5,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                total_epochs=cfg_optim.epochs,
                elapsed_time=elapsed,
            )

        # Always replace latest.pth.tar; versioned snapshots are opt-in.
        _checkpoint(epoch, (epoch + 1) * steps_per_epoch - 1, end_of_epoch=True)

    logger.info("Training completed!")
    cleanup_distributed()
    return None


if __name__ == "__main__":
    fire.Fire(run)