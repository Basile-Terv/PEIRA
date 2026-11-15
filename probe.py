"""
Linear probes on a frozen, pretrained backbone: ImageNet-1K and LeJEPA-8 transfer.

Both protocols share backbone loading, feature extraction and metrics; they differ by
dataset and readout optimizer.

``--datasets imagenet1k`` (default): linear evaluation of VICReg (Bardes et al., 2022,
Appendix E), the paper's ImageNet top-1:
- 100 epochs, batch size 1024, SGD, momentum=0.9, wd=1e-6
- base_lr 0.25 (VICReg); effective lr = base_lr * bs / 256, cosine LR annealing
- train augmentation: RandomResizedCrop(224, scale=0.08-1.0) + HorizontalFlip
- eval augmentation: Resize(256, bicubic) + CenterCrop(224)

``--datasets lejepa8`` (or a comma list of ``TRANSFER_DATASETS`` keys): features are
extracted once with the eval augmentation, standardized, and fed to
a linear head trained with Adam (lr 1e-3, 100 epochs, cosine); weight decay is picked in
{0, 1e-4, 1e-2} on a 10% train holdout, then the head is refit on the full train split.
Metric: top-1 (CIFAR, Food101) or mean per-class recall (other sets). The JSON reports
each dataset and the mean.

Usage:
    torchrun --nproc_per_node=8 probe.py \\
        --ckpt /path/to/latest.pth.tar --data-dir $EBJEPA_DSETS/ImageNet
    python probe.py --ckpt /path/to/latest.pth.tar \\
        --datasets lejepa8 --transfer-root $EBJEPA_DSETS/transfer --out transfer.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.datasets import TRANSFER_DATASETS, build_imagenet, build_transfer_dataset
from data.transforms import get_val_transforms
from eb_jepa.utils.distributed import (
    cleanup_distributed,
    local_batch_size,
    make_sampler,
    setup_distributed,
)
from eb_jepa.utils.logging import get_logger
from models import RESNET_FEATURES_DIM, TorchvisionResNet, load_pretrained

logger = get_logger(__name__)


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    """Paper's Appendix E train aug: RRC + HFlip only."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def load_backbone(ckpt_path: str, arch: str, image_size: int, device: torch.device):
    """Load a frozen TorchvisionResNet backbone from a ``train.py`` checkpoint or an
    exported release file (``scripts/export_hf.py``), see ``models.load_pretrained``.

    Returns:
        (backbone in eval mode, pretraining epoch of the checkpoint).
    """
    sd, ckpt = load_pretrained(ckpt_path)
    backbone_sd = {
        k.removeprefix("backbone."): v for k, v in sd.items() if k.startswith("backbone.")
    }
    backbone = TorchvisionResNet(arch=arch, image_size=image_size)
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing or unexpected:
        logger.warning(
            f"Backbone load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected"
        )
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, ckpt.get("epoch", -1)


@torch.no_grad()
def embed(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Frozen features ``[B, D]`` (float32) of images ``[B, 3, H, W]``."""
    feats = backbone(x)
    return feats.flatten(1).float()  # [B, D]


@torch.no_grad()
def extract_features(backbone, dataset, device, batch_size: int, num_workers: int):
    """Embed a whole dataset with the frozen backbone.

    Returns:
        feats ``[N, D]`` float32 and targets ``[N]`` long, on CPU.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    feats, targets = [], []
    for x, y in loader:
        feats.append(embed(backbone, x.to(device, non_blocking=True)).cpu())
        targets.append(torch.as_tensor(y))
    return torch.cat(feats), torch.cat(targets)


def accuracy(
    logits: torch.Tensor, y: torch.Tensor, k: int = 1, mean_per_class: bool = False
) -> float:
    """Top-k accuracy (%) of ``logits [N, C]`` against ``y [N]``, or mean per-class recall."""
    correct = logits.topk(k, dim=1).indices.eq(y.unsqueeze(1)).any(1).float()  # [N]
    if not mean_per_class:
        return 100.0 * correct.mean().item()
    return 100.0 * torch.stack([correct[y == c].mean() for c in y.unique()]).mean().item()


def cosine_lr(
    step: int, total_steps: int, warmup_steps: int, base_lr: float, end_lr: float = 0.0
) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    step -= warmup_steps
    total_steps -= warmup_steps
    q = 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
    return end_lr + (base_lr - end_lr) * q


# ---------------------------------------------------------------------------
# LeJEPA-8 transfer: linear head on precomputed, standardized features
# ---------------------------------------------------------------------------
def fit_linear(
    x, y, num_classes: int, wd: float, device, epochs: int = 100, seed: int = 0
) -> nn.Linear:
    """Adam (lr 1e-3, cosine) linear head on features ``x [N, D]`` with labels ``y [N]``."""
    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x, y = x.to(device), y.to(device)
    for _ in range(epochs):
        for i in range(0, x.shape[0], 4096):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(head(x[i : i + 4096]), y[i : i + 4096]).backward()
            opt.step()
        sched.step()
    return head


@torch.no_grad()
def predict(head: nn.Module, x: torch.Tensor, device, batch_size: int = 8192) -> torch.Tensor:
    """Head logits ``[N, C]`` on CPU for features ``x [N, D]``."""
    head.eval()
    return torch.cat(
        [head(x[i : i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)]
    )


def probe_transfer(name, backbone, device, args, wd_grid=(0.0, 1e-4, 1e-2), val_frac=0.1) -> dict:
    """Transfer linear probe on one ``TRANSFER_DATASETS`` entry; wd selected on a train holdout."""
    spec = TRANSFER_DATASETS[name]
    feats = {}
    for split in ("train", "test"):
        ds = build_transfer_dataset(
            name, split, os.path.join(args.transfer_root, name), get_val_transforms("imagenet1k")
        )
        feats[split] = extract_features(backbone, ds, device, args.batch_size, args.num_workers)
    (tr_x, tr_y), (te_x, te_y) = feats["train"], feats["test"]
    mu, sd = tr_x.mean(0, keepdim=True), tr_x.std(0, keepdim=True).clamp_min(1e-6)
    tr_x, te_x = (tr_x - mu) / sd, (te_x - mu) / sd
    perm = torch.randperm(len(tr_x), generator=torch.Generator().manual_seed(0))
    val_idx, fit_idx = perm[: round(val_frac * len(tr_x))], perm[round(val_frac * len(tr_x)) :]
    val_scores = {}
    for wd in wd_grid:
        head = fit_linear(tr_x[fit_idx], tr_y[fit_idx], spec.num_classes, wd, device)
        logits = predict(head, tr_x[val_idx], device)
        val_scores[wd] = accuracy(logits, tr_y[val_idx], mean_per_class=spec.mean_per_class)
    best_wd = max(val_scores, key=val_scores.get)
    head = fit_linear(tr_x, tr_y, spec.num_classes, best_wd, device)
    score = accuracy(predict(head, te_x, device), te_y, mean_per_class=spec.mean_per_class)
    metric = "mean_per_class" if spec.mean_per_class else "top1"
    logger.info(f"[{name}] linear ({metric}) = {score:.2f} (wd={best_wd})")
    return {
        "linear": score,
        "linear_wd": best_wd,
        "metric": metric,
        "n_train": len(tr_x),
        "n_test": len(te_x),
    }


# ---------------------------------------------------------------------------
# ImageNet-1K: SGD linear head on augmented, on-the-fly features (VICReg App. E)
# ---------------------------------------------------------------------------
def train_one_epoch(
    backbone, head, loader, optimizer, device, epoch, total_steps, warmup_steps, base_lr
):
    head.train()
    t0 = time.time()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for step_in_epoch, (x, y) in enumerate(loader):
        lr = cosine_lr(epoch * len(loader) + step_in_epoch, total_steps, warmup_steps, base_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        y = y.to(device, non_blocking=True)
        logits = head(embed(backbone, x.to(device, non_blocking=True)))
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += y.size(0)
    return {
        "train_loss": total_loss / max(1, total_n),
        "train_acc": 100.0 * total_correct / max(1, total_n),
        "lr": optimizer.param_groups[0]["lr"],
        "time_s": time.time() - t0,
    }


def probe_imagenet(backbone, device, args, resume_path: Path, is_main: bool) -> dict:
    """Paper's IN1K linear probe; auto-resumes from ``resume_path`` after a preemption.

    Validation features are deterministic (frozen backbone, eval transform), so they are
    extracted once, at the first evaluation, and reused.
    """
    train_ds = build_imagenet(args.data_dir, "train", build_train_transform(args.image_size))
    val_ds = build_imagenet(args.data_dir, "val", get_val_transforms("imagenet1k"))
    per_gpu_bs = local_batch_size(args.batch_size)
    sampler = make_sampler(train_ds)
    train_loader = DataLoader(
        train_ds,
        batch_size=per_gpu_bs,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    head = nn.Linear(RESNET_FEATURES_DIM[args.arch], 1000).to(device)
    head_module = head
    if dist.is_available() and dist.is_initialized():
        head = nn.parallel.DistributedDataParallel(head, device_ids=[device.index])
    effective_lr = args.base_lr * args.batch_size / 256.0
    optimizer = torch.optim.SGD(
        head.parameters(), lr=effective_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    logger.info(f"bs={args.batch_size} eff_lr={effective_lr:.4f} epochs={args.epochs}")

    # Resume state carries the run identity so a stale file from another config is rejected.
    identity = {"ckpt": str(args.ckpt), "epochs": args.epochs, "batch_size": args.batch_size}
    start_epoch, history, best_top1 = 0, [], 0.0
    if resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError(
                f"Resume file {resume_path} has {state['identity']}, not {identity}. Delete it."
            )
        head_module.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, history, best_top1 = state["epoch"] + 1, state["history"], state["best_top1"]
        logger.info(f"Resumed from epoch {start_epoch} ({resume_path})")

    val = {}

    def val_acc(k):
        if not val:
            val["x"], val["y"] = extract_features(
                backbone, val_ds, device, per_gpu_bs, args.num_workers
            )
        return accuracy(predict(head_module, val["x"], device), val["y"], k=k)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        stats = train_one_epoch(
            backbone,
            head,
            train_loader,
            optimizer,
            device,
            epoch,
            total_steps,
            warmup_steps,
            effective_lr,
        )
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            stats.update(epoch=epoch + 1, val_top1=val_acc(1), val_top5=val_acc(5))
            best_top1 = max(best_top1, stats["val_top1"])
            history.append(stats)
            if is_main:
                logger.info(
                    f"[ep {epoch + 1}/{args.epochs}] "
                    + " ".join(f"{k}={v:.4g}" for k, v in stats.items())
                )
        if is_main and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            state = {
                "head": head_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            state.update(identity=identity, history=history, best_top1=best_top1)
            torch.save(state, resume_path.with_suffix(".tmp"))
            os.replace(resume_path.with_suffix(".tmp"), resume_path)
    return {
        "final_top1": val_acc(1),
        "final_top5": val_acc(5),
        "best_top1": best_top1,
        "effective_lr": effective_lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        default="imagenet1k",
        help="'imagenet1k', 'lejepa8', or a comma list of imagenet1k / TRANSFER_DATASETS keys",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="ImageNet root (HF Arrow or ImageFolder)"
    )
    parser.add_argument(
        "--transfer-root", type=str, default=None, help="root holding <dataset>/ subdirs"
    )
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024, help="global batch size")
    parser.add_argument("--base-lr", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="JSON result path (default: offline_probe.json, or transfer.json without imagenet1k)",
    )
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--save-every", type=int, default=1, help="IN1K resume-state period (0 disables)"
    )
    args = parser.parse_args()
    names = list(TRANSFER_DATASETS) if args.datasets == "lejepa8" else args.datasets.split(",")

    local_rank, world_size, is_main = setup_distributed()
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    backbone, pretrain_epoch = load_backbone(args.ckpt, args.arch, args.image_size, device)
    logger.info(
        f"[rank {local_rank}] ckpt={args.ckpt} pretrain_epoch={pretrain_epoch} datasets={names}"
    )

    default_name = "offline_probe.json" if "imagenet1k" in names else "transfer.json"
    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / default_name
    resume_path = out_path.with_name(f"{out_path.stem}_resume.pth.tar")
    out = {"ckpt": str(args.ckpt), "pretrain_epoch": pretrain_epoch}
    if "imagenet1k" in names:
        out.update(probe_imagenet(backbone, device, args, resume_path, is_main))
        if is_main:
            logger.info(f"FINAL: top1={out['final_top1']:.2f} top5={out['final_top5']:.2f}")
    transfer = [n for n in names if n != "imagenet1k"]
    if transfer and is_main:
        out["transfer"] = {n: probe_transfer(n, backbone, device, args) for n in transfer}
        out["transfer_mean"] = sum(r["linear"] for r in out["transfer"].values()) / len(transfer)
        logger.info(
            f"FINAL: transfer mean over {len(transfer)} datasets = {out['transfer_mean']:.2f}"
        )
    if is_main:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        logger.info(f"Wrote {out_path}")
        resume_path.unlink(missing_ok=True)
    cleanup_distributed()


if __name__ == "__main__":
    main()
