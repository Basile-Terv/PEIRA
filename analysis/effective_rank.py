"""Held-out effective rank of backbone and projector features (paper App. G.6-G.7).

Two estimators, both ``erank = exp(entropy of the normalized singular values)``:

* held-out: features of the first ``--num-samples`` (20,480) ImageNet-1K validation
  images, larger than every projector width ``k`` (Table G.4, columns
  "Proj. erank/k" and "Backbone erank");
* window: the first ``10 B / 8`` of those features, the sample count of the
  in-training estimator (10 steps of the per-GPU batch on 8 GPUs), whose ceiling
  ``min(N, D)`` depends on ``B``.

Each erank is reported with its ceiling, the erank of i.i.d. Gaussian features of the
same shape under the same estimator, and the normalized erank = erank / ceiling.
The paper's "Norm. erank" uses the in-training estimator (augmented train images,
``train/backbone_effective_rank``); the window value here is its held-out proxy.

Usage:
    python -m analysis.effective_rank --ckpt <run>/latest.pth.tar --out erank.json
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from analysis.common import load_config, load_ssl_model, resolve_data_dir
from data.datasets import build_imagenet
from data.transforms import get_val_transforms
from eb_jepa.utils.rank import compute_effective_rank


def gaussian_ceiling(n: int, d: int, seed: int = 0) -> float:
    """Effective rank of i.i.d. standard Gaussian features ``[n, d]``."""
    g = torch.Generator().manual_seed(seed)
    return compute_effective_rank(torch.randn(n, d, generator=g, dtype=torch.float64))


def rank_summary(x: torch.Tensor) -> dict:
    """Erank, Gaussian ceiling and normalized erank of features ``[N, D]``."""
    erank, ceiling = compute_effective_rank(x), gaussian_ceiling(*x.shape)
    return {"n": x.shape[0], "dim": x.shape[1], "erank": erank, "ceiling": ceiling,
            "normalized": erank / ceiling}


@torch.no_grad()
def embed(model, loader: DataLoader, n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Backbone ``[n, D]`` and projector ``[n, k]`` features (float32, CPU) of ``n`` images."""
    feats, projs, seen = [], [], 0
    for x, _ in loader:
        f, z = model(x.to(device, non_blocking=True))
        feats.append(f.float().cpu())
        projs.append(z.float().cpu())
        seen += x.shape[0]
        if seen >= n:
            break
    return torch.cat(feats)[:n], torch.cat(projs)[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default=None, help="Default: config.yaml next to --ckpt.")
    p.add_argument("--data-dir", default=None, help="ImageNet root. Default: data.data_dir.")
    p.add_argument("--num-samples", type=int, default=20480)
    p.add_argument("--window", type=int, default=None, help="Default: 10 * data.batch_size / 8.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", default="effective_rank.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.ckpt, args.config)
    model, epoch = load_ssl_model(args.ckpt, cfg, device)
    dataset = build_imagenet(resolve_data_dir(cfg, args.data_dir), "val", get_val_transforms("imagenet1k"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    feats, projs = embed(model, loader, args.num_samples, device)

    window = args.window or 10 * cfg.data.batch_size // 8
    out = {
        "ckpt": args.ckpt,
        "epoch": epoch,
        "backbone": rank_summary(feats),
        "projector": rank_summary(projs),
        "backbone_window": rank_summary(feats[:window]),
    }
    out["projector"]["erank_over_k"] = out["projector"]["erank"] / projs.shape[1]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for name in ("backbone", "projector", "backbone_window"):
        r = out[name]
        print(f"{name:<16} N={r['n']:<6} D={r['dim']:<5} erank={r['erank']:8.2f} "
              f"ceiling={r['ceiling']:8.2f} normalized={r['normalized']:.3f}")
    print(f"projector erank/k = {out['projector']['erank_over_k']:.3f}; wrote {args.out}")


if __name__ == "__main__":
    main()
