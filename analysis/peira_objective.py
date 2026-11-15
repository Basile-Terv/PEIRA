r"""Closed-form PEIRA objective at saved checkpoints (paper Figure 2, right panel).

For projector embeddings ``u``, ``v`` of two augmented views and ridge ``lambda``:

    E_PEIRA = -0.5 Tr( (S + S^T) (N + lambda I)^-1 ) + 0.5 lambda ( E||u||^2 + E||v||^2 ),

with ``S = E[v u^T]`` and ``N = E[u u^T + v v^T]``. This is the value of the PEIRA loss
once its predictor has converged to the closed-form regressor; the paper plots
``-E_PEIRA`` against top-1 at epochs 25/50/75/100. Moments are accumulated in fp64 over
``--num-batches`` training batches (BN in eval mode) and the objective is evaluated on
the population moments.

Usage:
    python -m analysis.peira_objective --ckpt <run>/e-25.pth.tar <run>/latest.pth.tar \
        --out peira_objective.json
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from analysis.common import load_config, load_ssl_model, resolve_data_dir
from data.datasets import ImageDataset, build_imagenet
from data.transforms import make_transforms


def objective_from_moments(
    cross: torch.Tensor, cov_u: torch.Tensor, cov_v: torch.Tensor,
    sq_norm_u: torch.Tensor, sq_norm_v: torch.Tensor, reg: float,
) -> dict:
    """Closed form from second moments.

    Args:
        cross: ``E[v u^T]`` [k, k].
        cov_u: ``E[u u^T]`` [k, k].
        cov_v: ``E[v v^T]`` [k, k].
        sq_norm_u: ``E||u||^2`` (scalar).
        sq_norm_v: ``E||v||^2`` (scalar).
        reg: Ridge ``lambda``.

    Returns:
        ``objective`` = ``trace_term`` + ``reg_term`` (tensors).
    """
    eye = torch.eye(cross.shape[0], device=cross.device, dtype=cross.dtype)
    inv = torch.cholesky_inverse(torch.linalg.cholesky(cov_u + cov_v + reg * eye))
    trace_term = -0.5 * ((cross + cross.T) * inv).sum()
    reg_term = 0.5 * reg * (sq_norm_u + sq_norm_v)
    return {"objective": trace_term + reg_term, "trace_term": trace_term, "reg_term": reg_term}


def peira_objective(u: torch.Tensor, v: torch.Tensor, reg: float) -> torch.Tensor:
    """Batch closed form, equal to the PEIRA objective of the paper evaluated on one batch.

    Args:
        u: View-1 embeddings [B, k].
        v: View-2 embeddings [B, k].
        reg: Ridge ``lambda``.
    """
    B = u.shape[0]
    return objective_from_moments(
        v.T @ u / B, u.T @ u / B, v.T @ v / B,
        u.pow(2).sum(1).mean(), v.pow(2).sum(1).mean(), reg,
    )["objective"]


class MomentAccumulator:
    """fp64 running sums of the moments entering :func:`objective_from_moments`."""

    def __init__(self, dim: int, device: torch.device):
        z = lambda: torch.zeros(dim, dim, device=device, dtype=torch.float64)  # noqa: E731
        self.cross, self.cov_u, self.cov_v = z(), z(), z()
        self.sq_u = self.sq_v = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, u: torch.Tensor, v: torch.Tensor) -> None:
        u, v = u.double(), v.double()
        self.cross += v.T @ u
        self.cov_u += u.T @ u
        self.cov_v += v.T @ v
        self.sq_u += u.pow(2).sum().item()
        self.sq_v += v.pow(2).sum().item()
        self.n += u.shape[0]

    def finalize(self, reg: float) -> dict:
        n = self.n
        sq_u, sq_v = (torch.tensor(s / n, dtype=torch.float64) for s in (self.sq_u, self.sq_v))
        terms = objective_from_moments(self.cross / n, self.cov_u / n, self.cov_v / n, sq_u, sq_v, reg)
        out = {k: float(t) for k, t in terms.items()}
        out.update(neg_objective=-out["objective"], mean_sq_norm_u=self.sq_u / n,
                   mean_sq_norm_v=self.sq_v / n, n_samples=n, reg=reg)
        return out


def two_view_loader(data_dir: str, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    """Shuffled ImageNet train loader with the run's two augmented views."""
    dataset = ImageDataset(build_imagenet(data_dir, "train"), make_transforms("imagenet1k"))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=True,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model, loader: DataLoader, num_batches: int, reg: float, device: torch.device) -> dict:
    acc = MomentAccumulator(model.projector[-1].out_features, device)
    for i, (views, _) in enumerate(loader):
        if i == num_batches:
            break
        _, u = model(views[0].to(device, non_blocking=True))
        _, v = model(views[1].to(device, non_blocking=True))
        acc.update(u, v)
    return acc.finalize(reg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True, help="Checkpoints of one run (e.g. a ladder).")
    p.add_argument("--config", default=None, help="Default: config.yaml next to the first --ckpt.")
    p.add_argument("--data-dir", default=None, help="ImageNet root. Default: data.data_dir.")
    p.add_argument("--reg", type=float, default=None, help="Default: loss.reg of the run.")
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="peira_objective.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.ckpt[0], args.config)
    reg = args.reg if args.reg is not None else float(cfg.loss.reg)
    rows = []
    for ckpt in args.ckpt:
        model, epoch = load_ssl_model(ckpt, cfg, device)
        loader = two_view_loader(resolve_data_dir(cfg, args.data_dir), args.batch_size,
                                 args.num_workers, args.seed)
        row = {"ckpt": ckpt, "epoch": epoch, **evaluate(model, loader, args.num_batches, reg, device)}
        rows.append(row)
        print(f"epoch {epoch:>3}  -E_PEIRA = {row['neg_objective']:.4f}  "
              f"(trace {row['trace_term']:.4f}, reg {row['reg_term']:.4f}, n={row['n_samples']})")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
