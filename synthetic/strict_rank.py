"""Synthetic strict-rank staircase for PEIRA (paper appendix, "Synthetic two-view Gaussian data").

PEIRA keeps only the canonical directions whose square-root canonical correlation exceeds
the ridge ``lambda``, so the recovered rank should equal ``r_max = #{i : sqrt(c_i) > lambda}``.
For a single mode with correlation ``c`` and representation variance ``t`` the objective is
``-t c / (2t + lambda) + lambda t``, minimized at ``t* = max(0, (sqrt(c) - lambda) / 2)``.

We draw two-view Gaussian data with six prescribed canonical correlations, train two linear
encoders R^32 -> R^12 with ``PEIRALoss`` at each ``lambda`` of a sweep crossing the
thresholds ``sqrt(c_i)``, and count covariance eigenvalues of ``U(x)`` above ``5e-3``.

Usage (deterministic on CPU, paper figure):
    python synthetic/strict_rank.py --depth 0 --epochs 250 --seed 0
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from peira_loss import PEIRALoss

from eb_jepa.utils.rank import compute_effective_rank

SPECTRA = {
    "well-separated spectrum": (
        [0.90, 0.75, 0.60, 0.45, 0.30, 0.15],
        [0.30, 0.47, 0.61, 0.72, 0.82, 0.91, 0.97],
    ),
    "clustered spectrum": (
        [0.66, 0.64, 0.62, 0.10, 0.09, 0.08],
        [0.20, 0.29, 0.31, 0.50, 0.79, 0.805, 0.90],
    ),
}


def make_two_view_data(
    corrs: Sequence[float], n_samples: int, obs_dim: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Two-view Gaussian data with prescribed canonical correlations.

    Latent pairs ``y_i = c_i x_i + sqrt(1 - c_i^2) eps_i`` for ``i < k``, view-private noise
    elsewhere; each view is mixed by an independent random orthogonal matrix.

    Args:
        corrs: The ``k <= obs_dim`` canonical correlations ``c_i``.
        n_samples: Number of pairs ``N``.
        obs_dim: Dimension of each view.
        seed: RNG seed.

    Returns:
        ``(X, Y)``, each ``[N, obs_dim]``.
    """
    k = len(corrs)
    g = torch.Generator().manual_seed(seed)
    x_lat = torch.randn(n_samples, obs_dim, generator=g)  # [N, obs_dim]
    y_lat = torch.randn(n_samples, obs_dim, generator=g)  # [N, obs_dim]
    c = torch.tensor(corrs, dtype=torch.float32)  # [k]
    y_lat[:, :k] = c * x_lat[:, :k] + torch.sqrt(1.0 - c**2) * y_lat[:, :k]
    qx, _ = torch.linalg.qr(torch.randn(obs_dim, obs_dim, generator=g))
    qy, _ = torch.linalg.qr(torch.randn(obs_dim, obs_dim, generator=g))
    return x_lat @ qx, y_lat @ qy


def make_encoder(obs_dim: int, hidden: int, latent_dim: int, depth: int) -> nn.Sequential:
    """ReLU MLP with ``depth`` hidden layers; ``depth=0`` is a linear encoder."""
    layers: List[nn.Module] = []
    d_in = obs_dim
    for _ in range(depth):
        layers += [nn.Linear(d_in, hidden), nn.ReLU()]
        d_in = hidden
    return nn.Sequential(*layers, nn.Linear(d_in, latent_dim))


def measure_rank(z: torch.Tensor, abs_floor: float) -> Tuple[int, float]:
    """Recovered rank (covariance eigenvalues above ``abs_floor``) and entropy effective rank.

    Args:
        z: Representation ``[N, D]``.
        abs_floor: Absolute variance floor for a surviving direction.

    Returns:
        ``(recovered_rank, effective_rank)``; effective rank is 0 for a collapsed ``z``.
    """
    zc = z - z.mean(0, keepdim=True)  # [N, D]
    cov = zc.t().double() @ zc.double() / (zc.shape[0] - 1)  # [D, D]
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)  # [D]
    eff = 0.0 if float(eig.max()) < abs_floor else compute_effective_rank(zc)
    return int((eig > abs_floor).sum()), eff


def train_encoder(
    x: torch.Tensor, y: torch.Tensor, lam: float, args: argparse.Namespace
) -> nn.Sequential:
    """Train encoders ``(U, V)`` with ``PEIRALoss`` at constant ridge ``lam``; returns ``U``."""
    torch.manual_seed(args.seed)
    u = make_encoder(x.shape[1], args.hidden, args.latent_dim, args.depth)
    v = make_encoder(x.shape[1], args.hidden, args.latent_dim, args.depth)
    n = x.shape[0]
    steps_per_epoch = n // args.batch_size
    peira = PEIRALoss(
        latent_dim=args.latent_dim, T_max=max(1, args.epochs * steps_per_epoch), reg=lam,
        lr=0.9, lr_min=0.5, predictor_cov=args.predictor_cov,
    )
    opt = torch.optim.AdamW(list(u.parameters()) + list(v.parameters()), lr=args.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(args.seed + 1)
    for _ in range(args.epochs):
        perm = torch.randperm(n, generator=g)
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch_size : (s + 1) * args.batch_size]
            loss = peira(u(x[idx]), v(y[idx]))["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return u


def run_spectrum(corrs: Sequence[float], lams: Sequence[float], args: argparse.Namespace) -> dict:
    """Sweep ``lambda`` for one spectrum; returns predicted and recovered ranks per ``lambda``."""
    x, y = make_two_view_data(corrs, args.n_samples, args.obs_dim, args.seed)
    sqrt_c = np.sqrt(np.array(corrs))
    out = {"corrs": list(corrs), "sqrt_c": sqrt_c.tolist(), "lams": list(lams)}
    out.update(predicted=[], recovered=[], eff_rank=[])
    for lam in lams:
        u = train_encoder(x, y, lam, args)
        with torch.no_grad():
            rank, eff = measure_rank(u(x[: args.n_eval]), args.abs_floor)
        out["predicted"].append(int((sqrt_c > lam).sum()))
        out["recovered"].append(rank)
        out["eff_rank"].append(eff)
        print(f"  lambda={lam:5.3f}  r_max={out['predicted'][-1]}  recovered={rank}  eff_rank={eff:5.2f}", flush=True)
    return out


def make_figure(results: dict, out_pdf: Path) -> None:
    """Recovered rank vs ``lambda`` with the predicted step function, one panel per spectrum."""
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.6), squeeze=False)
    for ax, (name, r) in zip(axes[0], results.items()):
        sqrt_c = np.array(r["sqrt_c"])
        lam_grid = np.linspace(0, 1, 400)
        ax.step(
            lam_grid, [int((sqrt_c > lg).sum()) for lg in lam_grid], where="post", color="k", lw=1.8,
            label=r"predicted $r_{\max}=\#\{\sqrt{c_i}>\lambda\}$",
        )
        ax.plot(r["lams"], r["recovered"], "o", color="tab:red", ms=9, label="recovered rank (cov eig count)")
        ax.plot(r["lams"], r["eff_rank"], "s", color="tab:blue", ms=6, alpha=0.8, label="entropy effective rank")
        for c in sqrt_c:
            ax.axvline(c, color="grey", ls=":", lw=0.9, alpha=0.7)
        ax.set_xlabel(r"PEIRA ridge $\lambda$")
        ax.set_ylabel("recovered rank")
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.4, len(sqrt_c) + 0.6)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "PEIRA strict-rank prediction on synthetic two-view Gaussian data "
        r"(dotted grey = prescribed thresholds $\sqrt{c_i}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_pdf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=20000)
    p.add_argument("--n-eval", type=int, default=20000)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=12)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=0, help="hidden layers; 0 = linear encoders (paper)")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--abs-floor", type=float, default=5e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--predictor-cov", choices=("batch", "ema"), default="batch", help="Sigma in P (see PEIRALoss)")
    p.add_argument("--out-dir", type=Path, default=Path("synthetic/out"))
    args = p.parse_args()

    results = {}
    for name, (corrs, lams) in SPECTRA.items():
        print(f"{name}: c={corrs}", flush=True)
        results[name] = run_spectrum(corrs, lams, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "strict_rank.json").write_text(json.dumps(results, indent=2))
    make_figure(results, args.out_dir / "strict_rank.pdf")
    print(f"Wrote {args.out_dir / 'strict_rank.json'} and {args.out_dir / 'strict_rank.pdf'}")


if __name__ == "__main__":
    main()
