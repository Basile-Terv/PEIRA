"""Effective rank during pretraining (paper Figure 2 middle, Figure G.3, Figure G.2).

``--mode trajectory``: backbone effective rank vs epoch, one line per lambda (mean over seeds);
runs without a lambda (VICReg) get one line per ``LABEL=`` label instead.
``--mode end``: end-of-training backbone (left axis) and projector (right axis) effective rank
vs lambda, averaged over the last ``--last-epochs`` epochs; dots are seeds, lines are means.

Usage:
    python figures/effective_rank.py RUN_DIR [...] --mode trajectory --out figures/out/in1k_peira_eff_rank_trajectory.pdf
    python figures/effective_rank.py RUN_DIR [...] --mode end --out figures/out/in1k_peira_eff_rank_eot.pdf
"""

import argparse

import numpy as np
import pandas as pd
from _common import (
    BACKBONE_RANK,
    METHOD_STYLE,
    PROJECTOR_RANK,
    group_colors,
    group_name,
    group_of,
    load_runs,
    method_of,
    plt,
    save,
    setup_style,
)


def plot_trajectory(runs, out: str) -> None:
    method = method_of(runs)
    colors = group_colors(runs, METHOD_STYLE[method]["cmap"])
    setup_style(11)
    fig, ax = plt.subplots(figsize=(4.55, 4.0))
    for group, color in colors.items():
        members = [r for r in runs if group_of(r) == group]
        hist = pd.concat([r.history[["epoch", BACKBONE_RANK]] for r in members])
        traj = hist.groupby("epoch")[BACKBONE_RANK].mean()
        ax.plot(traj.index, traj.values, lw=1.5, color=color,
                label=f"{group_name(group)} ({len(members)})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("backbone effective rank")
    ax.set_title(f"{METHOD_STYLE[method]['name']} trajectory (mean across seeds)")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(loc="lower right", fontsize=8.0, frameon=True)
    save(fig, out)


def plot_end(runs, out: str, last_epochs: int, proj_dim: int) -> None:
    lams = sorted({r.lam for r in runs})
    setup_style(11)
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ax2 = ax.twinx()
    rng = np.random.default_rng(0)
    series = [
        (ax, BACKBONE_RANK, "backbone (RN50, $d{=}2048$)", "#1f5fa8", "o"),
        (ax2, PROJECTOR_RANK, f"projector ($d{{=}}{proj_dim}$)", "#c2502b", "s"),
    ]
    for axis, key, label, color, marker in series:
        means = []
        for lam in lams:
            ys = np.array([r.history[key].iloc[-last_epochs:].mean() for r in runs if r.lam == lam])
            xs = lam * (1.0 + rng.uniform(-0.03, 0.03, size=len(ys)))
            axis.scatter(xs, ys, color=color, alpha=0.7, s=34, marker=marker,
                         edgecolors="white", linewidths=0.5, zorder=3)
            means.append(ys.mean())
            print(f"{key} lambda={lam:g}: mean={ys.mean():.1f} over {len(ys)} seeds")
        axis.plot(lams, means, "-", color=color, lw=1.6, marker=marker, ms=7,
                  mfc=color, mec="white", mew=0.5, label=label, zorder=4)
        axis.tick_params(axis="y", labelcolor=color)
    ax.set_xscale("log")
    ax.set_xticks(lams)
    ax.set_xticklabels([f"{v:g}" for v in lams])
    ax.minorticks_off()
    ax.set_xlabel(r"PEIRA regularizer  $\lambda$")
    ax.set_ylabel("backbone effective rank", color="#1f5fa8")
    ax2.set_ylabel("projector effective rank", color="#c2502b")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.set_title(f"end-of-training effective rank (last {last_epochs} epochs)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9.0, frameon=True)
    save(fig, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="train.py run dirs")
    parser.add_argument("--mode", choices=("trajectory", "end"), default="trajectory")
    parser.add_argument("--last-epochs", type=int, default=5, help="end mode: epochs averaged")
    parser.add_argument("--proj-dim", type=int, default=256, help="end mode: legend label only")
    parser.add_argument("--out", default="figures/out/eff_rank.pdf")
    args = parser.parse_args()

    runs = load_runs(args.runs)
    if args.mode == "trajectory":
        plot_trajectory(runs, args.out)
    else:
        runs = [r for r in runs if r.lam is not None]
        if not runs:
            raise SystemExit("--mode end plots against lambda, and these runs have none (VICReg?)")
        plot_end(runs, args.out, args.last_epochs, args.proj_dim)


if __name__ == "__main__":
    main()
