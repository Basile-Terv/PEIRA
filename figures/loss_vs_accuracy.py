"""Negative training objective vs online linear-probe top-1 (paper Figure 2 right, Figure G.3).

One point per (run, epoch), coloured by lambda, or by run label when runs are given as
``LABEL=RUN_DIR`` or have no lambda (VICReg, labelled by method name); the per-colour Spearman correlation is reported in the legend.
The x-axis is ``-history[loss_key]``; by default ``train_peira_loss`` (the running estimate
of E_PEIRA) for PEIRA and ``train_loss`` otherwise. ``--epochs 24 49 74 99`` restricts the
points to logged epochs (0-indexed: 24 is the 25th epoch), as in the main-text PEIRA panel.

Usage:
    python figures/loss_vs_accuracy.py RUN_DIR [...] --out figures/out/in1k_peira_loss_vs_acc.pdf
"""

import argparse
from typing import List, Optional

import pandas as pd
from _common import (
    LABEL_COLORS,
    METHOD_STYLE,
    Run,
    group_of,
    lambda_colors,
    load_runs,
    method_of,
    plt,
    save,
    setup_style,
    spearman,
)


def points(runs: List[Run], loss_key: Optional[str], epochs: Optional[List[int]]) -> pd.DataFrame:
    rows = []
    for r in runs:
        key = loss_key or r.default_loss_key()
        h = r.history if epochs is None else r.history[r.history.epoch.isin(epochs)]
        rows.append(pd.DataFrame({
            "x": -h[key].astype(float),
            "acc": h[r.acc_key].astype(float),
            "lam": r.lam,
            "label": r.label,
            "group": group_of(r) if r.label is None and r.lam is None else r.label,
            "run": str(r.path),
        }))
    return pd.concat(rows, ignore_index=True)


def panel(ax, runs: List[Run], loss_key: Optional[str] = None, epochs=None) -> None:
    df = points(runs, loss_key, epochs)
    method = method_of(runs)
    by_label = df["group"].notna().all()
    if by_label:
        groups = list(dict.fromkeys(df["group"]))
        colors = {g: LABEL_COLORS[i % len(LABEL_COLORS)] for i, g in enumerate(groups)}
        col = "group"
    else:
        colors = lambda_colors(df["lam"], METHOD_STYLE[method]["cmap"])
        col = "lam"
    for g, color in colors.items():
        sub = df[df[col] == g]
        rho = spearman(sub["x"], sub["acc"]) if len(sub) > 2 else float("nan")
        name = g if by_label else rf"$\lambda$ = {g:g}"
        ax.scatter(sub["x"], sub["acc"], s=10, alpha=0.55, color=color,
                   label=rf"{name}  ($\rho$ = {rho:+.3f})", edgecolors="none")
        print(f"{name}: n={len(sub)} spearman={rho:+.4f}")
    rho_all = spearman(df["x"], df["acc"])
    key = loss_key or runs[0].default_loss_key()
    ax.set_xlabel(rf"{METHOD_STYLE[method]['name']}  $-$ {key}")
    ax.set_ylabel("online linear-probe top-1 (%)")
    ax.set_title(f"{METHOD_STYLE[method]['name']}: negative loss vs accuracy\n"
                 rf"(pooled Spearman $\rho$ = {rho_all:+.3f})")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="train.py run dirs, optionally LABEL=RUN_DIR")
    parser.add_argument("--loss-key", default=None, help="metrics.jsonl column for the x-axis")
    parser.add_argument("--epochs", type=int, nargs="*", default=None, help="keep only these epochs")
    parser.add_argument("--out", default="figures/out/loss_vs_acc.pdf")
    args = parser.parse_args()

    setup_style(11)
    fig, ax = plt.subplots(figsize=(4.55, 4.0))
    panel(ax, load_runs(args.runs), args.loss_key, args.epochs)
    save(fig, args.out)


if __name__ == "__main__":
    main()
