"""Final online linear-probe top-1 vs the regularizer lambda (paper Figure 2 left, Figure G.3).

Each dot is one run (its last-epoch online top-1); the line joins the per-lambda means.

Usage:
    python figures/lambda_sensitivity.py RUN_DIR [RUN_DIR ...] --out figures/out/in1k_peira_sensitivity.pdf
"""

import argparse

import numpy as np
from _common import METHOD_STYLE, load_runs, method_of, plt, save, setup_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="train.py run dirs (any lambda / seed)")
    parser.add_argument("--out", default="figures/out/sensitivity.pdf")
    parser.add_argument("--jitter", type=float, default=0.04, help="relative x jitter of dots")
    args = parser.parse_args()

    runs = [r for r in load_runs(args.runs) if r.lam is not None]
    if not runs:
        raise SystemExit("this figure plots against lambda, and these runs have none (VICReg?)")
    method = method_of(runs)
    style = METHOD_STYLE[method]
    finals = [(r.lam, float(r.history[r.acc_key].iloc[-1])) for r in runs]
    lams = sorted({lam for lam, _ in finals})

    setup_style(11)
    fig, ax = plt.subplots(figsize=(4.55, 4.0))
    rng = np.random.default_rng(0)
    means = []
    for lam in lams:
        ys = np.array([acc for l_, acc in finals if l_ == lam])
        xs = lam * (1.0 + rng.uniform(-args.jitter, args.jitter, size=len(ys)))
        ax.scatter(xs, ys, color=style["color"], alpha=0.55, s=28, marker=style["marker"],
                   edgecolors="white", linewidths=0.4, zorder=3)
        means.append(ys.mean())
        print(f"lambda={lam:g}: n={len(ys)} mean top-1={ys.mean():.2f}")
    ax.plot(lams, means, "-", color=style["color"], lw=1.6, marker=style["marker"], ms=7,
            mfc=style["color"], mec="white", mew=0.6, label=style["name"], zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(lams)
    ax.set_xticklabels([f"{v:g}" for v in lams])
    ax.minorticks_off()
    ax.set_xlim(lams[0] * 0.7, lams[-1] * 1.5)
    ax.set_xlabel(rf"{style['name']} regularizer  $\lambda$")
    ax.set_ylabel("ImageNet-1K linear-probe top-1 (%)")
    ax.set_title(rf"{style['name']}: sensitivity to $\lambda$")
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.text(0.06, 0.05,
            f"mean swing = {max(means) - min(means):.2f} pp\n"
            rf"$\lambda \in$ [{lams[0]:g}, {lams[-1]:g}]",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.3"))
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    save(fig, args.out)


if __name__ == "__main__":
    main()
