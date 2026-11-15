r"""VICReg diagnostics grouped by recipe (paper Figure G.4).

Left: backbone effective rank vs epoch (faint: per run; bold: per-label mean).
Right: negative training loss vs online linear-probe top-1, one colour per label, with the
per-label Spearman correlation. Runs are given as ``LABEL=RUN_DIR``; runs sharing a label are
pooled (e.g. the seeds of one recipe). For SIGReg, run ``lambda_sensitivity.py``,
``effective_rank.py`` and ``loss_vs_accuracy.py`` on the SIGReg run dirs instead.

Usage:
    python figures/baselines.py "LARS lr=0.3=RUN_A_SEED1" "LARS lr=0.3=RUN_A_SEED2" "SGD=RUN_B" \
        --out figures/out/in1k_vicreg.pdf
"""

import argparse

from _common import BACKBONE_RANK, load_runs, plt, save, setup_style
from loss_vs_accuracy import LABEL_COLORS, panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="LABEL=RUN_DIR (same label = same recipe)")
    parser.add_argument("--loss-key", default="train_loss")
    parser.add_argument("--out", default="figures/out/baselines.pdf")
    args = parser.parse_args()

    runs = load_runs(args.runs)
    for r in runs:
        r.label = r.label or r.path.name
    labels = list(dict.fromkeys(r.label for r in runs))

    setup_style(11)
    fig, (ax_rank, ax_loss) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for i, label in enumerate(labels):
        color = LABEL_COLORS[i % len(LABEL_COLORS)]
        group = [r for r in runs if r.label == label]
        for r in group:
            ax_rank.plot(r.history.epoch, r.history[BACKBONE_RANK], lw=0.8, color=color, alpha=0.3)
        mean = sum(r.history.set_index("epoch")[BACKBONE_RANK] for r in group) / len(group)
        ax_rank.plot(mean.index, mean.values, lw=1.7, color=color, label=f"{label} ({len(group)})")
    ax_rank.set_xlabel("epoch")
    ax_rank.set_ylabel("backbone effective rank")
    ax_rank.set_title("backbone effective rank trajectory")
    ax_rank.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax_rank.legend(loc="lower right", fontsize=7.5, frameon=True)
    panel(ax_loss, runs, args.loss_key)
    save(fig, args.out)


if __name__ == "__main__":
    main()
