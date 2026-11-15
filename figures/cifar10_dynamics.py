r"""PEIRA accuracy/objective, spectrum and alignment evolution for one CIFAR-10 run.

Reads metrics.jsonl and spectral.jsonl; no model loading or dataset access is needed.
The spectral arrays are used as logged: singular values of the uncentered average
second moment of both views, and inter-view alignment in that matrix's basis.
No centering, whitening, spectrum normalization or reordering is applied.

Usage:
    python figures/cifar10_dynamics.py checkpoints/c10_peira_seed42 \
        --out figures/out/cifar10_dynamics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import Run, load_run, plt, save, setup_style


def load_spectral_history(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read logged diagnostics, retaining the last record for repeated epochs."""
    history = {}
    dimension = None
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            spectrum = np.asarray(row["spectrum"], dtype=float)
            alignment = np.asarray(row["alignment"], dtype=float)
            if (
                spectrum.ndim != 1 or spectrum.size == 0
                or alignment.shape != spectrum.shape
                or not np.isfinite(spectrum).all() or not np.isfinite(alignment).all()
                or np.any(spectrum < 0) or not np.any(spectrum > 0)
            ):
                raise ValueError(f"Invalid spectral diagnostics at {path}:{line_number}")
            if dimension is not None and spectrum.size != dimension:
                raise ValueError(f"Spectral dimension changed at {path}:{line_number}")
            dimension = spectrum.size
            history[int(row["epoch"])] = {"spectrum": spectrum, "alignment": alignment}
    if not history:
        raise ValueError(f"No spectral diagnostics in {path}")
    return dict(sorted(history.items()))


def select_epochs(available: list[int], requested: list[int] | None, maximum: int) -> list[int]:
    """Choose round epoch intervals; 0..990 selects 0, 100, ..., 900 by default."""
    if requested is not None:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested spectral epochs are not logged: {missing}")
        return sorted(set(requested))
    if maximum < 1:
        raise ValueError("--max-epoch-traces must be positive")
    if len(available) <= maximum:
        return available
    if maximum == 1:
        return [available[-1]]
    raw_step = (available[-1] - available[0]) / maximum
    power = 10 ** np.floor(np.log10(raw_step))
    fraction = raw_step / power
    step = next(n for n in (1, 2, 5, 10) if fraction <= n) * power
    targets = np.arange(
        np.floor(available[0] / step) * step,
        available[-1] + 0.5 * step,
        step,
    )[:maximum]
    return sorted({min(available, key=lambda epoch: abs(epoch - target)) for target in targets})


def training_panel(ax, run: Run):
    """Online probe on the left, negative PEIRA objective on the right."""
    history = run.history
    for key in (run.acc_key, "train_peira_loss"):
        if key not in history:
            raise ValueError(f"Missing {key!r} in {run.path / 'metrics.jsonl'}")
    right = ax.twinx()
    accuracy, = ax.plot(
        history.epoch.to_numpy(), history[run.acc_key].to_numpy(), color="tab:blue", lw=2,
        label="Online linear-probe top-1 (%)",
    )
    objective, = right.plot(
        history.epoch.to_numpy(), -history.train_peira_loss.to_numpy(), color="tab:red", lw=2,
        label=r"$-\mathcal{E}_{\mathrm{PEIRA}}$",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Online linear-probe top-1 (%)")
    right.set_ylabel(r"$-\mathcal{E}_{\mathrm{PEIRA}}$")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(handles=[accuracy, objective], loc="lower right", frameon=False, fontsize=11)
    return right


def spectral_panel(ax, history, epochs: list[int], key: str) -> None:
    cmap = plt.get_cmap("viridis")
    lo, hi = epochs[0], epochs[-1]
    for epoch in epochs:
        color = cmap(0.5 if hi == lo else 0.15 + 0.77 * (epoch - lo) / (hi - lo))
        values = history[epoch][key]
        # Zero singular values have no logarithm; leave gaps rather than invent a floor.
        plotted = np.where(values > 0, values, np.nan) if key == "spectrum" else values
        ax.plot(np.arange(1, values.size + 1), plotted, color=color, lw=2, label=str(epoch))
    if key == "spectrum":
        ax.set_yscale("log")
        ax.set_ylabel("Spectrum Value")
    else:
        ax.set_ylabel("Alignment Value")
    ax.set_xlabel("Singular Value Index")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(title="Epoch", loc="best", frameon=False, fontsize=11, title_fontsize=12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="train.py run directory")
    parser.add_argument("--out", default="figures/out/cifar10_dynamics", help="output prefix")
    parser.add_argument("--epochs", type=int, nargs="+", help="explicit logged spectral epochs")
    parser.add_argument("--max-epoch-traces", type=int, default=10)
    args = parser.parse_args()

    run = load_run(args.run)
    if run.method != "peira":
        parser.error("This script requires a PEIRA run with train_peira_loss diagnostics")
    history = load_spectral_history(run.path / "spectral.jsonl")
    epochs = select_epochs(list(history), args.epochs, args.max_epoch_traces)
    print(f"Spectral epochs: {', '.join(map(str, epochs))}")
    prefix = str(Path(args.out).with_suffix(""))
    setup_style(14)

    fig, ax = plt.subplots(figsize=(7, 5))
    training_panel(ax, run)
    save(fig, f"{prefix}_both")
    for key in ("spectrum", "alignment"):
        fig, ax = plt.subplots(figsize=(7, 5))
        spectral_panel(ax, history, epochs, key)
        save(fig, f"{prefix}_{key}")


if __name__ == "__main__":
    main()
