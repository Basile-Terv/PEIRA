"""Shared loaders and style for the figure scripts.

Each run directory is one ``train.py`` output folder holding ``config.yaml`` and
``metrics.jsonl`` (one JSON record per epoch, written by ``train.py``).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

BACKBONE_RANK = "train/backbone_effective_rank"
PROJECTOR_RANK = "train/projector_effective_rank"
METHOD_STYLE = {
    "peira": dict(name="PEIRA", color="#1f5fa8", marker="o", cmap="viridis"),
    "sigreg": dict(name="SIGReg", color="#c2502b", marker="s", cmap="plasma"),
    "vicreg": dict(name="VICReg", color="#2a9d8f", marker="^", cmap="cividis"),
}
LABEL_COLORS = ["#1f5fa8", "#c2502b", "#2a9d8f", "#8e44ad", "#e9a23b", "#555555"]
LOSS_TYPE_TO_METHOD = {
    "peira": "peira",
    "peira_comp": "peira",
    "bcs": "sigreg",
    "sigreg": "sigreg",
    "vicreg": "vicreg",
    "vicreg_ref": "vicreg",
}


@dataclass
class Run:
    path: Path
    label: Optional[str]
    method: str
    lam: Optional[float]
    seed: int
    history: pd.DataFrame  # one row per epoch

    @property
    def acc_key(self) -> str:
        return next(c for c in self.history.columns if c.endswith("_val_acc"))

    def default_loss_key(self) -> str:
        return "train_peira_loss" if self.method == "peira" else "train_loss"


def load_run(spec: str) -> Run:
    """Load ``[LABEL=]RUN_DIR``; LABEL is optional (used for grouping/legends)."""
    label, path = (spec.rsplit("=", 1) if "=" in spec and not Path(spec).exists() else (None, spec))
    path = Path(path)
    cfg = OmegaConf.load(path / "config.yaml")
    method = LOSS_TYPE_TO_METHOD.get(cfg.loss.type, cfg.loss.type)
    lam = cfg.loss.get("reg") if method == "peira" else cfg.loss.get("lmbd")
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line]
    # A resumed run re-logs the epochs it repeats; keep the last record per epoch.
    history = pd.DataFrame(records).drop_duplicates("epoch", keep="last").sort_values("epoch")
    return Run(
        path=path,
        label=label,
        method=method,
        lam=None if lam is None else float(lam),
        seed=int(cfg.meta.seed),
        history=history.reset_index(drop=True),
    )


def load_runs(specs: List[str]) -> List[Run]:
    return [load_run(s) for s in specs]


def spearman(x, y) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    rx = pd.Series(np.asarray(x)).rank().to_numpy()
    ry = pd.Series(np.asarray(y)).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def lambda_colors(lams, cmap_name: str) -> dict:
    """Map each lambda to a colour on a log scale."""
    lams = sorted(set(lams))
    cmap = plt.get_cmap(cmap_name)
    if len(lams) == 1:
        return {lams[0]: cmap(0.5)}
    lo, hi = np.log10(lams[0]), np.log10(lams[-1])
    return {lam: cmap(0.15 + 0.77 * (np.log10(lam) - lo) / (hi - lo)) for lam in lams}


def group_of(run: Run):
    """Lambda when the run has one, else its label (the method name if unlabelled), e.g. VICReg."""
    return run.lam if run.lam is not None else (run.label or METHOD_STYLE[run.method]["name"])


def group_colors(runs: List[Run], cmap_name: str) -> dict:
    """Colours keyed by ``group_of``: a log-scale colormap over lambdas, else a fixed palette."""
    groups = list(dict.fromkeys(group_of(r) for r in runs))
    if all(isinstance(g, float) for g in groups):
        return lambda_colors(groups, cmap_name)
    return {g: LABEL_COLORS[i % len(LABEL_COLORS)] for i, g in enumerate(groups)}


def group_name(group) -> str:
    return rf"$\lambda$ = {group:g}" if isinstance(group, float) else str(group)


def method_of(runs: List[Run]) -> str:
    methods = {r.method for r in runs}
    return methods.pop() if len(methods) == 1 else "peira"


def setup_style(font_size: float = 11) -> None:
    plt.rcParams.update({"font.size": font_size, "mathtext.default": "regular"})


def save(fig, out: str) -> None:
    """Save ``out`` (any extension) as both .pdf and .png."""
    stem = Path(out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=240)
    print(f"Wrote {stem}.pdf and {stem}.png")
    plt.close(fig)
