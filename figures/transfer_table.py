r"""Linear-transfer table (8 datasets of LeJEPA) with a mean column (paper Table G.9).

Each row is ``LABEL=PATH[,PATH...]``; a PATH is a ``probe.py`` output JSON or a run dir.
Several paths in one row are averaged entry-wise (e.g. seeds). ``probe.py`` outputs hold
``final_top1`` (ImageNet-1K probe, ``offline_probe.json``) and ``transfer[name]["linear"]``
(transfer probe, ``transfer.json``); for a run dir, or a JSON file next to
``offline_probe.json``, both are read and merged. Rows given with ``--extra-row`` are printed
below a rule and excluded from the bolding of the best value per column.

Usage:
    python figures/transfer_table.py --row "VICReg (8192)=RUN_V" --row "PEIRA (256)=RUN_P" \
        --extra-row "PEIRA (1024)=RUN_S1000,RUN_S10000"
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

DATASETS = ["cifar10", "cifar100", "dtd", "flowers102", "pets", "food101", "aircraft", "cars"]
HEADERS = ["IN1K", "C10", "C100", "DTD", "Flowers", "Pets", "Food", "Aircraft", "Cars", "Mean"]


def read_json(path: Path) -> Dict[str, float]:
    """Return {"in1k": top1, <dataset>: linear score} from one ``probe.py`` JSON."""
    d = json.loads(path.read_text())
    out = {}
    if "final_top1" in d:
        out["in1k"] = float(d["final_top1"])
    for name, res in (d.get("transfer") or {}).items():
        out[name] = float(res["linear"])
    return out


def read_entry(path: str) -> Dict[str, float]:
    p = Path(path)
    files = [p] if p.is_file() else [p / "transfer.json", p / "offline_probe.json"]
    if p.is_file() and p.name != "offline_probe.json":
        files.append(p.parent / "offline_probe.json")
    out: Dict[str, float] = {}
    for f in files:
        if f.exists():
            for k, v in read_json(f).items():
                out.setdefault(k, v)
    missing = [k for k in DATASETS if k not in out]
    if missing:
        raise ValueError(f"{path}: missing transfer results for {missing}")
    return out


def row_values(paths: List[str]) -> List[Optional[float]]:
    entries = [read_entry(p) for p in paths]
    in1k = [e["in1k"] for e in entries if "in1k" in e]
    vals = [sum(in1k) / len(in1k) if len(in1k) == len(entries) else None]
    transfer = [sum(e[k] for e in entries) / len(entries) for k in DATASETS]
    return vals + transfer + [sum(transfer) / len(transfer)]


def fmt(v: Optional[float], bold: bool, latex: bool) -> str:
    if v is None:
        return "--"
    s = f"{v:.2f}"
    if latex:
        return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"
    return f"**{s}**" if bold else s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--row", action="append", default=[], help="LABEL=PATH[,PATH...]")
    parser.add_argument("--extra-row", action="append", default=[], help="like --row, not bolded")
    args = parser.parse_args()

    rows = []
    for spec, extra in [(s, False) for s in args.row] + [(s, True) for s in args.extra_row]:
        label, paths = spec.rsplit("=", 1)
        rows.append((label, row_values(paths.split(",")), extra))
    main_rows = [vals for _, vals, extra in rows if not extra]
    best = [max((v[j] for v in main_rows if v[j] is not None), default=None)
            for j in range(len(HEADERS))]

    print("| Method | " + " | ".join(HEADERS) + " |")
    print("|---" * (len(HEADERS) + 1) + "|")
    for label, vals, extra in rows:
        cells = [fmt(v, not extra and v is not None and v == best[j], False) for j, v in enumerate(vals)]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n\\begin{tabular}{l" + "c" * len(HEADERS) + "}\n\\toprule")
    print("Method ($k$) & " + " & ".join(HEADERS) + " \\\\\n\\midrule")
    in_extra = False
    for label, vals, extra in rows:
        if extra and not in_extra:
            print("\\midrule")
            in_extra = True
        cells = [fmt(v, not extra and v is not None and v == best[j], True) for j, v in enumerate(vals)]
        print(f"{label} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


if __name__ == "__main__":
    main()
