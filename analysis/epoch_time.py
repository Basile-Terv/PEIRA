"""Seconds per epoch from ``train.py`` logs (paper Table G.8).

``train.py`` logs one line per epoch through ``eb_jepa.utils.config.log_epoch``:

    [INFO ][2026-04-21 06:42:11][...] 📊 [Epoch 001/100] loss=... | time=1694.8s

where ``time`` is the wall-clock since the (re)start of the process. An epoch time is
the difference between consecutive epochs of one process; intervals across a resume
(``time`` drops) are skipped.

Usage:
    python -m analysis.epoch_time SIGReg='<sigreg_run>/train.log' PEIRA='<peira_run>/train.log'
"""

import argparse
import glob
import re
import statistics

EPOCH_RE = re.compile(r"\[Epoch (\d+)(?:/\d+)?\].*\btime=([\d.]+)s")


def epoch_intervals(lines) -> list[float]:
    """Per-epoch durations (s) from consecutive ``log_epoch`` lines of one log."""
    out, prev = [], None
    for line in lines:
        m = EPOCH_RE.search(line)
        if not m:
            continue
        epoch, t = int(m.group(1)), float(m.group(2))
        if prev and epoch == prev[0] + 1 and t > prev[1]:
            out.append(t - prev[1])
        prev = (epoch, t)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", help="LABEL=GLOB pairs, one per method.")
    args = p.parse_args()

    print(f"{'method':<10} {'logs':>5} {'epochs':>7} {'mean s/ep':>10} {'median s/ep':>12}")
    for spec in args.logs:
        label, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        intervals = []
        for path in files:
            with open(path, errors="replace") as f:
                intervals += epoch_intervals(f)
        if not intervals:
            print(f"{label:<10} {len(files):>5} {0:>7} {'-':>10} {'-':>12}")
            continue
        print(f"{label:<10} {len(files):>5} {len(intervals):>7} "
              f"{statistics.mean(intervals):>10.0f} {statistics.median(intervals):>12.0f}")


if __name__ == "__main__":
    main()
