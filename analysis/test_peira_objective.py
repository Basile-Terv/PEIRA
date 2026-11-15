"""Checks the moment accumulator of the PEIRA closed form and the epoch-time parser."""

import pytest
import torch

from analysis.epoch_time import epoch_intervals
from analysis.peira_objective import MomentAccumulator, peira_objective


def test_accumulator_equals_full_batch() -> None:
    u, v = torch.randn(1024, 64, dtype=torch.float64), torch.randn(1024, 64, dtype=torch.float64)
    acc = MomentAccumulator(64, torch.device("cpu"))
    for i in range(0, 1024, 256):
        acc.update(u[i : i + 256], v[i : i + 256])
    assert abs(acc.finalize(0.1)["objective"] - peira_objective(u, v, 0.1).item()) < 1e-9


def test_epoch_intervals_skip_resume() -> None:
    lines = [
        "[INFO][2026-04-21 06:28:31][x] 📊 [Epoch 000/100] loss=1.0 | time=874.8s",
        "[INFO][2026-04-21 06:42:11][x] 📊 [Epoch 001/100] loss=1.0 | time=1694.8s",
        "[INFO][2026-04-21 06:55:57][x] 📊 [Epoch 002/100] loss=1.0 | time=2520.6s",
        "[INFO][2026-04-21 08:00:00][x] 📊 [Epoch 003/100] loss=1.0 | time=900.0s",
        "[INFO][2026-04-21 08:14:00][x] 📊 [Epoch 004/100] loss=1.0 | time=1724.0s",
    ]
    assert epoch_intervals(lines) == pytest.approx([820.0, 825.8, 824.0])
