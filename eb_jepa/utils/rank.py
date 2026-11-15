"""Effective-rank utilities for representation diagnostics."""

from collections import deque

import torch


def effective_rank_from_spectrum(singular_values: torch.Tensor) -> float:
    """``exp(H)`` of the L1-normalized spectrum (singular or eigen values).

    Shared kernel of the raw-matrix and the image_jepa covariance-spectrum paths.

    Args:
        singular_values: 1-D tensor of non-negative values.

    Returns:
        Effective rank in ``[1, len(spectrum)]``; ``nan`` for an empty, non-finite
        or ~0-mass spectrum.
    """
    s = singular_values.float()
    total = s.sum()
    if s.numel() == 0 or not torch.isfinite(total) or total <= 0:
        return float("nan")
    p = s / total
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


def compute_effective_rank(matrix: torch.Tensor) -> float:
    """Entropy-based effective rank of a matrix.

    Non-finite rows (e.g. a diverging encoder that overflowed bf16) are dropped
    before the SVD, since ``svdvals`` raises on them and would crash training.

    Args:
        matrix: Input tensor [N, D].

    Returns:
        Effective rank in ``[1, D]``, or ``nan`` if no finite row remains.
    """
    mat = matrix.float()
    finite_rows = torch.isfinite(mat).all(dim=1)  # [N]
    if not bool(finite_rows.any()):
        return float("nan")
    if not bool(finite_rows.all()):
        mat = mat[finite_rows]  # [N_finite, D]
    return effective_rank_from_spectrum(torch.linalg.svdvals(mat))


class EffectiveRankAccumulator:
    """Accumulates embeddings over batches; ``compute()`` returns ranks and resets.

    Args:
        max_batches: Batches kept per name (oldest dropped first).
    """

    def __init__(self, max_batches: int = 10) -> None:
        self.max_batches = max_batches
        self._buffers: dict[str, deque] = {}

    def accumulate(self, name: str, embeddings: torch.Tensor) -> None:
        """Store one [N, D] chunk under ``name`` (e.g. ``train/visual_effective_rank``)."""
        if name not in self._buffers:
            self._buffers[name] = deque(maxlen=self.max_batches)
        self._buffers[name].append(embeddings.detach().cpu())

    def compute(self) -> dict[str, float]:
        """Effective rank of each name's concatenated chunks; clears the buffers."""
        results: dict[str, float] = {}
        for name, chunks in self._buffers.items():
            if chunks:
                cat = torch.cat(list(chunks), dim=0)  # [N_total, D]
                results[name] = compute_effective_rank(cat)
        self._buffers.clear()
        return results
