"""Anti-collapse losses of the PEIRA baselines: VICReg and SIGReg (LeJEPA BCS).

PEIRALoss itself lives in the ``peira_loss`` package.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@torch._dynamo.disable
def all_reduce(x, op):
    """All-reduce operation for distributed training.

    ``@torch._dynamo.disable`` is required: under ``torch.compile`` the in-place
    ``dist.all_reduce`` is functionalized into an op without an autograd formula,
    which silently detaches the result and removes the gradient of every loss
    built on top of it (e.g. SIGReg).
    """
    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.__dict__[op]
        dist.all_reduce(x, op=op)
        return x
    else:
        return x


def _total_batch_size(local_n: int) -> int:
    """Return ``local_n * world_size`` for the Epps-Pulley scaling factor."""
    if dist.is_available() and dist.is_initialized():
        return local_n * dist.get_world_size()
    return local_n


class FullGatherLayer(torch.autograd.Function):
    """Gather tensors from all processes with correct backward gradients."""

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()  # `dist.all_gather` requires contiguous tensors
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def batch_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Gather ``x`` across GPUs along dim 0; no-op when not distributed."""
    if dist.is_available() and dist.is_initialized():
        return torch.cat(FullGatherLayer.apply(x), dim=0)
    return x


class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = batch_all_gather(x)
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(torch.nn.Module):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__()

    def off_diagonal(self, x):
        n, m = x.shape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        x = batch_all_gather(x)
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        cov_loss = self.off_diagonal(cov).pow(2).mean()

        return cov_loss


class VICRegLoss(nn.Module):
    """VICReg loss combining invariance, variance (std), and covariance terms.

    Paper-faithful coefficient contract (Bardes et al. 2022, Appendix K):
    ``total = sim_coeff * invariance + std_coeff * variance + cov_coeff * covariance``
    with the reference recipe ``sim_coeff=25, std_coeff=25, cov_coeff=1``.

    The variance term sums the two per-view hinges (the reference halves each) and
    the covariance term averages the squared off-diagonal entries over ``D(D-1)``
    (the reference sums them and divides by ``D``). Hence, for ``D``-dim projections,
    ``VICRegLoss(s, v, c) == VICRegReferenceLoss(s, 2 v, c / (D - 1))`` in value and
    gradient, up to floating-point rounding. The CIFAR-10 recipe ``(1, 1, 80)`` at
    ``D = 2048`` is the reference ``(1, 2, 0.0391)``.
    """

    def __init__(self, sim_coeff=1.0, std_coeff=1.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> dict:
        """Compute VICReg loss.

        Args:
            z1: [B, D] first projection.
            z2: [B, D] second projection.

        Returns:
            dict with keys: loss, invariance_loss, var_loss, cov_loss
        """
        sim_loss = F.mse_loss(z1, z2)
        var_loss = self.std_loss_fn(z1) + self.std_loss_fn(z2)
        cov_loss = self.cov_loss_fn(z1) + self.cov_loss_fn(z2)

        total_loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        return {
            "loss": total_loss,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }


######################################################
# BCS (Batched Characteristic Slicing) loss for SIGReg


class EppsPulley(nn.Module):
    """Epps-Pulley test statistic for Gaussianity (LeJEPA-style).

    Uses real cos/sin decomposition on a half-domain [0, t_max] with
    trapezoidal quadrature weights doubled for symmetry. This avoids
    complex arithmetic and is compatible with ``torch.compile``.

    The statistic is scaled by the sample count N so that it has constant
    power regardless of batch size (LEWM/SIGReg convention).

    Args:
        t_max: Upper integration bound (symmetric around 0).
        n_points: Number of quadrature knots on [0, t_max].
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        t = torch.linspace(0, t_max, n_points)
        phi = torch.exp(-0.5 * t**2)
        # Trapezoidal weights, doubled for symmetric [−t_max, t_max]
        dt = t_max / (n_points - 1)
        w = torch.full((n_points,), 2.0 * dt)
        w[0] = dt
        w[-1] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("w", w)

    def forward(
        self, x: torch.Tensor, total_batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute Epps-Pulley statistic.

        Args:
            x: ``[N, M]`` projected samples (N = local batch, M = slices).
            total_batch_size: Global sample count across all GPUs.

        Returns:
            ``[M]`` statistic per slice, scaled by ``total_batch_size``.
        """
        if total_batch_size is None:
            total_batch_size = x.shape[0]
        x_t = x.unsqueeze(2) * self.t  # [N, M, T]
        cos_ecf = x_t.cos().mean(0)  # [M, T]
        sin_ecf = x_t.sin().mean(0)  # [M, T]
        cos_ecf = all_reduce(cos_ecf, op="AVG")
        sin_ecf = all_reduce(sin_ecf, op="AVG")
        err = self.w * self.phi * ((cos_ecf - self.phi) ** 2 + sin_ecf**2)
        return err.sum(1) * total_batch_size  # [M]


def _sliced_epps_pulley(
    x: torch.Tensor,
    step: int,
    num_slices: int,
    total_batch_size: int,
    epps: EppsPulley,
) -> Tuple[torch.Tensor, int]:
    """Random-project ``x`` and return (epps_pulley_mean, next_step).

    Args:
        x: ``[N, D]`` input embeddings.
        step: Current RNG step (used as seed for reproducible projections).
        num_slices: Number of random projection directions.
        total_batch_size: Global sample count for Epps-Pulley scaling.
        epps: ``EppsPulley`` module instance.

    Returns:
        ``(loss, step + 1)`` where ``loss`` is the mean Epps-Pulley statistic
        across slices.
    """
    with torch.no_grad():
        dev = x.device
        g = torch.Generator(device=dev)
        g.manual_seed(step)
        A = torch.randn(x.size(1), num_slices, device=dev, generator=g)
        A /= A.norm(p=2, dim=0)
    projected = x @ A  # [N, num_slices]
    loss = epps(projected, total_batch_size=total_batch_size).mean()
    return loss, step + 1


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg.

    The loss is ``invariance + lmbd * sigreg``; both views share one slice set per step.

    Args:
        num_slices: Random projection directions per step.
        lmbd: SIGReg weight.
    """

    def __init__(self, num_slices: int = 1024, lmbd: float = 0.1):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd
        self._total_n = None
        self.epps = EppsPulley()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> dict:
        """Compute BCS loss.

        Args:
            z1: [B, D] first projection.
            z2: [B, D] second projection.
        """
        if self._total_n is None:
            self._total_n = _total_batch_size(z1.shape[0])
        bcs1, _ = _sliced_epps_pulley(z1, self.step, self.num_slices, self._total_n, self.epps)
        bcs2, self.step = _sliced_epps_pulley(
            z2, self.step, self.num_slices, self._total_n, self.epps
        )
        bcs = (bcs1 + bcs2) / 2
        invariance_loss = F.mse_loss(z1, z2)
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}


class VICRegReferenceLoss(nn.Module):
    """VICReg loss, math-equivalent to the reference repo (facebookresearch/vicreg).

    Unlike :class:`VICRegLoss`, which composes this repo's ``HingeStdLoss`` and
    ``CovarianceLoss`` primitives (and therefore inherits their normalization
    quirks — ``/2`` missing on the hinge, ``mean`` instead of ``sum / D`` on the
    off-diagonal), this class inlines the math from ``main_vicreg.py`` so that
    the coefficients ``(sim_coeff, std_coeff, cov_coeff)`` mean exactly what
    they mean in the paper (Bardes et al. 2022, Appendix K):

        invariance = MSE(z1, z2)
        variance   = mean(relu(1 - sqrt(var(z1) + 1e-4))) / 2
                   + mean(relu(1 - sqrt(var(z2) + 1e-4))) / 2
        covariance = off_diag(cov_z1).pow(2).sum() / D
                   + off_diag(cov_z2).pow(2).sum() / D

    where ``cov_z*`` is computed on batch-all-gathered views (gradient-aware
    via :class:`FullGatherLayer`) and ``D`` is the projection dimension.

    Defaults ``(25, 25, 1)`` are the paper's Appendix K recipe (ImageNet-1K configs).
    ``VICRegReferenceLoss(s, v, c) == VICRegLoss(s, v / 2, c (D - 1))``, see
    :class:`VICRegLoss`.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    @staticmethod
    def _off_diagonal(m: torch.Tensor) -> torch.Tensor:
        n, cols = m.shape
        assert n == cols, "covariance matrix must be square"
        return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _per_view_std_cov(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_all = batch_all_gather(z)
        n, d = z_all.shape
        z_c = z_all - z_all.mean(dim=0)
        std = torch.sqrt(z_c.var(dim=0) + 1e-4)
        std_term = torch.mean(F.relu(1.0 - std))
        cov = (z_c.T @ z_c) / (n - 1)
        cov_term = self._off_diagonal(cov).pow(2).sum().div(d)
        return std_term, cov_term

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> dict:
        sim_loss = F.mse_loss(z1, z2)
        s1, c1 = self._per_view_std_cov(z1)
        s2, c2 = self._per_view_std_cov(z2)
        var_loss = s1 / 2 + s2 / 2
        cov_loss = c1 + c2

        total = (
            self.sim_coeff * sim_loss
            + self.std_coeff * var_loss
            + self.cov_coeff * cov_loss
        )
        return {
            "loss": total,
            "invariance_loss": sim_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
        }
