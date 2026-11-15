"""LARS optimizers: solo-learn style :class:`LARS` (CIFAR-10) and :class:`LARSReference` (IN1K)."""

import torch
from torch.optim.optimizer import Optimizer, required


class LARS(Optimizer):
    """LARS (Layer-wise Adaptive Rate Scaling) optimizer.

    Applies per-layer learning rate scaling based on the ratio of parameter
    norm to gradient norm, combined with SGD + momentum.

    Args:
        params: Iterable of parameters or param groups.
        lr: Base learning rate.
        momentum: Momentum factor (default: 0).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        eta: LARS trust coefficient (default: 1e-3).
        eps: Small constant for numerical stability (default: 1e-8).
        clip_lr: Clip the LARS-scaled LR to at most the base LR (default: False).
        exclude_bias_n_norm: Skip LARS scaling for 1-D params (biases and
            normalization layers) (default: False).
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        weight_decay=0,
        eta=1e-3,
        eps=1e-8,
        clip_lr=False,
        exclude_bias_n_norm=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            eps=eps,
            clip_lr=clip_lr,
            exclude_bias_n_norm=exclude_bias_n_norm,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if p.ndim != 1 or not group["exclude_bias_n_norm"]:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
                        lars_lr *= group["eta"]

                        # clip lr
                        if group["clip_lr"]:
                            lars_lr = min(lars_lr / group["lr"], 1)

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


def exclude_bias_and_norm(p: torch.Tensor) -> bool:
    """Reference-VICReg filter: True for biases and BatchNorm weights (ndim=1)."""
    return p.ndim == 1


class LARSReference(Optimizer):
    """LARS matching facebookresearch/vicreg/main_vicreg.py exactly.

    Differences from :class:`LARS` (the repo's older implementation):

    * Weight decay is applied to the gradient *before* the LARS trust-ratio
      is computed, so ``update_norm = ||g + wd * p||``. The repo class
      approximates that norm via the triangle inequality.
    * No LR clipping. :class:`LARS` has ``clip_lr=True`` by default, which
      is absent from the reference.
    * Biases and BN γ/β (:func:`exclude_bias_and_norm`) are excluded from both
      weight decay and LARS scaling, as at the reference's call site.

    Args:
        params: Iterable of parameters or param groups.
        lr: Base learning rate.
        weight_decay: L2 penalty coefficient (default ``0``).
        momentum: Momentum factor (default ``0.9``).
        eta: LARS trust coefficient (default ``0.001``).
    """

    def __init__(
        self,
        params,
        lr=required,
        weight_decay=0,
        momentum=0.9,
        eta=1e-3,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = {"lr": lr, "weight_decay": weight_decay, "momentum": momentum, "eta": eta}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad
                if dp is None:
                    continue

                if not exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0,
                            (g["eta"] * param_norm / update_norm),
                            one,
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                state = self.state[p]
                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)
                mu = state["mu"]
                mu.mul_(g["momentum"]).add_(dp)
                p.add_(mu, alpha=-g["lr"])

        return loss


def build_optimizer(cfg_optim, param_groups: list) -> torch.optim.Optimizer:
    """Build the optimizer named by ``cfg_optim.type`` (``"lars"`` or ``"lars_ref"``).

    Args:
        cfg_optim: Optimizer sub-config. Recognised keys: ``type``, ``weight_decay``,
            ``momentum``, ``lars_eta``, ``lars_clip_lr``, ``lars_exclude_bias_n_norm``.
            ``lars_ref`` is the reference VICReg LARS (biases and norm weights
            excluded from weight decay and adaptation).
        param_groups: Parameter group dicts, each with ``"params"`` and usually ``"lr"``.

    Returns:
        Configured ``torch.optim.Optimizer``.
    """
    optim_type = cfg_optim.type
    if optim_type == "lars":
        return LARS(
            param_groups,
            weight_decay=cfg_optim.get("weight_decay", 0),
            momentum=cfg_optim.get("momentum", 0.9),
            eta=cfg_optim.get("lars_eta", 0.001),
            clip_lr=cfg_optim.get("lars_clip_lr", True),
            exclude_bias_n_norm=cfg_optim.get("lars_exclude_bias_n_norm", True),
        )
    if optim_type == "lars_ref":
        return LARSReference(
            param_groups,
            weight_decay=cfg_optim.get("weight_decay", 0),
            momentum=cfg_optim.get("momentum", 0.9),
            eta=cfg_optim.get("lars_eta", 0.001),
        )
    raise ValueError(
        f"Unknown optimizer type: {optim_type!r}. Supported: 'lars', 'lars_ref'."
    )
