"""Evaluation metrics during pretraining: online linear-probe accuracy and spectral diagnostics.

Spectral diagnostics: spectrum of the intra-view covariance ``E[z1 z1^T]`` and alignment
of the inter-view covariance ``E[z2 z1^T]`` with the intra-view eigenbasis.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch.amp import autocast


def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
    """Evaluate linear probe on validation set.

    Returns:
        top1: Top-1 accuracy (%).
        top5: Top-5 accuracy (%).
        avg_loss: Mean cross-entropy over batches.
    """
    model.eval()
    linear_probe.eval()

    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                probe_features, _ = model(data)

            outputs = linear_probe(probe_features.float())
            loss = F.cross_entropy(outputs, target)

            total_loss += loss.item()
            total += target.size(0)

            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(target).sum().item()

            _, top5_pred = outputs.topk(5, dim=1)
            correct_top5 += top5_pred.eq(target.unsqueeze(1)).any(1).sum().item()

    top1 = 100.0 * correct_top1 / total
    top5 = 100.0 * correct_top5 / total
    avg_loss = total_loss / len(val_loader)

    return top1, top5, avg_loss


def compute_alignment(M: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of each column ``u_i`` of ``U`` with ``M @ u_i``.

    Args:
        M: ``[k, k]`` matrix (e.g. inter-view covariance).
        U: ``[k, d]`` matrix whose columns are the directions to probe.

    Returns:
        ``[d]`` tensor of per-column cosine similarities.
    """
    M_ui = M @ U
    norms_M_ui = torch.sqrt(torch.diag(M_ui.t() @ M_ui))
    norms_U = torch.sqrt(torch.diag(U.t() @ U))
    dot_products = torch.sum(U * M_ui, dim=0)
    return dot_products / ((norms_M_ui + 1e-8) * (norms_U + 1e-8))


@torch.no_grad()
def compute_spectral_metrics(
    model,
    data_loader,
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, torch.Tensor]:
    """Estimate intra-view spectrum and inter-view alignment.

    Args:
        model: ``ImageSSL``; ``model(x) -> (features, projections)``.
        data_loader: yields ``((view1, view2), target)``.
        device: device for matmuls.
        num_batches: number of batches to average covariances over.

    Returns:
        ``{"spectrum": [k], "alignment": [k]}`` tensors (CPU-resident).
    """
    was_training = model.training
    model.eval()

    inter_cov = None
    intra_cov = None
    for batch_idx, (views, _) in enumerate(data_loader):
        if batch_idx == num_batches:
            break
        view1 = views[0].to(device, non_blocking=True)
        view2 = views[1].to(device, non_blocking=True)
        _, z1 = model(view1)
        _, z2 = model(view2)
        z1 = z1.float()
        z2 = z2.float()
        B = z1.shape[0]
        if inter_cov is None:
            k = z1.shape[1]
            inter_cov = torch.zeros((k, k), device=z1.device)
            intra_cov = torch.zeros((k, k), device=z1.device)
        inter_cov += (z2.t() @ z1) / (B * num_batches)
        intra_cov += (z1.t() @ z1 + z2.t() @ z2) / (2 * B * num_batches)

    if was_training:
        model.train()

    R, S, _ = torch.linalg.svd(intra_cov)
    alignment = compute_alignment(inter_cov, R)
    return {"spectrum": S.detach().cpu(), "alignment": alignment.detach().cpu()}
