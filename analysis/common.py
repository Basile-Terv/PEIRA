"""Shared helpers for the analysis scripts: load a pretrained ImageSSL and its config."""

import os
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from models import ImageSSL, TorchvisionResNet, load_pretrained


def load_config(ckpt: str, config: Optional[str] = None) -> DictConfig:
    """Run config, by default the ``config.yaml`` next to the checkpoint."""
    return OmegaConf.load(config or Path(ckpt).parent / "config.yaml")


def resolve_data_dir(cfg: DictConfig, data_dir: Optional[str]) -> str:
    """CLI value, else ``data.data_dir``, else ``$EBJEPA_DSETS`` (as in ``train.py``)."""
    return data_dir or cfg.data.get("data_dir") or os.environ["EBJEPA_DSETS"]


def load_ssl_model(ckpt: str, cfg: DictConfig, device: torch.device) -> tuple[ImageSSL, int]:
    """Frozen backbone + projector of a ``train.py`` checkpoint, in eval mode.

    Strips the ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) key prefixes (see
    ``models.load_pretrained``). The analyses read projector outputs, so the backbone-only
    release files of ``scripts/export_hf.py`` are rejected.

    Returns:
        (model, pretraining epoch stored in the checkpoint).
    """
    m = cfg.model
    image_size = 224 if cfg.data.dataset.startswith("imagenet") else 32
    backbone = TorchvisionResNet(arch=m.get("resnet_arch", "resnet50"), image_size=image_size)
    model = ImageSSL(
        backbone,
        features_dim=backbone.features_dim,
        proj_hidden_dim=m.proj_hidden_dim,
        proj_output_dim=m.proj_output_dim,
        proj_final_bias=m.get("proj_final_bias", True),
    )
    sd, blob = load_pretrained(ckpt)
    if not any(k.startswith("projector.") for k in sd):
        raise ValueError(
            f"{ckpt} has no projector weights (a backbone-only release file?); the analysis "
            "scripts need a full train.py checkpoint."
        )
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, int(blob.get("epoch", -1))
