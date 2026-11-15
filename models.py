"""ResNet backbone, SSL projector head and linear probe."""

import math

import torch
import torch.nn as nn
import torchvision

RESNET_FEATURES_DIM = {"resnet18": 512, "resnet50": 2048}


class TorchvisionResNet(nn.Module):
    """Torchvision ResNet backbone (resnet18 or resnet50).

    Args:
        arch: Architecture name ('resnet18' or 'resnet50').
        image_size: Input image resolution. Uses standard 7x7 conv1 + maxpool
            for image_size >= 64 (e.g. ImageNet), and 3x3 conv1 without maxpool
            for smaller images (e.g. CIFAR-10).
    """

    def __init__(self, arch: str = "resnet18", image_size: int = 32):
        super().__init__()
        build_fn = getattr(torchvision.models, arch)
        self.backbone = build_fn()
        self.backbone.fc = nn.Identity()
        if image_size < 64:
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            self.backbone.maxpool = nn.Identity()
        self.features_dim = RESNET_FEATURES_DIM[arch]

    def forward(self, x):
        return self.backbone(x)


class ImageSSL(nn.Module):
    """Backbone followed by a 3-layer BN+ReLU MLP projector.

    ``forward`` returns ``(features [B, features_dim], projections [B, proj_output_dim])``;
    the features feed the online linear probe, the projections feed the SSL loss.
    """

    def __init__(
        self,
        backbone,
        features_dim,
        proj_hidden_dim=2048,
        proj_output_dim=2048,
        proj_final_bias: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.features_dim = features_dim

        # ``proj_final_bias=False`` matches facebookresearch/vicreg.
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim, bias=proj_final_bias),
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.squeeze(-1).squeeze(-1)
        projections = self.projector(features)
        return features, projections


def init_weights(m: nn.Module, scale: float = 1.0) -> None:
    """Re-initialize conv/linear/BN layers with optional scaling on the weight magnitudes.

    Used to study robustness to initialization scale (``cfg.init.scale``).
    Scale 1.0 gives Kaiming-normal init; train.py skips re-initialization at scale 1.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            m.weight.mul_(scale)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=math.sqrt(5))
        with torch.no_grad():
            m.weight.mul_(scale)
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)
            with torch.no_grad():
                m.bias.mul_(scale)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias) 


def init_last_linear(model: nn.Module, scale: float = 1.0) -> None:
    """Re-initialize only the last registered Linear layer, including its bias.

    Call once on the model, not through ``model.apply``. For ``ImageSSL`` with
    a projector, this selects the projector's output layer.
    """
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise ValueError("Model has no Linear layer to initialize")
    init_weights(linear_layers[-1], scale=scale)


class LinearProbe(nn.Module):
    """Linear classifier on frozen features (online probe during pretraining)."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def load_pretrained(path: str) -> tuple[dict[str, torch.Tensor], dict]:
    """``ImageSSL`` weights of a ``train.py`` checkpoint or a ``scripts/export_hf.py`` file.

    Strips any nesting of the ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) key
    prefixes and keeps the ``backbone.*`` and ``projector.*`` entries, dropping the online
    probe, optimizer, scheduler and loss states of training checkpoints. Exported release
    files hold the backbone only, so their state dict has no ``projector.*`` entries.

    Returns:
        (state dict, loaded checkpoint dict). Exported files also carry ``config`` (the
        training config) and ``metadata`` (method, dims, seed, epochs, offline top-1).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    for prefix in ("module.", "_orig_mod."):
        while any(k.startswith(prefix) for k in sd):
            sd = {k.removeprefix(prefix): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k.startswith(("backbone.", "projector."))}
    if not sd:
        raise KeyError(f"No 'backbone.*' or 'projector.*' keys in {path}")
    return sd, ckpt
