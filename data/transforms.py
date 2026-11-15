"""Two-view pretraining augmentations (CIFAR-10 at 32x32, ImageNet-1K at 224x224)."""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2

# (image_size, mean, std, num_classes)
DATASET_INFO = {
    "cifar10": (32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 10),
    "imagenet1k": (224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1000),
}


DATASET_ABBREV = {"cifar10": "c10", "imagenet1k": "in1k"}


def get_dataset_info(dataset: str) -> tuple[int, int]:
    """Return (image_size, num_classes) for a dataset.

    Args:
        dataset: Dataset name (e.g. "cifar10", "imagenet1k").

    Returns:
        Tuple of (image_size, num_classes).
    """
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(DATASET_INFO)}")
    info = DATASET_INFO[dataset]
    return info[0], info[3]


class RandomResizedCrop:
    """Random resized crop augmentation."""

    def __init__(
        self,
        size,
        scale=(0.2, 1.0),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.transform = transforms.RandomResizedCrop(
            size, scale=scale, interpolation=interpolation
        )

    def __call__(self, img):
        return self.transform(img)


class ColorJitter:
    """Color jitter augmentation."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Grayscale:
    """Grayscale augmentation."""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.Grayscale(num_output_channels=3)(img)
        return img


class Solarization:
    """Solarization augmentation."""

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = transforms.functional.solarize(img, threshold=128)
        return img


class HorizontalFlip:
    """Horizontal flip augmentation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.functional.hflip(img)
        return img


def get_cifar_train_transforms() -> transforms.Compose:
    """CIFAR-10 training transforms (PIL ``torchvision.transforms`` v1 pipeline)."""
    image_size, mean, std, _ = DATASET_INFO["cifar10"]
    return transforms.Compose(
        [
            RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8),
            Grayscale(prob=0.2),
            Solarization(prob=0.1),
            HorizontalFlip(prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def get_imagenet_train_transforms(blur_prob: float, solar_prob: float) -> v2.Compose:
    """ImageNet-1K training transforms (VICReg/BYOL recipe, ``torchvision.transforms.v2``).

    Uses the tensor-native v2 API (uint8 throughout, one float cast at the end), which is
    typically 2-10x faster than a PIL pipeline. Accepts PIL input from the Arrow loader.

    Args:
        blur_prob: GaussianBlur probability.
        solar_prob: Solarization probability.
    """
    image_size, mean, std, _ = DATASET_INFO["imagenet1k"]
    return v2.Compose(
        [
            v2.PILToTensor(),
            v2.RandomResizedCrop(
                image_size,
                scale=(0.08, 1.0),
                interpolation=v2.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
            ),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=blur_prob),
            v2.RandomSolarize(threshold=128, p=solar_prob),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


def make_transforms(dataset: str) -> list:
    """Per-view train transforms ``[t1, t2]``.

    ImageNet-1K uses the asymmetric BYOL schedule of ``facebookresearch/vicreg``: view 1
    blur p=1.0 / solar p=0.0, view 2 blur p=0.1 / solar p=0.2. CIFAR-10 shares one
    transform between both views.

    Args:
        dataset: ``"cifar10"`` or ``"imagenet1k"``.
    """
    if dataset == "imagenet1k":
        return [
            get_imagenet_train_transforms(blur_prob=1.0, solar_prob=0.0),
            get_imagenet_train_transforms(blur_prob=0.1, solar_prob=0.2),
        ]
    t = get_cifar_train_transforms()
    return [t, t]


def get_val_transforms(dataset: str = "cifar10") -> transforms.Compose:
    """Get validation transforms.

    For IN1K-scale images the resize step uses BICUBIC to match the train-side
    ``RandomResizedCrop`` interpolation (and what the reference VICReg repo
    does). Small-image datasets (CIFAR) stay on torchvision's default bilinear.

    Args:
        dataset: Dataset name. Determines normalization stats and resize/crop.
    """
    image_size, mean, std, _ = DATASET_INFO[dataset]
    ops = []
    if image_size > 32:
        ops += [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)
