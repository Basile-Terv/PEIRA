"""Datasets: multi-view pretraining wrapper, ImageNet-1K (HF Arrow or ImageFolder), CIFAR-10
and the LeJEPA-8 transfer suite of ``probe.py``."""

import os
from typing import Callable, NamedTuple, Optional

import torch
import torch.utils.data
import torchvision.datasets
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    """Dataset that applies per-view augmentations to create multiple views.

    Args:
        dataset: Base dataset returning (image, label) tuples.
        transforms: List of transform pipelines, one per view.
    """

    def __init__(self, dataset, transforms: list):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        views = [t(image) for t in self.transforms]
        return views, label


# Torchvision's ImageNet1k split names ("val") differ from Hugging Face's
# ("validation"); normalize the former to the latter.
_HF_SPLIT_ALIASES = {"val": "validation"}


def is_hf_arrow_dataset_dir(root: str) -> bool:
    """Return True iff ``root`` is a Hugging Face ``DatasetDict`` saved to disk."""
    return os.path.isfile(os.path.join(root, "dataset_dict.json"))


class HFImageDataset(torch.utils.data.Dataset):
    """Map-style ``(image, int-label)`` view of one Hugging Face image-classification split.

    Exposes the same contract as ``torchvision.datasets.ImageFolder``.

    Args:
        dataset: A ``datasets.Dataset`` with ``image`` (PIL) and ``label`` (int) columns.
        transform: Optional callable applied to each RGB PIL image.
    """

    def __init__(self, dataset, transform: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


class HFImageNet(HFImageDataset):
    """ImageNet from a HuggingFace DatasetDict saved with ``save_to_disk``.

    Args:
        root: Directory with ``dataset_dict.json`` and per-split Arrow shards.
        split: Split name. ``"val"`` is translated to ``"validation"`` automatically.
        transform: Optional callable applied to each PIL image before it is returned.
    """

    def __init__(self, root: str, split: str, transform: Optional[Callable] = None):
        from datasets import load_from_disk

        dset = load_from_disk(root)
        resolved = split if split in dset else _HF_SPLIT_ALIASES.get(split, split)
        if resolved not in dset:
            raise ValueError(
                f"Split {split!r} not found under {root!r}. Available splits: {sorted(dset.keys())}"
            )
        super().__init__(dset[resolved], transform)
        self.split = resolved


def build_imagenet(data_dir: str, split: str, transform: Optional[Callable] = None):
    """ImageNet-1K ``split`` ("train" or "val") from an HF Arrow dir or an ImageFolder tree."""
    if data_dir is None or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"ImageNet1k root directory not found at {data_dir!r}. "
            "Set data.data_dir (or EBJEPA_DSETS) to the ImageNet root."
        )
    if is_hf_arrow_dataset_dir(data_dir):
        return HFImageNet(data_dir, split=split, transform=transform)
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"ImageNet1k {split} directory not found at {split_dir}. Expected either a "
            "Hugging Face DatasetDict (data_dir/dataset_dict.json) or an ImageFolder layout "
            "(data_dir/{train,val}/<wnid>/*.JPEG)."
        )
    return torchvision.datasets.ImageFolder(split_dir, transform=transform)


class TransferSpec(NamedTuple):
    """One frozen-feature transfer dataset.

    Attributes:
        tv_class: ``torchvision.datasets`` class name (None for the HF-hosted Stanford Cars).
        splits: (train, test) values of the constructor's split argument.
        num_classes: Linear-head output dimension.
        mean_per_class: Report mean per-class recall instead of overall top-1.
        kwargs: Extra constructor kwargs.
    """

    tv_class: Optional[str]
    splits: tuple
    num_classes: int
    mean_per_class: bool
    kwargs: dict = {}


# The LeJEPA-8 transfer suite (LeJEPA Table 2), with each dataset's canonical metric.
TRANSFER_DATASETS = {
    "cifar10": TransferSpec("CIFAR10", (True, False), 10, False),
    "cifar100": TransferSpec("CIFAR100", (True, False), 100, False),
    "dtd": TransferSpec("DTD", ("train", "test"), 47, True, {"partition": 1}),
    "flowers102": TransferSpec("Flowers102", ("train", "test"), 102, True),
    "pets": TransferSpec("OxfordIIITPet", ("trainval", "test"), 37, True),
    "food101": TransferSpec("Food101", ("train", "test"), 101, False),
    "aircraft": TransferSpec(
        "FGVCAircraft", ("trainval", "test"), 100, True, {"annotation_level": "variant"}
    ),
    "cars": TransferSpec(None, ("train", "test"), 196, True),
}


def build_transfer_dataset(name: str, split: str, root: str, transform: Optional[Callable] = None):
    """Instantiate ``split`` ("train" or "test") of a ``TRANSFER_DATASETS`` entry under ``root``."""
    spec = TRANSFER_DATASETS[name]
    value = spec.splits[split != "train"]
    if spec.tv_class is None:  # torchvision's StanfordCars download is broken upstream
        from datasets import load_dataset

        hf = load_dataset("tanganke/stanford_cars", split=value, cache_dir=os.path.join(root, "hf"))
        return HFImageDataset(hf, transform)
    split_kw = {"train": value} if isinstance(value, bool) else {"split": value}
    cls = getattr(torchvision.datasets, spec.tv_class)
    return cls(root=root, download=True, transform=transform, **split_kw, **spec.kwargs)
