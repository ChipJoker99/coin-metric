"""
Coin dataset loader.

Expects the following folder structure:

    data/raw/
        <class_name>/
            image1.jpg
            image2.jpg
            ...

Each subdirectory name is used as the class label.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_augmentation_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _build_eval_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CoinDataset(Dataset):
    """
    PyTorch Dataset for coin images organized by class folder.

    Args:
        root_dir: Path to the root data directory (e.g. data/raw/).
        augment: If True, apply training augmentations. Default False.
        image_size: Target image size in pixels.
    """

    def __init__(
        self,
        root_dir: str | Path,
        augment: bool = False,
        image_size: int = 224,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = (
            _build_augmentation_transform(image_size)
            if augment
            else _build_eval_transform(image_size)
        )

        self.samples: list[tuple[Path, int]] = []
        self.classes: list[str] = sorted(
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        )
        self.class_to_idx: dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        for cls in self.classes:
            label = self.class_to_idx[cls]
            class_dir = self.root_dir / cls
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, label
