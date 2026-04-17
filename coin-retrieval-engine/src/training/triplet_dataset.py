"""
Triplet sampling dataset for metric learning.

Wraps CoinDataset and yields (anchor, positive, negative) tensor triplets.
Each class must have at least 2 samples to form valid anchor/positive pairs.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from training.dataset import CoinDataset


class TripletCoinDataset(Dataset):
    """
    Yields (anchor, positive, negative) image triplets for triplet loss training.

    Sampling strategy:
    - anchor: random sample from class C
    - positive: different sample from same class C (anchor_idx ≠ positive_idx)
    - negative: random sample from any class ≠ C

    Args:
        root_dir: Path to the root data directory (same structure as CoinDataset).
        augment: If True, apply training augmentations.
        image_size: Target image size in pixels.
        seed: Optional integer seed for reproducible sampling.
    """

    def __init__(
        self,
        root_dir: str | Path,
        augment: bool = True,
        image_size: int = 224,
        seed: int | None = None,
    ) -> None:
        self._dataset = CoinDataset(root_dir=root_dir, augment=augment, image_size=image_size)
        self._rng = random.Random(seed)

        if len(self._dataset.classes) < 2:
            raise ValueError(
                f"TripletCoinDataset requires at least 2 classes, "
                f"found {len(self._dataset.classes)}."
            )

        self._class_to_indices: dict[int, list[int]] = {}
        for idx, (_, label) in enumerate(self._dataset.samples):
            self._class_to_indices.setdefault(label, []).append(idx)

        for cls_idx, indices in self._class_to_indices.items():
            cls_name = self._dataset.classes[cls_idx]
            if len(indices) < 2:
                raise ValueError(
                    f"Class '{cls_name}' has only {len(indices)} sample(s). "
                    f"Each class needs at least 2 images to form anchor/positive pairs."
                )

        self._class_ids: list[int] = list(self._class_to_indices.keys())

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, anchor_label = self._dataset.samples[idx]

        positive_pool = [i for i in self._class_to_indices[anchor_label] if i != idx]
        positive_idx = self._rng.choice(positive_pool)

        negative_classes = [c for c in self._class_ids if c != anchor_label]
        negative_label = self._rng.choice(negative_classes)
        negative_idx = self._rng.choice(self._class_to_indices[negative_label])

        anchor, _ = self._dataset[idx]
        positive, _ = self._dataset[positive_idx]
        negative, _ = self._dataset[negative_idx]

        return anchor, positive, negative

    @property
    def classes(self) -> list[str]:
        return self._dataset.classes

    @property
    def num_classes(self) -> int:
        return len(self._dataset.classes)
