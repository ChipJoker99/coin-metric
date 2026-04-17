"""
Image preprocessing utilities.

Provides a single entry point for converting raw images (file paths or PIL
Images) into normalized PyTorch tensors ready for model inference.
"""

from pathlib import Path
from typing import Union

import torch
from torchvision import transforms
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess_image(
    source: Union[str, Path, Image.Image],
    size: int = 224,
) -> torch.Tensor:
    """
    Convert an image to a normalized tensor suitable for model input.

    Args:
        source: File path (str or Path) or an already-loaded PIL Image.
        size: Target square size in pixels (default 224).

    Returns:
        Float tensor of shape (3, size, size), ImageNet-normalized.
    """
    if isinstance(source, (str, Path)):
        image = Image.open(source).convert("RGB")
    elif isinstance(source, Image.Image):
        image = source.convert("RGB")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if size == 224:
        transform = _DEFAULT_TRANSFORM
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transform(image)
