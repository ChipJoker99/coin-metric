import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


class CoinEmbeddingModel(nn.Module):
    """
    Args:
        embedding_dim: Output embedding dimension (default 128).
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, embedding_dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.projection = nn.Linear(in_features, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            L2-normalized embeddings of shape (B, embedding_dim).
        """
        features = self.backbone(x)
        projected = self.projection(features)
        return F.normalize(projected, p=2, dim=1)

    def extract(self, image_path: str | Path, device: str = "cpu") -> np.ndarray:
        """
        Args:
            image_path: Path to the image file.
            device: Torch device string.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        self.eval()
        self.to(device)
        with torch.no_grad():
            embedding = self(tensor)

        return embedding.squeeze(0).cpu().numpy()
