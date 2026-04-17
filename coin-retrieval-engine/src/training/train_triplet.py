"""
Triplet loss training loop for the coin embedding model.

Trains CoinEmbeddingModel using TripletMarginLoss on (anchor, positive, negative)
triplets sampled from TripletCoinDataset.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from embeddings.model import CoinEmbeddingModel
from training.triplet_dataset import TripletCoinDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict) -> tuple[CoinEmbeddingModel, list[float]]:
    """
    Train the embedding model with triplet loss.

    Args:
        config: Training configuration dict with the following keys:
            - data_dir (str): Path to data/raw/ directory.
            - embedding_dim (int): Embedding output dimension. Default 128.
            - lr (float): Learning rate. Default 1e-4.
            - epochs (int): Number of training epochs. Default 20.
            - batch_size (int): Batch size. Default 16.
            - margin (float): Triplet loss margin. Default 0.3.
            - checkpoint_dir (str): Directory for model checkpoints.
            - checkpoint_every (int): Save checkpoint every N epochs. Default 5.
            - pretrained (bool): Use pretrained backbone. Default True.
            - freeze_backbone (bool): Freeze backbone weights, train only the
              projection head. Dramatically faster on CPU. Default False.
            - augment (bool): Apply augmentations. Default True.
            - seed (int | None): Optional seed for reproducibility. Default None.

    Returns:
        Tuple of (trained CoinEmbeddingModel, list of mean loss values per epoch).
    """
    seed = config.get("seed", None)
    if seed is not None:
        _set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = TripletCoinDataset(
        root_dir=config["data_dir"],
        augment=config.get("augment", True),
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 16),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = CoinEmbeddingModel(
        embedding_dim=config.get("embedding_dim", 128),
        pretrained=config.get("pretrained", True),
    ).to(device)

    if config.get("freeze_backbone", False):
        for param in model.backbone.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Backbone frozen — trainable params: {trainable:,} / {total:,}")

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.get("lr", 1e-4))
    criterion = nn.TripletMarginLoss(margin=config.get("margin", 0.3))

    checkpoint_dir = Path(config.get("checkpoint_dir", "models/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = config.get("checkpoint_every", 5)

    epoch_losses: list[float] = []

    for epoch in range(1, config.get("epochs", 20) + 1):
        batch_losses: list[float] = []

        for anchors, positives, negatives in loader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            emb_a = model(anchors)
            emb_p = model(positives)
            emb_n = model(negatives)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(mean_loss)
        print(f"Epoch [{epoch:>3}/{config.get('epochs', 20)}]  loss: {mean_loss:.4f}")

        if epoch % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {  # noqa: E501
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": mean_loss,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  → checkpoint saved: {ckpt_path}")

    model.eval()
    return model, epoch_losses
