"""
Build the retrieval index from raw coin images.

Automatically uses the latest trained checkpoint from models/checkpoints/ if
available. Falls back to pretrained ImageNet weights with a warning.

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from embeddings.model import CoinEmbeddingModel
from retrieval.index import CoinIndex
from training.dataset import CoinDataset

DATA_DIR = ROOT / "data" / "raw"
INDEX_PATH = ROOT / "data" / "embeddings" / "index.pkl"
CHECKPOINTS_DIR = ROOT / "models" / "checkpoints"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _latest_checkpoint() -> Path | None:
    if not CHECKPOINTS_DIR.exists():
        return None
    checkpoints = sorted(CHECKPOINTS_DIR.glob("*.pt"))
    return checkpoints[-1] if checkpoints else None


def _load_model(checkpoint: Path | None) -> CoinEmbeddingModel:
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        embedding_dim = ckpt.get("config", {}).get("embedding_dim", 128)
        model = CoinEmbeddingModel(embedding_dim=embedding_dim, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {checkpoint.name}  (embedding_dim={embedding_dim})")
        return model

    print("  ⚠  No checkpoint found — using pretrained ImageNet weights.")
    print("     Run 'python scripts/train.py' first for better retrieval quality.")
    return CoinEmbeddingModel(embedding_dim=128, pretrained=True)


def build_index() -> None:
    print(f"Loading dataset from: {DATA_DIR}")
    dataset = CoinDataset(root_dir=DATA_DIR, augment=False)

    if len(dataset) == 0:
        print("No images found. Ensure data/raw/<class>/<image> structure exists.")
        sys.exit(1)

    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    checkpoint = _latest_checkpoint()
    model = _load_model(checkpoint)
    model.eval().to(DEVICE)

    all_embeddings = []
    all_metadata = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            embeddings = model(images).cpu().numpy()
            all_embeddings.extend(embeddings)

            start = batch_idx * BATCH_SIZE
            for i, label_idx in enumerate(labels.tolist()):
                sample_path, _ = dataset.samples[start + i]
                all_metadata.append({
                    "label": dataset.classes[label_idx],
                    "path": str(sample_path),
                })

    import numpy as np  # noqa: F811 — already imported at top
    embeddings_array = np.array(all_embeddings, dtype="float32")

    index = CoinIndex()
    index.build(embeddings_array, all_metadata)
    index.save(INDEX_PATH)

    print(f"Index saved to: {INDEX_PATH}  ({len(index)} entries)")


if __name__ == "__main__":
    build_index()
