"""
Build the retrieval index from raw coin images.

Usage:
    python scripts/build_index.py

Expects images organised as:
    data/raw/<class_name>/<image_file>

Saves the index to:
    data/embeddings/index.pkl
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings.model import CoinEmbeddingModel
from src.retrieval.index import CoinIndex
from src.training.dataset import CoinDataset

DATA_DIR = ROOT / "data" / "raw"
INDEX_PATH = ROOT / "data" / "embeddings" / "index.pkl"
BATCH_SIZE = 32
EMBEDDING_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_index() -> None:
    print(f"Loading dataset from: {DATA_DIR}")
    dataset = CoinDataset(root_dir=DATA_DIR, augment=False)

    if len(dataset) == 0:
        print("No images found. Ensure data/raw/<class>/<image> structure exists.")
        sys.exit(1)

    print(f"Found {len(dataset)} images across {len(dataset.classes)} classes.")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CoinEmbeddingModel(embedding_dim=EMBEDDING_DIM, pretrained=True)
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

    import numpy as np
    embeddings_array = np.array(all_embeddings, dtype="float32")

    index = CoinIndex()
    index.build(embeddings_array, all_metadata)
    index.save(INDEX_PATH)

    print(f"Index saved to: {INDEX_PATH}  ({len(index)} entries)")


if __name__ == "__main__":
    build_index()
