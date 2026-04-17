"""
Full training pipeline for the Coin Retrieval Engine.

Reads configs/train_config.yaml, trains the embedding model with triplet loss,
evaluates Recall@K, and rebuilds the retrieval index with the trained weights.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
"""
from embeddings.model import CoinEmbeddingModel
from metrics.retrieval_metrics import evaluate
from retrieval.index import CoinIndex
from training.dataset import CoinDataset
from training.train_triplet import train

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CONFIG = ROOT / "configs" / "train_config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)

    config["data_dir"] = str(ROOT / config["data_dir"])
    config["checkpoint_dir"] = str(ROOT / config["checkpoint_dir"])
    config["index_path"] = str(ROOT / config.get("index_path", "data/embeddings/index.pkl"))
    return config


def build_index_from_model(
    model: CoinEmbeddingModel,
    data_dir: str,
    index_path: str,
    device: str,
) -> None:
    dataset = CoinDataset(root_dir=data_dir, augment=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval().to(device)
    all_embeddings = []
    all_metadata = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            embeddings = model(images.to(device)).cpu().numpy()
            all_embeddings.extend(embeddings)
            start = batch_idx * 32
            for i, label_idx in enumerate(labels.tolist()):
                sample_path, _ = dataset.samples[start + i]
                all_metadata.append({
                    "label": dataset.classes[label_idx],
                    "path": str(sample_path),
                })

    embeddings_array = np.array(all_embeddings, dtype="float32")
    index = CoinIndex()
    index.build(embeddings_array, all_metadata)
    index.save(index_path)
    print(f"  Index saved → {index_path}  ({len(index)} entries)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the coin embedding model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to train_config.yaml (default: configs/train_config.yaml)",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    print("=" * 60)
    print("  Coin Retrieval Engine — Training")
    print("=" * 60)
    print(f"  backbone      : {config.get('backbone', 'resnet18')}")
    print(f"  embedding_dim : {config.get('embedding_dim', 128)}")
    print(f"  epochs        : {config.get('epochs', 80)}")
    print(f"  lr            : {config.get('lr', 1e-4)}")
    print(f"  margin        : {config.get('margin', 0.5)}")
    print(f"  batch_size    : {config.get('batch_size', 16)}")
    print(f"  augment       : {config.get('augment', True)}")
    print(f"  pretrained    : {config.get('pretrained', True)}")
    print(f"  freeze_backbone: {config.get('freeze_backbone', False)}")
    print(f"  data_dir      : {config['data_dir']}")
    print("=" * 60)

    print("\n[1/3] Training...")
    trained_model, losses = train(config)

    print(f"\n  Loss: {losses[0]:.4f} → {losses[-1]:.4f}  "
          f"({'↓ improved' if losses[-1] < losses[0] else '⚠ did not decrease'})")

    print("\n[2/3] Evaluating retrieval metrics...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_dataset = CoinDataset(root_dir=config["data_dir"], augment=False)
    metrics = evaluate(trained_model, eval_dataset, device=device, top_k_list=[1, 5])
    print(f"  Recall@1  : {metrics['recall@1']:.3f}")
    print(f"  Recall@5  : {metrics['recall@5']:.3f}")
    print(f"  DistRatio : {metrics['distance_ratio']:.3f}  (< 1.0 = well-separated)")

    print("\n[3/3] Rebuilding retrieval index with trained weights...")
    build_index_from_model(
        model=trained_model,
        data_dir=config["data_dir"],
        index_path=config["index_path"],
        device=device,
    )

    print("\n" + "=" * 60)
    print("  Training complete. Run Streamlit to test the updated model:")
    print("  streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
