"""
Overfitting sanity test for the triplet training pipeline.

Trains on a tiny synthetic dataset (3 classes x 15 images) and verifies that:
1. Loss decreases consistently over epochs.
2. Recall@1 improves significantly after training (model can memorize).

This validates pipeline correctness, not generalisation.
Runs with pretrained=False for speed.
"""

from pathlib import Path

from embeddings.model import CoinEmbeddingModel
import numpy as np
import pytest
from PIL import Image

from metrics.retrieval_metrics import evaluate
from training.dataset import CoinDataset
from training.train_triplet import train


NUM_CLASSES = 3
IMAGES_PER_CLASS = 15
EPOCHS = 30
SEED = 42


def _create_synthetic_dataset(root: Path) -> None:
    """Write NUM_CLASSES × IMAGES_PER_CLASS random JPEGs to disk."""
    rng = np.random.default_rng(SEED)
    for cls_id in range(NUM_CLASSES):
        class_dir = root / f"coin_class_{cls_id}"
        class_dir.mkdir(parents=True)
        for img_id in range(IMAGES_PER_CLASS):
            pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(class_dir / f"img_{img_id:03d}.jpg")


@pytest.fixture(scope="module")
def synthetic_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synthetic_coins")
    _create_synthetic_dataset(root)
    return root


def test_loss_decreases(synthetic_data_dir: Path) -> None:
    config = {
        "data_dir": str(synthetic_data_dir),
        "embedding_dim": 64,
        "lr": 1e-3,
        "epochs": EPOCHS,
        "batch_size": 9,
        "margin": 0.3,
        "pretrained": False,
        "augment": False,
        "seed": SEED,
        "checkpoint_dir": str(synthetic_data_dir / "checkpoints"),
        "checkpoint_every": 999,
    }
    _, losses = train(config)

    assert len(losses) == EPOCHS
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


def test_recall_improves_after_training(synthetic_data_dir: Path) -> None:
    eval_dataset = CoinDataset(root_dir=synthetic_data_dir, augment=False)

    untrained_model = CoinEmbeddingModel(embedding_dim=64, pretrained=False)
    baseline_metrics = evaluate(untrained_model, eval_dataset, device="cpu", top_k_list=[1])
    baseline_recall = baseline_metrics["recall@1"]

    config = {
        "data_dir": str(synthetic_data_dir),
        "embedding_dim": 64,
        "lr": 1e-3,
        "epochs": EPOCHS,
        "batch_size": 9,
        "margin": 0.3,
        "pretrained": False,
        "augment": False,
        "seed": SEED,
        "checkpoint_dir": str(synthetic_data_dir / "checkpoints"),
        "checkpoint_every": 999,
    }
    trained_model, _ = train(config)

    trained_metrics = evaluate(trained_model, eval_dataset, device="cpu", top_k_list=[1])
    trained_recall = trained_metrics["recall@1"]

    assert trained_recall >= 0.7, (
        f"Expected Recall@1 ≥ 0.7 after overfitting. "
        f"Baseline: {baseline_recall:.3f}, Trained: {trained_recall:.3f}"
    )
