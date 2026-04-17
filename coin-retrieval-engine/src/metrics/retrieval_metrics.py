"""
Retrieval evaluation metrics for metric learning.

All functions operate on precomputed L2-normalized embeddings.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from embeddings.model import CoinEmbeddingModel
from training.dataset import CoinDataset


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: list[int] | np.ndarray,
    k: int,
) -> float:
    """
    Compute Recall@K for a set of embeddings.

    For each query, the top-K nearest neighbors (excluding self) are retrieved.
    A query is considered a hit if at least one of the top-K results shares
    the same class label.

    Args:
        embeddings: Float array of shape (N, D). Assumed L2-normalised.
        labels: Integer class label for each embedding, length N.
        k: Number of nearest neighbors to retrieve.

    Returns:
        Recall@K as a float in [0, 1].
    """
    labels = np.array(labels)
    n = len(embeddings)

    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)

    hits = 0
    for i in range(n):
        top_k_indices = np.argpartition(similarity[i], -k)[-k:]
        if np.any(labels[top_k_indices] == labels[i]):
            hits += 1

    return hits / n


def compute_mean_distance_ratio(
    embeddings: np.ndarray,
    labels: list[int] | np.ndarray,
) -> float:
    """
    Compute the ratio of mean intra-class distance to mean inter-class distance.

    A value < 1.0 indicates that same-class embeddings are closer together
    than different-class embeddings (desired for retrieval systems).

    Args:
        embeddings: Float array of shape (N, D). Assumed L2-normalised.
        labels: Integer class label for each embedding, length N.

    Returns:
        Float ratio: mean_intra_dist / mean_inter_dist.
        Returns 1.0 if either set is empty (degenerate case).
    """
    labels = np.array(labels)
    distances = np.sqrt(np.maximum(2 - 2 * (embeddings @ embeddings.T), 0.0))

    intra: list[float] = []
    inter: list[float] = []

    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra.append(distances[i, j])
            else:
                inter.append(distances[i, j])

    if not intra or not inter:
        return 1.0

    return float(np.mean(intra) / np.mean(inter))


def evaluate(
    model: CoinEmbeddingModel,
    dataset: CoinDataset,
    device: str = "cpu",
    top_k_list: list[int] | None = None,
) -> dict[str, float]:
    """
    Extract embeddings for the full dataset and compute retrieval metrics.

    Args:
        model: Trained CoinEmbeddingModel.
        dataset: CoinDataset to evaluate on.
        device: Torch device string.
        top_k_list: List of K values for Recall@K. Default [1, 5].

    Returns:
        Dict with keys like 'recall@1', 'recall@5', 'distance_ratio'.
    """
    if top_k_list is None:
        top_k_list = [1, 5]

    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_embeddings: list[np.ndarray] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            embeddings = model(images.to(device))
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.tolist())

    embeddings_array = np.vstack(all_embeddings)
    results: dict[str, float] = {}

    for k in top_k_list:
        effective_k = min(k, len(embeddings_array) - 1)
        results[f"recall@{k}"] = compute_recall_at_k(embeddings_array, all_labels, effective_k)

    results["distance_ratio"] = compute_mean_distance_ratio(embeddings_array, all_labels)

    return results
