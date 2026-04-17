"""
In-memory retrieval index using cosine similarity.

Phase 1 implementation — no FAISS, no external vector DB.
Embeddings are stored as a numpy matrix; similarity is computed via dot
product (valid because all embeddings are L2-normalised).
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class CoinIndex:
    """
    Stores embeddings and associated metadata, and supports Top-K retrieval
    using cosine similarity.

    Usage::

        index = CoinIndex()
        index.build(embeddings, metadata)
        results = index.search(query_embedding, top_k=5)
        index.save("data/embeddings/index.pkl")

        index2 = CoinIndex.load("data/embeddings/index.pkl")
    """

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []

    def build(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """
        Populate the index.

        Args:
            embeddings: Float array of shape (N, D). Assumed L2-normalised.
            metadata: List of N dicts, one per embedding (e.g. label, path).
        """
        if len(embeddings) != len(metadata):
            raise ValueError(
                f"embeddings and metadata length mismatch: "
                f"{len(embeddings)} vs {len(metadata)}"
            )
        self._embeddings = np.array(embeddings, dtype=np.float32)
        self._metadata = list(metadata)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve the Top-K most similar items to the query embedding.

        Cosine similarity is computed as dot product (embeddings are L2-normalised).

        Args:
            query: 1-D float array of shape (D,). Should be L2-normalised.
            top_k: Number of results to return.

        Returns:
            List of dicts, each containing all metadata fields plus a
            'score' key (float in [-1, 1], higher = more similar).
        """
        if self._embeddings is None:
            raise RuntimeError("Index is empty. Call build() first.")

        query_norm = query / (np.linalg.norm(query) + 1e-10)
        scores = self._embeddings @ query_norm

        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            entry = dict(self._metadata[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)

        return results

    def save(self, path: str | Path) -> None:
        """Persist the index to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self._embeddings, "metadata": self._metadata}, f)

    @classmethod
    def load(cls, path: str | Path) -> "CoinIndex":
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._embeddings = data["embeddings"]
        instance._metadata = data["metadata"]
        return instance

    def __len__(self) -> int:
        return len(self._metadata)
