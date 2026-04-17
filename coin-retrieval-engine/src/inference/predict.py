"""
Inference pipeline for the Coin Retrieval Engine.

Orchestrates: image preprocessing → embedding extraction → similarity search.
"""

from pathlib import Path
from typing import Any, Union

import torch

from embeddings.model import CoinEmbeddingModel
from retrieval.index import CoinIndex
from utils.image_utils import preprocess_image


class CoinPredictor:
    """
    End-to-end inference: given an image path, returns the Top-K
    most similar coins from the retrieval index.

    Args:
        model: A loaded CoinEmbeddingModel instance.
        index: A populated CoinIndex instance.
        device: Torch device string (default 'cpu').
    """

    def __init__(
        self,
        model: CoinEmbeddingModel,
        index: CoinIndex,
        device: str = "cpu",
    ) -> None:
        self.model = model.eval().to(device)
        self.index = index
        self.device = device

    def predict(
        self,
        image_path: Union[str, Path],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Run the full retrieval pipeline on a single image.

        Args:
            image_path: Path to the query coin image.
            top_k: Number of similar coins to return.

        Returns:
            List of dicts with metadata fields and a 'score' key,
            sorted by descending similarity.
        """
        tensor = preprocess_image(image_path).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        query = embedding.squeeze(0).cpu().numpy()
        return self.index.search(query, top_k=top_k)

    @classmethod
    def from_paths(
        cls,
        index_path: Union[str, Path],
        embedding_dim: int = 128,
        device: str = "cpu",
    ) -> "CoinPredictor":
        """
        Convenience constructor: loads model and index from disk.

        Args:
            index_path: Path to the saved CoinIndex pickle file.
            embedding_dim: Embedding dimension used when the index was built.
            device: Torch device string.
        """
        model = CoinEmbeddingModel(embedding_dim=embedding_dim, pretrained=False)
        index = CoinIndex.load(index_path)
        return cls(model=model, index=index, device=device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        index_path: Union[str, Path],
        device: str = "cpu",
    ) -> "CoinPredictor":
        """
        Load a trained model from a checkpoint file and a saved index.

        The checkpoint must have been saved by train_triplet.train(), which
        stores a dict with 'model_state_dict' and 'config' keys.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            index_path: Path to the saved CoinIndex pickle file.
            device: Torch device string.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        embedding_dim = checkpoint.get("config", {}).get("embedding_dim", 128)
        model = CoinEmbeddingModel(embedding_dim=embedding_dim, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        index = CoinIndex.load(index_path)
        return cls(model=model, index=index, device=device)
