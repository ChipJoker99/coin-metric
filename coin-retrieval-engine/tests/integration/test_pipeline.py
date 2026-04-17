"""
Integration tests for the full Phase 1 retrieval pipeline.

Tests the end-to-end flow:
    synthetic image → preprocessing → embedding → index → top-k results

No real dataset is required. PIL images are generated synthetically.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from embeddings.model import CoinEmbeddingModel
from inference.predict import CoinPredictor
from retrieval.index import CoinIndex
from utils.image_utils import preprocess_image


EMBEDDING_DIM = 128
NUM_INDEX_ENTRIES = 20


def _synthetic_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a random RGB PIL image."""
    array = np.random.default_rng().integers(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def _save_synthetic_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _synthetic_image().save(path)


@pytest.fixture(scope="module")
def model() -> CoinEmbeddingModel:
    m = CoinEmbeddingModel(embedding_dim=EMBEDDING_DIM, pretrained=False)
    m.eval()
    return m


@pytest.fixture(scope="module")
def populated_index(model: CoinEmbeddingModel) -> CoinIndex:
    rng = np.random.default_rng(0)
    raw = rng.random((NUM_INDEX_ENTRIES, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms

    metadata = [{"label": f"coin_{i}", "path": f"data/raw/coin_{i}/img.jpg"} for i in range(NUM_INDEX_ENTRIES)]

    index = CoinIndex()
    index.build(embeddings, metadata)
    return index


@pytest.fixture(scope="module")
def predictor(model: CoinEmbeddingModel, populated_index: CoinIndex) -> CoinPredictor:
    return CoinPredictor(model=model, index=populated_index, device="cpu")


@pytest.fixture
def temp_image_path(tmp_path: Path) -> Path:
    path = tmp_path / "query.jpg"
    _save_synthetic_image(path)
    return path


def test_preprocess_image_returns_tensor(temp_image_path: Path) -> None:
    tensor = preprocess_image(temp_image_path)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_embedding_extracted_from_image(model: CoinEmbeddingModel, temp_image_path: Path) -> None:
    tensor = preprocess_image(temp_image_path).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
    assert embedding.shape == (1, EMBEDDING_DIM)
    norm = torch.linalg.norm(embedding, dim=1)
    assert pytest.approx(norm.item(), abs=1e-5) == 1.0


def test_predict_returns_top_k_results(predictor: CoinPredictor, temp_image_path: Path) -> None:
    results = predictor.predict(temp_image_path, top_k=5)
    assert len(results) == 5


def test_predict_results_have_required_fields(predictor: CoinPredictor, temp_image_path: Path) -> None:
    results = predictor.predict(temp_image_path, top_k=3)
    for r in results:
        assert "label" in r
        assert "score" in r


def test_predict_scores_sorted_descending(predictor: CoinPredictor, temp_image_path: Path) -> None:
    results = predictor.predict(temp_image_path, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_predict_score_in_valid_range(predictor: CoinPredictor, temp_image_path: Path) -> None:
    results = predictor.predict(temp_image_path, top_k=5)
    for r in results:
        assert -1.0 <= r["score"] <= 1.0 + 1e-5


def test_index_save_load_preserves_pipeline(
    model: CoinEmbeddingModel,
    populated_index: CoinIndex,
    temp_image_path: Path,
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.pkl"
    populated_index.save(index_path)

    loaded_index = CoinIndex.load(index_path)
    predictor = CoinPredictor(model=model, index=loaded_index, device="cpu")

    results = predictor.predict(temp_image_path, top_k=3)
    assert len(results) == 3
    assert all("score" in r for r in results)
