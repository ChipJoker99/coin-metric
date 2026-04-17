"""
Unit tests for CoinIndex.

All tests use synthetic numpy arrays — no real dataset required.
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest
from retrieval.index import CoinIndex


DIM = 128
N = 10


def _random_l2_embeddings(n: int, dim: int) -> np.ndarray:
    """Generate L2-normalised random embeddings."""
    raw = np.random.default_rng(42).random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def _make_metadata(n: int) -> list[dict]:
    return [{"label": f"coin_{i}", "path": f"data/raw/coin_{i}/img.jpg"} for i in range(n)]


@pytest.fixture
def populated_index() -> CoinIndex:
    index = CoinIndex()
    embeddings = _random_l2_embeddings(N, DIM)
    metadata = _make_metadata(N)
    index.build(embeddings, metadata)
    return index


def test_build_sets_length(populated_index: CoinIndex) -> None:
    assert len(populated_index) == N


def test_search_returns_top_k(populated_index: CoinIndex) -> None:
    query = _random_l2_embeddings(1, DIM)[0]
    results = populated_index.search(query, top_k=3)
    assert len(results) == 3


def test_search_results_have_score(populated_index: CoinIndex) -> None:
    query = _random_l2_embeddings(1, DIM)[0]
    results = populated_index.search(query, top_k=5)
    for r in results:
        assert "score" in r
        assert -1.0 <= r["score"] <= 1.0 + 1e-5


def test_search_results_sorted_descending(populated_index: CoinIndex) -> None:
    query = _random_l2_embeddings(1, DIM)[0]
    results = populated_index.search(query, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_identical_query_scores_1(populated_index: CoinIndex) -> None:
    """A query identical to an indexed embedding must return score ≈ 1."""
    embeddings = _random_l2_embeddings(N, DIM)
    index = CoinIndex()
    index.build(embeddings, _make_metadata(N))

    query = embeddings[0]
    results = index.search(query, top_k=1)
    assert pytest.approx(results[0]["score"], abs=1e-5) == 1.0


def test_search_top_k_capped_at_index_size(populated_index: CoinIndex) -> None:
    query = _random_l2_embeddings(1, DIM)[0]
    results = populated_index.search(query, top_k=100)
    assert len(results) == N


def test_metadata_preserved_in_results(populated_index: CoinIndex) -> None:
    query = _random_l2_embeddings(1, DIM)[0]
    results = populated_index.search(query, top_k=1)
    assert "label" in results[0]
    assert "path" in results[0]


def test_build_mismatched_lengths_raises() -> None:
    index = CoinIndex()
    embeddings = _random_l2_embeddings(5, DIM)
    metadata = _make_metadata(3)
    with pytest.raises(ValueError):
        index.build(embeddings, metadata)


def test_search_on_empty_index_raises() -> None:
    index = CoinIndex()
    query = _random_l2_embeddings(1, DIM)[0]
    with pytest.raises(RuntimeError):
        index.search(query)


def test_save_and_load_roundtrip(populated_index: CoinIndex) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "index.pkl"
        populated_index.save(path)
        loaded = CoinIndex.load(path)

    assert len(loaded) == N
    query = _random_l2_embeddings(1, DIM)[0]
    original_results = populated_index.search(query, top_k=3)
    loaded_results = loaded.search(query, top_k=3)
    for orig, loaded_r in zip(original_results, loaded_results):
        assert pytest.approx(orig["score"], abs=1e-6) == loaded_r["score"]
