import torch
import pytest
from embeddings.model import CoinEmbeddingModel


EMBEDDING_DIM = 128
BATCH_SIZE = 4
IMAGE_SIZE = 224


@pytest.fixture(scope="module")
def model() -> CoinEmbeddingModel:
    m = CoinEmbeddingModel(embedding_dim=EMBEDDING_DIM, pretrained=False)
    m.eval()
    return m


@pytest.fixture
def random_batch() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_forward_does_not_crash(model: CoinEmbeddingModel, random_batch: torch.Tensor) -> None:
    with torch.no_grad():
        output = model(random_batch)
    assert output is not None


def test_output_shape(model: CoinEmbeddingModel, random_batch: torch.Tensor) -> None:
    with torch.no_grad():
        output = model(random_batch)
    assert output.shape == (BATCH_SIZE, EMBEDDING_DIM)


def test_output_is_l2_normalized(model: CoinEmbeddingModel, random_batch: torch.Tensor) -> None:
    with torch.no_grad():
        output = model(random_batch)
    norms = torch.linalg.norm(output, dim=1)
    assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5)


def test_single_sample_shape(model: CoinEmbeddingModel) -> None:
    single = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        output = model(single)
    assert output.shape == (1, EMBEDDING_DIM)


def test_custom_embedding_dim() -> None:
    custom_dim = 256
    m = CoinEmbeddingModel(embedding_dim=custom_dim, pretrained=False)
    m.eval()
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        output = m(x)
    assert output.shape == (2, custom_dim)
