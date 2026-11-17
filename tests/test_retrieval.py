"""Deterministic tests for exhaustive NumPy semantic retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.retrieval import search_books


class FakeModel:
    """Minimal sentence-transformers stand-in with no network dependency."""

    def __init__(self, vector: object) -> None:
        self.vector = vector
        self.calls = 0

    def encode(self, sentences: str, **kwargs: object) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.vector)


def books(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "book_id": [f"book-{index}" for index in range(count)],
            "title": [f"Title {index}" for index in range(count)],
            "description": [f"Description {index}" for index in range(count)],
        }
    )


def test_exact_match_ranks_first() -> None:
    embeddings = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)

    result = search_books("east", FakeModel([1, 0]), embeddings, books(3), k=3)

    assert result.loc[0, "book_id"] == "book-0"
    assert result.loc[0, "similarity_score"] == pytest.approx(1.0)


def test_results_are_in_descending_score_order() -> None:
    embeddings = np.array([[1, 0], [1, 1], [0, 1], [-1, 0]], dtype=np.float32)

    result = search_books("east", FakeModel([1, 0]), embeddings, books(4), k=4)

    scores = result["similarity_score"].to_numpy()
    assert np.all(scores[:-1] >= scores[1:])
    assert result["rank"].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(("k", "expected"), [(5, 5), (10, 10), (20, 12)])
def test_k_returns_requested_count_up_to_corpus_size(k: int, expected: int) -> None:
    angles = np.linspace(0, np.pi, 12, dtype=np.float32)
    embeddings = np.column_stack((np.cos(angles), np.sin(angles)))

    result = search_books("east", FakeModel([1, 0]), embeddings, books(12), k=k)

    assert len(result) == expected


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k_is_rejected(k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        search_books("query", FakeModel([1, 0]), np.eye(2), books(2), k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", "  \n\t  "])
def test_blank_query_is_rejected_before_model_inference(query: str) -> None:
    model = FakeModel([1, 0])

    with pytest.raises(ValueError, match="query"):
        search_books(query, model, np.eye(2), books(2))

    assert model.calls == 0


def test_embedding_and_metadata_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        search_books("query", FakeModel([1, 0]), np.eye(2), books(1))


def test_query_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="dimensions do not match"):
        search_books("query", FakeModel([1, 0, 0]), np.eye(2), books(2))


@pytest.mark.parametrize(
    "embeddings", [np.array([[1.0, np.nan]]), np.array([[1.0, np.inf]])]
)
def test_nonfinite_corpus_embeddings_are_rejected(embeddings: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite"):
        search_books("query", FakeModel([1, 0]), embeddings, books(1))


@pytest.mark.parametrize("vector", [[np.nan, 0], [np.inf, 0]])
def test_nonfinite_query_embedding_is_rejected(vector: list[float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        search_books("query", FakeModel(vector), np.eye(2), books(2))


def test_zero_norm_vectors_are_rejected() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        search_books("query", FakeModel([1, 0]), np.array([[0, 0]]), books(1))

    with pytest.raises(ValueError, match="nonzero norm"):
        search_books("query", FakeModel([0, 0]), np.eye(2), books(2))


def test_top_k_never_returns_duplicate_indices() -> None:
    embeddings = np.array(
        [[1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]], dtype=np.float32
    )

    result = search_books("east", FakeModel([1, 0]), embeddings, books(4), k=3)

    assert len(result) == 3
    assert result["book_id"].is_unique


def test_empty_corpus_returns_empty_result_without_inference() -> None:
    model = FakeModel([1, 0])

    result = search_books("query", model, np.empty((0, 2)), books(0), k=5)

    assert result.empty
    assert list(result.columns) == [
        "rank",
        "book_id",
        "title",
        "description",
        "similarity_score",
    ]
    assert model.calls == 0
