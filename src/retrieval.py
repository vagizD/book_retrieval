"""NumPy semantic retrieval module.

Cosine scoring and top-k selection are intentionally implemented in Phase 2,
before integration with the real embedding model.
"""

from __future__ import annotations

from numbers import Integral
from typing import Protocol

import numpy as np
import pandas as pd


class EmbeddingModel(Protocol):
    """Narrow interface needed from a sentence-transformers model."""

    def encode(self, sentences: str, **kwargs: object) -> np.ndarray:
        """Encode one query string as a vector."""
        ...


def search_books(
    query: str,
    model: EmbeddingModel,
    embeddings: np.ndarray,
    books: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """Return books ranked by cosine similarity to a free-text query.

    Both the corpus and query vectors are normalized inside this function, even
    though production corpus embeddings are expected to be pre-normalized. An
    empty corpus returns an empty frame without invoking the model. Invalid
    inputs raise ``ValueError`` before ranking.

    Args:
        query: Non-blank natural-language description of the desired book.
        model: Model capable of encoding the query as one numeric vector.
        embeddings: Two-dimensional corpus embedding matrix, one row per book.
        books: Corpus metadata in the same row order as ``embeddings``.
        k: Requested positive number of results.

    Returns:
        A copy of the selected metadata rows in descending similarity order,
        with one-based ``rank`` and cosine ``similarity_score`` columns.

    Raises:
        ValueError: If an input is malformed, nonfinite, zero-norm, or
            inconsistent with another input.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must contain non-whitespace text")
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")

    corpus = np.asarray(embeddings, dtype=np.float32)
    if corpus.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if len(books) != corpus.shape[0]:
        raise ValueError("embeddings and books must contain the same number of rows")
    if corpus.shape[1] == 0:
        raise ValueError("embeddings must have at least one dimension")
    if not np.isfinite(corpus).all():
        raise ValueError("embeddings must contain only finite values")

    if corpus.shape[0] == 0:
        result = books.iloc[0:0].copy().reset_index(drop=True)
        result.insert(0, "rank", pd.Series(dtype="int64"))
        result["similarity_score"] = pd.Series(dtype="float32")
        return result

    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    if np.any(corpus_norms == 0) or not np.isfinite(corpus_norms).all():
        raise ValueError("embeddings must not contain zero-norm rows")
    normalized_corpus = corpus / corpus_norms

    encoded = model.encode(
        query.strip(), convert_to_numpy=True, normalize_embeddings=False
    )
    query_vector = np.asarray(encoded, dtype=np.float32)
    if query_vector.ndim == 2 and query_vector.shape[0] == 1:
        query_vector = query_vector[0]
    if query_vector.ndim != 1:
        raise ValueError("model must return exactly one query vector")
    if query_vector.shape[0] != corpus.shape[1]:
        raise ValueError("query and corpus embedding dimensions do not match")
    if not np.isfinite(query_vector).all():
        raise ValueError("query embedding must contain only finite values")

    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0 or not np.isfinite(query_norm):
        raise ValueError("query embedding must have a finite, nonzero norm")
    normalized_query = query_vector / query_norm

    scores = normalized_corpus @ normalized_query
    if not np.isfinite(scores).all():
        raise ValueError("similarity calculation produced nonfinite scores")

    result_count = min(int(k), corpus.shape[0])
    if result_count == corpus.shape[0]:
        candidate_indices = np.arange(corpus.shape[0])
    else:
        candidate_indices = np.argpartition(scores, -result_count)[-result_count:]

    # The secondary index key makes ties reproducible within the selected set.
    order = np.lexsort((candidate_indices, -scores[candidate_indices]))
    ranked_indices = candidate_indices[order]
    result = books.iloc[ranked_indices].copy().reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, result_count + 1))
    result["similarity_score"] = scores[ranked_indices]
    return result
