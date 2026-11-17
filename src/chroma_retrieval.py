"""Online semantic retrieval backed by a precomputed Chroma collection."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Protocol

import numpy as np
import pandas as pd


class QueryEmbeddingModel(Protocol):
    """Narrow sentence-transformers interface needed for online queries."""

    def encode(self, sentences: str, **kwargs: object) -> np.ndarray:
        """Encode one query string."""
        ...


def search_chroma(
    query: str,
    model: QueryEmbeddingModel,
    collection: Any,
    *,
    k: int = 5,
) -> pd.DataFrame:
    """Query Chroma and return ranked book metadata with cosine similarities."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must contain non-whitespace text")
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")

    corpus_size = collection.count()
    if corpus_size == 0:
        return _empty_results()

    encoded = model.encode(
        query.strip(), convert_to_numpy=True, normalize_embeddings=True
    )
    query_vector = np.asarray(encoded, dtype=np.float32)
    if query_vector.ndim == 2 and query_vector.shape[0] == 1:
        query_vector = query_vector[0]
    if query_vector.ndim != 1 or query_vector.size == 0:
        raise ValueError("model must return exactly one query vector")
    if not np.isfinite(query_vector).all():
        raise ValueError("query embedding must contain only finite values")
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0 or not np.isfinite(query_norm):
        raise ValueError("query embedding must have a finite, nonzero norm")
    query_vector = query_vector / query_norm

    expected_dimension = (collection.metadata or {}).get("embedding_dimension")
    if expected_dimension is not None and query_vector.size != expected_dimension:
        raise ValueError("query and Chroma embedding dimensions do not match")

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(int(k), corpus_size),
        include=["documents", "metadatas", "distances"],
    )
    ids = _first_query_values(result, "ids")
    documents = _first_query_values(result, "documents")
    metadatas = _first_query_values(result, "metadatas")
    distances = _first_query_values(result, "distances")
    if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
        raise ValueError("Chroma returned misaligned result fields")

    rows: list[dict[str, object]] = []
    for rank, (book_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        numeric_distance = float(distance)
        if not np.isfinite(numeric_distance):
            raise ValueError("Chroma returned a nonfinite distance")
        metadata = metadata or {}
        rows.append(
            {
                "rank": rank,
                "book_id": str(book_id),
                "title": str(metadata.get("title") or "Untitled"),
                "description": str(document or ""),
                "similarity_score": 1.0 - numeric_distance,
            }
        )
    return pd.DataFrame(rows, columns=_empty_results().columns)


def _first_query_values(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key)
    if not isinstance(values, list) or not values or not isinstance(values[0], list):
        raise ValueError(f"Chroma response is missing {key}")
    return values[0]


def _empty_results() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["rank", "book_id", "title", "description", "similarity_score"]
    )
