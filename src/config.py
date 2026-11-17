"""Shared project constants and artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DATASET_ID = "booksouls/goodreads-book-descriptions"
DATASET_REVISION = "main"


@dataclass(frozen=True)
class IndexManifest:
    """Metadata required to validate corpus/index alignment.

    Serialization and validation are added with the Phase 3 indexing pipeline.
    """

    dataset_id: str
    dataset_revision: str
    model_id: str
    row_count: int
    embedding_dimension: int
    corpus_sha256: str
    seed: int

