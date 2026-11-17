"""Shared project constants and artifact contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
DATASET_ID = "booksouls/goodreads-book-descriptions"
DATASET_REVISION = "main"
COLLECTION_NAME = "goodreads_books"
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexManifest:
    """Metadata required to validate corpus, embeddings, and Chroma alignment."""

    schema_version: int
    dataset_id: str
    dataset_revision: str
    model_id: str
    collection_name: str
    row_count: int
    embedding_dimension: int
    corpus_sha256: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "IndexManifest":
        """Build a manifest while rejecting missing or unknown fields."""
        try:
            manifest = cls(**values)
        except TypeError as error:
            raise ValueError(f"invalid manifest fields: {error}") from error
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Validate values that form the persisted index contract."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema version")
        if self.row_count < 0:
            raise ValueError("manifest row_count must not be negative")
        if self.embedding_dimension <= 0:
            raise ValueError("manifest embedding_dimension must be positive")
        if len(self.corpus_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.corpus_sha256
        ):
            raise ValueError("manifest corpus_sha256 must be a lowercase SHA-256")
        for field_name in (
            "dataset_id",
            "dataset_revision",
            "model_id",
            "collection_name",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"manifest {field_name} must not be blank")
