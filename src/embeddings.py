"""Offline embedding, artifact validation, and Chroma ingestion."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd

from .config import IndexManifest
from .preprocessing import OUTPUT_COLUMNS

DEFAULT_BATCH_SIZE: Final[int] = 64


class BatchEmbeddingModel(Protocol):
    """Subset of the sentence-transformers interface used by the pipeline."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> np.ndarray:
        """Encode a batch of texts."""
        ...


def compute_corpus_sha256(books: pd.DataFrame) -> str:
    """Compute a stable fingerprint over ordered corpus fields."""
    missing = set(OUTPUT_COLUMNS) - set(books.columns)
    if missing:
        raise ValueError(f"books are missing required columns: {sorted(missing)}")

    digest = hashlib.sha256()
    for row in books.loc[:, list(OUTPUT_COLUMNS)].itertuples(index=False, name=None):
        serialized = json.dumps(
            row, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(serialized)
        digest.update(b"\n")
    return digest.hexdigest()


def encode_books(
    books: pd.DataFrame,
    model: BatchEmbeddingModel,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Encode corpus descriptions as normalized float32 vectors."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if "embedding_text" not in books:
        raise ValueError("books must contain an embedding_text column")
    texts = books["embedding_text"].tolist()
    if not texts:
        raise ValueError("cannot encode an empty corpus")

    encoded = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(encoded, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(books):
        raise ValueError("model returned an embedding matrix with an invalid shape")
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("model returned empty or nonfinite embeddings")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms == 0) or not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("model embeddings must be L2-normalized")
    return embeddings


def save_artifacts(
    output_dir: str | Path,
    books: pd.DataFrame,
    embeddings: np.ndarray,
    manifest: IndexManifest,
) -> None:
    """Persist aligned corpus, NumPy embeddings, and manifest files."""
    output_path = Path(output_dir)
    matrix = np.asarray(embeddings, dtype=np.float32)
    _validate_alignment(books, matrix, manifest)
    output_path.mkdir(parents=True, exist_ok=True)
    books.loc[:, list(OUTPUT_COLUMNS)].to_csv(output_path / "books.csv", index=False)
    np.save(output_path / "embeddings.npy", matrix, allow_pickle=False)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_artifacts(
    output_dir: str | Path,
    *,
    expected_model_id: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, IndexManifest]:
    """Load artifacts and reject stale or misaligned files."""
    output_path = Path(output_dir)
    try:
        manifest_values = json.loads(
            (output_path / "manifest.json").read_text(encoding="utf-8")
        )
        books = pd.read_csv(output_path / "books.csv", keep_default_na=False)
        embeddings = np.load(output_path / "embeddings.npy", allow_pickle=False)
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load index artifacts: {error}") from error

    if not isinstance(manifest_values, dict):
        raise ValueError("manifest must contain a JSON object")
    manifest = IndexManifest.from_dict(manifest_values)
    _validate_alignment(books, embeddings, manifest)
    if expected_model_id is not None and manifest.model_id != expected_model_id:
        raise ValueError("manifest model does not match the requested model")
    return books, embeddings, manifest


def load_manifest(output_dir: str | Path) -> IndexManifest:
    """Load and validate only the lightweight index manifest."""
    manifest_path = Path(output_dir) / "manifest.json"
    try:
        values = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load index manifest: {error}") from error
    if not isinstance(values, dict):
        raise ValueError("manifest must contain a JSON object")
    return IndexManifest.from_dict(values)


def _validate_alignment(
    books: pd.DataFrame, embeddings: np.ndarray, manifest: IndexManifest
) -> None:
    """Validate the one-to-one artifact contract."""
    manifest.validate()
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2:
        raise ValueError("embeddings artifact must be two-dimensional")
    if len(books) != matrix.shape[0] or len(books) != manifest.row_count:
        raise ValueError("corpus, embeddings, and manifest row counts do not match")
    if matrix.shape[1] != manifest.embedding_dimension:
        raise ValueError("embedding dimension does not match the manifest")
    if matrix.dtype != np.float32:
        raise ValueError("embeddings artifact must use float32")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings artifact contains nonfinite values")
    if compute_corpus_sha256(books) != manifest.corpus_sha256:
        raise ValueError("corpus fingerprint does not match the manifest")


def rebuild_chroma_collection(
    client: Any,
    books: pd.DataFrame,
    embeddings: np.ndarray,
    manifest: IndexManifest,
    *,
    batch_size: int = 100,
) -> Any:
    """Recreate and populate a cosine-distance Chroma collection.

    The caller supplies either a persistent or ephemeral Chroma client, which
    keeps this module testable without binding deployment configuration here.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    matrix = np.asarray(embeddings, dtype=np.float32)
    _validate_alignment(books, matrix, manifest)

    existing_names = {
        item if isinstance(item, str) else item.name for item in client.list_collections()
    }
    if manifest.collection_name in existing_names:
        client.delete_collection(manifest.collection_name)
    collection = client.create_collection(
        name=manifest.collection_name,
        metadata={
            "hnsw:space": "cosine",
            "model_id": manifest.model_id,
            "corpus_sha256": manifest.corpus_sha256,
            "row_count": manifest.row_count,
            "embedding_dimension": manifest.embedding_dimension,
        },
        embedding_function=None,
    )

    for start in range(0, len(books), batch_size):
        stop = min(start + batch_size, len(books))
        batch = books.iloc[start:stop]
        collection.add(
            ids=batch["book_id"].tolist(),
            documents=batch["embedding_text"].tolist(),
            metadatas=[{"title": title} for title in batch["title"].tolist()],
            embeddings=matrix[start:stop].tolist(),
        )

    validate_chroma_collection(collection, manifest)
    return collection


def validate_chroma_collection(collection: Any, manifest: IndexManifest) -> None:
    """Check that a Chroma collection matches the persisted manifest."""
    if collection.count() != manifest.row_count:
        raise ValueError("Chroma collection count does not match the manifest")
    metadata = collection.metadata or {}
    for key, expected in (
        ("model_id", manifest.model_id),
        ("corpus_sha256", manifest.corpus_sha256),
        ("row_count", manifest.row_count),
        ("embedding_dimension", manifest.embedding_dimension),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"Chroma collection {key} does not match the manifest")
