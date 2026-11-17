"""Artifact and Chroma smoke tests without model or dataset downloads."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import pytest

from src.config import MANIFEST_SCHEMA_VERSION, IndexManifest
from src.chroma_retrieval import search_chroma
from src.embeddings import (
    compute_corpus_sha256,
    encode_books,
    load_artifacts,
    rebuild_chroma_collection,
    save_artifacts,
)
from src.preprocessing import prepare_books


class FakeBatchModel:
    """Small deterministic normalized embedder for pipeline tests."""

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        raw = np.array(
            [[len(text), sum(character.isalpha() for character in text)] for text in sentences],
            dtype=np.float32,
        )
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)


class FakeQueryModel:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls = 0

    def encode(self, sentences: str, **kwargs: object) -> np.ndarray:
        self.calls += 1
        return np.asarray(self.vector, dtype=np.float32)


def sample_books() -> pd.DataFrame:
    return prepare_books(
        [
            {"title": "Ocean", "description": "A voyage across a dangerous ocean."},
            {"title": "Forest", "description": "A quiet mystery in an ancient forest."},
            {"title": "Space", "description": "Explorers journey beyond the known stars."},
        ],
        max_books=3,
        min_description_characters=10,
    )


def manifest_for(books: pd.DataFrame, embeddings: np.ndarray) -> IndexManifest:
    return IndexManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset_id="synthetic/books",
        dataset_revision="fixture-v1",
        model_id="fake-model",
        collection_name="test_books",
        row_count=len(books),
        embedding_dimension=embeddings.shape[1],
        corpus_sha256=compute_corpus_sha256(books),
        seed=42,
    )


def test_encode_books_produces_aligned_float32_matrix() -> None:
    books = sample_books()

    embeddings = encode_books(books, FakeBatchModel(), batch_size=2)

    assert embeddings.shape == (3, 2)
    assert embeddings.dtype == np.float32
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)


def test_artifact_round_trip_validates_alignment(tmp_path: Path) -> None:
    books = sample_books()
    embeddings = encode_books(books, FakeBatchModel())
    manifest = manifest_for(books, embeddings)

    save_artifacts(tmp_path, books, embeddings, manifest)
    loaded_books, loaded_embeddings, loaded_manifest = load_artifacts(
        tmp_path, expected_model_id="fake-model"
    )

    assert loaded_books.equals(books)
    assert np.array_equal(loaded_embeddings, embeddings)
    assert loaded_manifest == manifest


def test_artifact_loader_rejects_stale_manifest(tmp_path: Path) -> None:
    books = sample_books()
    embeddings = encode_books(books, FakeBatchModel())
    manifest = manifest_for(books, embeddings)
    save_artifacts(tmp_path, books, embeddings, manifest)
    manifest_path = tmp_path / "manifest.json"
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    values["row_count"] += 1
    manifest_path.write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError, match="row counts"):
        load_artifacts(tmp_path)


def test_artifact_loader_rejects_wrong_dimension_and_model(tmp_path: Path) -> None:
    books = sample_books()
    embeddings = encode_books(books, FakeBatchModel())
    manifest = manifest_for(books, embeddings)
    save_artifacts(tmp_path, books, embeddings, manifest)

    with pytest.raises(ValueError, match="requested model"):
        load_artifacts(tmp_path, expected_model_id="different-model")

    manifest_path = tmp_path / "manifest.json"
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    values["embedding_dimension"] += 1
    manifest_path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="dimension"):
        load_artifacts(tmp_path)


def test_persistent_chroma_ingestion_and_query_smoke(tmp_path: Path) -> None:
    books = sample_books()
    embeddings = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    manifest = manifest_for(books, embeddings)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))

    collection = rebuild_chroma_collection(
        client, books, embeddings, manifest, batch_size=2
    )
    result = collection.query(query_embeddings=[[1.0, 0.0]], n_results=2)
    recommendations = search_chroma(
        "an eastern journey", FakeQueryModel([1, 0]), collection, k=2
    )

    assert collection.count() == 3
    assert result["ids"][0][0] == books.loc[0, "book_id"]
    assert result["distances"][0] == sorted(result["distances"][0])
    assert recommendations["book_id"].tolist()[0] == books.loc[0, "book_id"]
    assert recommendations["similarity_score"].tolist() == sorted(
        recommendations["similarity_score"].tolist(), reverse=True
    )


def test_chroma_search_rejects_blank_query_before_inference(tmp_path: Path) -> None:
    books = sample_books()
    embeddings = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    manifest = manifest_for(books, embeddings)
    collection = rebuild_chroma_collection(
        chromadb.PersistentClient(path=str(tmp_path / "chroma")),
        books,
        embeddings,
        manifest,
    )
    model = FakeQueryModel([1, 0])

    with pytest.raises(ValueError, match="query"):
        search_chroma("   ", model, collection)

    assert model.calls == 0


def test_chroma_search_rejects_dimension_mismatch(tmp_path: Path) -> None:
    books = sample_books()
    embeddings = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    manifest = manifest_for(books, embeddings)
    collection = rebuild_chroma_collection(
        chromadb.PersistentClient(path=str(tmp_path / "chroma")),
        books,
        embeddings,
        manifest,
    )

    with pytest.raises(ValueError, match="dimensions do not match"):
        search_chroma("query", FakeQueryModel([1, 0, 0]), collection)
