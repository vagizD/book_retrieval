"""Checks for the Phase 1 project and command contracts."""

from scripts.build_index import build_parser
from src.config import EMBEDDING_DIMENSION, MODEL_ID, IndexManifest


def test_default_build_arguments() -> None:
    args = build_parser().parse_args([])

    assert args.max_books == 20_000
    assert args.seed == 42
    assert args.output_dir == "data"
    assert args.batch_size == 64


def test_index_manifest_records_alignment_metadata() -> None:
    manifest = IndexManifest(
        dataset_id="dataset",
        dataset_revision="revision",
        model_id=MODEL_ID,
        row_count=3,
        embedding_dimension=EMBEDDING_DIMENSION,
        corpus_sha256="a" * 64,
        seed=42,
    )

    assert manifest.row_count == 3
    assert manifest.embedding_dimension == 384

