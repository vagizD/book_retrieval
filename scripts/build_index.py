"""Command-line contract for the offline index builder."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    COLLECTION_NAME,
    DATASET_ID,
    DATASET_REVISION,
    MANIFEST_SCHEMA_VERSION,
    MODEL_ID,
    IndexManifest,
)

DEFAULT_MIN_DESCRIPTION_CHARACTERS = 80


def build_parser() -> argparse.ArgumentParser:
    """Create the index-builder argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare a reproducible book corpus and embedding index."
    )
    parser.add_argument(
        "--max-books",
        type=int,
        default=20_000,
        help="Maximum number of cleaned books to retain (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic corpus selection (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for generated corpus, embeddings, and manifest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    parser.add_argument(
        "--chroma-batch-size",
        type=int,
        default=100,
        help="Records per Chroma write (default: 100).",
    )
    parser.add_argument(
        "--min-description-characters",
        type=int,
        default=DEFAULT_MIN_DESCRIPTION_CHARACTERS,
        help="Minimum cleaned description length (default: 80).",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Development-only cap on source rows scanned; full builds omit it.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="sentence-transformers device (default: cpu).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build validated flat-file artifacts and a persistent Chroma index."""
    args = build_parser().parse_args(argv)
    if args.max_books <= 0:
        raise SystemExit("--max-books must be positive")
    if args.batch_size <= 0 or args.chroma_batch_size <= 0:
        raise SystemExit("batch sizes must be positive")
    if args.min_description_characters <= 0:
        raise SystemExit("--min-description-characters must be positive")
    if args.source_limit is not None and args.source_limit <= 0:
        raise SystemExit("--source-limit must be positive when provided")

    # Heavy dependencies remain inside the command so unit tests can exercise
    # preprocessing and artifact contracts without downloading a model.
    import chromadb
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    from src.embeddings import (
        compute_corpus_sha256,
        encode_books,
        rebuild_chroma_collection,
        save_artifacts,
    )
    from src.preprocessing import prepare_books

    started = time.perf_counter()
    dataset_revision = HfApi().dataset_info(
        DATASET_ID, revision=DATASET_REVISION
    ).sha
    source = load_dataset(
        DATASET_ID,
        split="train",
        revision=dataset_revision,
        streaming=True,
    )
    records = (
        itertools.islice(source, args.source_limit)
        if args.source_limit is not None
        else source
    )
    books = prepare_books(
        records,
        max_books=args.max_books,
        seed=args.seed,
        min_description_characters=args.min_description_characters,
    )
    if books.empty:
        raise SystemExit("No usable books remained after cleaning.")

    model = SentenceTransformer(MODEL_ID, device=args.device)
    embeddings = encode_books(books, model, batch_size=args.batch_size)
    manifest = IndexManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset_id=DATASET_ID,
        dataset_revision=dataset_revision,
        model_id=MODEL_ID,
        collection_name=COLLECTION_NAME,
        row_count=len(books),
        embedding_dimension=embeddings.shape[1],
        corpus_sha256=compute_corpus_sha256(books),
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    save_artifacts(output_dir, books, embeddings, manifest)
    client = chromadb.PersistentClient(path=str(output_dir / "chroma"))
    rebuild_chroma_collection(
        client,
        books,
        embeddings,
        manifest,
        batch_size=args.chroma_batch_size,
    )

    elapsed = time.perf_counter() - started
    artifact_bytes = sum(
        path.stat().st_size for path in output_dir.rglob("*") if path.is_file()
    )
    print(f"Corpus rows: {len(books):,}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Elapsed: {elapsed:.1f} seconds")
    print(f"Output size: {artifact_bytes / (1024 * 1024):.1f} MiB")
    print(f"Output directory: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    main()
