"""Command-line contract for the offline index builder."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments and report the current implementation phase."""
    args = build_parser().parse_args(argv)
    if args.max_books <= 0:
        raise SystemExit("--max-books must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    raise SystemExit("Index construction is implemented in Phase 3.")


if __name__ == "__main__":
    main()

