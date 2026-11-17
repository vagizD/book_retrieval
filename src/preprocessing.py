"""Cleaning, deduplication, and deterministic corpus selection."""

from __future__ import annotations

import hashlib
import heapq
import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

import pandas as pd

SOURCE_COLUMNS: Final[tuple[str, str]] = ("title", "description")
MIN_DESCRIPTION_CHARACTERS: Final[int] = 80
OUTPUT_COLUMNS: Final[tuple[str, str, str, str]] = (
    "book_id",
    "title",
    "description",
    "embedding_text",
)

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Collapse whitespace and case-fold text for identity comparisons."""
    return _WHITESPACE.sub(" ", value).strip().casefold()


def clean_text(value: object) -> str:
    """Return display text with surrounding and repeated whitespace removed."""
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", value).strip()


def _record_identity(title: str, description: str) -> tuple[str, str]:
    normalized_key = f"{normalize_text(title)}\0{normalize_text(description)}"
    digest = hashlib.sha256(normalized_key.encode("utf-8")).hexdigest()
    return f"book-{digest}", normalized_key


def prepare_books(
    records: Iterable[Mapping[str, Any]],
    *,
    max_books: int = 20_000,
    seed: int = 42,
    min_description_characters: int = MIN_DESCRIPTION_CHARACTERS,
) -> pd.DataFrame:
    """Clean, deduplicate, and deterministically select book records.

    Selection uses a seeded SHA-256 priority rather than source order or a
    version-specific pseudo-random sampler. Only the best ``max_books`` records
    are retained in memory, making this suitable for a streaming source.

    Args:
        records: Mappings containing source ``title`` and ``description``.
        max_books: Maximum number of usable unique records to return.
        seed: Salt for deterministic selection priorities.
        min_description_characters: Minimum cleaned description length.

    Returns:
        A dataframe in stable order with an internal ID and exact embedding text.
    """
    if max_books <= 0:
        raise ValueError("max_books must be positive")
    if min_description_characters <= 0:
        raise ValueError("min_description_characters must be positive")

    # A max-heap represented with negative priorities retains the smallest
    # deterministic priority values without materializing the full dataset.
    selected: list[tuple[int, str, str, str]] = []
    selected_ids: set[str] = set()
    seed_prefix = f"{seed}\0".encode("utf-8")

    for record in records:
        title = clean_text(record.get("title"))
        description = clean_text(record.get("description"))
        if not title or len(description) < min_description_characters:
            continue

        book_id, normalized_key = _record_identity(title, description)
        if book_id in selected_ids:
            continue
        priority = int.from_bytes(
            hashlib.sha256(seed_prefix + normalized_key.encode("utf-8")).digest(),
            "big",
        )
        entry = (-priority, book_id, title, description)
        if len(selected) < max_books:
            heapq.heappush(selected, entry)
            selected_ids.add(book_id)
            continue

        largest_selected_priority = -selected[0][0]
        if priority < largest_selected_priority:
            removed = heapq.heapreplace(selected, entry)
            selected_ids.remove(removed[1])
            selected_ids.add(book_id)

    ordered = sorted(selected, key=lambda entry: (-entry[0], entry[1]))
    rows = [
        {
            "book_id": book_id,
            "title": title,
            "description": description,
            "embedding_text": description,
        }
        for _, book_id, title, description in ordered
    ]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
