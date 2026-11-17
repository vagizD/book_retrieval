"""Tests for deterministic, streaming-friendly corpus preparation."""

from __future__ import annotations

from src.preprocessing import prepare_books


def record(index: int, description: str | None = None) -> dict[str, object]:
    return {
        "title": f" Book {index} ",
        "description": description
        or f"A sufficiently informative description for book number {index}.",
    }


def test_cleaning_removes_unusable_rows_and_preserves_embedding_text() -> None:
    records = [
        {"title": "", "description": "A long enough description"},
        {"title": "Missing description", "description": None},
        {"title": "Short", "description": "tiny"},
        {"title": "  Useful   Book ", "description": " A useful\n description. "},
    ]

    result = prepare_books(records, max_books=10, min_description_characters=10)

    assert len(result) == 1
    assert result.loc[0, "title"] == "Useful Book"
    assert result.loc[0, "description"] == "A useful description."
    assert result.loc[0, "embedding_text"] == result.loc[0, "description"]


def test_deduplication_uses_normalized_title_and_description() -> None:
    records = [
        {"title": "The Book", "description": "A meaningful description here."},
        {"title": " the   book ", "description": "a MEANINGFUL description here."},
        {"title": "The Book", "description": "A different description is retained."},
    ]

    result = prepare_books(records, max_books=10, min_description_characters=10)

    assert len(result) == 2
    assert result["book_id"].is_unique


def test_selection_is_deterministic_and_independent_of_source_order() -> None:
    records = [record(index) for index in range(30)]

    first = prepare_books(records, max_books=8, seed=17, min_description_characters=10)
    second = prepare_books(
        reversed(records), max_books=8, seed=17, min_description_characters=10
    )

    assert first.equals(second)
    assert len(first) == 8


def test_seed_changes_selected_subset() -> None:
    records = [record(index) for index in range(30)]

    first = prepare_books(records, max_books=8, seed=1, min_description_characters=10)
    second = prepare_books(records, max_books=8, seed=2, min_description_characters=10)

    assert set(first["book_id"]) != set(second["book_id"])

