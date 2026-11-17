"""Tests for packaged deployment artifact restoration."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from src.deployment import materialize_chroma_dir


def test_materialize_packaged_database(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    payload = b"sqlite payload for a deployment test"
    archive = chroma / "chroma.sqlite3.gz"
    with gzip.open(archive, "wb") as handle:
        handle.write(payload)
    (chroma / "chroma.sqlite3.gz.sha256").write_text(
        hashlib.sha256(archive.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )

    fingerprint = hashlib.sha256(str(tmp_path).encode()).hexdigest()
    restored = materialize_chroma_dir(tmp_path, fingerprint)

    assert (restored / "chroma.sqlite3").read_bytes() == payload


def test_materialize_rejects_bad_checksum(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    archive = chroma / "chroma.sqlite3.gz"
    archive.write_bytes(b"not a valid archive")
    (chroma / "chroma.sqlite3.gz.sha256").write_text(
        "0" * 64 + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checksum"):
        materialize_chroma_dir(
            tmp_path, hashlib.sha256(str(tmp_path).encode()).hexdigest()
        )
