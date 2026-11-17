"""Materialize packaged Chroma artifacts for read-only cloud deployments."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

DATABASE_NAME = "chroma.sqlite3"
ARCHIVE_NAME = f"{DATABASE_NAME}.gz"
CHECKSUM_NAME = f"{ARCHIVE_NAME}.sha256"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_chroma_dir(index_dir: str | Path, corpus_sha256: str) -> Path:
    """Return a usable Chroma directory, expanding its packaged database if needed."""
    source = Path(index_dir) / "chroma"
    if (source / DATABASE_NAME).is_file():
        return source

    archive = source / ARCHIVE_NAME
    checksum_path = source / CHECKSUM_NAME
    if not archive.is_file() or not checksum_path.is_file():
        raise ValueError("Chroma persistence database is missing")
    expected = checksum_path.read_text(encoding="utf-8").strip()
    if len(expected) != 64 or _sha256(archive) != expected:
        raise ValueError("packaged Chroma database checksum does not match")

    target = Path(tempfile.gettempdir()) / f"between-lines-{corpus_sha256[:16]}"
    database = target / DATABASE_NAME
    if database.is_file():
        return target

    staging = target.with_name(f"{target.name}-{os.getpid()}.tmp")
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(
        source,
        staging,
        ignore=shutil.ignore_patterns(ARCHIVE_NAME, CHECKSUM_NAME),
    )
    with gzip.open(archive, "rb") as compressed, (staging / DATABASE_NAME).open(
        "wb"
    ) as restored:
        shutil.copyfileobj(compressed, restored, length=1024 * 1024)

    try:
        staging.rename(target)
    except FileExistsError:
        shutil.rmtree(staging)
    if not database.is_file():
        raise ValueError("unable to materialize the packaged Chroma database")
    return target
