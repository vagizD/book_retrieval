"""Compress the large Chroma SQLite file for GitHub-based deployment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
from pathlib import Path

GITHUB_FILE_LIMIT = 100 * 1024 * 1024


def main() -> int:
    """Create a deterministic gzip file and checksum beside the Chroma database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data")
    args = parser.parse_args()

    chroma_dir = Path(args.index_dir) / "chroma"
    source = chroma_dir / "chroma.sqlite3"
    archive = chroma_dir / "chroma.sqlite3.gz"
    temporary = chroma_dir / "chroma.sqlite3.gz.tmp"
    if not source.is_file():
        raise SystemExit(f"Missing Chroma database: {source}")

    with source.open("rb") as raw, temporary.open("wb") as destination:
        with gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as compressed:
            shutil.copyfileobj(raw, compressed, length=1024 * 1024)
    os.replace(temporary, archive)
    if archive.stat().st_size >= GITHUB_FILE_LIMIT:
        raise SystemExit("Compressed database still exceeds GitHub's 100 MiB limit")

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".gz.sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"Packaged {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
