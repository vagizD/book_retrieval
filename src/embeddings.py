"""Offline embedding generation and artifact-loading contracts.

Implementation belongs to Phase 3. The online app must only load artifacts
created by that pipeline; it must never rebuild the corpus index at startup.
"""

from __future__ import annotations

from typing import Final

DEFAULT_BATCH_SIZE: Final[int] = 64

