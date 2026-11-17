"""Dataset cleaning and deterministic sampling contracts.

Implementation belongs to Phase 3 after retrieval behavior is tested.
"""

from __future__ import annotations

from typing import Final

SOURCE_COLUMNS: Final[tuple[str, str]] = ("title", "description")
MIN_DESCRIPTION_CHARACTERS: Final[int] = 80

