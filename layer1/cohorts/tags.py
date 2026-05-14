"""Map symbols to zero-or-more cohort labels (injection only, no scoring)."""

from __future__ import annotations

from .seed_sets import COHORT_TO_SYMBOLS


def cohort_labels_for_symbol(symbol: str) -> list[str]:
    """Return stable sorted cohort keys for ``symbol`` (uppercased)."""
    sym = symbol.strip().upper()
    labels = [key for key, members in COHORT_TO_SYMBOLS.items() if sym in members]
    return sorted(labels)


__all__ = ["cohort_labels_for_symbol"]
