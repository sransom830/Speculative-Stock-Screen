"""
Behavioral cohort injection for Layer 1 (tags / provenance only).
"""

from .seed_sets import ALL_COHORT_KEYS, COHORT_TO_SYMBOLS
from .tags import cohort_labels_for_symbol

__all__ = [
    "ALL_COHORT_KEYS",
    "COHORT_TO_SYMBOLS",
    "cohort_labels_for_symbol",
]
