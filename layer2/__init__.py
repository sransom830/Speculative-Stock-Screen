"""
Layer 2 — What is attracting speculative crowd attention?

Market-native attention scoring only (no fragility, confirmation, execution).
"""

from .fetch_population import fetch_layer2_speculative_population_df
from .models import (
    NARRATIVE_COHORT_KEYS,
    SPECULATIVE_BATTLEFIELD_COHORT_KEYS,
    SpeculativeAttentionConfig,
    SpeculativeAttentionState,
)
from .persistence import (
    append_population_history,
    population_rows_from_attention_df,
    upsert_current_population,
)
from .speculative_attention_score import compute_speculative_attention

__all__ = [
    "NARRATIVE_COHORT_KEYS",
    "SPECULATIVE_BATTLEFIELD_COHORT_KEYS",
    "SpeculativeAttentionConfig",
    "SpeculativeAttentionState",
    "append_population_history",
    "compute_speculative_attention",
    "fetch_layer2_speculative_population_df",
    "population_rows_from_attention_df",
    "upsert_current_population",
]
