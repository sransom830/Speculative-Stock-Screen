"""
External narrative-attention observability (enrichment overlay).

Normalized factors only — no sentiment polarity, no speculative state mutation,
no Layer 3 fragility. Market-native Layer 2 scoring remains authoritative.
"""

from __future__ import annotations

from .models import (
    ExternalAttentionWindow,
    ExternalNarrativeAttentionEnrichment,
    NewsAttentionFactors,
    NewsAttentionObservation,
    NewsRawCounts,
    NewsThemeMass,
    RedditAttentionFactors,
    RedditAttentionObservation,
    RedditRawCounts,
    aggregate_thematic_attention_factor,
    clamp_unit_interval,
    merge_external_enrichment,
)
from .news_attention import NewsNormalizationPolicy, compute_news_attention_observation
from .reddit_attention import RedditNormalizationPolicy, compute_reddit_attention_observation

__all__ = [
    "ExternalAttentionWindow",
    "ExternalNarrativeAttentionEnrichment",
    "NewsAttentionFactors",
    "NewsAttentionObservation",
    "NewsNormalizationPolicy",
    "NewsRawCounts",
    "NewsThemeMass",
    "RedditAttentionFactors",
    "RedditAttentionObservation",
    "RedditNormalizationPolicy",
    "RedditRawCounts",
    "aggregate_thematic_attention_factor",
    "clamp_unit_interval",
    "compute_news_attention_observation",
    "compute_reddit_attention_observation",
    "merge_external_enrichment",
]
