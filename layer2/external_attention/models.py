"""
External narrative-attention observations — enrichment overlay models only.

These structures describe observational counts and normalized factors in [0, 1].
They do not encode bullish/bearish sentiment, predictions, or Layer 2 scoring hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


def clamp_unit_interval(x: float) -> float:
    """Deterministic saturation into ``[0.0, 1.0]``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


@dataclass(frozen=True)
class ExternalAttentionWindow:
    """
    Half-open semantic window ``[interval_start_utc, interval_end_utc)`` for replay alignment.

    ``provider_slug`` identifies the upstream ingestion implementation without binding scoring.
    """

    symbol: str
    interval_start_utc: datetime
    interval_end_utc: datetime
    provider_slug: str

    def __post_init__(self) -> None:
        if self.interval_end_utc <= self.interval_start_utc:
            raise ValueError("interval_end_utc must be strictly after interval_start_utc.")
        sym = str(self.symbol).strip().upper()
        if not sym:
            raise ValueError("symbol must be non-empty.")
        object.__setattr__(self, "symbol", sym)


@dataclass(frozen=True)
class RedditRawCounts:
    """Provider-supplied Reddit observables for one symbol/window (counts only)."""

    mention_count_window: int
    mention_count_prior_window: int
    comment_count_window: int
    comment_count_prior_window: int
    distinct_subreddit_count: int

    def __post_init__(self) -> None:
        for name, val in (
            ("mention_count_window", self.mention_count_window),
            ("mention_count_prior_window", self.mention_count_prior_window),
            ("comment_count_window", self.comment_count_window),
            ("comment_count_prior_window", self.comment_count_prior_window),
            ("distinct_subreddit_count", self.distinct_subreddit_count),
        ):
            if val < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class RedditAttentionFactors:
    """Normalized Reddit enrichment factors (no sentiment polarity)."""

    reddit_attention_factor: float
    mention_velocity_factor: float
    subreddit_breadth_factor: float
    comment_velocity_factor: float


@dataclass(frozen=True)
class RedditAttentionObservation:
    window: ExternalAttentionWindow
    raw: RedditRawCounts
    factors: RedditAttentionFactors


@dataclass(frozen=True)
class NewsThemeMass:
    """Single thematic bucket mass (headline or article-equivalent unit counts)."""

    label: str
    mass: int

    def __post_init__(self) -> None:
        if self.mass < 0:
            raise ValueError("mass must be >= 0")
        lab = str(self.label).strip()
        if not lab:
            raise ValueError("label must be non-empty.")
        object.__setattr__(self, "label", lab)


@dataclass(frozen=True)
class NewsRawCounts:
    """Provider-supplied news observables (density + thematic masses only)."""

    headline_count_window: int
    headline_count_prior_window: int
    theme_masses: tuple[NewsThemeMass, ...]

    def __post_init__(self) -> None:
        if self.headline_count_window < 0 or self.headline_count_prior_window < 0:
            raise ValueError("headline counts must be >= 0")


@dataclass(frozen=True)
class NewsAttentionFactors:
    """Normalized news enrichment factors (no sentiment polarity)."""

    news_attention_factor: float
    headline_density_factor: float
    narrative_concentration_factor: float
    thematic_clustering_factor: float


@dataclass(frozen=True)
class NewsAttentionObservation:
    window: ExternalAttentionWindow
    raw: NewsRawCounts
    factors: NewsAttentionFactors


@dataclass(frozen=True)
class ExternalNarrativeAttentionEnrichment:
    """
    Combined enrichment overlay for one symbol/window.

    Market-native speculative ontology remains authoritative; these factors are observability-only.
    """

    window: ExternalAttentionWindow
    reddit_attention_factor: float
    news_attention_factor: float
    thematic_attention_factor: float
    reddit: RedditAttentionObservation | None
    news: NewsAttentionObservation | None

    def as_flat_dict(self) -> dict[str, Any]:
        """Stable key order for logging, replay manifests, or downstream persistence prototypes."""
        out: dict[str, Any] = {
            "symbol": self.window.symbol,
            "interval_start_utc": self.window.interval_start_utc.isoformat(),
            "interval_end_utc": self.window.interval_end_utc.isoformat(),
            "provider_slug": self.window.provider_slug,
            "reddit_attention_factor": self.reddit_attention_factor,
            "news_attention_factor": self.news_attention_factor,
            "thematic_attention_factor": self.thematic_attention_factor,
        }
        if self.reddit is not None:
            out["reddit_mention_velocity_factor"] = self.reddit.factors.mention_velocity_factor
            out["reddit_subreddit_breadth_factor"] = self.reddit.factors.subreddit_breadth_factor
            out["reddit_comment_velocity_factor"] = self.reddit.factors.comment_velocity_factor
        if self.news is not None:
            out["news_headline_density_factor"] = self.news.factors.headline_density_factor
            out["news_narrative_concentration_factor"] = self.news.factors.narrative_concentration_factor
            out["news_thematic_clustering_factor"] = self.news.factors.thematic_clustering_factor
        return out


def aggregate_thematic_attention_factor(
    *,
    reddit_factors: RedditAttentionFactors | None,
    news_factors: NewsAttentionFactors | None,
    breadth_weight: float = 0.34,
    concentration_weight: float = 0.33,
    clustering_weight: float = 0.33,
) -> float:
    """
    Combine cross-provider thematic observability — breadth (Reddit subs), news concentration/clustering.

    Weights sum to ``1.0`` by default; unused inputs contribute ``0.0`` mass (renormalization keeps determinism).
    """
    components: list[float] = []
    weights: list[float] = []

    if reddit_factors is not None:
        components.append(reddit_factors.subreddit_breadth_factor)
        weights.append(breadth_weight)
    if news_factors is not None:
        components.append(news_factors.narrative_concentration_factor)
        weights.append(concentration_weight)
        components.append(news_factors.thematic_clustering_factor)
        weights.append(clustering_weight)

    if not components:
        return 0.0

    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("aggregate weights must sum to a positive value.")
    acc = sum(c * (w / w_sum) for c, w in zip(components, weights))
    return clamp_unit_interval(acc)


def merge_external_enrichment(
    *,
    window: ExternalAttentionWindow,
    reddit: RedditAttentionObservation | None,
    news: NewsAttentionObservation | None,
) -> ExternalNarrativeAttentionEnrichment:
    """Attach normalized headline Reddit/news composites plus thematic overlay."""
    if reddit is not None:
        if reddit.window.symbol != window.symbol:
            raise ValueError("reddit observation symbol disagrees with merge window.")
        if reddit.window.interval_start_utc != window.interval_start_utc:
            raise ValueError("reddit observation interval_start disagrees with merge window.")
        if reddit.window.interval_end_utc != window.interval_end_utc:
            raise ValueError("reddit observation interval_end disagrees with merge window.")
    if news is not None:
        if news.window.symbol != window.symbol:
            raise ValueError("news observation symbol disagrees with merge window.")
        if news.window.interval_start_utc != window.interval_start_utc:
            raise ValueError("news observation interval_start disagrees with merge window.")
        if news.window.interval_end_utc != window.interval_end_utc:
            raise ValueError("news observation interval_end disagrees with merge window.")

    rf = reddit.factors if reddit is not None else None
    nf = news.factors if news is not None else None

    reddit_f = rf.reddit_attention_factor if rf is not None else 0.0
    news_f = nf.news_attention_factor if nf is not None else 0.0
    thematic_f = aggregate_thematic_attention_factor(reddit_factors=rf, news_factors=nf)

    return ExternalNarrativeAttentionEnrichment(
        window=window,
        reddit_attention_factor=float(clamp_unit_interval(reddit_f)),
        news_attention_factor=float(clamp_unit_interval(news_f)),
        thematic_attention_factor=float(clamp_unit_interval(thematic_f)),
        reddit=reddit,
        news=news,
    )
