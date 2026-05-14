"""
Reddit observational enrichment — counts → bounded factors only.

Provider adapters supply ``RedditRawCounts``; this module applies deterministic normalization.
No sentiment polarity, no prediction, no coupling to speculative_attention_score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import (
    ExternalAttentionWindow,
    RedditAttentionFactors,
    RedditAttentionObservation,
    RedditRawCounts,
    clamp_unit_interval,
)


@dataclass(frozen=True)
class RedditNormalizationPolicy:
    """
    Saturation references for mapping observables into ``[0, 1]``.

    Tune per deployment — defaults are conservative observability ceilings only.
    """

    mention_velocity_ratio_cap: float = 5.0
    comment_velocity_ratio_cap: float = 5.0
    cold_start_mention_reference: int = 50
    cold_start_comment_reference: int = 200
    subreddit_breadth_ceiling: int = 48


def _velocity_ratio_factor(recent: int, prior: int, *, ratio_cap: float, cold_ref: int) -> float:
    if ratio_cap <= 0:
        raise ValueError("ratio_cap must be positive.")
    if cold_ref < 1:
        raise ValueError("cold_ref must be >= 1.")
    if prior == 0:
        ratio = recent / float(cold_ref)
    else:
        ratio = (recent - prior) / float(prior)
    return clamp_unit_interval(ratio / ratio_cap)


def _breadth_factor(distinct_subs: int, *, ceiling: int) -> float:
    if ceiling < 1:
        raise ValueError("subreddit breadth ceiling must be >= 1.")
    return clamp_unit_interval(math.log1p(distinct_subs) / math.log1p(ceiling))


def compute_reddit_attention_observation(
    *,
    window: ExternalAttentionWindow,
    raw: RedditRawCounts,
    policy: RedditNormalizationPolicy | None = None,
    composite_weights: tuple[float, float, float] | None = None,
) -> RedditAttentionObservation:
    """
    Map Reddit observables → normalized enrichment factors.

    Composite weights ``(mention_velocity, comment_velocity, subreddit_breadth)`` sum to ``1.0``.
    """
    pol = policy or RedditNormalizationPolicy()
    w_mv, w_cv, w_br = composite_weights or (0.34, 0.33, 0.33)
    w_sum = w_mv + w_cv + w_br
    if w_sum <= 0:
        raise ValueError("composite weights must sum to a positive value.")
    w_mv, w_cv, w_br = w_mv / w_sum, w_cv / w_sum, w_br / w_sum

    mv = _velocity_ratio_factor(
        raw.mention_count_window,
        raw.mention_count_prior_window,
        ratio_cap=pol.mention_velocity_ratio_cap,
        cold_ref=pol.cold_start_mention_reference,
    )
    cv = _velocity_ratio_factor(
        raw.comment_count_window,
        raw.comment_count_prior_window,
        ratio_cap=pol.comment_velocity_ratio_cap,
        cold_ref=pol.cold_start_comment_reference,
    )
    br = _breadth_factor(raw.distinct_subreddit_count, ceiling=pol.subreddit_breadth_ceiling)

    composite = clamp_unit_interval(w_mv * mv + w_cv * cv + w_br * br)

    factors = RedditAttentionFactors(
        reddit_attention_factor=composite,
        mention_velocity_factor=mv,
        subreddit_breadth_factor=br,
        comment_velocity_factor=cv,
    )
    return RedditAttentionObservation(window=window, raw=raw, factors=factors)
