"""Layer 2 data models — speculative attention only (no fragility or execution)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet


class SpeculativeAttentionState(str, Enum):
    """
    Interpretable attention regime (bullish or bearish agnostic).
    Assignment is rule-ordered and deterministic given inputs.
    """

    NEUTRAL = "NEUTRAL"
    HIGH_SPECULATION = "HIGH_SPECULATION"
    RETAIL_FRENZY = "RETAIL_FRENZY"
    NARRATIVE_ACCELERATION = "NARRATIVE_ACCELERATION"
    VOLATILITY_CLUSTER = "VOLATILITY_CLUSTER"


# Cohorts that imply “story / theme” acceleration for state rules (not a sentiment score).
NARRATIVE_COHORT_KEYS: FrozenSet[str] = frozenset(
    {
        "ai_growth",
        "recent_ipo_spac",
        "short_squeeze_ecosystem",
        "space_aerospace_speculative",
        "crypto_linked",
        "leveraged_crypto_beta",
    }
)


# Behavioral “speculative battlefield” tags — identity matters for scoring + states.
SPECULATIVE_BATTLEFIELD_COHORT_KEYS: FrozenSet[str] = frozenset(
    {
        "retail_meme",
        "leveraged_crypto_beta",
        "short_squeeze_ecosystem",
        "space_aerospace_speculative",
        "volatility_ecosystems",
    }
)


@dataclass(frozen=True)
class SpeculativeAttentionConfig:
    """Tunable windows and thresholds (all deterministic)."""

    # Windows (trading days implied by row count; calendar daily bars).
    relative_volume_window: int = 20
    dollar_volume_short: int = 5
    dollar_volume_long: int = 20
    rv_short: int = 5
    rv_long: int = 20
    accel_short: int = 5
    accel_long: int = 20
    momentum_days: int = 10

    # Composite weights (sum to 1.0). Participation + vol dominate vs raw price path.
    weight_relative_volume: float = 0.20
    weight_dollar_volume_expansion: float = 0.26
    weight_realized_vol_expansion: float = 0.26
    weight_price_acceleration: float = 0.14
    weight_momentum_persistence: float = 0.14

    # Ecosystem multiplier cap (each cohort adds a bump; see ecosystem module).
    ecosystem_cap: float = 1.48

    # State thresholds (0–1 factor space where noted; score is 0–100).
    # Generic names need stronger scores; battlefield cohorts can label HIGH_SPECULATION earlier.
    score_high_speculation_generic: float = 68.0
    score_high_speculation_ecosystem: float = 54.0
    ecosystem_high_spec_liquidity_floor: float = 0.40
    score_retail_frenzy_participation: float = 0.52
    score_retail_frenzy_min: float = 55.0
    factor_vol_cluster: float = 0.68
    factor_vol_cluster_strong: float = 0.82
    factor_narrative_accel: float = 0.62
    cohort_volatility_tag: str = "volatility_ecosystems"
