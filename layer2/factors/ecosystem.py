"""Layer 1 cohort → interpretable multipliers (behavioral ecosystems, not sentiment)."""

from __future__ import annotations

from typing import Iterable, Mapping

# Per-cohort boosts additive to 1.0 before capping (deterministic).
# Battlefield ecosystems carry materially more weight than peripheral themes.
_COHORT_INCREMENTS: Mapping[str, float] = {
    "retail_meme": 0.12,
    "short_squeeze_ecosystem": 0.12,
    "volatility_ecosystems": 0.10,
    "leveraged_crypto_beta": 0.11,
    "space_aerospace_speculative": 0.10,
    "crypto_linked": 0.05,
    "recent_ipo_spac": 0.05,
    "ai_growth": 0.04,
}


def ecosystem_multiplier(
    cohorts: Iterable[str],
    *,
    cap: float = 1.48,
) -> float:
    """
    ``1 + sum(increments)`` for known cohort keys, hard-capped (default 1.48).

    Unknown cohorts contribute 0. Multipliers are *not* bearish/bullish; they
    up-weight attention where behavioral ecosystems concentrate by construction.
    """
    bump = 0.0
    seen: set[str] = set()
    for c in cohorts:
        key = str(c).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        bump += float(_COHORT_INCREMENTS.get(key, 0.0))
    return float(min(cap, 1.0 + bump))
