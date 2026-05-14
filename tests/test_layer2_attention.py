"""Tests for market-native speculative attention scoring."""

from __future__ import annotations

import pandas as pd

from layer2 import SpeculativeAttentionConfig, compute_speculative_attention
from layer2.models import SpeculativeAttentionState


def _rising_bars(symbol: str, n: int = 25, vol_mult_last: float = 4.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(range(100, 100 + n), dtype=float)
    vol = pd.Series([1_000_000.0] * n)
    vol.iloc[-1] *= vol_mult_last
    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp_utc": idx,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": vol,
        }
    )


def test_attention_rising_volume_symbol_scores_above_zero() -> None:
    layer1 = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "company_name": ["Test"],
            "exchange": ["NASDAQ"],
            "asset_class": ["equity"],
            "source_cohorts": [[]],
        }
    )
    bars = _rising_bars("AAA", vol_mult_last=5.0)
    out = compute_speculative_attention(layer1, bars)
    assert len(out) == 1
    assert out.iloc[0]["speculative_attention_score"] > 0
    assert isinstance(out.iloc[0]["speculative_attention_state"], str)


def test_retail_meme_triggers_frenzy_with_high_participation() -> None:
    layer1 = pd.DataFrame(
        {
            "symbol": ["GME"],
            "company_name": ["GameStop"],
            "exchange": ["NYSE"],
            "asset_class": ["equity"],
            "source_cohorts": [["retail_meme"]],
        }
    )
    bars = _rising_bars("GME", n=30, vol_mult_last=10.0)
    cfg = SpeculativeAttentionConfig(score_retail_frenzy_min=40.0)
    out = compute_speculative_attention(layer1, bars, config=cfg)
    assert len(out) == 1
    assert (
        out.iloc[0]["speculative_attention_state"]
        == SpeculativeAttentionState.RETAIL_FRENZY.value
    )


def test_empty_layer1_returns_empty() -> None:
    out = compute_speculative_attention(pd.DataFrame(), pd.DataFrame())
    assert out.empty
