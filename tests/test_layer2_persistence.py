"""Layer 2 persistence mapping (no live Supabase)."""

from __future__ import annotations

import pandas as pd

from layer2.persistence import population_rows_from_attention_df, upsert_current_population


def test_population_rows_from_attention_df() -> None:
    df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "speculative_attention_score": 55.5,
                "speculative_attention_state": "HIGH_SPECULATION",
                "source_cohorts": ["retail_meme"],
                "factor_relative_volume": 0.8,
                "factor_realized_vol_expansion": 0.3,
                "factor_acceleration_composite": 0.5,
                "ecosystem_multiplier": 1.1,
            }
        ]
    )
    rows = population_rows_from_attention_df(df)
    assert rows[0]["symbol"] == "AAA"
    assert rows[0]["attention_state"] == "HIGH_SPECULATION"
    assert rows[0]["relative_volume_factor"] == 0.8
    assert rows[0]["volatility_factor"] == 0.3
    assert rows[0]["acceleration_factor"] == 0.5


def test_upsert_current_population_noop_empty(monkeypatch) -> None:
    def boom(**_kwargs):
        raise AssertionError("should not call supabase")

    monkeypatch.setattr(
        "layer2.persistence.get_supabase_client",
        boom,
    )
    assert upsert_current_population([]) is None
