"""Diagnostic output from compute_speculative_attention."""

from __future__ import annotations

import pandas as pd

from layer2.speculative_attention_score import compute_speculative_attention


def test_diagnostics_insufficient_bars() -> None:
    layer1 = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "source_cohorts": [[]],
        }
    )
    bars = pd.DataFrame(
        {
            "symbol": ["AAA"] * 5,
            "timestamp_utc": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "close": [100.0] * 5,
            "volume": [1e6] * 5,
        }
    )
    diags: list = []
    out = compute_speculative_attention(layer1, bars, diagnostics_out=diags)
    assert out.empty
    assert len(diags) == 1
    assert diags[0]["symbol"] == "AAA"
    assert diags[0]["status"] == "skipped"
    assert "insufficient_bars" in diags[0]["skip_reason"]
    assert diags[0]["entered_scoring_loop"] is True
    assert diags[0]["bar_row_count"] == 5


def test_diagnostics_no_rows_for_symbol() -> None:
    layer1 = pd.DataFrame({"symbol": ["ZZZ"], "source_cohorts": [[]]})
    bars = pd.DataFrame(
        {
            "symbol": ["OTHER"],
            "timestamp_utc": pd.Timestamp("2024-01-01", tz="UTC"),
            "close": [10.0],
            "volume": [1.0],
        }
    )
    diags: list = []
    compute_speculative_attention(layer1, bars, diagnostics_out=diags)
    assert len(diags) == 2
    by_s = {d["symbol"]: d for d in diags}
    assert by_s["ZZZ"]["skip_reason"] == "no_bar_rows_for_symbol_in_request"
    assert by_s["OTHER"]["skip_reason"] == "symbol_not_in_layer1_snapshot"


def test_diagnostics_symbol_not_in_layer1() -> None:
    layer1 = pd.DataFrame({"symbol": ["AAA"], "source_cohorts": [[]]})
    bars = pd.DataFrame(
        {
            "symbol": ["BBB"],
            "timestamp_utc": pd.Timestamp("2024-01-01", tz="UTC"),
            "close": [10.0],
            "volume": [1.0],
        }
    )
    diags: list = []
    compute_speculative_attention(layer1, bars, diagnostics_out=diags)
    assert len(diags) == 2
    by_s = {d["symbol"]: d for d in diags}
    assert by_s["AAA"]["skip_reason"] == "no_bar_rows_for_symbol_in_request"
    assert by_s["BBB"]["skip_reason"] == "symbol_not_in_layer1_snapshot"
