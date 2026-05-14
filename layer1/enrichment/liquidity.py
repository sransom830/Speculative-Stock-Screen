"""Liquidity metrics from daily OHLCV (Layer 1 filtering only)."""

from __future__ import annotations

import pandas as pd


def metrics_from_daily_bars(
    bars: pd.DataFrame,
    *,
    min_avg_daily_dollar_volume: float,
) -> pd.DataFrame:
    """
    From normalized Alpaca daily bars, compute mean dollar volume and last close (price).

    ``bars`` must include ``symbol``, ``timestamp_utc``, ``close``, ``volume``.
    """
    if bars.empty:
        return pd.DataFrame(
            columns=["symbol", "avg_daily_dollar_volume", "price"],
        )
    work = bars.sort_values(["symbol", "timestamp_utc"]).copy()
    work["close_f"] = pd.to_numeric(work["close"], errors="coerce")
    work["volume_f"] = pd.to_numeric(work["volume"], errors="coerce")
    work["dollar_vol"] = work["close_f"] * work["volume_f"]
    agg = (
        work.groupby("symbol", sort=False)
        .agg(
            avg_daily_dollar_volume=("dollar_vol", "mean"),
            price=("close_f", "last"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["avg_daily_dollar_volume", "price"])
    agg = agg.loc[agg["avg_daily_dollar_volume"] >= float(min_avg_daily_dollar_volume)]
    return agg
