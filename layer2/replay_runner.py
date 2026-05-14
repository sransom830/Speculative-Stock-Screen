"""
Shared Layer 2 historical replay orchestration (bars through replay-date only).

Used by CLI replay + behavioral validation scripts. Speculative attention only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, MutableMapping

import pandas as pd
from alpaca.data.enums import DataFeed

from data.providers import AlpacaMarketDataProvider
from data.providers.base_provider import AssetClass, AssetRef
from layer1 import fetch_layer1_broad_universe_df

from .speculative_attention_score import compute_speculative_attention


@dataclass(frozen=True)
class Layer2ReplayArtifacts:
    """Deterministic replay outputs for one ``replay_day`` (UTC calendar)."""

    replay_day: date
    lookback_calendar_days: int
    feed_label: str
    fetch_start_utc: datetime
    fetch_end_exclusive_utc: datetime
    layer1_df: pd.DataFrame
    layer1_load_stats: dict[str, Any]
    bars_rows_fetched: int
    bars_unique_symbols_fetched: int
    bars_df: pd.DataFrame
    bars_rows_after_clip: int
    bars_unique_symbols_after_clip: int
    diagnostics: list[dict[str, Any]]
    attention_df: pd.DataFrame


def refs_from_layer1(layer1_df: pd.DataFrame) -> list[AssetRef]:
    refs: list[AssetRef] = []
    for _, row in layer1_df.iterrows():
        sym = str(row["symbol"])
        ex = row.get("exchange")
        ex_s = None if ex is None or (isinstance(ex, float) and pd.isna(ex)) else str(ex)
        refs.append(AssetRef(sym, AssetClass.EQUITY, exchange=ex_s))
    return refs


def fetch_window_utc(replay_day: date, lookback_calendar_days: int) -> tuple[datetime, datetime]:
    """
    Alpaca daily bars ``end`` is exclusive (REST convention).

    ``replay_day`` sessions are included in the clipped frame.
    """
    if lookback_calendar_days < 1:
        raise ValueError("lookback window must be >= 1 calendar day")
    end_exclusive = datetime(replay_day.year, replay_day.month, replay_day.day, tzinfo=timezone.utc) + timedelta(days=1)
    start_inclusive = end_exclusive - timedelta(days=lookback_calendar_days)
    return start_inclusive, end_exclusive


def clip_bars_no_future_leak(bars_df: pd.DataFrame, replay_day: date) -> pd.DataFrame:
    """Keep rows whose UTC calendar date is <= ``replay_day``."""
    if bars_df.empty:
        return bars_df
    ts = pd.to_datetime(bars_df["timestamp_utc"], utc=True)
    mask = ts.dt.date <= replay_day
    return bars_df.loc[mask].copy()


def run_layer2_replay(
    *,
    replay_day: date,
    lookback_calendar_days: int = 90,
    feed: DataFeed | None = None,
    layer1_df: pd.DataFrame | None = None,
    layer1_load_stats_out: MutableMapping[str, Any] | None = None,
    provider: AlpacaMarketDataProvider | None = None,
) -> Layer2ReplayArtifacts:
    """
    Full replay: Layer 1 universe (or supplied frame), Alpaca daily bars clipped to
    ``replay_day``, then ``compute_speculative_attention``.
    """
    feed_eff = feed if feed is not None else DataFeed.IEX
    feed_label = getattr(feed_eff, "value", str(feed_eff))

    stats_sink: dict[str, Any] = {}
    if layer1_df is None:
        layer1_df = fetch_layer1_broad_universe_df(load_stats_out=stats_sink)

    if layer1_load_stats_out is not None:
        layer1_load_stats_out.clear()
        layer1_load_stats_out.update(stats_sink)
        persisted_layer1_stats = dict(layer1_load_stats_out)
    else:
        persisted_layer1_stats = dict(stats_sink)

    if layer1_df.empty:
        return Layer2ReplayArtifacts(
            replay_day=replay_day,
            lookback_calendar_days=lookback_calendar_days,
            feed_label=feed_label,
            fetch_start_utc=datetime.min.replace(tzinfo=timezone.utc),
            fetch_end_exclusive_utc=datetime.min.replace(tzinfo=timezone.utc),
            layer1_df=layer1_df,
            layer1_load_stats=persisted_layer1_stats,
            bars_rows_fetched=0,
            bars_unique_symbols_fetched=0,
            bars_df=pd.DataFrame(),
            bars_rows_after_clip=0,
            bars_unique_symbols_after_clip=0,
            diagnostics=[],
            attention_df=pd.DataFrame(),
        )

    start_utc, end_exclusive_utc = fetch_window_utc(replay_day, lookback_calendar_days)
    prov = provider or AlpacaMarketDataProvider(stock_feed=feed_eff)
    refs = refs_from_layer1(layer1_df)
    bars_raw = prov.fetch_daily_bars_df(refs, start_utc, end_exclusive_utc, feed=feed_eff)
    bars_rows_fetched = int(len(bars_raw))
    bars_unique_symbols_fetched = (
        int(bars_raw["symbol"].astype(str).str.upper().nunique())
        if bars_rows_fetched and "symbol" in bars_raw.columns
        else 0
    )

    bars_df = clip_bars_no_future_leak(bars_raw, replay_day)
    bars_rows_after_clip = int(len(bars_df))
    bars_unique_symbols_after_clip = (
        int(bars_df["symbol"].astype(str).str.upper().nunique())
        if bars_rows_after_clip and "symbol" in bars_df.columns
        else 0
    )

    diagnostics: list[dict[str, Any]] = []
    attention_df = compute_speculative_attention(layer1_df, bars_df, diagnostics_out=diagnostics)

    return Layer2ReplayArtifacts(
        replay_day=replay_day,
        lookback_calendar_days=lookback_calendar_days,
        feed_label=feed_label,
        fetch_start_utc=start_utc,
        fetch_end_exclusive_utc=end_exclusive_utc,
        layer1_df=layer1_df,
        layer1_load_stats=persisted_layer1_stats,
        bars_rows_fetched=bars_rows_fetched,
        bars_unique_symbols_fetched=bars_unique_symbols_fetched,
        bars_df=bars_df,
        bars_rows_after_clip=bars_rows_after_clip,
        bars_unique_symbols_after_clip=bars_unique_symbols_after_clip,
        diagnostics=diagnostics,
        attention_df=attention_df,
    )
