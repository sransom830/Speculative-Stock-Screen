#!/usr/bin/env python3
"""Layer 2: load Layer 1 from Supabase, score attention, persist current + history."""

from __future__ import annotations

import sys
import time
from typing import Any
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.enums import DataFeed  # noqa: E402

from data.providers import AlpacaMarketDataProvider  # noqa: E402
from data.providers.base_provider import AssetClass, AssetRef  # noqa: E402
from layer1 import fetch_layer1_broad_universe_df  # noqa: E402
from layer2 import (  # noqa: E402
    append_population_history,
    compute_speculative_attention,
    population_rows_from_attention_df,
    upsert_current_population,
)

# Enough calendar history for longest Layer 2 window (default ~20 sessions + buffer).
BAR_LOOKBACK_DAYS = 90
TOP_N = 25


def _refs_from_layer1(layer1_df: pd.DataFrame) -> list[AssetRef]:
    refs: list[AssetRef] = []

    for _, row in layer1_df.iterrows():
        sym = str(row["symbol"])
        ex = row.get("exchange")
        ex_s = None if ex is None or (isinstance(ex, float) and pd.isna(ex)) else str(ex)
        refs.append(AssetRef(sym, AssetClass.EQUITY, exchange=ex_s))
    return refs


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    stats: dict[str, Any] = {}

    t0 = time.perf_counter()

    t_l1 = time.perf_counter()
    layer1_load_stats: dict[str, Any] = {}
    layer1_df = fetch_layer1_broad_universe_df(load_stats_out=layer1_load_stats)
    stats["layer1_rows"] = int(len(layer1_df))
    stats["layer1_fetch_pagination_chunks"] = int(
        layer1_load_stats.get("layer1_fetch_pagination_chunks", 0)
    )
    stats["layer1_fetch_chunk_row_counts"] = layer1_load_stats.get(
        "layer1_fetch_chunk_row_counts", []
    )
    stats["fetch_layer1_s"] = round(time.perf_counter() - t_l1, 3)

    if layer1_df.empty:
        print("layer1_rows: 0 (nothing to score)")
        print(
            "layer1_fetch_pagination_chunks:",
            stats["layer1_fetch_pagination_chunks"],
            "chunk_row_counts:",
            stats["layer1_fetch_chunk_row_counts"],
        )
        print("elapsed_total_s:", round(time.perf_counter() - t0, 3))
        return

    provider = AlpacaMarketDataProvider(stock_feed=DataFeed.IEX)
    refs = _refs_from_layer1(layer1_df)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=BAR_LOOKBACK_DAYS)

    t_bars = time.perf_counter()
    bars_df = provider.fetch_daily_bars_df(refs, start, end, feed=DataFeed.IEX)
    stats["bars_rows"] = int(len(bars_df))
    stats["bars_symbols"] = int(bars_df["symbol"].nunique()) if not bars_df.empty else 0
    stats["fetch_bars_s"] = round(time.perf_counter() - t_bars, 3)

    t_score = time.perf_counter()
    attention_df = compute_speculative_attention(layer1_df, bars_df)
    stats["scored_rows"] = int(len(attention_df))
    stats["score_s"] = round(time.perf_counter() - t_score, 3)

    pop_rows: list = []
    t_persist = time.perf_counter()
    if not attention_df.empty:
        pop_rows = population_rows_from_attention_df(attention_df)
        run_ts = datetime.now(timezone.utc).isoformat()
        for r in pop_rows:
            r["last_updated"] = run_ts
        upsert_current_population(pop_rows)
        append_population_history(pop_rows, snapshot_timestamp=run_ts)
    stats["persist_s"] = round(time.perf_counter() - t_persist, 3)
    stats["elapsed_total_s"] = round(time.perf_counter() - t0, 3)

    print("--- runtime ---")
    for k in (
        "layer1_rows",
        "layer1_fetch_pagination_chunks",
        "layer1_fetch_chunk_row_counts",
        "fetch_layer1_s",
        "bars_rows",
        "bars_symbols",
        "fetch_bars_s",
        "scored_rows",
        "score_s",
        "persist_s",
        "elapsed_total_s",
    ):
        print(f"{k}: {stats[k]}")

    if attention_df.empty:
        print("\n(no rows scored — check bars vs Layer 1 overlap)")
        return

    top = attention_df.sort_values("speculative_attention_score", ascending=False).head(TOP_N)
    cols = [
        "symbol",
        "speculative_attention_score",
        "speculative_attention_state",
        "ecosystem_multiplier",
        "factor_relative_volume",
        "factor_realized_vol_expansion",
        "factor_acceleration_composite",
    ]
    cols = [c for c in cols if c in top.columns]
    print(f"\n--- top {TOP_N} by speculative_attention_score ---")
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
