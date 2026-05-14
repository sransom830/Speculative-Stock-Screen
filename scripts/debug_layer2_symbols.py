#!/usr/bin/env python3
"""
Debug why specific Layer 1 symbols never reach Layer 2 persistence.

Mirrors production: full layer1_broad_universe, batched Alpaca daily bars (IEX),
then compute_speculative_attention(..., diagnostics_out=...).
"""

from __future__ import annotations

import argparse
import json
import sys
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
from layer2 import compute_speculative_attention  # noqa: E402

BAR_LOOKBACK_DAYS = 90


def _refs_from_layer1(layer1_df: pd.DataFrame) -> list[AssetRef]:
    refs: list[AssetRef] = []
    for _, row in layer1_df.iterrows():
        sym = str(row["symbol"])
        ex = row.get("exchange")
        ex_s = None if ex is None or (isinstance(ex, float) and pd.isna(ex)) else str(ex)
        refs.append(AssetRef(sym, AssetClass.EQUITY, exchange=ex_s))
    return refs


def _print_block(title: str, d: dict) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    for k in sorted(d.keys()):
        v = d[k]
        if k == "source_cohorts" and isinstance(v, list):
            print(f"{k}: {json.dumps(v)}")
        else:
            print(f"{k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 2 per-symbol diagnostics (production path).")
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated tickers, e.g. MSTR,GME,PLTR,RKLB,CVNA",
    )
    args = parser.parse_args()
    targets = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not targets:
        print("Provide at least one symbol.")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    layer1_load_stats: dict = {}
    layer1_df = fetch_layer1_broad_universe_df(load_stats_out=layer1_load_stats)
    if layer1_df.empty:
        print("layer1_broad_universe is empty.")
        print(
            "layer1_fetch_pagination_chunks:",
            layer1_load_stats.get("layer1_fetch_pagination_chunks"),
            "chunk_row_counts:",
            layer1_load_stats.get("layer1_fetch_chunk_row_counts"),
        )
        return

    sym_col = layer1_df["symbol"].astype(str).str.upper()
    in_l1: dict[str, bool] = {t: t in set(sym_col) for t in targets}

    print("Production-equivalent path: full Layer 1 universe + shared bars batch.")
    print(f"Layer 1 rows loaded: {len(layer1_df)}")
    print(
        f"Layer 1 fetch: pagination_chunks={layer1_load_stats.get('layer1_fetch_pagination_chunks')} "
        f"chunk_row_counts={layer1_load_stats.get('layer1_fetch_chunk_row_counts')} "
        f"chunk_size_requested={layer1_load_stats.get('layer1_fetch_chunk_size_requested')}"
    )

    provider = AlpacaMarketDataProvider(stock_feed=DataFeed.IEX)
    refs = _refs_from_layer1(layer1_df)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=BAR_LOOKBACK_DAYS)
    bars_df = provider.fetch_daily_bars_df(refs, start, end, feed=DataFeed.IEX)

    bar_counts = (
        bars_df.groupby(bars_df["symbol"].astype(str).str.upper()).size().to_dict()
        if not bars_df.empty and "symbol" in bars_df.columns
        else {}
    )

    diagnostics: list[dict] = []
    compute_speculative_attention(layer1_df, bars_df, diagnostics_out=diagnostics)
    by_sym = {str(d["symbol"]).upper(): d for d in diagnostics}

    print(f"Bars frame rows: {len(bars_df)} unique symbols in bars: {len(bar_counts)}")

    for t in targets:
        print()
        print("#" * 72)
        print(f"# SYMBOL: {t}")
        print("#" * 72)
        if not in_l1[t]:
            print(f"NOT in layer1_broad_universe (Layer 2 never considers this symbol).")
            print(f"Alpaca daily bar rows fetched (batch may still include it): {bar_counts.get(t, 0)}")
            continue

        row_l1 = layer1_df.loc[sym_col == t].iloc[0]
        cohorts_l1 = row_l1.get("source_cohorts")
        print(f"In layer1_broad_universe: yes | source_cohorts (Layer1 row): {cohorts_l1}")
        print(f"Alpaca daily bar rows in this run: {bar_counts.get(t, 0)}")

        d = by_sym.get(t)
        if d is None:
            print("INTERNAL: no diagnostic row (unexpected).")
            continue
        _print_block(f"{t} — Layer 2 diagnostic record", d)
        if d.get("status") == "skipped":
            print("\n>>> Outcome: SKIPPED from scored universe — persistence will omit this symbol.")
            print(f">>> Reason: {d.get('skip_reason')}")
        else:
            print("\n>>> Outcome: SCORED — run_layer2 upserts the full scored set (TOP_N is stdout only).")
            print(">>> If this symbol is still missing in Supabase after a successful run, investigate DB/errors, not score filtering.")


if __name__ == "__main__":
    main()
