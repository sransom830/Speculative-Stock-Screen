#!/usr/bin/env python3
"""Rebuild Layer 1 broad US equity universe and persist to Supabase."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.enums import DataFeed  # noqa: E402

from data.providers import AlpacaMarketDataProvider  # noqa: E402
from layer1 import build_broad_us_equity_universe  # noqa: E402


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    provider = AlpacaMarketDataProvider(stock_feed=DataFeed.IEX)

    t0 = time.perf_counter()
    df = build_broad_us_equity_universe(
        provider,
        bar_feed=DataFeed.IEX,
        persist=True,
    )
    elapsed = time.perf_counter() - t0

    print("rows:", len(df))
    print("columns:", list(df.columns))
    if df.empty:
        print("(empty DataFrame)")
    else:
        print(df.head(15).to_string())
    print("elapsed_s:", round(elapsed, 3))


if __name__ == "__main__":
    main()
