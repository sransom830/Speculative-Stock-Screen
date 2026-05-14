#!/usr/bin/env python3
"""Smoke-test Alpaca read-only provider (loads .env, no trading logic)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.enums import DataFeed  # noqa: E402

from data.providers import AlpacaMarketDataProvider  # noqa: E402
from data.providers.base_provider import AssetClass, AssetRef  # noqa: E402


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    # IEX avoids SIP feed (many accounts get 403 on consolidated SIP for bars).
    provider = AlpacaMarketDataProvider(stock_feed=DataFeed.IEX)

    print("provider_name:", provider.provider_name())
    print("healthcheck:", provider.healthcheck())

    symbols = ("AAPL", "TSLA", "PLTR", "MSTR", "GME")
    bars_refs = [AssetRef(s, AssetClass.EQUITY) for s in symbols]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=90)

    print("\n--- Tradable assets (normalized) ---")
    assets_df = provider.list_tradable_assets_df()
    print("columns:", list(assets_df.columns))
    print("rows:", len(assets_df))
    print(assets_df.head(10).to_string())
    print("...")

    print("\n--- Daily bars ---")
    bars_df = provider.fetch_daily_bars_df(bars_refs, start=start, end=end)
    print("columns:", list(bars_df.columns))
    print("rows:", len(bars_df))
    print(bars_df.head(15).to_string())


if __name__ == "__main__":
    main()
