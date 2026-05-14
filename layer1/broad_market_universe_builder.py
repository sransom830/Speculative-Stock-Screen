"""
Broad US equity universe builder (Layer 1: what exists?).

Universe → liquidity gate → optional enrichment gaps (market_cap, beta) → cohort tags → persist.
No scoring, fragility, confirmation, shorts, or execution.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from data.providers.base_provider import AssetClass, AssetRef

from layer1.cohorts import cohort_labels_for_symbol
from layer1.enrichment import metrics_from_daily_bars
from layer1.filters import filter_us_equity_major_listed
from layer1.layer1_provider_protocol import Layer1ObservableProvider
from layer1.persistence import upsert_broad_universe

LAYER1_FRAME_COLUMNS: tuple[str, ...] = (
    "symbol",
    "company_name",
    "exchange",
    "asset_class",
    "market_cap",
    "avg_daily_dollar_volume",
    "price",
    "beta",
    "source_cohorts",
)


def _layer1_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = str(r["symbol"])
        company = r["company_name"]
        exch = r["exchange"]
        ac = r["asset_class"]
        adv = r["avg_daily_dollar_volume"]
        price = r["price"]
        cohorts = r["source_cohorts"]
        rows.append(
            {
                "symbol": sym,
                "company_name": None if pd.isna(company) else str(company),
                "exchange": None if pd.isna(exch) else str(exch),
                "asset_class": None if pd.isna(ac) else str(ac),
                "market_cap": None,
                "avg_daily_dollar_volume": None if adv is None or (isinstance(adv, float) and math.isnan(adv)) else float(adv),
                "price": None if price is None or (isinstance(price, float) and math.isnan(price)) else float(price),
                "beta": None,
                "source_cohorts": list(cohorts) if cohorts is not None else [],
            }
        )
    return rows


def build_broad_us_equity_universe(
    provider: Layer1ObservableProvider,
    *,
    min_avg_daily_dollar_volume: float = 1_000_000.0,
    adv_lookback_calendar_days: int = 56,
    bar_batch_size: int = 100,
    bar_feed: Any | None = None,
    persist: bool = True,
    supabase_client: Any | None = None,
) -> pd.DataFrame:
    """
    Build the canonical Layer 1 DataFrame and optionally persist it.

    Parameters
    ----------
    provider :
        Any implementation satisfying ``Layer1ObservableProvider`` (e.g. Alpaca).
    min_avg_daily_dollar_volume :
        Liquidity floor on trailing average daily ``close * volume`` from daily bars.
    adv_lookback_calendar_days :
        History window pulled for that average (calendar days, not sessions).
    bar_feed :
        Optional Alpaca ``DataFeed`` (e.g. ``DataFeed.IEX`` on free data tiers).
    persist :
        When True and the universe is non-empty, upserts into Supabase via Layer 1 persistence.
    supabase_client :
        Injected client for tests; defaults to env-based client when None.

    Notes
    -----
    ``market_cap`` and ``beta`` are present for schema alignment; they are left null until a
    fundamentals source is wired in.
    """
    assets = provider.list_tradable_assets_df(asset_classes=[AssetClass.EQUITY])
    us = filter_us_equity_major_listed(assets)

    if us.empty:
        return pd.DataFrame(columns=list(LAYER1_FRAME_COLUMNS))

    refs: list[AssetRef] = []
    for _, row in us.iterrows():
        sym = str(row["symbol"])
        ex = row.get("exchange")
        ex_s = None if ex is None or (isinstance(ex, float) and pd.isna(ex)) else str(ex)
        refs.append(AssetRef(sym, AssetClass.EQUITY, exchange=ex_s))

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(adv_lookback_calendar_days))

    bars = provider.fetch_daily_bars_df(
        refs,
        start,
        end,
        batch_size=bar_batch_size,
        feed=bar_feed,
    )
    metrics = metrics_from_daily_bars(
        bars,
        min_avg_daily_dollar_volume=min_avg_daily_dollar_volume,
    )

    meta = us[["symbol", "name", "exchange", "asset_class"]].copy()
    layered = meta.merge(metrics, on="symbol", how="inner")
    layered["company_name"] = layered["name"]
    layered["asset_class"] = layered["asset_class"].astype(str).str.lower()
    layered["market_cap"] = pd.NA
    layered["beta"] = pd.NA
    layered["source_cohorts"] = layered["symbol"].map(lambda s: cohort_labels_for_symbol(str(s)))

    out = layered[list(LAYER1_FRAME_COLUMNS)].copy()

    if persist and not out.empty:
        upsert_broad_universe(_layer1_records(out), client=supabase_client)

    return out
