"""US listed equity filters (major listing venues; excludes OTC-style names)."""

from __future__ import annotations

from typing import Any

import pandas as pd

# Substrings seen in Alpaca ``AssetExchange.*`` string forms.
_MAJOR_LISTING_MARKERS: frozenset[str] = frozenset(
    {
        "NASDAQ",
        "NYSE",
        "AMEX",
        "ARCA",
        "BATS",
    }
)


def normalize_exchange_field(value: Any) -> str:
    """Turn ``AssetExchange.NASDAQ``-style values into ``NASDAQ``."""
    s = str(value).strip().upper()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s


def is_us_major_listed_exchange(exchange_raw: Any) -> bool:
    ex = str(exchange_raw).upper()
    if "OTC" in ex or "PINK" in ex:
        return False
    return any(marker in ex for marker in _MAJOR_LISTING_MARKERS)


def filter_us_equity_major_listed(assets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep rows that are US ``equity`` class on a major listing venue.

    Expects columns from ``list_tradable_assets_df``: ``symbol``, ``asset_class``, ``exchange``.
    """
    if assets_df.empty:
        return assets_df
    df = assets_df.copy()
    if "asset_class" not in df.columns:
        raise ValueError("assets_df must include 'asset_class'")
    df = df[df["asset_class"].astype(str).str.lower() == "equity"]
    if "exchange" not in df.columns:
        raise ValueError("assets_df must include 'exchange'")
    mask = df["exchange"].map(is_us_major_listed_exchange)
    df = df.loc[mask].copy()
    df["exchange"] = df["exchange"].map(normalize_exchange_field)
    return df.drop_duplicates(subset=["symbol"], keep="first")
