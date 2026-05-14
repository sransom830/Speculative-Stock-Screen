"""
Typing protocol: Layer 1 only depends on this read-only surface, not on Alpaca types.

Concrete vendors (e.g. ``AlpacaMarketDataProvider``) should satisfy this protocol.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence

import pandas as pd

from data.providers.base_provider import AssetClass, AssetRef


class Layer1ObservableProvider(Protocol):
    """Market data access needed to construct the Layer 1 US equity universe."""

    def provider_name(self) -> str: ...

    def list_tradable_assets_df(
        self,
        *,
        asset_classes: Sequence[AssetClass] | None = None,
    ) -> pd.DataFrame: ...

    def fetch_daily_bars_df(
        self,
        assets: Sequence[AssetRef],
        start: datetime,
        end: datetime,
        *,
        batch_size: int | None = None,
        feed: Any | None = None,
    ) -> pd.DataFrame: ...
