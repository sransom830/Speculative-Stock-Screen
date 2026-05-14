"""
Alpaca Markets read-only adapter (alpaca-py).

Fetches and normalizes universe, quotes, and daily bars. No orders, signals,
or Layer 2+ logic.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from itertools import islice
from typing import Any, Callable, Iterable, List, Optional, Sequence, TypeVar

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoLatestQuoteRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass as AlpacaAssetClass
from alpaca.trading.enums import AssetStatus
from alpaca.trading.models import Asset as AlpacaAsset
from alpaca.trading.requests import GetAssetsRequest

from data.providers.base_provider import AssetClass, AssetRef, BaseMarketDataProvider, QuoteSnapshot

T = TypeVar("T")

PROVIDER_NAME = "alpaca"

DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_BACKOFF_S = 0.75
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


ASSETS_DF_COLUMNS = (
    "provider",
    "provider_asset_id",
    "symbol",
    "asset_class",
    "exchange",
    "name",
    "status",
    "tradable",
    "shortable",
    "easy_to_borrow",
    "fractionable",
)

QUOTES_DF_COLUMNS = (
    "provider",
    "symbol",
    "asset_class",
    "bid",
    "ask",
    "last",
    "timestamp_utc",
)

BARS_DF_COLUMNS = (
    "provider",
    "symbol",
    "timestamp_utc",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
)


def _batched(items: Sequence[T], batch_size: int) -> Iterable[List[T]]:
    it = iter(items)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk


def _call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_backoff_s: float = DEFAULT_BASE_BACKOFF_S,
) -> T:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except APIError as e:
            last_exc = e
            code = e.status_code
            if code is not None and code not in RETRYABLE_STATUS_CODES:
                raise
        except (ConnectionError, TimeoutError) as e:
            last_exc = e
        if attempt == max_retries - 1:
            break
        time.sleep(base_backoff_s * (2**attempt))
    if last_exc:
        raise last_exc
    raise RuntimeError("retry loop exited without result")


def _empty_df(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _alpaca_asset_class_to_domain(ac: AlpacaAssetClass) -> AssetClass:
    if ac == AlpacaAssetClass.US_EQUITY:
        return AssetClass.EQUITY
    if ac == AlpacaAssetClass.CRYPTO or ac == AlpacaAssetClass.CRYPTO_PERP:
        return AssetClass.CRYPTO
    if ac == AlpacaAssetClass.US_OPTION:
        return AssetClass.OPTION
    return AssetClass.OTHER


def _domain_asset_class_matches(asset: AlpacaAsset, want: frozenset[AssetClass]) -> bool:
    return _alpaca_asset_class_to_domain(asset.asset_class) in want


def _alpaca_asset_to_ref(asset: AlpacaAsset) -> AssetRef:
    return AssetRef(
        symbol=asset.symbol,
        asset_class=_alpaca_asset_class_to_domain(asset.asset_class),
        exchange=str(asset.exchange) if asset.exchange is not None else None,
        provider_asset_id=str(asset.id) if asset.id is not None else None,
    )


def _normalize_assets_df(assets: Sequence[AlpacaAsset]) -> pd.DataFrame:
    if not assets:
        return _empty_df(ASSETS_DF_COLUMNS)
    rows = []
    for a in assets:
        rows.append(
            {
                "provider": PROVIDER_NAME,
                "provider_asset_id": str(a.id) if a.id is not None else None,
                "symbol": a.symbol,
                "asset_class": _alpaca_asset_class_to_domain(a.asset_class).value,
                "exchange": str(a.exchange) if a.exchange is not None else None,
                "name": a.name,
                "status": str(a.status) if a.status is not None else None,
                "tradable": bool(a.tradable),
                "shortable": bool(a.shortable),
                "easy_to_borrow": bool(a.easy_to_borrow),
                "fractionable": bool(a.fractionable),
            }
        )
    return pd.DataFrame(rows, columns=list(ASSETS_DF_COLUMNS))


def _barset_to_normalized_df(barset: BarSet, timeframe_label: str) -> pd.DataFrame:
    if not barset.data:
        return _empty_df(BARS_DF_COLUMNS)
    df = barset.df.reset_index()
    df.insert(0, "provider", PROVIDER_NAME)
    df = df.rename(columns={"timestamp": "timestamp_utc"})
    df["timeframe"] = timeframe_label
    df = df[
        [
            "provider",
            "symbol",
            "timestamp_utc",
            "timeframe",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "trade_count",
            "vwap",
        ]
    ]
    return df


def _merge_quote_results_to_df(
    *,
    stock_rows: List[dict],
    crypto_rows: List[dict],
) -> pd.DataFrame:
    rows = stock_rows + crypto_rows
    if not rows:
        return _empty_df(QUOTES_DF_COLUMNS)
    return pd.DataFrame(rows, columns=list(QUOTES_DF_COLUMNS))


class AlpacaMarketDataProvider(BaseMarketDataProvider):
    """Synchronous Alpaca REST adapter for Layer 1."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        *,
        paper: Optional[bool] = None,
        quote_batch_size: int = DEFAULT_BATCH_SIZE,
        bar_batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        stock_feed: Optional[DataFeed] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        if not self._api_key or not self._secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set (or pass api_key/secret_key).")
        if paper is None:
            paper = os.environ.get("ALPACA_PAPER", "true").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        self._quote_batch_size = quote_batch_size
        self._bar_batch_size = bar_batch_size
        self._max_retries = max_retries
        self._stock_feed = stock_feed

        self._trading = TradingClient(
            self._api_key,
            self._secret_key,
            paper=paper,
        )
        self._stock_data = StockHistoricalDataClient(self._api_key, self._secret_key)
        self._crypto_data = CryptoHistoricalDataClient(self._api_key, self._secret_key)

    def provider_name(self) -> str:
        return PROVIDER_NAME

    def _retry(self, fn: Callable[[], T]) -> T:
        return _call_with_retry(fn, max_retries=self._max_retries)

    def healthcheck(self) -> dict[str, Any]:
        try:
            clock = self._retry(lambda: self._trading.get_clock())
            return {
                "ok": True,
                "provider": PROVIDER_NAME,
                "is_open": clock.is_open,
                "timestamp": clock.timestamp.isoformat() if clock.timestamp else None,
            }
        except Exception as e:
            return {"ok": False, "provider": PROVIDER_NAME, "error": str(e)}

    def _load_tradable_alpaca_assets(
        self,
        *,
        asset_classes: Sequence[AssetClass] | None,
    ) -> List[AlpacaAsset]:
        req = GetAssetsRequest(status=AssetStatus.ACTIVE)
        raw = self._retry(lambda: self._trading.get_all_assets(req))
        if not isinstance(raw, list):
            raise TypeError(f"unexpected get_all_assets response: {type(raw)}")
        tradable = [a for a in raw if a.tradable]
        if asset_classes:
            want = frozenset(asset_classes)
            tradable = [a for a in tradable if _domain_asset_class_matches(a, want)]
        return tradable

    def list_tradable_assets(
        self,
        *,
        asset_classes: Sequence[AssetClass] | None = None,
    ) -> Sequence[AssetRef]:
        assets = self._load_tradable_alpaca_assets(asset_classes=asset_classes)
        return [_alpaca_asset_to_ref(a) for a in assets]

    def list_tradable_assets_df(
        self,
        *,
        asset_classes: Sequence[AssetClass] | None = None,
    ) -> pd.DataFrame:
        """Normalized tradable universe (one row per instrument)."""
        assets = self._load_tradable_alpaca_assets(asset_classes=asset_classes)
        return _normalize_assets_df(assets)

    def fetch_quotes(self, assets: Sequence[AssetRef]) -> Sequence[QuoteSnapshot]:
        df = self.fetch_quotes_df(assets)
        if df.empty:
            return []
        out: List[QuoteSnapshot] = []
        sym_to_ref = {a.symbol: a for a in assets}
        for _, row in df.iterrows():
            ref = sym_to_ref.get(str(row["symbol"]))
            if ref is None:
                continue
            ts = row["timestamp_utc"]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            out.append(
                QuoteSnapshot(
                    asset=ref,
                    bid=float(row["bid"]) if pd.notna(row["bid"]) else None,
                    ask=float(row["ask"]) if pd.notna(row["ask"]) else None,
                    last=float(row["last"]) if pd.notna(row["last"]) else None,
                    timestamp_utc=ts,
                )
            )
        return out

    def fetch_quotes_df(
        self,
        assets: Sequence[AssetRef],
        *,
        batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Latest quote snapshots; columns defined by ``QUOTES_DF_COLUMNS``."""
        if not assets:
            return _empty_df(QUOTES_DF_COLUMNS)
        bsz = batch_size or self._quote_batch_size
        stock: List[AssetRef] = []
        crypto: List[AssetRef] = []
        for a in assets:
            if a.asset_class == AssetClass.CRYPTO:
                crypto.append(a)
            else:
                stock.append(a)

        stock_rows: List[dict] = []
        for chunk in _batched(stock, bsz):
            syms = [a.symbol for a in chunk]
            req = StockLatestQuoteRequest(
                symbol_or_symbols=syms,
                feed=self._stock_feed,
            )
            res = self._retry(lambda: self._stock_data.get_stock_latest_quote(req))
            if not isinstance(res, dict):
                raise TypeError(f"unexpected stock quote response: {type(res)}")
            by_sym = {a.symbol: a for a in chunk}
            for sym, quote in res.items():
                ref = by_sym.get(sym)
                if ref is None:
                    continue
                ac_value = ref.asset_class.value
                bid = float(quote.bid_price) if quote.bid_price is not None else None
                ask = float(quote.ask_price) if quote.ask_price is not None else None
                last = None
                if bid is not None and ask is not None:
                    last = (bid + ask) / 2.0
                stock_rows.append(
                    {
                        "provider": PROVIDER_NAME,
                        "symbol": sym,
                        "asset_class": ac_value,
                        "bid": bid,
                        "ask": ask,
                        "last": last,
                        "timestamp_utc": quote.timestamp,
                    }
                )

        crypto_rows: List[dict] = []
        for chunk in _batched(crypto, bsz):
            syms = [a.symbol for a in chunk]
            req = CryptoLatestQuoteRequest(symbol_or_symbols=syms)
            res = self._retry(lambda: self._crypto_data.get_crypto_latest_quote(req))
            if not isinstance(res, dict):
                raise TypeError(f"unexpected crypto quote response: {type(res)}")
            by_sym = {a.symbol: a for a in chunk}
            for sym, quote in res.items():
                ref = by_sym.get(sym)
                if ref is None:
                    continue
                ac_value = ref.asset_class.value
                bid = float(quote.bid_price) if quote.bid_price is not None else None
                ask = float(quote.ask_price) if quote.ask_price is not None else None
                last = None
                if bid is not None and ask is not None:
                    last = (bid + ask) / 2.0
                crypto_rows.append(
                    {
                        "provider": PROVIDER_NAME,
                        "symbol": sym,
                        "asset_class": ac_value,
                        "bid": bid,
                        "ask": ask,
                        "last": last,
                        "timestamp_utc": quote.timestamp,
                    }
                )

        df = _merge_quote_results_to_df(stock_rows=stock_rows, crypto_rows=crypto_rows)
        if not df.empty:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        return df

    def fetch_daily_bars_df(
        self,
        assets: Sequence[AssetRef],
        start: datetime,
        end: datetime,
        *,
        batch_size: Optional[int] = None,
        feed: Optional[DataFeed] = None,
    ) -> pd.DataFrame:
        """
        Daily OHLCV bars. Routes US equities/options through the stock data client
        and crypto through the crypto client. ``timeframe`` column is ``1Day``.
        """
        if not assets:
            return _empty_df(BARS_DF_COLUMNS)
        bsz = batch_size or self._bar_batch_size
        feed_eff = feed if feed is not None else self._stock_feed
        stock_like = [a for a in assets if a.asset_class != AssetClass.CRYPTO]
        crypto = [a for a in assets if a.asset_class == AssetClass.CRYPTO]

        parts: List[pd.DataFrame] = []
        tf_label = str(TimeFrame.Day)

        for chunk in _batched(stock_like, bsz):
            syms = [a.symbol for a in chunk]
            req = StockBarsRequest(
                symbol_or_symbols=syms,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=feed_eff,
            )
            barset = self._retry(lambda: self._stock_data.get_stock_bars(req))
            if not isinstance(barset, BarSet):
                raise TypeError(f"unexpected stock bars response: {type(barset)}")
            parts.append(_barset_to_normalized_df(barset, tf_label))

        for chunk in _batched(crypto, bsz):
            syms = [a.symbol for a in chunk]
            req = CryptoBarsRequest(
                symbol_or_symbols=syms,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            barset = self._retry(lambda: self._crypto_data.get_crypto_bars(req))
            if not isinstance(barset, BarSet):
                raise TypeError(f"unexpected crypto bars response: {type(barset)}")
            parts.append(_barset_to_normalized_df(barset, tf_label))

        if not parts:
            return _empty_df(BARS_DF_COLUMNS)
        out = pd.concat(parts, ignore_index=True)
        if not out.empty:
            out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
        return out
