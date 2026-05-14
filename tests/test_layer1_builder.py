"""Layer 1 universe builder unit tests (mocked provider, no live APIs)."""

from __future__ import annotations

import pandas as pd

from data.providers.base_provider import AssetClass, AssetRef

from layer1.broad_market_universe_builder import build_broad_us_equity_universe
from layer1.cohorts import cohort_labels_for_symbol
from layer1.enrichment import metrics_from_daily_bars
from layer1.filters import filter_us_equity_major_listed


def test_cohort_labels_sorted_and_meme() -> None:
    assert "retail_meme" in cohort_labels_for_symbol("gme")
    assert "short_squeeze_ecosystem" in cohort_labels_for_symbol("gme")
    assert "leveraged_crypto_beta" in cohort_labels_for_symbol("mstr")
    assert "space_aerospace_speculative" in cohort_labels_for_symbol("rklb")
    assert cohort_labels_for_symbol("ZZZZZZ") == []


def test_filter_us_equity_drops_otc() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "AAPL", "asset_class": "equity", "exchange": "AssetExchange.NASDAQ", "name": "Apple"},
            {"symbol": "PINK", "asset_class": "equity", "exchange": "AssetExchange.OTC", "name": "Pink"},
        ]
    )
    out = filter_us_equity_major_listed(df)
    assert list(out["symbol"]) == ["AAPL"]


def test_metrics_from_daily_bars() -> None:
    bars = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "timestamp_utc": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "close": [10.0, 12.0, 5.0, 5.0],
            "volume": [1_000_000.0, 2_000_000.0, 2_000_000.0, 0.0],
        }
    )
    m = metrics_from_daily_bars(bars, min_avg_daily_dollar_volume=1.0)
    assert set(m["symbol"]) == {"A", "B"}
    a = m.loc[m["symbol"] == "A"].iloc[0]
    assert a["price"] == 12.0
    assert a["avg_daily_dollar_volume"] == (10 * 1_000_000 + 12 * 2_000_000) / 2


class _FakeLayer1Provider:
    def provider_name(self) -> str:
        return "fake"

    def list_tradable_assets_df(self, *, asset_classes=None) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "provider": "fake",
                    "provider_asset_id": "1",
                    "symbol": "AAA",
                    "asset_class": "equity",
                    "exchange": "AssetExchange.NASDAQ",
                    "name": "Test A",
                    "status": "active",
                    "tradable": True,
                    "shortable": False,
                    "easy_to_borrow": False,
                    "fractionable": True,
                }
            ]
        )

    def fetch_daily_bars_df(self, assets, start, end, **kwargs) -> pd.DataFrame:
        assert isinstance(assets[0], AssetRef)
        return pd.DataFrame(
            {
                "provider": ["fake", "fake"],
                "symbol": ["AAA", "AAA"],
                "timestamp_utc": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
                "timeframe": ["1Day", "1Day"],
                "open": [10.0, 11.0],
                "high": [11.0, 12.0],
                "low": [9.0, 10.0],
                "close": [10.0, 12.0],
                "volume": [2_000_000.0, 2_000_000.0],
                "trade_count": [100.0, 110.0],
                "vwap": [10.0, 11.0],
            }
        )


def test_build_broad_us_equity_universe_mocked(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_upsert(rows, *, client=None):
        calls.append("upsert")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "AAA"
        assert rows[0]["company_name"] == "Test A"
        assert rows[0]["avg_daily_dollar_volume"] > 0
        assert rows[0]["price"] == 12.0
        return None

    monkeypatch.setattr(
        "layer1.broad_market_universe_builder.upsert_broad_universe",
        _fake_upsert,
    )

    df = build_broad_us_equity_universe(
        _FakeLayer1Provider(),
        min_avg_daily_dollar_volume=1.0,
        persist=True,
    )
    assert len(df) == 1
    assert "source_cohorts" in df.columns
    assert calls == ["upsert"]


def test_build_skips_persist_when_empty(monkeypatch) -> None:
    class EmptyProv(_FakeLayer1Provider):
        def list_tradable_assets_df(self, *, asset_classes=None) -> pd.DataFrame:
            return pd.DataFrame(
                columns=[
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
                ]
            )

    def _boom(_rows, *, client=None):
        raise AssertionError("should not persist empty")

    monkeypatch.setattr(
        "layer1.broad_market_universe_builder.upsert_broad_universe",
        _boom,
    )

    df = build_broad_us_equity_universe(EmptyProv(), persist=True)
    assert df.empty
