"""Unit tests for Alpaca provider utilities (no live API calls)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from alpaca.common.exceptions import APIError
from alpaca.data.models.bars import BarSet

from data.providers.alpaca_provider import (
    AlpacaMarketDataProvider,
    _barset_to_normalized_df,
    _batched,
    _call_with_retry,
    _empty_df,
    ASSETS_DF_COLUMNS,
    BARS_DF_COLUMNS,
)


def _http_err(code: int):
    class Resp:
        status_code = code

    class Err:
        response = Resp()

    return Err()


def test_batched() -> None:
    assert list(_batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(_batched([], 3)) == []


def test_call_with_retry_succeeds_after_transient() -> None:
    attempts = {"n": 0}

    def fn():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise APIError('{"code":429,"message":"limit"}', http_error=_http_err(429))
        return "ok"

    with patch("data.providers.alpaca_provider.time.sleep"):
        assert _call_with_retry(fn, max_retries=4) == "ok"
    assert attempts["n"] == 2


def test_call_with_retry_non_retryable_raises_immediately() -> None:
    def fn():
        raise APIError('{"code":400,"message":"bad"}', http_error=_http_err(400))

    with pytest.raises(APIError):
        _call_with_retry(fn, max_retries=4)


def test_barset_to_normalized_df() -> None:
    raw = {
        "AAPL": [
            {
                "t": datetime(2024, 1, 2, tzinfo=timezone.utc),
                "o": 10.0,
                "h": 11.0,
                "l": 9.0,
                "c": 10.5,
                "v": 1000.0,
                "n": 5.0,
                "vw": 10.2,
            }
        ]
    }
    df = _barset_to_normalized_df(BarSet(raw), "1Day")
    assert list(df.columns) == list(BARS_DF_COLUMNS)
    assert df.iloc[0]["symbol"] == "AAPL"
    assert df.iloc[0]["close"] == 10.5


def test_empty_assets_df_shape() -> None:
    df = _empty_df(ASSETS_DF_COLUMNS)
    assert df.empty
    assert list(df.columns) == list(ASSETS_DF_COLUMNS)


def test_alpaca_provider_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="ALPACA_API_KEY"):
        AlpacaMarketDataProvider(api_key="", secret_key="")
