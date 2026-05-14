import pytest

from layer1.persistence import _normalize_row, upsert_broad_universe


def test_normalize_row_defaults() -> None:
    row = _normalize_row({"symbol": "AAPL", "company_name": "Apple Inc."})
    assert row["symbol"] == "AAPL"
    assert row["company_name"] == "Apple Inc."
    assert row["source_cohorts"] == []
    assert "last_updated" in row


def test_normalize_row_cohorts() -> None:
    row = _normalize_row(
        {
            "symbol": "TSLA",
            "source_cohorts": ["alpaca_tradable"],
        }
    )
    assert row["source_cohorts"] == ["alpaca_tradable"]


def test_normalize_row_rejects_bad_symbol() -> None:
    with pytest.raises(ValueError):
        _normalize_row({"symbol": ""})


def test_normalize_row_rejects_bad_cohorts() -> None:
    with pytest.raises(TypeError):
        _normalize_row({"symbol": "X", "source_cohorts": "not-a-list"})


def test_upsert_broad_universe_empty_returns_none() -> None:
    assert upsert_broad_universe([]) is None
