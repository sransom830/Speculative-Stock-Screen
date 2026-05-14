import pytest

from data.providers.base_provider import AssetClass, BaseMarketDataProvider


def test_base_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseMarketDataProvider()  # type: ignore[abstract]


def test_asset_class_enum_str() -> None:
    assert AssetClass.EQUITY.value == "equity"
