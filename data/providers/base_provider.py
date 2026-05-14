"""
Abstract interface for Layer 1 market data providers.

Implementations (e.g. Alpaca) supply observable ecosystem facts only:
tradable instruments, classification metadata, and read-only snapshots.
No order routing, no signal generation, no fragility or unwind logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, Sequence


class AssetClass(str, Enum):
    """Coarse asset bucket for universe filtering (provider-agnostic)."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    OPTION = "option"
    OTHER = "other"


@dataclass(frozen=True)
class AssetRef:
    """Stable identifier for an instrument within a provider namespace."""

    symbol: str
    asset_class: AssetClass
    exchange: str | None = None
    provider_asset_id: str | None = None


@dataclass(frozen=True)
class QuoteSnapshot:
    """Point-in-time top-of-book or last-trade style observation."""

    asset: AssetRef
    bid: float | None
    ask: float | None
    last: float | None
    timestamp_utc: datetime


class BaseMarketDataProvider(ABC):
    """
    Read-only contract for listing and observing a speculative-capable universe.

    Subclasses encapsulate vendor specifics (REST/WebSocket, pagination, auth).
    Callers in `data.ingestion` coordinate fetching; this module stays free of
    orchestration and business rules beyond data access.
    """

    @abstractmethod
    def provider_name(self) -> str:
        """Short stable name, e.g. ``alpaca``."""

    @abstractmethod
    def list_tradable_assets(
        self,
        *,
        asset_classes: Sequence[AssetClass] | None = None,
    ) -> Sequence[AssetRef]:
        """
        Return instruments the provider considers active and tradable.

        Used for Layer 1 “what exists?” universe construction. Filtering by
        liquidity or narrative happens in higher layers, not here.
        """

    @abstractmethod
    def fetch_quotes(
        self,
        assets: Sequence[AssetRef],
    ) -> Sequence[QuoteSnapshot]:
        """Batch read-only quotes for the given assets (best-effort per vendor)."""

    def healthcheck(self) -> dict[str, Any]:
        """
        Optional connectivity / credential probe. Default: subclasses may override.

        Returns a small JSON-serializable dict for logging and startup checks.
        """
        return {"ok": True, "provider": self.provider_name()}


class SupportsWireFormat(Protocol):
    """
    Optional hook for providers that map vendor JSON into domain types.

    Not required on the abstract base; Alpaca (or other) adapters may implement.
    """

    @classmethod
    def from_vendor_asset(cls, payload: dict[str, Any]) -> AssetRef: ...


__all__ = [
    "AssetClass",
    "AssetRef",
    "BaseMarketDataProvider",
    "QuoteSnapshot",
    "SupportsWireFormat",
]
