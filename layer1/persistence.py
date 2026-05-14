"""
Layer 1 persistence: canonical broad-universe Upserts only.

No signals, fragility, confirmation, prediction, or execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from data.storage.supabase_client import get_supabase_client

TABLE_NAME = "layer1_broad_universe"

_COLUMNS = frozenset(
    {
        "symbol",
        "company_name",
        "exchange",
        "asset_class",
        "market_cap",
        "avg_daily_dollar_volume",
        "price",
        "beta",
        "source_cohorts",
        "last_updated",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _COLUMNS:
        if key not in row:
            continue
        out[key] = row[key]
    if "symbol" not in out or not out["symbol"]:
        raise ValueError("each row must include a non-empty 'symbol'")

    cohorts = out.get("source_cohorts", [])
    if cohorts is None:
        out["source_cohorts"] = []
    elif isinstance(cohorts, (list, tuple)):
        out["source_cohorts"] = list(cohorts)
    else:
        raise TypeError("source_cohorts must be a list/tuple of cohort labels")

    if "last_updated" not in out or out["last_updated"] is None:
        out["last_updated"] = _utc_now_iso()

    return out


def upsert_broad_universe(
    rows: Sequence[Mapping[str, Any]],
    *,
    client: Any | None = None,
) -> Any:
    """
    Upsert rows into ``layer1_broad_universe`` on the primary key ``symbol``.

    Unknown keys on input rows are ignored. ``source_cohorts`` is stored as JSONB.
    """
    if not rows:
        return None
    sb = client or get_supabase_client()
    payload: list[dict[str, Any]] = [_normalize_row(r) for r in rows]
    return sb.table(TABLE_NAME).upsert(payload, on_conflict="symbol").execute()
