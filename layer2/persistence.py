"""
Layer 2 persistence: current population upsert + append-only history.

No fragility, confirmation, execution, or external narrative ingestion.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd

from data.storage.supabase_client import get_supabase_client

TABLE_CURRENT = "layer2_speculative_population"
TABLE_HISTORY = "layer2_speculative_population_history"

_HISTORY_INSERT_KEYS = frozenset(
    {
        "symbol",
        "speculative_attention_score",
        "attention_state",
        "source_cohorts",
        "relative_volume_factor",
        "volatility_factor",
        "acceleration_factor",
        "ecosystem_multiplier",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_or_none(val: Any) -> float | None:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _normalize_cohorts(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    raise TypeError("source_cohorts must be a list of strings")


def population_rows_from_attention_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Map ``compute_speculative_attention`` output into Layer 2 persistence rows (before UPSERT).

    Uses ``factor_relative_volume``, ``factor_realized_vol_expansion``,
    ``factor_acceleration_composite`` for the three persisted factors.
    """
    required = (
        "symbol",
        "speculative_attention_score",
        "speculative_attention_state",
        "source_cohorts",
        "factor_relative_volume",
        "factor_realized_vol_expansion",
        "factor_acceleration_composite",
        "ecosystem_multiplier",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"attention DataFrame missing columns: {missing}")
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "symbol": str(r["symbol"]),
                "speculative_attention_score": float(r["speculative_attention_score"]),
                "attention_state": str(r["speculative_attention_state"]),
                "source_cohorts": _normalize_cohorts(r["source_cohorts"]),
                "relative_volume_factor": _float_or_none(r["factor_relative_volume"]),
                "volatility_factor": _float_or_none(r["factor_realized_vol_expansion"]),
                "acceleration_factor": _float_or_none(r["factor_acceleration_composite"]),
                "ecosystem_multiplier": _float_or_none(r["ecosystem_multiplier"]),
            }
        )
    return rows


def _normalize_current_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sym = row.get("symbol")
    if not sym:
        raise ValueError("each row needs a non-empty symbol")
    out: dict[str, Any] = {
        "symbol": str(sym),
        "speculative_attention_score": float(row["speculative_attention_score"]),
        "attention_state": str(row["attention_state"]),
        "source_cohorts": _normalize_cohorts(row.get("source_cohorts", [])),
        "relative_volume_factor": _float_or_none(row.get("relative_volume_factor")),
        "volatility_factor": _float_or_none(row.get("volatility_factor")),
        "acceleration_factor": _float_or_none(row.get("acceleration_factor")),
        "ecosystem_multiplier": _float_or_none(row.get("ecosystem_multiplier")),
        "last_updated": row.get("last_updated") or _utc_now_iso(),
    }
    return out


def upsert_current_population(
    rows: Sequence[Mapping[str, Any]],
    *,
    client: Any | None = None,
    batch_size: int = 500,
) -> Any | None:
    """
    Upsert rows into ``layer2_speculative_population`` on primary key ``symbol``.
    """
    if not rows:
        return None
    sb = client or get_supabase_client()
    payload = [_normalize_current_row(dict(r)) for r in rows]
    last: Any = None
    for i in range(0, len(payload), batch_size):
        chunk = payload[i : i + batch_size]
        last = sb.table(TABLE_CURRENT).upsert(chunk, on_conflict="symbol").execute()
    return last


def append_population_history(
    rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_timestamp: str | None = None,
    client: Any | None = None,
    batch_size: int = 500,
) -> Any | None:
    """
    Insert historical snapshots into ``layer2_speculative_population_history``.

    All rows share the same ``snapshot_timestamp`` (default: now UTC ISO).
    """
    if not rows:
        return None
    ts = snapshot_timestamp or _utc_now_iso()
    sb = client or get_supabase_client()
    payload: list[dict[str, Any]] = []
    for r in rows:
        base = {k: r[k] for k in _HISTORY_INSERT_KEYS}
        if "symbol" not in base or not base["symbol"]:
            raise ValueError("history row needs symbol")
        base["snapshot_timestamp"] = ts
        base["source_cohorts"] = _normalize_cohorts(base.get("source_cohorts", []))
        base["speculative_attention_score"] = float(base["speculative_attention_score"])
        base["attention_state"] = str(base["attention_state"])
        for k in (
            "relative_volume_factor",
            "volatility_factor",
            "acceleration_factor",
            "ecosystem_multiplier",
        ):
            if k in base and base[k] is not None:
                base[k] = float(base[k])
        payload.append(base)
    last: Any = None
    for i in range(0, len(payload), batch_size):
        chunk = payload[i : i + batch_size]
        last = sb.table(TABLE_HISTORY).insert(chunk).execute()
    return last
