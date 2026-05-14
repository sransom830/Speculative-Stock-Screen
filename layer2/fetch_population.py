"""Read Layer 2 speculative population snapshot from Supabase (read-only)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from data.storage.supabase_client import get_supabase_client

from layer2.persistence import TABLE_CURRENT

_CHUNK = 1000


def _parse_cohorts(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            return []
    return []


def fetch_layer2_speculative_population_df(*, client: Any | None = None) -> pd.DataFrame:
    """
    Load all rows from ``layer2_speculative_population`` (paginated).

    ``source_cohorts`` normalized to ``list[str]``.
    """
    sb = client or get_supabase_client()
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        resp = (
            sb.table(TABLE_CURRENT)
            .select("*")
            .range(start, start + _CHUNK - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < _CHUNK:
            break
        start += _CHUNK
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "source_cohorts" in df.columns:
        df["source_cohorts"] = df["source_cohorts"].map(_parse_cohorts)
    return df
