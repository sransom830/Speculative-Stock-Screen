"""Load Layer 1 broad universe from Supabase (read-only)."""

from __future__ import annotations

import json
from typing import Any, MutableMapping

import pandas as pd

from data.storage.supabase_client import get_supabase_client

TABLE_NAME = "layer1_broad_universe"

# PostgREST range is inclusive; (chunk_size) rows per request.
_DEFAULT_PAGE_SIZE = 1000


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


def fetch_layer1_broad_universe_df(
    *,
    client: Any | None = None,
    load_stats_out: MutableMapping[str, Any] | None = None,
    chunk_size: int = _DEFAULT_PAGE_SIZE,
) -> pd.DataFrame:
    """
    Return all rows from ``layer1_broad_universe`` as a DataFrame.

    Paginates until a short page is returned (full universe — not limited to
    PostgREST's default max rows per request).

    Rows are ordered by ``symbol`` ascending so pagination is deterministic.

    Normalizes ``source_cohorts`` to ``list[str]`` when possible.

    Parameters
    ----------
    load_stats_out :
        If provided, populated in-place with fetch diagnostics:

        - ``layer1_fetch_total_rows`` — final row count
        - ``layer1_fetch_pagination_chunks`` — number of range requests made
        - ``layer1_fetch_chunk_row_counts`` — rows returned per chunk (in order)
        - ``layer1_fetch_chunk_size_requested`` — ``chunk_size`` used
    chunk_size :
        PostgREST page size (max rows per request). Must be >= 1.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    sb = client or get_supabase_client()
    rows: list[dict[str, Any]] = []
    chunk_row_counts: list[int] = []
    start = 0
    pagination_chunks = 0

    while True:
        end = start + chunk_size - 1
        resp = (
            sb.table(TABLE_NAME)
            .select("*")
            .order("symbol", desc=False)
            .range(start, end)
            .execute()
        )
        batch = resp.data or []
        pagination_chunks += 1
        chunk_row_counts.append(len(batch))
        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        start += chunk_size

    if load_stats_out is not None:
        load_stats_out["layer1_fetch_total_rows"] = len(rows)
        load_stats_out["layer1_fetch_pagination_chunks"] = pagination_chunks
        load_stats_out["layer1_fetch_chunk_row_counts"] = list(chunk_row_counts)
        load_stats_out["layer1_fetch_chunk_size_requested"] = chunk_size

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "source_cohorts" in df.columns:
        df["source_cohorts"] = df["source_cohorts"].map(_parse_cohorts)
    return df
