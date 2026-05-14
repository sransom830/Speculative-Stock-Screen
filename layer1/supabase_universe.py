"""Load Layer 1 broad universe from Supabase (read-only)."""

from __future__ import annotations

import json
from typing import Any, MutableMapping

import pandas as pd

from data.storage.supabase_client import get_supabase_client

TABLE_NAME = "layer1_broad_universe"

# Rows per Supabase request (server max-rows still caps payload size per call).
_DEFAULT_PAGE_SIZE = 1000

_KEYSET_STRATEGY = "keyset_symbol_gt"


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

    Loads the **full** table using deterministic **keyset** pagination on
    ``symbol`` (``WHERE symbol > cursor ORDER BY symbol ASC LIMIT chunk_size``).
    This avoids ``offset``/``.range()`` pagination, which maps to PostgREST
    ``offset`` query params and can stop returning rows after the first chunk on
    some deployments despite more rows existing server-side.

    Normalizes ``source_cohorts`` to ``list[str]`` when possible.

    Parameters
    ----------
    load_stats_out :
        If provided, populated in-place with fetch diagnostics:

        - ``layer1_fetch_total_rows`` — final row count
        - ``layer1_fetch_pagination_chunks`` — number of chunk requests made
        - ``layer1_fetch_chunk_row_counts`` — rows returned per chunk (in order)
        - ``layer1_fetch_chunk_size_requested`` — ``chunk_size`` used
        - ``layer1_fetch_pagination_strategy`` — ``\"keyset_symbol_gt\"``
    chunk_size :
        Maximum rows per request. Must be >= 1.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    sb = client or get_supabase_client()
    rows: list[dict[str, Any]] = []
    chunk_row_counts: list[int] = []
    pagination_chunks = 0
    cursor_after_symbol: Any | None = None
    prev_chunk_tail_symbol: Any | None = None

    while True:
        q = sb.table(TABLE_NAME).select("*")
        if cursor_after_symbol is not None:
            q = q.gt("symbol", cursor_after_symbol)
        resp = q.order("symbol", desc=False).limit(chunk_size).execute()
        batch = resp.data or []
        pagination_chunks += 1
        chunk_row_counts.append(len(batch))
        rows.extend(batch)
        if len(batch) < chunk_size:
            break
        tail_symbol = batch[-1]["symbol"]
        if tail_symbol == prev_chunk_tail_symbol:
            raise RuntimeError(
                "layer1_broad_universe pagination stalled: got another full chunk "
                f"but tail symbol {tail_symbol!r} did not advance — refusing infinite loop."
            )
        prev_chunk_tail_symbol = tail_symbol
        cursor_after_symbol = tail_symbol

    if load_stats_out is not None:
        load_stats_out["layer1_fetch_total_rows"] = len(rows)
        load_stats_out["layer1_fetch_pagination_chunks"] = pagination_chunks
        load_stats_out["layer1_fetch_chunk_row_counts"] = list(chunk_row_counts)
        load_stats_out["layer1_fetch_chunk_size_requested"] = chunk_size
        load_stats_out["layer1_fetch_pagination_strategy"] = _KEYSET_STRATEGY

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "source_cohorts" in df.columns:
        df["source_cohorts"] = df["source_cohorts"].map(_parse_cohorts)
    return df
