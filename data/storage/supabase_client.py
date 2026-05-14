"""Thin Supabase (PostgREST) client for Layer 1 persistence."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from supabase import Client, create_client


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Return a cached Supabase client (reads ``SUPABASE_URL`` and ``SUPABASE_KEY``).

    Use a **service role** key for server-side upserts; never expose it in clients.
    """
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_KEY", "").strip()
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
    return create_client(url, key)


def reset_supabase_client_cache() -> None:
    """Clear the cached client (useful in tests)."""
    get_supabase_client.cache_clear()  # type: ignore[attr-defined]
