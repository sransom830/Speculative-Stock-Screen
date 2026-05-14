"""Layer 1 Supabase universe fetch (pagination)."""

from __future__ import annotations

from layer1.supabase_universe import fetch_layer1_broad_universe_df


class _FakeResp:
    __slots__ = ("data",)

    def __init__(self, data: list):
        self.data = data


class _FakeQuery:
    __slots__ = ("_pages", "_call")

    def __init__(self, pages: list[list[dict]]) -> None:
        self._pages = pages
        self._call = 0

    def select(self, *args: object, **kwargs: object) -> _FakeQuery:
        return self

    def order(self, *args: object, **kwargs: object) -> _FakeQuery:
        return self

    def range(self, start: int, end: int) -> _FakeQuery:
        return self

    def execute(self) -> _FakeResp:
        batch = self._pages[self._call] if self._call < len(self._pages) else []
        self._call += 1
        return _FakeResp(batch)


class _FakeClient:
    __slots__ = ("_pages",)

    def __init__(self, pages: list[list[dict]]) -> None:
        self._pages = pages

    def table(self, name: str) -> _FakeQuery:
        assert name == "layer1_broad_universe"
        return _FakeQuery(self._pages)


def test_fetch_layer1_paginates_until_short_page() -> None:
    rows = [{"symbol": f"S{i}", "source_cohorts": []} for i in range(5)]
    pages = [rows[0:2], rows[2:4], rows[4:5]]
    stats: dict = {}
    df = fetch_layer1_broad_universe_df(client=_FakeClient(pages), load_stats_out=stats, chunk_size=2)
    assert len(df) == 5
    assert stats["layer1_fetch_total_rows"] == 5
    assert stats["layer1_fetch_pagination_chunks"] == 3
    assert stats["layer1_fetch_chunk_row_counts"] == [2, 2, 1]
    assert stats["layer1_fetch_chunk_size_requested"] == 2
    assert list(df["symbol"]) == [f"S{i}" for i in range(5)]


def test_fetch_layer1_exact_multiple_triggers_probe_page() -> None:
    rows = [{"symbol": f"A{i}", "source_cohorts": []} for i in range(4)]
    pages = [rows[0:2], rows[2:4], []]
    stats: dict = {}
    df = fetch_layer1_broad_universe_df(client=_FakeClient(pages), load_stats_out=stats, chunk_size=2)
    assert len(df) == 4
    assert stats["layer1_fetch_chunk_row_counts"] == [2, 2, 0]


def test_fetch_layer1_empty_table() -> None:
    stats: dict = {}
    df = fetch_layer1_broad_universe_df(client=_FakeClient([[]]), load_stats_out=stats, chunk_size=10)
    assert df.empty
    assert stats["layer1_fetch_total_rows"] == 0
    assert stats["layer1_fetch_pagination_chunks"] == 1
    assert stats["layer1_fetch_chunk_row_counts"] == [0]


def test_fetch_layer1_rejects_bad_chunk_size() -> None:
    try:
        fetch_layer1_broad_universe_df(client=_FakeClient([[]]), chunk_size=0)
    except ValueError as e:
        assert "chunk_size" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
