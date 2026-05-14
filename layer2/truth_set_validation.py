"""
Historical truth-set validation: event JSON definitions vs ``run_layer2_replay``.

Behavioral calibration only — shared by ``validate_layer2_replay`` and analysis tools.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from alpaca.data.enums import DataFeed

from layer2.models import SpeculativeAttentionState
from layer2.replay_runner import Layer2ReplayArtifacts, run_layer2_replay

VALID_STATES = frozenset(s.value for s in SpeculativeAttentionState)


def parse_truth_replay_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def parse_truth_feed(name: str) -> DataFeed:
    n = name.strip().lower()
    if n == "sip":
        return DataFeed.SIP
    return DataFeed.IEX


def load_event_files(events_dir: Path, stems: frozenset[str] | None) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(events_dir.glob("*.json")):
        stem = path.stem
        if stems is not None and stem not in stems:
            continue
        with path.open(encoding="utf-8") as f:
            payload = json.load(f)
        out.append((stem, payload))
    return out


def validate_event_payload(payload: dict[str, Any], path_hint: str) -> None:
    required_top = ("event_name", "symbols", "event_dates", "expectations")
    for k in required_top:
        if k not in payload:
            raise ValueError(f"{path_hint}: missing top-level key {k!r}")
    if not isinstance(payload["symbols"], list):
        raise ValueError(f"{path_hint}: symbols must be a list")
    if not isinstance(payload["event_dates"], list):
        raise ValueError(f"{path_hint}: event_dates must be a list")
    exps = payload["expectations"]
    if not isinstance(exps, list) or not exps:
        raise ValueError(f"{path_hint}: expectations must be a non-empty list")
    for i, row in enumerate(exps):
        if not isinstance(row, dict):
            raise ValueError(f"{path_hint}: expectation[{i}] must be object")
        for rk in ("replay_date", "symbol", "expected_speculative_states", "behavioral_tags"):
            if rk not in row:
                raise ValueError(f"{path_hint}: expectation[{i}] missing {rk!r}")
        states = row["expected_speculative_states"]
        if not isinstance(states, list) or not states:
            raise ValueError(f"{path_hint}: expectation[{i}] expected_speculative_states must be non-empty list")
        for st in states:
            if st not in VALID_STATES:
                raise ValueError(
                    f"{path_hint}: invalid state {st!r}; must be one of {sorted(VALID_STATES)}"
                )
        tags = row["behavioral_tags"]
        if not isinstance(tags, list):
            raise ValueError(f"{path_hint}: expectation[{i}] behavioral_tags must be a list")


def attention_row(attention_df: pd.DataFrame, sym_upper: str) -> pd.Series | None:
    if attention_df.empty:
        return None
    sub = attention_df.loc[attention_df["symbol"].astype(str).str.upper() == sym_upper]
    if sub.empty:
        return None
    return sub.iloc[0]


def evaluate_expectation(
    *,
    event_name: str,
    exp: dict[str, Any],
    artifacts: Layer2ReplayArtifacts,
) -> dict[str, Any]:
    sym_u = str(exp["symbol"]).strip().upper()
    replay_day = artifacts.replay_day
    expected = frozenset(str(x) for x in exp["expected_speculative_states"])
    tags = list(exp["behavioral_tags"])

    layer1_syms = set(artifacts.layer1_df["symbol"].astype(str).str.upper())
    diag_by_sym = {str(d["symbol"]).upper(): d for d in artifacts.diagnostics}
    diag = diag_by_sym.get(sym_u)

    row_att = attention_row(artifacts.attention_df, sym_u)

    observed_state: str | None = None
    observed_score: Any = None
    if row_att is not None:
        observed_state = str(row_att.get("speculative_attention_state"))
        observed_score = row_att.get("speculative_attention_score")

    verdict = "UNKNOWN"
    detail = ""

    if sym_u not in layer1_syms:
        verdict = "SKIP_NOT_IN_LAYER1"
        detail = "symbol missing from current Layer 1 universe snapshot"
    elif diag is None:
        verdict = "SKIP_INTERNAL"
        detail = "no diagnostic row for symbol"
    elif diag.get("status") == "skipped":
        verdict = "SKIP_NOT_SCORED"
        detail = str(diag.get("skip_reason") or "skipped")
    elif observed_state is None:
        verdict = "SKIP_INTERNAL"
        detail = "diagnostic not skipped but no attention_df row"
    elif observed_state in expected:
        verdict = "PASS"
    else:
        verdict = "FAIL"
        detail = f"observed {observed_state!r} not in expected set {sorted(expected)}"

    return {
        "event_name": event_name,
        "replay_date": replay_day.isoformat(),
        "symbol": sym_u,
        "expected_speculative_states": sorted(expected),
        "observed_speculative_state": observed_state,
        "observed_score": observed_score,
        "verdict": verdict,
        "detail": detail,
        "behavioral_tags": tags,
        "bars_after_clip": artifacts.bars_rows_after_clip,
    }


def run_truth_set_validation(
    *,
    events_dir: Path,
    event_filters: frozenset[str] | None,
    lookback_override: int | None,
    feed_override: str | None,
) -> tuple[list[dict[str, Any]], dict[tuple[date, int, str], Layer2ReplayArtifacts]]:
    """
    Load event JSON files, run cached replays, return (result rows, replay_cache).

    Each result row includes ``event_file_stem``, ``lookback_calendar_days``, ``alpaca_feed``.
    """
    loaded = load_event_files(events_dir, event_filters)
    if not loaded:
        return [], {}

    replay_cache: dict[tuple[date, int, str], Layer2ReplayArtifacts] = {}

    def get_artifacts(replay_day: date, lookback: int, feed: DataFeed) -> Layer2ReplayArtifacts:
        key = (replay_day, lookback, feed.value)
        if key not in replay_cache:
            replay_cache[key] = run_layer2_replay(
                replay_day=replay_day,
                lookback_calendar_days=lookback,
                feed=feed,
            )
        return replay_cache[key]

    all_results: list[dict[str, Any]] = []

    for stem, payload in loaded:
        validate_event_payload(payload, stem + ".json")
        event_name = str(payload["event_name"])
        lookback = int(lookback_override or payload.get("lookback_calendar_days") or 90)
        feed_name = feed_override or payload.get("alpaca_feed") or "iex"
        feed = parse_truth_feed(str(feed_name))

        by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
        for exp in payload["expectations"]:
            rd = parse_truth_replay_date(str(exp["replay_date"]))
            by_date[rd].append(exp)

        for replay_day in sorted(by_date.keys()):
            artifacts = get_artifacts(replay_day, lookback, feed)
            for exp in sorted(by_date[replay_day], key=lambda r: str(r["symbol"]).upper()):
                row = evaluate_expectation(event_name=event_name, exp=exp, artifacts=artifacts)
                row["event_file_stem"] = stem
                row["lookback_calendar_days"] = lookback
                row["alpaca_feed"] = feed.value
                all_results.append(row)

    return all_results, replay_cache


def replay_cache_key(row: Mapping[str, Any]) -> tuple[date, int, str]:
    """Lookup key into ``replay_cache`` from a validation result row."""
    return (
        parse_truth_replay_date(str(row["replay_date"])),
        int(row["lookback_calendar_days"]),
        str(row["alpaca_feed"]),
    )