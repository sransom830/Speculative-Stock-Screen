#!/usr/bin/env python3
"""
Historical Layer 2 replay — speculative attention concentration only.

Recomputes speculative_attention_score / speculative_attention_state using bars
available through a chosen replay calendar date (UTC). Does not predict,
detect reversals, measure fragility, or optimize ML.

Hindsight caveats (explicit):
    • Bars are clipped so timestamps never fall after ``replay-date`` (UTC
      calendar day).
    • Layer 1 universe and ``source_cohorts`` come from the **current**
      Supabase snapshot unless you replace ``layer1_df`` downstream — membership
      and cohort tags are not time-traveled here (historical Layer 1 snapshots
      are not wired yet).

Deterministic given fixed inputs: sorted diagnostics aggregation and stable
sort keys for leaderboard output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.enums import DataFeed  # noqa: E402

from layer2.replay_runner import run_layer2_replay  # noqa: E402

DEFAULT_LOOKBACK_DAYS = 90
DEFAULT_TOP_N = 25
DEFAULT_BATTLEFIELD_COMPARE = ("GME", "MSTR", "PLTR", "RKLB", "CVNA")


def _parse_replay_date(s: str) -> date:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError("replay-date must be YYYY-MM-DD") from e


def _cohorts_cell(v: Any) -> str:
    if isinstance(v, list):
        return json.dumps(v)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v)


def _print_sorted_dict(title: str, d: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    for k in sorted(d.keys()):
        v = d[k]
        if k == "source_cohorts" and isinstance(v, list):
            print(f"{k}: {json.dumps(v)}")
        else:
            print(f"{k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Layer 2 speculative attention as of a historical date "
            "(bars through replay-date UTC only)."
        )
    )
    parser.add_argument(
        "--replay-date",
        type=_parse_replay_date,
        required=True,
        help="UTC calendar date through which bars may inform scores (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        metavar="N",
        help=f"Calendar-day span backward from replay-date for bar fetch (default {DEFAULT_LOOKBACK_DAYS}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        metavar="N",
        help=f"Leaderboard size by speculative_attention_score (default {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--compare-symbols",
        type=str,
        default="",
        help="Comma-separated tickers for detailed comparison blocks.",
    )
    parser.add_argument(
        "--battlefield",
        action="store_true",
        help=f"Merge default battlefield symbols into comparison: {', '.join(DEFAULT_BATTLEFIELD_COMPARE)}.",
    )
    parser.add_argument(
        "--feed",
        type=str,
        choices=("iex", "sip"),
        default="iex",
        help="Alpaca stock feed for daily bars (default iex).",
    )
    args = parser.parse_args()

    replay_day: date = args.replay_date
    today_utc = datetime.now(timezone.utc).date()
    if replay_day > today_utc:
        print(f"WARNING: replay-date {replay_day} is after UTC today {today_utc} — bars may be empty.", file=sys.stderr)

    compare_raw = [s.strip().upper() for s in args.compare_symbols.split(",") if s.strip()]
    compare_set = set(compare_raw)
    if args.battlefield:
        compare_set |= set(DEFAULT_BATTLEFIELD_COMPARE)
    compare_symbols = sorted(compare_set)

    load_dotenv(PROJECT_ROOT / ".env")

    feed = DataFeed.IEX if args.feed.lower() == "iex" else DataFeed.SIP

    layer1_stats: dict[str, Any] = {}
    artifacts = run_layer2_replay(
        replay_day=replay_day,
        lookback_calendar_days=args.lookback_days,
        feed=feed,
        layer1_load_stats_out=layer1_stats,
    )
    layer1_df = artifacts.layer1_df
    attention_df = artifacts.attention_df
    diagnostics = artifacts.diagnostics
    bars_df = artifacts.bars_df
    bars_rows_fetched = artifacts.bars_rows_fetched
    bars_unique_raw = artifacts.bars_unique_symbols_fetched
    bars_rows_after_clip = artifacts.bars_rows_after_clip
    bars_unique_after = artifacts.bars_unique_symbols_after_clip
    start_utc = artifacts.fetch_start_utc
    end_exclusive_utc = artifacts.fetch_end_exclusive_utc

    if layer1_df.empty:
        print("layer1_broad_universe is empty — nothing to replay.")
        print(f"layer1_fetch diagnostics: {layer1_stats}")
        return

    sym_col = layer1_df["symbol"].astype(str).str.upper()
    layer1_syms_upper = set(sym_col)

    layer1_diag = [
        d for d in diagnostics if str(d.get("symbol", "")).upper() in layer1_syms_upper
    ]
    symbols_scored = sum(1 for d in layer1_diag if d.get("status") == "scored")
    symbols_skipped_layer1 = sum(1 for d in layer1_diag if d.get("status") == "skipped")
    diag_non_layer1 = len(diagnostics) - len(layer1_diag)

    skip_reasons = Counter(
        str(d.get("skip_reason") or "unknown")
        for d in layer1_diag
        if d.get("status") == "skipped"
    )

    print()
    print("=" * 72)
    print("Layer 2 historical replay — speculative attention only")
    print("=" * 72)
    print(f"replay_date_utc (calendar): {replay_day.isoformat()}")
    print(f"lookback_calendar_days: {args.lookback_days}")
    print(f"bars_fetch_start_utc_inclusive: {start_utc.isoformat()}")
    print(f"bars_fetch_end_utc_exclusive: {end_exclusive_utc.isoformat()}")
    print(f"alpaca_feed: {artifacts.feed_label.upper()}")
    print()
    print("--- hindsight / data snapshot caveats ---")
    print(
        "Bars are clipped to timestamps with UTC date <= replay-date. "
        "Layer 1 universe and cohorts are the current Supabase snapshot "
        "(not time-traveled)."
    )
    print()
    print("--- replay diagnostics ---")
    print(f"layer1_rows_loaded: {len(layer1_df)}")
    print(f"layer1_fetch_strategy: {layer1_stats.get('layer1_fetch_pagination_strategy')}")
    print(f"layer1_fetch_pagination_chunks: {layer1_stats.get('layer1_fetch_pagination_chunks')}")
    print(f"bars_rows_fetched: {bars_rows_fetched}")
    print(f"bars_unique_symbols_fetched: {bars_unique_raw}")
    print(f"bars_rows_after_as_of_clip: {bars_rows_after_clip}")
    print(f"bars_unique_symbols_after_clip: {bars_unique_after}")
    print(f"diagnostic_records_total: {len(diagnostics)}")
    print(f"diagnostic_rows_symbols_only_in_bars_not_layer1: {diag_non_layer1}")
    print(f"symbols_scored_layer1: {symbols_scored}")
    print(f"symbols_skipped_layer1: {symbols_skipped_layer1}")
    if skip_reasons:
        print("skipped_reason_counts (layer1 symbols only):")
        for reason, cnt in sorted(skip_reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {reason}: {cnt}")

    if attention_df.empty:
        print("\n(no scored rows — widen lookback or check Layer 1 ∩ bars overlap)")
        return

    top_n = max(1, args.top_n)
    display_cols = [
        "symbol",
        "speculative_attention_score",
        "speculative_attention_state",
        "ecosystem_multiplier",
        "factor_relative_volume",
        "factor_dollar_volume_expansion",
        "factor_realized_vol_expansion",
        "factor_price_acceleration",
        "factor_momentum_persistence",
        "factor_participation_composite",
        "factor_acceleration_composite",
        "source_cohorts",
    ]
    cols_existing = [c for c in display_cols if c in attention_df.columns]
    leaderboard = attention_df.sort_values(
        ["speculative_attention_score", "symbol"],
        ascending=[False, True],
    ).head(top_n)
    print()
    print(f"--- top {top_n} by speculative_attention_score (replay as-of {replay_day}) ---")
    lb_show = leaderboard[cols_existing].copy()
    if "source_cohorts" in lb_show.columns:
        lb_show["source_cohorts"] = lb_show["source_cohorts"].map(_cohorts_cell)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(lb_show.to_string(index=False))

    if compare_symbols:
        bars_counts = (
            bars_df.groupby(bars_df["symbol"].astype(str).str.upper()).size().to_dict()
            if bars_rows_after_clip and "symbol" in bars_df.columns
            else {}
        )
        diag_by_sym = {str(d["symbol"]).upper(): d for d in diagnostics}

        print()
        print("=" * 72)
        print("Historical comparison symbols (battlefield / calibration)")
        print("=" * 72)

        att_by_sym = (
            attention_df.assign(_u=attention_df["symbol"].astype(str).str.upper()).set_index("_u")
            if not attention_df.empty
            else pd.DataFrame()
        )

        for sym_u in compare_symbols:
            print()
            print("#" * 72)
            print(f"# SYMBOL: {sym_u}")
            print("#" * 72)

            if sym_u not in layer1_syms_upper:
                print("NOT in current layer1_broad_universe snapshot — Layer 2 replay ignores this symbol.")
                print(f"bars_rows_through_replay_date (may still fetch): {bars_counts.get(sym_u, 0)}")
                continue

            row_l1 = layer1_df.loc[sym_col == sym_u].iloc[0]
            cohorts_l1 = row_l1.get("source_cohorts")
            print(f"In layer1_broad_universe: yes | source_cohorts (Layer1 row): {cohorts_l1}")
            print(f"bars_rows_through_replay_date: {bars_counts.get(sym_u, 0)}")

            d = diag_by_sym.get(sym_u)
            if d is None:
                print("INTERNAL: missing diagnostic row for Layer 1 symbol (unexpected).")
                continue

            _print_sorted_dict(f"{sym_u} — Layer 2 diagnostic record (replay)", dict(d))

            if d.get("status") == "skipped":
                print("\n>>> Replay outcome: SKIPPED")
                print(f">>> Reason: {d.get('skip_reason')}")
            elif sym_u in att_by_sym.index:
                row = att_by_sym.loc[sym_u]
                extra = row[
                    [
                        c
                        for c in (
                            "factor_participation_composite",
                            "factor_acceleration_composite",
                            "factor_volatility_composite",
                        )
                        if c in row.index
                    ]
                ]
                print("\n>>> Replay outcome: SCORED (attention_df row)")
                if len(extra):
                    print(extra.to_string())
            else:
                print("\n>>> Replay outcome: diagnostic scored path inconsistent with attention_df")


if __name__ == "__main__":
    main()
