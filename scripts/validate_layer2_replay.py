#!/usr/bin/env python3
"""
Layer 2 behavioral calibration — canonical historical events vs replayed states.

Loads frozen JSON truth-sets under ``data/historical_speculative_events/``, runs the
same replay pipeline as ``scripts/replay_layer2.py`` (``layer2.replay_runner``),
and compares ``speculative_attention_state`` to labeled acceptable states.

Not ML training, not prediction optimization, not Layer 3 fragility.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from layer2.truth_set_validation import run_truth_set_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Layer 2 replay against historical event truth-sets.")
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "historical_speculative_events",
        help="Directory of event JSON definitions.",
    )
    parser.add_argument(
        "--event",
        action="append",
        dest="event_filters",
        default=None,
        metavar="STEM",
        help="Limit to event file stem(s) without .json (repeatable).",
    )
    parser.add_argument(
        "--lookback-override",
        type=int,
        default=None,
        metavar="N",
        help="Override lookback_calendar_days for all loaded events.",
    )
    parser.add_argument(
        "--feed-override",
        choices=("iex", "sip"),
        default=None,
        help="Override alpaca_feed from each event file.",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")

    events_dir = args.events_dir
    if not events_dir.is_dir():
        print(f"Events directory not found: {events_dir}", file=sys.stderr)
        sys.exit(1)

    stems = frozenset(args.event_filters) if args.event_filters else None
    all_results, replay_cache = run_truth_set_validation(
        events_dir=events_dir,
        event_filters=stems,
        lookback_override=args.lookback_override,
        feed_override=args.feed_override,
    )
    if not all_results:
        print("No event JSON files matched.", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 72)
    print("Layer 2 historical truth-set validation (behavioral calibration)")
    print("=" * 72)
    print(f"events_dir: {events_dir}")
    print(f"unique_replays_run: {len(replay_cache)}")
    print()

    verdict_counts = Counter(str(r["verdict"]) for r in all_results)
    print("--- aggregate verdict counts ---")
    for v, c in sorted(verdict_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {v}: {c}")

    print()
    print("--- per-check results (event_file_stem, replay_date, symbol) ---")
    for r in sorted(all_results, key=lambda x: (x["event_file_stem"], x["replay_date"], x["symbol"])):
        tags_s = ", ".join(r["behavioral_tags"])
        score_disp = "None" if r["observed_score"] is None else f"{float(r['observed_score']):.4g}"
        obs_disp = str(r["observed_speculative_state"])
        print(
            f"{r['verdict']:18} | {r['event_file_stem']:26} | {r['replay_date']} | {r['symbol']:6} | "
            f"obs={obs_disp:22} | score={score_disp:>8} | bars_clip={r['bars_after_clip']:7} | tags=[{tags_s}]"
        )
        if r["detail"]:
            print(f"         └─ {r['detail']}")

    print()
    print("--- replay diagnostics (cached runs) ---")
    for key in sorted(replay_cache.keys(), key=lambda k: (k[0].isoformat(), k[1], k[2])):
        art = replay_cache[key]
        rd, lb, fl = key
        scored_n = len(art.attention_df)
        layer1_syms = set(art.layer1_df["symbol"].astype(str).str.upper())
        layer1_diag = [d for d in art.diagnostics if str(d.get("symbol", "")).upper() in layer1_syms]
        skipped_l1 = sum(1 for d in layer1_diag if d.get("status") == "skipped")
        scored_l1 = sum(1 for d in layer1_diag if d.get("status") == "scored")
        print(
            f"{rd.isoformat()} lookback={lb} feed={fl} | layer1_rows={len(art.layer1_df)} "
            f"bars_clip={art.bars_rows_after_clip} symbols_scored_layer1={scored_l1} "
            f"symbols_skipped_layer1={skipped_l1} attention_rows={scored_n}"
        )


if __name__ == "__main__":
    main()
