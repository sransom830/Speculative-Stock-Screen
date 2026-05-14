#!/usr/bin/env python3
"""
Historical replay: Layer 2 speculative attention + external narrative-attention overlay.

Runs market-native Layer 2 replay (bars through replay-date UTC), then applies
provider-supplied Reddit/news **count snapshots** from JSON to compute normalized
external factors. External signals are observability-only — **not** fused into
``speculative_attention_score``.

Snapshot formats (UTF-8 JSON):

**Curated (replay-aligned, preferred)** — ``schema_version`` ≥ 2 or ``symbols`` as an array.
See ``layer2/external_attention/snapshot_loader.py`` and examples under
``data/external_attention_snapshots/`` (e.g. ``gme_january_2021.json``).
Fields include ``replay_date``, ``lookback_calendar_days``, ``provider_slug``, and per-symbol
``reddit_mentions``, ``reddit_comment_count``, ``subreddit_count``, ``subreddit_list``,
``news_headline_count``, ``thematic_topic_masses``.

**Legacy v1** — ``symbols`` as an object keyed by ticker, with nested ``reddit_raw`` /
``news_raw`` count blobs (same shapes as Layer 2 enrichment models).

Either Reddit or news blocks may be omitted per symbol where the loader allows it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.enums import DataFeed  # noqa: E402

from layer2.external_attention import (  # noqa: E402
    ExternalAttentionWindow,
    NewsRawCounts,
    RedditRawCounts,
    compute_news_attention_observation,
    compute_reddit_attention_observation,
    merge_external_enrichment,
)
from layer2.external_attention.snapshot_loader import (  # noqa: E402
    json_snapshot_preview,
    load_external_snapshot_for_replay,
    validate_replay_window_alignment,
    validate_snapshot_records_consistency,
)
from layer2.truth_set_validation import attention_row as layer2_attention_row  # noqa: E402


def _parse_replay_date(s: str) -> date:
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError("replay-date must be YYYY-MM-DD") from e


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay Layer 2 + historical external narrative-attention snapshots (observational)."
    )
    parser.add_argument("--replay-date", type=_parse_replay_date, required=True)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        metavar="N",
        help="Calendar-day span backward from replay-date (Layer 2 bar fetch). Default 90.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        required=True,
        help="Comma-separated tickers to analyze (must intersect Layer 1 for Layer 2 scoring).",
    )
    parser.add_argument(
        "--external-snapshot",
        type=Path,
        default=None,
        help=(
            "JSON snapshot (curated replay-aligned schema v2+ under data/external_attention_snapshots/, "
            "or legacy v1 nested reddit_raw/news_raw)."
        ),
    )
    parser.add_argument(
        "--strict-external-alignment",
        action="store_true",
        help="Exit non-zero when curated snapshot replay_date / lookback / fetch interval mismatches Layer 2 replay.",
    )
    parser.add_argument(
        "--feed",
        type=str,
        choices=("iex", "sip"),
        default="iex",
        help="Alpaca feed for Layer 2 replay.",
    )
    args = parser.parse_args()

    targets = sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
    if not targets:
        print("Provide at least one symbol.", file=sys.stderr)
        sys.exit(1)

    load_dotenv(PROJECT_ROOT / ".env")

    feed = DataFeed.IEX if args.feed.lower() == "iex" else DataFeed.SIP

    # Deferred import keeps CLI `--help` usable without pulling Alpaca env errors early.
    from layer2.replay_runner import run_layer2_replay  # noqa: E402

    replay_day: date = args.replay_date
    today_utc = datetime.now(timezone.utc).date()
    if replay_day > today_utc:
        print(
            f"WARNING: replay-date {replay_day} is after UTC today {today_utc} — Layer 2 bars may be sparse.",
            file=sys.stderr,
        )

    artifacts = run_layer2_replay(
        replay_day=replay_day,
        lookback_calendar_days=args.lookback_days,
        feed=feed,
    )

    curated_doc = None
    ext_provider = "historical_snapshot_v1"
    ext_blob: dict[str, tuple[RedditRawCounts | None, NewsRawCounts | None]] = {}
    if args.external_snapshot is not None:
        if not args.external_snapshot.is_file():
            print(f"External snapshot not found: {args.external_snapshot}", file=sys.stderr)
            sys.exit(1)
        try:
            ext_provider, ext_blob, curated_doc = load_external_snapshot_for_replay(args.external_snapshot)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"External snapshot load failed: {e}", file=sys.stderr)
            sys.exit(1)

    start_utc = artifacts.fetch_start_utc
    end_exc = artifacts.fetch_end_exclusive_utc

    curated_by_sym = {r.symbol: r for r in curated_doc.symbols} if curated_doc else {}

    if curated_doc:
        align_errs = validate_replay_window_alignment(
            curated_doc,
            replay_day=replay_day,
            lookback_calendar_days=args.lookback_days,
            fetch_start_utc=start_utc,
            fetch_end_exclusive_utc=end_exc,
        )
        if align_errs:
            for msg in align_errs:
                print(f"WARNING external replay alignment: {msg}", file=sys.stderr)
            if args.strict_external_alignment:
                print("Strict external alignment requested — exiting.", file=sys.stderr)
                sys.exit(1)
        for w in validate_snapshot_records_consistency(curated_doc):
            print(f"WARNING snapshot consistency: {w}", file=sys.stderr)

    layer1_syms = set(artifacts.layer1_df["symbol"].astype(str).str.upper())
    att_df = artifacts.attention_df
    diag_by = {str(d["symbol"]).upper(): d for d in artifacts.diagnostics}

    print()
    print("=" * 72)
    print("Historical replay — Layer 2 speculative attention + external narrative overlay")
    print("=" * 72)
    print(f"replay_date_utc (calendar): {replay_day.isoformat()}")
    print(f"lookback_calendar_days: {args.lookback_days}")
    print(f"interval_utc: [{start_utc.isoformat()}, {end_exc.isoformat()})")
    print(f"alpaca_feed: {artifacts.feed_label.upper()}")
    print(f"requested_symbols: {targets}")
    print(f"external_snapshot: {args.external_snapshot or '(none — external section per symbol will be omitted)'}")
    if args.external_snapshot:
        print(f"external_provider_slug: {ext_provider}")
        if curated_doc:
            print(f"snapshot_manifest: {json.dumps(json_snapshot_preview(curated_doc))}")
    print()
    print("External attention is observational enrichment only — not integrated into Layer 2 scoring.")
    print()

    deltas_r_p: list[float] = []
    deltas_n_v: list[float] = []
    deltas_t_a: list[float] = []

    for sym in targets:
        print()
        print("#" * 72)
        print(f"# SYMBOL: {sym}")
        print("#" * 72)

        if sym not in layer1_syms:
            print("Layer 1: NOT in current universe snapshot — Layer 2 does not score this symbol.")
            if sym in ext_blob:
                print("External snapshot: present for symbol (counts not shown here without Layer 2 join).")
            continue

        row = layer2_attention_row(att_df, sym) if not att_df.empty else None
        d = diag_by.get(sym)

        print("--- Layer 2 (market-native) ---")
        if row is not None:
            print(f"  speculative_attention_score: {row.get('speculative_attention_score')}")
            print(f"  speculative_attention_state: {row.get('speculative_attention_state')}")
            part = _safe_float(row.get("factor_participation_composite"))
            rv = _safe_float(row.get("factor_realized_vol_expansion"))
            acc = _safe_float(row.get("factor_acceleration_composite"))
            print(f"  factor_participation_composite: {part}")
            print(f"  factor_realized_vol_expansion: {rv}")
            print(f"  factor_acceleration_composite: {acc}")
            print(f"  ecosystem_multiplier: {_safe_float(row.get('ecosystem_multiplier'))}")
            sc = row.get("source_cohorts")
            if isinstance(sc, list):
                print(f"  source_cohorts: {json.dumps(sc)}")
            else:
                print(f"  source_cohorts: {sc}")
        elif d is not None and d.get("status") == "skipped":
            print(f"  status: skipped | reason: {d.get('skip_reason')}")
        else:
            print("  (no scored row — check diagnostics / bars overlap)")

        print("--- External narrative attention (normalized) ---")
        if sym not in ext_blob:
            print("  (no snapshot entry for this symbol — supply JSON under --external-snapshot)")
        else:
            r_raw, n_raw = ext_blob[sym]
            win = ExternalAttentionWindow(sym, start_utc, end_exc, ext_provider)
            r_obs = (
                compute_reddit_attention_observation(window=win, raw=r_raw)
                if r_raw is not None
                else None
            )
            n_obs = (
                compute_news_attention_observation(window=win, raw=n_raw)
                if n_raw is not None
                else None
            )
            bundle = merge_external_enrichment(window=win, reddit=r_obs, news=n_obs)
            if r_obs:
                print(f"  mention_velocity_factor:        {r_obs.factors.mention_velocity_factor:.6f}")
                print(f"  subreddit_breadth_factor:       {r_obs.factors.subreddit_breadth_factor:.6f}")
                print(f"  comment_velocity_factor:        {r_obs.factors.comment_velocity_factor:.6f}")
                print(f"  reddit_attention_factor:        {r_obs.factors.reddit_attention_factor:.6f}")
            if n_obs:
                print(f"  headline_density_factor:        {n_obs.factors.headline_density_factor:.6f}")
                print(f"  narrative_concentration_factor: {n_obs.factors.narrative_concentration_factor:.6f}")
                print(f"  thematic_clustering_factor:     {n_obs.factors.thematic_clustering_factor:.6f}")
                print(f"  news_attention_factor:          {n_obs.factors.news_attention_factor:.6f}")
            print(f"  thematic_attention_factor (overlay): {bundle.thematic_attention_factor:.6f}")

            rec = curated_by_sym.get(sym)
            if rec:
                tpm = [{"topic": t, "mass": m} for t, m in rec.thematic_topic_masses]
                print("  --- curated snapshot observables (replay-aligned file) ---")
                print(
                    f"  reddit_mentions (window | prior): {rec.reddit_mentions} | {rec.reddit_mentions_prior}"
                )
                print(
                    f"  reddit_comment_count (window | prior): "
                    f"{rec.reddit_comment_count} | {rec.reddit_comment_count_prior}"
                )
                print(f"  subreddit_count / len(list): {rec.subreddit_count} / {len(rec.subreddit_list)}")
                print(f"  subreddit_list: {json.dumps(list(rec.subreddit_list))}")
                print(
                    f"  news_headline_count (window | prior): "
                    f"{rec.news_headline_count} | {rec.news_headline_count_prior}"
                )
                print(f"  thematic_topic_masses: {json.dumps(tpm)}")

        print("--- comparative diagnostics (convergence / divergence) ---")
        if sym not in ext_blob or row is None:
            print("  (needs both Layer2 scored row and external snapshot for numeric comparison)")
            continue

        r_raw, n_raw = ext_blob[sym]
        win = ExternalAttentionWindow(sym, start_utc, end_exc, ext_provider)
        r_obs = compute_reddit_attention_observation(window=win, raw=r_raw) if r_raw is not None else None
        n_obs = compute_news_attention_observation(window=win, raw=n_raw) if n_raw is not None else None
        bundle = merge_external_enrichment(window=win, reddit=r_obs, news=n_obs)

        part = _safe_float(row.get("factor_participation_composite"))
        rv = _safe_float(row.get("factor_realized_vol_expansion"))
        acc = _safe_float(row.get("factor_acceleration_composite"))

        lines: list[str] = []

        reddit_f = r_obs.factors.reddit_attention_factor if r_obs is not None else None
        news_conc = n_obs.factors.narrative_concentration_factor if n_obs is not None else None
        thematic_f = bundle.thematic_attention_factor

        if reddit_f is not None and part is not None:
            del_rp = reddit_f - part
            deltas_r_p.append(abs(del_rp))
            lines.append(
                f"reddit_attention_factor ({reddit_f:.4f}) vs Layer2_participation_composite ({part:.4f}) → Δ={del_rp:+.4f}"
            )
        if news_conc is not None and rv is not None:
            del_nv = news_conc - rv
            deltas_n_v.append(abs(del_nv))
            lines.append(
                f"news_narrative_concentration ({news_conc:.4f}) vs Layer2_realized_vol_expansion ({rv:.4f}) → Δ={del_nv:+.4f}"
            )
        if thematic_f is not None and acc is not None:
            del_ta = thematic_f - acc
            deltas_t_a.append(abs(del_ta))
            lines.append(
                f"thematic_attention_factor ({thematic_f:.4f}) vs Layer2_acceleration_composite ({acc:.4f}) → Δ={del_ta:+.4f}"
            )

        if not lines:
            print("  (insufficient overlapping series for automated comparison)")
        else:
            for ln in sorted(lines):
                print(f"  • {ln}")

    if args.external_snapshot and deltas_r_p:
        print()
        print("--- aggregate |Δ| across symbols with both Reddit+participation data ---")
        print(f"  mean_abs(reddit_vs_participation): {sum(deltas_r_p) / len(deltas_r_p):.6f} (n={len(deltas_r_p)})")
    if args.external_snapshot and deltas_n_v:
        print(f"  mean_abs(news_concentration_vs_rv): {sum(deltas_n_v) / len(deltas_n_v):.6f} (n={len(deltas_n_v)})")
    if args.external_snapshot and deltas_t_a:
        print(f"  mean_abs(thematic_vs_acceleration): {sum(deltas_t_a) / len(deltas_t_a):.6f} (n={len(deltas_t_a)})")


if __name__ == "__main__":
    main()
