#!/usr/bin/env python3
"""
Observational speculative saturation replay analysis — Layer 2 vs external narrative.

Joins market-native speculative concentration (Layer 2 replay) with curated external
attention snapshots and prints convergence diagnostics plus ranked saturation summaries.

This script does **not** predict reversals, classify tops/bearishness, apply Layer 3
fragility scoring, or encode trading logic.

Provide ``--preset`` (bundled historical examples) or explicit ``--replay-date`` +
``--external-snapshot``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    ExternalAttentionSnapshotRecord,
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


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _mean_present(vals: Sequence[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _normalized_mass_concentration(masses: Sequence[int]) -> float | None:
    """Herfindahl excess vs uniform baseline on raw masses → ``[0, 1]`` (same spirit as news module)."""
    if not masses:
        return None
    total = sum(int(m) for m in masses)
    if total <= 0:
        return None
    shares = [int(m) / total for m in masses]
    n = len(shares)
    if n <= 1:
        return 1.0 if shares and shares[0] > 0 else None
    hhi = sum(s * s for s in shares)
    h_min = 1.0 / n
    if hhi <= h_min:
        return 0.0
    return _clamp01((hhi - h_min) / (1.0 - h_min))


@dataclass(frozen=True)
class HistoricalReplayPreset:
    label: str
    replay_date: date
    lookback_calendar_days: int
    snapshot_relpath: str


PRESETS: dict[str, HistoricalReplayPreset] = {
    "gme-jan-2021": HistoricalReplayPreset(
        label="GME January 2021 (illustrative snapshot)",
        replay_date=date(2021, 1, 27),
        lookback_calendar_days=90,
        snapshot_relpath="data/external_attention_snapshots/gme_january_2021.json",
    ),
    "amc-jan-2021": HistoricalReplayPreset(
        label="AMC January 2021 (illustrative snapshot)",
        replay_date=date(2021, 1, 27),
        lookback_calendar_days=90,
        snapshot_relpath="data/external_attention_snapshots/amc_january_2021.json",
    ),
    "nvda-ai-narrative": HistoricalReplayPreset(
        label="NVDA AI narrative window (illustrative snapshot)",
        replay_date=date(2024, 5, 23),
        lookback_calendar_days=90,
        snapshot_relpath="data/external_attention_snapshots/nvda_ai_narrative.json",
    ),
    "mstr-crypto-reflexivity": HistoricalReplayPreset(
        label="MSTR crypto reflexivity window (illustrative snapshot)",
        replay_date=date(2021, 5, 19),
        lookback_calendar_days=90,
        snapshot_relpath="data/external_attention_snapshots/mstr_crypto_reflexivity.json",
    ),
}


@dataclass
class SymbolSaturationRow:
    symbol: str
    layer2_scored: bool
    participation: float | None
    volatility_expansion: float | None
    acceleration: float | None
    mention_velocity: float | None
    subreddit_breadth: float | None
    reddit_attention: float | None
    narrative_concentration: float | None
    thematic_clustering: float | None
    thematic_overlay: float | None
    headline_density: float | None
    news_attention: float | None
    thematic_mass_concentration_raw: float | None
    convergence: Mapping[str, float | None]
    saturation: Mapping[str, float | None]
    market_native_blend: float | None
    external_narrative_blend: float | None
    combined_saturation_index: float | None


def _collect_symbol_analysis(
    *,
    sym: str,
    layer2_scored: bool,
    row: Mapping[str, Any] | None,
    rec: ExternalAttentionSnapshotRecord | None,
    start_utc: datetime,
    end_exc: datetime,
    ext_provider: str,
    ext_blob: dict[str, tuple[RedditRawCounts | None, NewsRawCounts | None]],
) -> SymbolSaturationRow:
    part = _safe_float(row.get("factor_participation_composite")) if row is not None else None
    rv = _safe_float(row.get("factor_realized_vol_expansion")) if row is not None else None
    acc = _safe_float(row.get("factor_acceleration_composite")) if row is not None else None

    mention_vel = sub_breadth = reddit_att = narrative_conc = thematic_clust = headline_den = news_att = None
    thematic_overlay = None

    r_raw, n_raw = ext_blob[sym]
    win = ExternalAttentionWindow(sym, start_utc, end_exc, ext_provider)
    r_obs = compute_reddit_attention_observation(window=win, raw=r_raw) if r_raw is not None else None
    n_obs = compute_news_attention_observation(window=win, raw=n_raw) if n_raw is not None else None
    bundle = merge_external_enrichment(window=win, reddit=r_obs, news=n_obs)

    if r_obs:
        mention_vel = r_obs.factors.mention_velocity_factor
        sub_breadth = r_obs.factors.subreddit_breadth_factor
        reddit_att = r_obs.factors.reddit_attention_factor
    if n_obs:
        narrative_conc = n_obs.factors.narrative_concentration_factor
        thematic_clust = n_obs.factors.thematic_clustering_factor
        headline_den = n_obs.factors.headline_density_factor
        news_att = n_obs.factors.news_attention_factor
    thematic_overlay = bundle.thematic_attention_factor

    masses_raw: float | None = None
    if rec is not None:
        masses_raw = _normalized_mass_concentration([m for _, m in rec.thematic_topic_masses])

    # Signed gaps: external_factor − market_factor (positive ⇒ external narrative hotter).
    convergence: dict[str, float | None] = {
        "reddit_velocity_minus_participation": (
            _gap(mention_vel, part) if mention_vel is not None and part is not None else None
        ),
        "reddit_composite_minus_participation": (
            _gap(reddit_att, part) if reddit_att is not None and part is not None else None
        ),
        "subreddit_breadth_minus_participation": (
            _gap(sub_breadth, part) if sub_breadth is not None and part is not None else None
        ),
        "narrative_concentration_minus_volatility": (
            _gap(narrative_conc, rv) if narrative_conc is not None and rv is not None else None
        ),
        "thematic_clustering_minus_volatility": (
            _gap(thematic_clust, rv) if thematic_clust is not None and rv is not None else None
        ),
        "reddit_velocity_minus_acceleration": (
            _gap(mention_vel, acc) if mention_vel is not None and acc is not None else None
        ),
        "thematic_clustering_minus_acceleration": (
            _gap(thematic_clust, acc) if thematic_clust is not None and acc is not None else None
        ),
        "thematic_overlay_minus_acceleration": (
            _gap(thematic_overlay, acc) if thematic_overlay is not None and acc is not None else None
        ),
    }

    ext_blend_core = _mean_present([reddit_att, narrative_conc, thematic_clust])
    market_blend = _mean_present([part, rv, acc])

    external_participation_alignment = (
        _clamp01(1.0 - abs(float(reddit_att) - float(part)))
        if reddit_att is not None and part is not None
        else None
    )
    crowd_fixation_intensity = _mean_present([mention_vel, thematic_clust])

    accel_without_participation = (
        max(0.0, float(acc) - float(part)) if acc is not None and part is not None else None
    )
    participation_without_external_amp = (
        max(0.0, float(part) - float(ext_blend_core))
        if part is not None and ext_blend_core is not None
        else None
    )

    saturation: dict[str, float | None] = {
        "external_participation_alignment": external_participation_alignment,
        "narrative_concentration_dominance": narrative_conc,
        "crowd_fixation_intensity": crowd_fixation_intensity,
        "attention_share_concentration_thematic_mass": masses_raw,
        "acceleration_without_participation_excess": accel_without_participation,
        "participation_without_narrative_amplification_excess": participation_without_external_amp,
    }

    combined = (
        _clamp01(0.5 * float(market_blend) + 0.5 * float(ext_blend_core))
        if market_blend is not None and ext_blend_core is not None
        else None
    )

    return SymbolSaturationRow(
        symbol=sym,
        layer2_scored=layer2_scored,
        participation=part,
        volatility_expansion=rv,
        acceleration=acc,
        mention_velocity=mention_vel,
        subreddit_breadth=sub_breadth,
        reddit_attention=reddit_att,
        narrative_concentration=narrative_conc,
        thematic_clustering=thematic_clust,
        thematic_overlay=thematic_overlay,
        headline_density=headline_den,
        news_attention=news_att,
        thematic_mass_concentration_raw=masses_raw,
        convergence=convergence,
        saturation=saturation,
        market_native_blend=market_blend,
        external_narrative_blend=ext_blend_core,
        combined_saturation_index=combined,
    )


def _gap(external: float | None, market: float | None) -> float | None:
    if external is None or market is None:
        return None
    return float(external) - float(market)


def _print_convergence_block(row: SymbolSaturationRow) -> None:
    print("  --- convergence gaps (external − market-native factor) ---")
    for k, v in sorted(row.convergence.items()):
        if v is None:
            continue
        print(f"    {k}: {v:+.6f}")


def _print_saturation_block(row: SymbolSaturationRow) -> None:
    print("  --- speculative saturation diagnostics (observational) ---")
    for k, v in sorted(row.saturation.items()):
        if v is None:
            continue
        print(f"    {k}: {v:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Observational saturation replay: Layer 2 speculative concentration × external narrative factors."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Historical presets (bundled JSON under data/external_attention_snapshots/):\n"
            + "\n".join(f"  {name}: {PRESETS[name].label}" for name in PRESETS)
        ),
    )
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None)
    parser.add_argument("--replay-date", type=_parse_replay_date, default=None)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=90,
        metavar="N",
        help="Calendar-day span backward from replay-date (Layer 2 bar fetch). Default 90.",
    )
    parser.add_argument(
        "--external-snapshot",
        type=Path,
        default=None,
        help="Curated or legacy external-attention JSON (required unless --preset sets default path).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated tickers (default: all symbols listed in the snapshot).",
    )
    parser.add_argument(
        "--strict-external-alignment",
        action="store_true",
        help="Exit non-zero when curated snapshot metadata mismatches replay window.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        metavar="N",
        help="How many ranked symbols to print (default 25).",
    )
    parser.add_argument(
        "--feed",
        type=str,
        choices=("iex", "sip"),
        default="iex",
        help="Alpaca feed for Layer 2 replay.",
    )
    args = parser.parse_args()

    preset_meta: HistoricalReplayPreset | None = PRESETS[args.preset] if args.preset else None

    replay_day = args.replay_date
    ext_path = args.external_snapshot
    if preset_meta:
        replay_day = replay_day or preset_meta.replay_date
        ext_path = ext_path or (PROJECT_ROOT / preset_meta.snapshot_relpath)

    if replay_day is None:
        parser.error("Provide --replay-date or --preset (preset supplies replay-date).")
    if ext_path is None:
        parser.error("Provide --external-snapshot or --preset (preset supplies bundled snapshot path).")

    ext_path = ext_path.resolve()
    if not ext_path.is_file():
        print(f"External snapshot not found: {ext_path}", file=sys.stderr)
        sys.exit(1)

    user_targets = (
        sorted({s.strip().upper() for s in args.symbols.split(",") if s.strip()})
        if args.symbols
        else None
    )

    load_dotenv(PROJECT_ROOT / ".env")
    feed = DataFeed.IEX if args.feed.lower() == "iex" else DataFeed.SIP

    from layer2.replay_runner import run_layer2_replay  # noqa: E402

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

    try:
        ext_provider, ext_blob, curated_doc = load_external_snapshot_for_replay(ext_path)
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

    snap_syms = sorted(ext_blob.keys())
    targets = sorted(set(snap_syms) & set(user_targets)) if user_targets else snap_syms
    missing_requested = sorted(set(user_targets or []) - set(snap_syms))
    if missing_requested:
        for m in missing_requested:
            print(f"WARNING: requested symbol {m} absent from snapshot — skipped.", file=sys.stderr)

    layer1_syms = set(artifacts.layer1_df["symbol"].astype(str).str.upper())
    att_df = artifacts.attention_df

    print()
    print("=" * 78)
    print("Observational speculative saturation replay — convergence diagnostics")
    print("=" * 78)
    if preset_meta:
        print(f"preset: {args.preset} | {preset_meta.label}")
    print(f"replay_date (calendar): {replay_day.isoformat()}")
    print(f"lookback_calendar_days: {args.lookback_days}")
    print(f"interval_utc: [{start_utc.isoformat()}, {end_exc.isoformat()})")
    print(f"external_snapshot: {ext_path}")
    print(f"external_provider_slug: {ext_provider}")
    if curated_doc:
        print(f"snapshot_manifest: {json.dumps(json_snapshot_preview(curated_doc))}")
    print(f"snapshot_symbols: {snap_syms}")
    print(f"analysis_symbols: {targets}")
    print()
    print(
        "Diagnostics are descriptive only — not forecasts, trade signals, Layer 3 fragility, "
        "or reversal detection."
    )
    print()

    rows: list[SymbolSaturationRow] = []
    for sym in targets:
        row_map = layer2_attention_row(att_df, sym) if not att_df.empty else None
        layer2_ok = sym in layer1_syms and row_map is not None
        rec = curated_by_sym.get(sym)

        if sym not in ext_blob:
            print(f"# {sym}: no snapshot payload — skipping.")
            continue

        sar = _collect_symbol_analysis(
            sym=sym,
            layer2_scored=layer2_ok,
            row=row_map,
            rec=rec,
            start_utc=start_utc,
            end_exc=end_exc,
            ext_provider=ext_provider,
            ext_blob=ext_blob,
        )
        rows.append(sar)

        print()
        print("#" * 78)
        print(f"# SYMBOL: {sym}")
        print("#" * 78)
        print(f"  layer2_scored: {sar.layer2_scored}")
        print("  --- market-native (Layer 2 factors) ---")
        print(f"    participation_composite:      {sar.participation}")
        print(f"    realized_vol_expansion:       {sar.volatility_expansion}")
        print(f"    acceleration_composite:       {sar.acceleration}")
        print("  --- external normalized factors ---")
        print(f"    mention_velocity:             {sar.mention_velocity}")
        print(f"    subreddit_breadth:            {sar.subreddit_breadth}")
        print(f"    reddit_attention_composite:   {sar.reddit_attention}")
        print(f"    headline_density:             {sar.headline_density}")
        print(f"    narrative_concentration:      {sar.narrative_concentration}")
        print(f"    thematic_clustering:          {sar.thematic_clustering}")
        print(f"    news_attention_composite:     {sar.news_attention}")
        print(f"    thematic_overlay (breadth×conc×cluster blend): {sar.thematic_overlay}")
        _print_convergence_block(sar)
        _print_saturation_block(sar)
        print("  --- ranking blends ---")
        print(f"    market_native_blend (mean P,Rv,Acc): {sar.market_native_blend}")
        print(f"    external_core_blend (mean Reddit,NConc,TCluster): {sar.external_narrative_blend}")
        print(f"    combined_saturation_index:    {sar.combined_saturation_index}")

    ranked = sorted(
        [r for r in rows if r.combined_saturation_index is not None],
        key=lambda r: float(r.combined_saturation_index),
        reverse=True,
    )
    partial = [r for r in rows if r.combined_saturation_index is None]

    print()
    print("=" * 78)
    print(f"Top saturation candidates by combined_saturation_index (market∩external blend), top {args.top}")
    print("=" * 78)
    print(
        "(combined_saturation_index = 0.5×mean(participation,rv,acceleration) "
        "+ 0.5×mean(reddit_attention,narrative_concentration,thematic_clustering))"
    )
    print()

    hdr = (
        f"{'rank':>4}  {'symbol':^6}  {'combined':>10}  {'mkt_blend':>10}  {'ext_blend':>10}  "
        f"{'L2_scored':^9}  {'ext∩part_align':>14}"
    )
    print(hdr)
    print("-" * len(hdr))

    for i, r in enumerate(ranked[: max(0, args.top)], start=1):
        align = r.saturation.get("external_participation_alignment")
        align_s = f"{align:.4f}" if align is not None else "—"
        scored = "yes" if r.layer2_scored else "no"
        print(
            f"{i:4d}  {r.symbol:^6}  {float(r.combined_saturation_index):10.6f}  "
            f"{r.market_native_blend or 0.0:10.6f}  {r.external_narrative_blend or 0.0:10.6f}  "
            f"{scored:^9}  {align_s:>14}"
        )

    if partial:
        print()
        print(
            "Symbols excluded from ranking (missing Layer 2 row or external blend components): "
            + ", ".join(p.symbol for p in partial)
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
