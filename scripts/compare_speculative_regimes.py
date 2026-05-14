#!/usr/bin/env python3
"""
Observational cross-regime comparison — Layer 2 × external narrative snapshots.

Runs multiple historical replay presets side-by-side and prints comparison tables plus
deterministic archetype tags. No prediction, reversal detection, Layer 3 scoring, or
trading logic.

Reuses saturation row construction from ``analyze_speculative_saturation.py`` so
metrics match that script exactly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.enums import DataFeed  # noqa: E402

from layer2.external_attention.snapshot_loader import (  # noqa: E402
    json_snapshot_preview,
    load_external_snapshot_for_replay,
    validate_replay_window_alignment,
    validate_snapshot_records_consistency,
)
from layer2.truth_set_validation import attention_row as layer2_attention_row  # noqa: E402


def _load_analyze_saturation_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "analyze_speculative_saturation.py"
    name = "_analyze_speculative_saturation"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load saturation analysis module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_SAT = _load_analyze_saturation_module()
PRESETS: Mapping[str, Any] = _SAT.PRESETS
_collect_symbol_analysis = _SAT._collect_symbol_analysis
SymbolSaturationRow = _SAT.SymbolSaturationRow


DEFAULT_COMPARE_PRESETS: tuple[str, ...] = (
    "gme-jan-2021",
    "nvda-ai-narrative",
    "mstr-crypto-reflexivity",
)


@dataclass(frozen=True)
class RegimeBundle:
    preset_key: str
    label: str
    replay_date_iso: str
    symbol: str
    row: SymbolSaturationRow


def _fmt_cell(x: float | None, ndigits: int = 6) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{ndigits}f}"


def _primary_symbol(rows: Sequence[SymbolSaturationRow]) -> SymbolSaturationRow | None:
    if not rows:
        return None
    return sorted(rows, key=lambda r: r.symbol)[0]


def _metric_cells(row: SymbolSaturationRow | None) -> dict[str, float | None]:
    if row is None:
        return {k: None for k in _METRIC_KEYS}
    align = row.saturation.get("external_participation_alignment")
    fixation = row.saturation.get("crowd_fixation_intensity")
    return {
        "participation": row.participation,
        "realized_volatility_expansion": row.volatility_expansion,
        "acceleration": row.acceleration,
        "reddit_attention_velocity": row.mention_velocity,
        "subreddit_breadth": row.subreddit_breadth,
        "thematic_clustering": row.thematic_clustering,
        "external_participation_alignment": align,
        "crowd_fixation_intensity": fixation,
        "combined_saturation_index": row.combined_saturation_index,
    }


_METRIC_KEYS: tuple[str, ...] = (
    "participation",
    "realized_volatility_expansion",
    "acceleration",
    "reddit_attention_velocity",
    "subreddit_breadth",
    "thematic_clustering",
    "external_participation_alignment",
    "crowd_fixation_intensity",
    "combined_saturation_index",
)


def _median_nonempty(vals: Sequence[float]) -> float | None:
    xs = sorted(vals)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])


def _max_key_for_metric(
    bundles: Sequence[RegimeBundle],
    metric: str,
) -> str | None:
    best_k: str | None = None
    best_v: float | None = None
    for b in bundles:
        cells = _metric_cells(b.row)
        v = cells.get(metric)
        if v is None:
            continue
        if best_v is None or float(v) > float(best_v):
            best_v = float(v)
            best_k = b.preset_key
    return best_k


def archetype_tags_for_bundle(bundle: RegimeBundle, all_bundles: Sequence[RegimeBundle]) -> list[str]:
    """Deterministic observational labels — threshold + batch-relative rules."""
    row = bundle.row
    tags: list[str] = []
    if row is None:
        return tags

    cells_by_preset = {b.preset_key: _metric_cells(b.row) for b in all_bundles}

    parts = [cells_by_preset[k]["participation"] for k in cells_by_preset]
    parts_nn = [float(x) for x in parts if x is not None]
    med_part = _median_nonempty(parts_nn)

    aligns = [
        cells_by_preset[k]["external_participation_alignment"]
        for k in cells_by_preset
        if cells_by_preset[k]["external_participation_alignment"] is not None
    ]
    aligns_nn = [float(x) for x in aligns]
    med_align = _median_nonempty(aligns_nn)

    combineds = [
        cells_by_preset[k]["combined_saturation_index"]
        for k in cells_by_preset
        if cells_by_preset[k]["combined_saturation_index"] is not None
    ]
    combineds_nn = [float(x) for x in combineds]
    med_combined = _median_nonempty(combineds_nn)

    ncs = [b.row.narrative_concentration for b in all_bundles if b.row is not None]
    ncs_nn = [float(x) for x in ncs if x is not None]
    max_nc = max(ncs_nn) if ncs_nn else None

    tmasses = [
        b.row.thematic_mass_concentration_raw for b in all_bundles if b.row is not None
    ]
    tmasses_nn = [float(x) for x in tmasses if x is not None]
    max_tm = max(tmasses_nn) if tmasses_nn else None

    part = row.participation
    rv = row.volatility_expansion
    acc = row.acceleration
    alignment = row.saturation.get("external_participation_alignment")
    combined = row.combined_saturation_index
    nc = row.narrative_concentration
    tm = row.thematic_mass_concentration_raw

    # Participation leads market-native stack vs acceleration/vol when all defined.
    if (
        part is not None
        and acc is not None
        and rv is not None
        and float(part) >= float(acc)
        and float(part) >= float(rv)
        and med_part is not None
        and float(part) >= float(med_part)
    ):
        tags.append("participation_dominant_reflexivity")

    # External Reddit stack and market stack both elevated and aligned vs batch.
    if (
        alignment is not None
        and combined is not None
        and med_align is not None
        and med_combined is not None
        and float(alignment) >= float(med_align)
        and float(combined) >= float(med_combined)
        and float(alignment) >= 0.65
    ):
        tags.append("narrative_synchronized_saturation")

    # Acceleration and vol expansion sit at or above participation (co-expanded activity).
    if (
        part is not None
        and acc is not None
        and rv is not None
        and float(acc) >= float(part)
        and float(rv) >= float(part)
    ):
        tags.append("coherent_leverage_reflexivity")

    # Thematic / news structure concentrates mass at the top of the batch distribution.
    if nc is not None and max_nc is not None and float(nc) >= float(max_nc):
        tags.append("thematic_concentration_dominance")
    elif tm is not None and max_tm is not None and float(tm) >= float(max_tm):
        tags.append("thematic_concentration_dominance")

    return sorted(set(tags))


def _print_cross_regime_table(bundles: Sequence[RegimeBundle], title: str) -> None:
    keys = [b.preset_key for b in bundles]
    col_w = max(12, max(len(k) for k in keys) if keys else 12)
    metric_w = 38

    print()
    print("=" * 88)
    print(title)
    print("=" * 88)

    header = f"{'metric':<{metric_w}}" + "".join(f"  {k:>{col_w}}" for k in keys)
    print(header)
    print("-" * len(header))

    for mk in _METRIC_KEYS:
        row_cells: list[str] = []
        for b in bundles:
            m = _metric_cells(b.row).get(mk)
            row_cells.append(_fmt_cell(m))
        label = mk
        line = f"{label:<{metric_w}}" + "".join(f"  {c:>{col_w}}" for c in row_cells)
        print(line)


def _print_archetype_block(bundles: Sequence[RegimeBundle]) -> None:
    print()
    print("=" * 88)
    print("Comparative archetype diagnostics (observational tags)")
    print("=" * 88)
    print(
        "Tags use batch-relative medians/maxima across this run only — not forecasts "
        "or Layer 3 instability classifications."
    )
    print()

    for b in sorted(bundles, key=lambda x: x.preset_key):
        tags = archetype_tags_for_bundle(b, bundles)
        tag_str = "; ".join(tags) if tags else "(none matched)"
        print(f"  [{b.preset_key}] {b.symbol} @ {b.replay_date_iso}")
        print(f"      archetypes: {tag_str}")

    print()
    print("--- Leaders per metric (preset key with highest numeric value among non-null) ---")
    for mk in _METRIC_KEYS:
        leader = _max_key_for_metric(bundles, mk)
        print(f"  {mk}: {leader or '—'}")


def run_one_preset(
    preset_key: str,
    *,
    feed: DataFeed,
    strict_alignment: bool,
) -> tuple[RegimeBundle, Any | None] | None:
    if preset_key not in PRESETS:
        print(f"ERROR: unknown preset {preset_key!r}", file=sys.stderr)
        return None

    cfg = PRESETS[preset_key]
    replay_day = cfg.replay_date
    lookback = cfg.lookback_calendar_days
    ext_path = (PROJECT_ROOT / cfg.snapshot_relpath).resolve()

    if not ext_path.is_file():
        print(f"ERROR: snapshot missing for {preset_key}: {ext_path}", file=sys.stderr)
        return None

    from layer2.replay_runner import run_layer2_replay  # noqa: E402

    artifacts = run_layer2_replay(
        replay_day=replay_day,
        lookback_calendar_days=lookback,
        feed=feed,
    )

    try:
        ext_provider, ext_blob, curated_doc = load_external_snapshot_for_replay(ext_path)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: snapshot load failed ({preset_key}): {e}", file=sys.stderr)
        return None

    start_utc = artifacts.fetch_start_utc
    end_exc = artifacts.fetch_end_exclusive_utc

    curated_by_sym = {r.symbol: r for r in curated_doc.symbols} if curated_doc else {}

    if curated_doc:
        align_errs = validate_replay_window_alignment(
            curated_doc,
            replay_day=replay_day,
            lookback_calendar_days=lookback,
            fetch_start_utc=start_utc,
            fetch_end_exclusive_utc=end_exc,
        )
        if align_errs:
            for msg in align_errs:
                print(f"WARNING [{preset_key}] external replay alignment: {msg}", file=sys.stderr)
            if strict_alignment:
                print(f"Strict alignment failed for preset {preset_key}.", file=sys.stderr)
                return None
        for w in validate_snapshot_records_consistency(curated_doc):
            print(f"WARNING [{preset_key}] snapshot consistency: {w}", file=sys.stderr)

    layer1_syms = set(artifacts.layer1_df["symbol"].astype(str).str.upper())
    att_df = artifacts.attention_df

    rows_out: list[SymbolSaturationRow] = []
    for sym in sorted(ext_blob.keys()):
        row_map = layer2_attention_row(att_df, sym) if not att_df.empty else None
        layer2_ok = sym in layer1_syms and row_map is not None
        rec = curated_by_sym.get(sym)
        rows_out.append(
            _collect_symbol_analysis(
                sym=sym,
                layer2_scored=layer2_ok,
                row=row_map,
                rec=rec,
                start_utc=start_utc,
                end_exc=end_exc,
                ext_provider=ext_provider,
                ext_blob=ext_blob,
            )
        )

    primary = _primary_symbol(rows_out)
    if primary is None:
        print(f"ERROR: no symbols in snapshot for {preset_key}", file=sys.stderr)
        return None

    return (
        RegimeBundle(
            preset_key=preset_key,
            label=str(cfg.label),
            replay_date_iso=replay_day.isoformat(),
            symbol=primary.symbol,
            row=primary,
        ),
        curated_doc,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare speculative regime structures across replay presets (observational)."
    )
    parser.add_argument(
        "presets",
        nargs="*",
        default=list(DEFAULT_COMPARE_PRESETS),
        metavar="PRESET",
        help=(
            "Replay preset keys (default: gme-jan-2021 nvda-ai-narrative mstr-crypto-reflexivity). "
            "See analyze_speculative_saturation PRESETS for full list."
        ),
    )
    parser.add_argument(
        "--feed",
        type=str,
        choices=("iex", "sip"),
        default="iex",
        help="Alpaca feed for Layer 2 replay.",
    )
    parser.add_argument(
        "--strict-external-alignment",
        action="store_true",
        help="Skip a preset when curated snapshot metadata mismatches Layer 2 fetch window.",
    )
    args = parser.parse_args()

    preset_keys = [str(p).strip() for p in args.presets if str(p).strip()]
    if not preset_keys:
        preset_keys = list(DEFAULT_COMPARE_PRESETS)

    load_dotenv(PROJECT_ROOT / ".env")
    feed = DataFeed.IEX if args.feed.lower() == "iex" else DataFeed.SIP

    today_utc = datetime.now(timezone.utc).date()

    bundles: list[RegimeBundle] = []
    print()
    print("=" * 88)
    print("Cross-regime speculative comparison — observational ontology research")
    print("=" * 88)
    print(f"presets (order preserved): {preset_keys}")
    print()

    for pk in preset_keys:
        cfg = PRESETS.get(pk)
        if cfg is None:
            print(f"SKIP: unknown preset {pk!r}", file=sys.stderr)
            continue
        if cfg.replay_date > today_utc:
            print(
                f"WARNING [{pk}] replay-date {cfg.replay_date} is after UTC today — bars may be sparse.",
                file=sys.stderr,
            )

        info_path = (PROJECT_ROOT / cfg.snapshot_relpath).resolve()
        print(f"--- loading regime: {pk} | {cfg.label}")
        print(f"    replay_date={cfg.replay_date.isoformat()} lookback={cfg.lookback_calendar_days} snapshot={info_path}")

        out = run_one_preset(pk, feed=feed, strict_alignment=args.strict_external_alignment)
        if out is None:
            continue
        b, curated_doc = out
        bundles.append(b)

        if curated_doc:
            print(f"    snapshot_manifest: {json.dumps(json_snapshot_preview(curated_doc))}")
        print(f"    primary_symbol: {b.symbol} (alphabetically first in snapshot)")
        cells = _metric_cells(b.row)
        print(f"    combined_saturation_index: {_fmt_cell(cells.get('combined_saturation_index'))}")
        print()

    if len(bundles) < 2:
        print("Need at least two successful preset runs for side-by-side comparison.", file=sys.stderr)
        sys.exit(1)

    _print_cross_regime_table(bundles, "Cross-regime metric table (primary symbol per preset)")
    _print_archetype_block(bundles)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
