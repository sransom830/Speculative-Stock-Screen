#!/usr/bin/env python3
"""
Large-scale historical replay orchestration — observational ontology matrix.

Loads replay definitions from the speculative regime catalog (preset links),
external-attention snapshots, and historical truth-set events; runs Layer 2 replay
with cached artifacts; computes saturation rows, emission tags, and catalog archetype
linkage; persists deterministic summaries under ``data/replay_matrix/``.

No ML training, prediction targets, Layer 3 fragility scoring, or trading logic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpaca.data.enums import DataFeed  # noqa: E402

from layer2.external_attention.snapshot_loader import (  # noqa: E402
    load_external_snapshot_for_replay,
    validate_replay_window_alignment,
    validate_snapshot_records_consistency,
)
from layer2.truth_set_validation import (  # noqa: E402
    attention_row as layer2_attention_row,
    load_event_files,
    parse_truth_feed,
    parse_truth_replay_date,
    validate_event_payload,
)

OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "replay_matrix"
CATALOG_DIR_DEFAULT = PROJECT_ROOT / "data" / "speculative_regime_catalog"
SNAPSHOTS_DIR_DEFAULT = PROJECT_ROOT / "data" / "external_attention_snapshots"
EVENTS_DIR_DEFAULT = PROJECT_ROOT / "data" / "historical_speculative_events"


def _load_compare_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "compare_speculative_regimes.py"
    name = "_compare_speculative_regimes"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load compare module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_CMP = _load_compare_module()
PRESETS = _CMP.PRESETS
_collect_symbol_analysis = _CMP._collect_symbol_analysis
SymbolSaturationRow = _CMP.SymbolSaturationRow
RegimeBundle = _CMP.RegimeBundle
archetype_tags_for_bundle = _CMP.archetype_tags_for_bundle


def _replay_id(canonical: Mapping[str, Any]) -> str:
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"rm_{h[:28]}"


def _snapshot_quick_meta(path: Path) -> tuple[date, int, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: root must be object")
    rd = parse_truth_replay_date(str(raw["replay_date"]))
    lb = int(raw.get("lookback_calendar_days", 90))
    syms_raw = raw.get("symbols")
    symbols: list[str] = []
    if isinstance(syms_raw, list):
        for row in syms_raw:
            if isinstance(row, dict) and "symbol" in row:
                symbols.append(str(row["symbol"]).strip().upper())
    elif isinstance(syms_raw, dict):
        symbols.extend(str(k).strip().upper() for k in syms_raw)
    if not symbols:
        raise ValueError(f"{path}: could not derive symbols")
    return rd, lb, tuple(sorted(set(symbols)))


def _load_preset_links(path: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("links")
    if not isinstance(rows, list):
        raise ValueError("preset_links.links must be array")
    out: list[tuple[str, tuple[str, ...]]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"links[{i}] must be object")
        pk = str(row["preset_key"]).strip()
        aids = row.get("catalog_archetype_ids")
        if not isinstance(aids, list) or not aids:
            raise ValueError(f"links[{i}].catalog_archetype_ids invalid")
        arch = tuple(sorted({str(x).strip() for x in aids if str(x).strip()}))
        out.append((pk, arch))
    return tuple(sorted(out, key=lambda t: t[0]))


def _resolve_truth_snapshot(replay_day: date, symbol: str, snapshots_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for path in sorted(snapshots_dir.glob("*.json")):
        try:
            rd, _lb, syms = _snapshot_quick_meta(path)
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            continue
        if rd == replay_day and symbol.upper() in syms:
            candidates.append(path)
    return candidates[0] if candidates else None


def _saturation_row_to_json(row: SymbolSaturationRow) -> dict[str, Any]:
    return {
        "symbol": row.symbol,
        "layer2_scored": row.layer2_scored,
        "participation": row.participation,
        "volatility_expansion": row.volatility_expansion,
        "acceleration": row.acceleration,
        "mention_velocity": row.mention_velocity,
        "subreddit_breadth": row.subreddit_breadth,
        "reddit_attention": row.reddit_attention,
        "narrative_concentration": row.narrative_concentration,
        "thematic_clustering": row.thematic_clustering,
        "thematic_overlay": row.thematic_overlay,
        "headline_density": row.headline_density,
        "news_attention": row.news_attention,
        "thematic_mass_concentration_raw": row.thematic_mass_concentration_raw,
        "convergence": {k: row.convergence[k] for k in sorted(row.convergence.keys())},
        "saturation": {k: row.saturation[k] for k in sorted(row.saturation.keys())},
        "market_native_blend": row.market_native_blend,
        "external_narrative_blend": row.external_narrative_blend,
        "combined_saturation_index": row.combined_saturation_index,
    }


def _layer2_row_to_json(series: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if series is None:
        return None
    keys = (
        "speculative_attention_score",
        "speculative_attention_state",
        "factor_participation_composite",
        "factor_realized_vol_expansion",
        "factor_acceleration_composite",
        "factor_volatility_composite",
        "ecosystem_multiplier",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in series.index:
            v = series[k]
            if hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    pass
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                out[k] = None
            else:
                out[k] = v
    return out


def _bucket_unit_interval(x: float) -> str:
    x = max(0.0, min(1.0, float(x)))
    if x >= 1.0:
        return "[0.75,1.00]"
    idx = min(3, int(x * 4.0))
    low = idx * 0.25
    high = (idx + 1) * 0.25
    return f"[{low:.2f},{high:.2f})"


@dataclass(frozen=True)
class MatrixStudy:
    replay_id: str
    provenance: tuple[str, ...]
    replay_day: date
    lookback: int
    feed: DataFeed
    symbols: tuple[str, ...]
    snapshot_path: Path
    catalog_archetypes: tuple[str, ...]
    preset_key: str | None
    truth_meta: dict[str, Any] | None


def discover_studies(
    *,
    include_catalog: bool,
    include_snapshots: bool,
    include_truth: bool,
    catalog_dir: Path,
    snapshots_dir: Path,
    events_dir: Path,
    truth_event_filter: frozenset[str] | None,
) -> tuple[MatrixStudy, ...]:
    studies: list[MatrixStudy] = []

    preset_links_path = catalog_dir / "preset_links.json"
    preset_links: tuple[tuple[str, tuple[str, ...]], ...] = ()
    if preset_links_path.is_file():
        preset_links = _load_preset_links(preset_links_path)

    arch_by_preset = {pk: arch for pk, arch in preset_links}

    if include_catalog:
        for preset_key, catalog_arch in preset_links:
            if preset_key not in PRESETS:
                continue
            cfg = PRESETS[preset_key]
            snap = (PROJECT_ROOT / cfg.snapshot_relpath).resolve()
            if not snap.is_file():
                continue
            rd, lb_snap, syms_snap = _snapshot_quick_meta(snap)
            lb = cfg.lookback_calendar_days
            rid = _replay_id(
                {
                    "kind": "catalog_preset",
                    "preset_key": preset_key,
                    "replay_date": rd.isoformat(),
                    "lookback_calendar_days": lb,
                    "snapshot_relpath": cfg.snapshot_relpath,
                }
            )
            studies.append(
                MatrixStudy(
                    replay_id=rid,
                    provenance=(f"catalog:{preset_key}",),
                    replay_day=cfg.replay_date,
                    lookback=lb,
                    feed=DataFeed.IEX,
                    symbols=syms_snap,
                    snapshot_path=snap,
                    catalog_archetypes=catalog_arch,
                    preset_key=preset_key,
                    truth_meta=None,
                )
            )

    if include_snapshots:
        for path in sorted(snapshots_dir.glob("*.json")):
            stem = path.stem
            try:
                rd, lb, syms = _snapshot_quick_meta(path)
            except (ValueError, KeyError, OSError, json.JSONDecodeError):
                continue
            snap = path.resolve()
            rel_posix = str(snap.relative_to(PROJECT_ROOT).as_posix())
            matched_preset = None
            for pk, cfg in PRESETS.items():
                if cfg.snapshot_relpath == rel_posix:
                    matched_preset = pk
                    break
            catalog_arch = arch_by_preset.get(matched_preset, ()) if matched_preset else ()
            rid = _replay_id(
                {
                    "kind": "snapshot_file",
                    "stem": stem,
                    "replay_date": rd.isoformat(),
                    "lookback_calendar_days": lb,
                    "relative_path": rel_posix,
                }
            )
            feed_name = "iex"
            studies.append(
                MatrixStudy(
                    replay_id=rid,
                    provenance=(f"snapshot:{stem}",),
                    replay_day=rd,
                    lookback=lb,
                    feed=parse_truth_feed(feed_name),
                    symbols=syms,
                    snapshot_path=snap,
                    catalog_archetypes=catalog_arch,
                    preset_key=matched_preset,
                    truth_meta=None,
                )
            )

    if include_truth:
        loaded = load_event_files(events_dir, truth_event_filter)
        for stem, payload in loaded:
            validate_event_payload(payload, stem + ".json")
            event_name = str(payload["event_name"])
            lb = int(payload.get("lookback_calendar_days") or 90)
            feed = parse_truth_feed(str(payload.get("alpaca_feed") or "iex"))
            for idx, exp in enumerate(payload["expectations"]):
                sym_u = str(exp["symbol"]).strip().upper()
                rd = parse_truth_replay_date(str(exp["replay_date"]))
                snap_path = _resolve_truth_snapshot(rd, sym_u, snapshots_dir)
                if snap_path is None:
                    print(
                        f"WARNING: truth expectation skipped — no snapshot for "
                        f"{event_name} {rd.isoformat()} {sym_u}",
                        file=sys.stderr,
                    )
                    continue
                _snap_rd_meta, _snap_lb_meta, syms_snap = _snapshot_quick_meta(snap_path)
                catalog_arch: tuple[str, ...] = ()
                preset_match: str | None = None
                rel_posix = str(snap_path.resolve().relative_to(PROJECT_ROOT).as_posix())
                for pk, cfg in PRESETS.items():
                    if cfg.snapshot_relpath == rel_posix:
                        preset_match = pk
                        catalog_arch = arch_by_preset.get(pk, ())
                        break
                rid = _replay_id(
                    {
                        "kind": "truth_expectation",
                        "event_file_stem": stem,
                        "expectation_index": idx,
                        "symbol": sym_u,
                        "replay_date": rd.isoformat(),
                        "lookback_calendar_days": lb,
                        "snapshot_relative_path": rel_posix,
                    }
                )
                truth_meta = {
                    "event_file_stem": stem,
                    "event_name": event_name,
                    "expectation_index": idx,
                    "expected_speculative_states": sorted(
                        str(x) for x in exp["expected_speculative_states"]
                    ),
                    "behavioral_tags": [str(x) for x in exp["behavioral_tags"]],
                }
                studies.append(
                    MatrixStudy(
                        replay_id=rid,
                        provenance=(
                            f"truth:{stem}:{idx}",
                            f"truth_event:{event_name}",
                        ),
                        replay_day=rd,
                        lookback=lb,
                        feed=feed,
                        symbols=(sym_u,),
                        snapshot_path=snap_path.resolve(),
                        catalog_archetypes=catalog_arch,
                        preset_key=preset_match,
                        truth_meta=truth_meta,
                    )
                )

    studies_sorted = tuple(sorted(studies, key=lambda s: s.replay_id))
    seen: set[str] = set()
    for s in studies_sorted:
        if s.replay_id in seen:
            raise RuntimeError(f"duplicate replay_id collision: {s.replay_id}")
        seen.add(s.replay_id)
    return studies_sorted


def _run_matrix(
    studies: Sequence[MatrixStudy],
    *,
    strict_alignment: bool,
    runs_dir: Path,
) -> tuple[list[dict[str, Any]], list[RegimeBundle]]:
    from layer2.replay_runner import run_layer2_replay  # noqa: E402

    artifact_cache: dict[tuple[str, int, str], Any] = {}

    def artifacts_for(day: date, lb: int, feed: DataFeed) -> Any:
        key = (day.isoformat(), lb, feed.value)
        if key not in artifact_cache:
            artifact_cache[key] = run_layer2_replay(
                replay_day=day,
                lookback_calendar_days=lb,
                feed=feed,
            )
        return artifact_cache[key]

    partial_rows: list[dict[str, Any]] = []
    bundles: list[RegimeBundle] = []

    for study in studies:
        arts = artifacts_for(study.replay_day, study.lookback, study.feed)
        ext_provider, ext_blob, curated_doc = load_external_snapshot_for_replay(study.snapshot_path)

        start_utc = arts.fetch_start_utc
        end_exc = arts.fetch_end_exclusive_utc

        if curated_doc:
            errs = validate_replay_window_alignment(
                curated_doc,
                replay_day=study.replay_day,
                lookback_calendar_days=study.lookback,
                fetch_start_utc=start_utc,
                fetch_end_exclusive_utc=end_exc,
            )
            if errs:
                for e in errs:
                    print(f"WARNING [{study.replay_id}] alignment: {e}", file=sys.stderr)
                if strict_alignment:
                    print(
                        f"SKIP persist [{study.replay_id}] — strict-external-alignment",
                        file=sys.stderr,
                    )
                    continue
            for w in validate_snapshot_records_consistency(curated_doc):
                print(f"WARNING [{study.replay_id}] consistency: {w}", file=sys.stderr)

        curated_by_sym = {r.symbol: r for r in curated_doc.symbols} if curated_doc else {}
        layer1_syms = set(arts.layer1_df["symbol"].astype(str).str.upper())
        att_df = arts.attention_df

        symbols_scan = tuple(sorted(set(study.symbols) & set(ext_blob.keys())))
        if not symbols_scan:
            print(
                f"WARNING [{study.replay_id}] no symbols intersect snapshot keys — skipping persist.",
                file=sys.stderr,
            )
            continue

        per_symbol: dict[str, Any] = {}

        for sym in symbols_scan:
            row_map = layer2_attention_row(att_df, sym) if not att_df.empty else None
            layer2_ok = sym in layer1_syms and row_map is not None
            sar = _collect_symbol_analysis(
                sym=sym,
                layer2_scored=layer2_ok,
                row=row_map,
                rec=curated_by_sym.get(sym),
                start_utc=start_utc,
                end_exc=end_exc,
                ext_provider=ext_provider,
                ext_blob=ext_blob,
            )
            bundle_key = f"{study.replay_id}:{sym}"
            bundles.append(
                RegimeBundle(
                    preset_key=bundle_key,
                    label=" | ".join(study.provenance),
                    replay_date_iso=study.replay_day.isoformat(),
                    symbol=sym,
                    row=sar,
                )
            )

            per_symbol[sym] = {
                "layer2_attention": _layer2_row_to_json(row_map),
                "saturation": _saturation_row_to_json(sar),
            }

        partial_rows.append(
            {
                "schema_version": 1,
                "replay_id": study.replay_id,
                "provenance": list(study.provenance),
                "replay_date": study.replay_day.isoformat(),
                "lookback_calendar_days": study.lookback,
                "alpaca_feed": study.feed.value,
                "external_snapshot_path": str(study.snapshot_path.relative_to(PROJECT_ROOT)),
                "external_provider_slug": ext_provider,
                "preset_key": study.preset_key,
                "catalog_archetype_ids": list(study.catalog_archetypes),
                "truth_expectation": study.truth_meta,
                "symbols_analyzed": list(symbols_scan),
                "per_symbol": per_symbol,
            }
        )

    tag_by_bundle = {b.preset_key: archetype_tags_for_bundle(b, bundles) for b in bundles}

    persisted_rows: list[dict[str, Any]] = []
    for row in partial_rows:
        rid = row["replay_id"]
        emission_preview: dict[str, list[str]] = {}
        for sym in row["symbols_analyzed"]:
            bk = f"{rid}:{sym}"
            emission_preview[sym] = list(tag_by_bundle.get(bk, []))
        row["replay_emission_tags"] = emission_preview
        persisted_rows.append(row)
        out_path = runs_dir / f"{rid}.json"
        out_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return persisted_rows, bundles


def _build_aggregate(
    persisted_rows: Sequence[Mapping[str, Any]],
    bundles: Sequence[RegimeBundle],
) -> dict[str, Any]:
    arch_ctr: Counter[str] = Counter()
    tag_ctr: Counter[str] = Counter()
    align_bucket: Counter[str] = Counter()
    sat_bucket: Counter[str] = Counter()
    conv_accum: dict[str, list[float]] = defaultdict(list)

    tag_pairs: Counter[str] = Counter()
    arch_pairs: Counter[str] = Counter()

    stability: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)

    for row in persisted_rows:
        for aid in row.get("catalog_archetype_ids", []):
            arch_ctr[str(aid)] += 1
        tags_by_sym = row.get("replay_emission_tags") or {}
        for sym, tags in tags_by_sym.items():
            for t in tags:
                tag_ctr[str(t)] += 1
            ts = tuple(sorted(tags))
            stability[(row["replay_date"], sym)].append(ts)
            for pair in combinations(ts, 2):
                tag_pairs[f"{pair[0]}||{pair[1]}"] += 1

        cats = tuple(sorted(row.get("catalog_archetype_ids") or []))
        for pair in combinations(cats, 2):
            arch_pairs[f"{pair[0]}||{pair[1]}"] += 1

        per_sym = row.get("per_symbol") or {}
        for sym, sym_blob in per_sym.items():
            if not isinstance(sym_blob, dict):
                continue
            core = sym_blob.get("saturation")
            if not isinstance(core, dict):
                continue
            diag_sat = core.get("saturation")
            if isinstance(diag_sat, dict):
                align = diag_sat.get("external_participation_alignment")
                if isinstance(align, (int, float)):
                    align_bucket[_bucket_unit_interval(float(align))] += 1
            csi = core.get("combined_saturation_index")
            if isinstance(csi, (int, float)):
                sat_bucket[_bucket_unit_interval(float(csi))] += 1
            conv_inner = core.get("convergence")
            if isinstance(conv_inner, dict):
                for ck, cv in conv_inner.items():
                    if cv is None:
                        continue
                    try:
                        conv_accum[str(ck)].append(float(cv))
                    except (TypeError, ValueError):
                        continue

    convergence_summary: dict[str, dict[str, float | int]] = {}
    for ck, vals in sorted(conv_accum.items()):
        if not vals:
            continue
        convergence_summary[ck] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    stability_summary: dict[str, Any] = {}
    for (rd, sym), variant_rows in sorted(stability.items()):
        uniq = sorted(set(variant_rows))
        stability_summary[f"{rd}|{sym}"] = {
            "distinct_emission_tag_sets": len(uniq),
            "variants": [list(u) for u in uniq],
            "run_rows_observed": len(variant_rows),
        }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(persisted_rows),
        "bundle_count": len(bundles),
        "catalog_archetype_id_frequency": dict(sorted(arch_ctr.items())),
        "replay_emission_tag_frequency_across_symbol_rows": dict(sorted(tag_ctr.items())),
        "external_participation_alignment_buckets": dict(sorted(align_bucket.items())),
        "combined_saturation_index_buckets": dict(sorted(sat_bucket.items())),
        "convergence_numeric_summary": convergence_summary,
        "cross_regime": {
            "emission_tag_pairs_within_symbol_row": dict(sorted(tag_pairs.items())),
            "catalog_archetype_pairs_within_study": dict(sorted(arch_pairs.items())),
            "replay_date_symbol_emission_variants": stability_summary,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate observational replay matrix batches.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--catalog-dir", type=Path, default=CATALOG_DIR_DEFAULT)
    parser.add_argument("--snapshots-dir", type=Path, default=SNAPSHOTS_DIR_DEFAULT)
    parser.add_argument("--events-dir", type=Path, default=EVENTS_DIR_DEFAULT)
    parser.add_argument(
        "--sources",
        type=str,
        default="catalog,snapshots,truth",
        help="Comma-separated: catalog, snapshots, truth",
    )
    parser.add_argument(
        "--truth-events",
        type=str,
        default=None,
        help="Comma-separated event JSON stems (without .json); default all.",
    )
    parser.add_argument("--strict-external-alignment", action="store_true")
    args = parser.parse_args()

    src_parts = {s.strip().lower() for s in args.sources.split(",") if s.strip()}
    include_catalog = "catalog" in src_parts
    include_snapshots = "snapshots" in src_parts
    include_truth = "truth" in src_parts

    filt = None
    if args.truth_events:
        filt = frozenset(x.strip() for x in args.truth_events.split(",") if x.strip())

    load_dotenv(PROJECT_ROOT / ".env")

    studies = discover_studies(
        include_catalog=include_catalog,
        include_snapshots=include_snapshots,
        include_truth=include_truth,
        catalog_dir=args.catalog_dir.resolve(),
        snapshots_dir=args.snapshots_dir.resolve(),
        events_dir=args.events_dir.resolve(),
        truth_event_filter=filt,
    )

    out_root = args.output_root.resolve()
    runs_dir = out_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    persisted, bundles = _run_matrix(
        studies,
        strict_alignment=args.strict_external_alignment,
        runs_dir=runs_dir,
    )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_enabled": sorted(src_parts),
        "matrix_study_specs_discovered": len(studies),
        "replay_ids_persisted": sorted(r["replay_id"] for r in persisted),
        "persisted_run_count": len(persisted),
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    aggregate = _build_aggregate(persisted, bundles)
    (out_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(persisted)} runs under {runs_dir}")
    print(f"Manifest: {out_root / 'manifest.json'}")
    print(f"Aggregate: {out_root / 'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
