#!/usr/bin/env python3
"""
Observational speculative regime catalog — taxonomy summaries and preset linkage diagnostics.

Loads curated archetypes under ``data/speculative_regime_catalog/`` and optionally overlays
replay-derived emission tags from ``compare_speculative_regimes.py``. No Layer 3 scoring,
prediction, fragility estimation, or trading logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CATALOG_DIR = PROJECT_ROOT / "data" / "speculative_regime_catalog"


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


_CMP_MOD_CACHE: Any | None = None


def _load_compare_module_cached() -> Any:
    global _CMP_MOD_CACHE
    if _CMP_MOD_CACHE is None:
        _CMP_MOD_CACHE = _load_compare_module()
    return _CMP_MOD_CACHE


def _try_preset_registry_keys() -> tuple[str, ...] | None:
    try:
        return tuple(sorted(_load_compare_module_cached().PRESETS.keys()))
    except Exception:
        return None


@dataclass(frozen=True)
class CatalogArchetype:
    id: str
    label: str
    dominant_structure: str
    expected_convergence_behavior: str
    expected_participation_behavior: str
    expected_narrative_behavior: str
    related_replay_emission_tags: tuple[str, ...]


@dataclass(frozen=True)
class PresetArchetypeLink:
    preset_key: str
    catalog_archetype_ids: tuple[str, ...]
    curator_notes: str | None


@dataclass(frozen=True)
class RegimeCatalogDocument:
    schema_version: int
    ontology_scope: str | None
    disclaimer: str | None
    archetypes: tuple[CatalogArchetype, ...]


@dataclass(frozen=True)
class PresetLinksDocument:
    schema_version: int
    disclaimer: str | None
    links: tuple[PresetArchetypeLink, ...]


def _require_str(obj: Mapping[str, Any], key: str, *, ctx: str) -> str:
    if key not in obj:
        raise ValueError(f"{ctx}: missing {key!r}")
    v = obj[key]
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{ctx}: {key!r} must be non-empty string")
    return v.strip()


def load_archetypes_json(path: Path) -> RegimeCatalogDocument:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("archetypes root must be object")
    ver = int(raw.get("schema_version", 1))
    scope_raw = raw.get("ontology_scope")
    scope_s = str(scope_raw).strip() if isinstance(scope_raw, str) else None

    disc_raw = raw.get("disclaimer")
    disc_s = str(disc_raw).strip() if isinstance(disc_raw, str) else None

    rows_in = raw.get("archetypes")
    if not isinstance(rows_in, list) or not rows_in:
        raise ValueError("archetypes.archetypes must be non-empty array")

    parsed: list[CatalogArchetype] = []
    for i, row in enumerate(rows_in):
        ctx = f"archetypes[{i}]"
        if not isinstance(row, dict):
            raise ValueError(f"{ctx} must be object")
        aid = _require_str(row, "id", ctx=ctx)
        lab = _require_str(row, "label", ctx=ctx)
        dom = _require_str(row, "dominant_structure", ctx=ctx)
        ec = _require_str(row, "expected_convergence_behavior", ctx=ctx)
        ep = _require_str(row, "expected_participation_behavior", ctx=ctx)
        en = _require_str(row, "expected_narrative_behavior", ctx=ctx)
        tags_in = row.get("related_replay_emission_tags", [])
        if tags_in is None:
            tags_in = []
        if not isinstance(tags_in, list):
            raise ValueError(f"{ctx}.related_replay_emission_tags must be array")
        tags = tuple(sorted({str(t).strip() for t in tags_in if str(t).strip()}))

        parsed.append(
            CatalogArchetype(
                id=aid,
                label=lab,
                dominant_structure=dom,
                expected_convergence_behavior=ec,
                expected_participation_behavior=ep,
                expected_narrative_behavior=en,
                related_replay_emission_tags=tags,
            )
        )

    by_id = {a.id for a in parsed}
    if len(by_id) != len(parsed):
        raise ValueError("duplicate archetype id")

    parsed_sorted = tuple(sorted(parsed, key=lambda a: a.id))
    return RegimeCatalogDocument(
        schema_version=ver,
        ontology_scope=scope_s,
        disclaimer=disc_s,
        archetypes=parsed_sorted,
    )


def load_preset_links_json(path: Path) -> PresetLinksDocument:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("preset_links root must be object")
    ver = int(raw.get("schema_version", 1))
    disc_raw = raw.get("disclaimer")
    disc_s = str(disc_raw).strip() if isinstance(disc_raw, str) else None

    links_in = raw.get("links")
    if not isinstance(links_in, list):
        raise ValueError("preset_links.links must be array")

    links: list[PresetArchetypeLink] = []
    for i, row in enumerate(links_in):
        ctx = f"links[{i}]"
        if not isinstance(row, dict):
            raise ValueError(f"{ctx} must be object")
        pk = _require_str(row, "preset_key", ctx=ctx)
        ids_in = row.get("catalog_archetype_ids")
        if not isinstance(ids_in, list) or not ids_in:
            raise ValueError(f"{ctx}.catalog_archetype_ids must be non-empty array")
        aids = tuple(sorted({str(x).strip() for x in ids_in if str(x).strip()}))
        notes = row.get("curator_notes")
        notes_s = str(notes).strip() if isinstance(notes, str) else None

        links.append(
            PresetArchetypeLink(
                preset_key=pk,
                catalog_archetype_ids=aids,
                curator_notes=notes_s,
            )
        )

    links_sorted = tuple(sorted(links, key=lambda x: x.preset_key))
    seen_pk = [x.preset_key for x in links_sorted]
    if len(set(seen_pk)) != len(seen_pk):
        raise ValueError("duplicate preset_key in preset_links")

    return PresetLinksDocument(
        schema_version=ver,
        disclaimer=disc_s,
        links=links_sorted,
    )


def _validate_links_against_catalog(
    catalog: RegimeCatalogDocument,
    links: PresetLinksDocument,
    *,
    known_preset_keys: Iterable[str] | None,
) -> list[str]:
    warns: list[str] = []
    ids_by = {a.id for a in catalog.archetypes}
    for lk in links.links:
        unknown = sorted(set(lk.catalog_archetype_ids) - ids_by)
        if unknown:
            warns.append(f"preset {lk.preset_key!r} references unknown archetype ids: {unknown}")
    if known_preset_keys is not None:
        ks = set(known_preset_keys)
        for lk in links.links:
            if lk.preset_key not in ks:
                warns.append(f"preset link {lk.preset_key!r} not found in replay PRESETS registry")
    return sorted(warns)


def _emission_tag_index(catalog: RegimeCatalogDocument) -> dict[str, tuple[str, ...]]:
    inv: dict[str, list[str]] = defaultdict(list)
    for a in catalog.archetypes:
        for t in a.related_replay_emission_tags:
            inv[t].append(a.id)
    return {t: tuple(sorted(set(ids))) for t, ids in sorted(inv.items())}


def _print_taxonomy_summary(catalog: RegimeCatalogDocument) -> None:
    print()
    print("=" * 88)
    print("Regime catalog — archetype taxonomy (observational)")
    print("=" * 88)
    if catalog.disclaimer:
        print(catalog.disclaimer)
        print()
    if catalog.ontology_scope:
        print(f"ontology_scope: {catalog.ontology_scope}")
        print()

    for a in catalog.archetypes:
        print("-" * 88)
        print(f"id: {a.id}")
        print(f"label: {a.label}")
        print(f"dominant_structure: {a.dominant_structure}")
        print(f"expected_convergence_behavior: {a.expected_convergence_behavior}")
        print(f"expected_participation_behavior: {a.expected_participation_behavior}")
        print(f"expected_narrative_behavior: {a.expected_narrative_behavior}")
        if a.related_replay_emission_tags:
            print(f"related_replay_emission_tags: {json.dumps(list(a.related_replay_emission_tags))}")
        else:
            print("related_replay_emission_tags: []")


def _print_preset_link_table(links: PresetLinksDocument) -> None:
    print()
    print("=" * 88)
    print("Preset → catalog archetype linkage")
    print("=" * 88)
    if links.disclaimer:
        print(links.disclaimer)
        print()

    for lk in links.links:
        print(f"[{lk.preset_key}]")
        print(f"  archetypes: {json.dumps(list(lk.catalog_archetype_ids))}")
        if lk.curator_notes:
            print(f"  curator_notes: {lk.curator_notes}")
        print()


def _print_cross_archetype_matrix(
    catalog: RegimeCatalogDocument,
    links: PresetLinksDocument,
) -> None:
    presets = [lk.preset_key for lk in links.links]
    arch_ids = [a.id for a in catalog.archetypes]
    cell_w = max(16, max(len(a) for a in arch_ids) if arch_ids else 16)

    preset_link_map = {lk.preset_key: set(lk.catalog_archetype_ids) for lk in links.links}

    print()
    print("=" * 88)
    print("Cross-regime coverage matrix (preset × catalog archetype)")
    print("=" * 88)
    header = f"{'preset':<26}" + "".join(f"  {aid:>{cell_w}}" for aid in arch_ids)
    print(header)
    print("-" * len(header))
    for pk in sorted(presets):
        linked = preset_link_map.get(pk, set())
        row_cells = ["X" if aid in linked else "·" for aid in arch_ids]
        line = f"{pk:<26}" + "".join(f"  {c:>{cell_w}}" for c in row_cells)
        print(line)


def _print_emission_bridge(catalog: RegimeCatalogDocument) -> None:
    print()
    print("=" * 88)
    print("Replay emission tag bridge (deterministic tags ↔ catalog archetypes)")
    print("=" * 88)
    idx = _emission_tag_index(catalog)
    for tag in sorted(idx.keys()):
        print(f"  {tag}: {json.dumps(list(idx[tag]))}")


def _catalog_emission_union_for_preset(
    catalog_by_id: Mapping[str, CatalogArchetype],
    link: PresetArchetypeLink | None,
) -> frozenset[str]:
    if link is None:
        return frozenset()
    out: set[str] = set()
    for aid in link.catalog_archetype_ids:
        a = catalog_by_id.get(aid)
        if a is None:
            continue
        out.update(a.related_replay_emission_tags)
    return frozenset(out)


def _print_taxonomy_diagnostics(
    catalog: RegimeCatalogDocument,
    links: PresetLinksDocument,
    *,
    preset_registry_keys: tuple[str, ...] | None,
) -> None:
    all_arch_ids = {a.id for a in catalog.archetypes}
    linked_arch: set[str] = set()
    for lk in links.links:
        linked_arch.update(lk.catalog_archetype_ids)
    orphans = sorted(all_arch_ids - linked_arch)

    linked_presets = {lk.preset_key for lk in links.links}
    missing_preset_links: list[str] = []
    if preset_registry_keys:
        missing_preset_links = sorted(set(preset_registry_keys) - linked_presets)

    arch_presets: dict[str, list[str]] = defaultdict(list)
    for lk in links.links:
        for aid in lk.catalog_archetype_ids:
            arch_presets[aid].append(lk.preset_key)

    print()
    print("=" * 88)
    print("Taxonomy diagnostics (catalog coverage)")
    print("=" * 88)
    print(f"  archetypes_defined: {len(all_arch_ids)}")
    print(f"  archetypes_linked_to_preset: {len(linked_arch)}")
    print(f"  presets_with_catalog_links: {len(linked_presets)}")
    if orphans:
        print(f"  archetypes_without_preset_links: {json.dumps(orphans)}")
    else:
        print("  archetypes_without_preset_links: []")
    if missing_preset_links:
        print(
            "  replay_presets_without_catalog_links: "
            f"{json.dumps(missing_preset_links)}"
        )
    else:
        if preset_registry_keys:
            print("  replay_presets_without_catalog_links: []")

    print()
    print("  archetype_preset_fanout (deterministic preset sort):")
    for aid in sorted(arch_presets.keys()):
        ps = sorted(set(arch_presets[aid]))
        print(f"    {aid}: {json.dumps(ps)} (count={len(ps)})")


def _run_replay_overlay(
    preset_keys: Sequence[str],
    *,
    feed: str,
    strict_alignment: bool,
    catalog: RegimeCatalogDocument,
    links: PresetLinksDocument,
) -> None:
    from dotenv import load_dotenv  # noqa: WPS433

    load_dotenv(PROJECT_ROOT / ".env")

    cmp = _load_compare_module_cached()
    run_one_preset = cmp.run_one_preset
    archetype_tags_for_bundle = cmp.archetype_tags_for_bundle
    PRESETS = cmp.PRESETS

    from alpaca.data.enums import DataFeed  # noqa: WPS433

    fd = DataFeed.IEX if feed.lower() == "iex" else DataFeed.SIP

    bundles: list[Any] = []
    for pk in preset_keys:
        if pk not in PRESETS:
            print(f"SKIP replay: unknown preset {pk!r}", file=sys.stderr)
            continue
        out = run_one_preset(pk, feed=fd, strict_alignment=strict_alignment)
        if out is None:
            continue
        b, _cur = out
        bundles.append(b)

    if len(bundles) < 2:
        print(
            "Replay overlay needs at least two successful preset runs for batch-relative "
            "emission tags — fewer succeeded; tags may be empty or single-regime only.",
            file=sys.stderr,
        )

    catalog_by_id = {a.id: a for a in catalog.archetypes}
    link_by_pk = {lk.preset_key: lk for lk in links.links}

    print()
    print("=" * 88)
    print("Replay emission summary vs catalog linkage (observational overlay)")
    print("=" * 88)

    if not bundles:
        print("  (no bundles — nothing to summarize)")
        return

    for b in sorted(bundles, key=lambda x: x.preset_key):
        rt = archetype_tags_for_bundle(b, bundles)
        lk = link_by_pk.get(b.preset_key)
        catalog_archetypes = list(lk.catalog_archetype_ids) if lk else []
        emission_union = sorted(_catalog_emission_union_for_preset(catalog_by_id, lk))
        rt_set = set(rt)
        em_set = set(emission_union)
        inter = sorted(rt_set & em_set)
        only_rt = sorted(rt_set - em_set)
        only_cat = sorted(em_set - rt_set)

        print()
        print(f"preset: {b.preset_key} | symbol: {b.symbol} | replay_date: {b.replay_date_iso}")
        print(f"  catalog_archetype_ids: {json.dumps(catalog_archetypes)}")
        print(f"  runtime_replay_emission_tags: {json.dumps(rt)}")
        print(f"  catalog_emission_tag_union: {json.dumps(emission_union)}")
        print(f"  intersection: {json.dumps(inter)}")
        print(f"  runtime_only: {json.dumps(only_rt)}")
        print(f"  catalog_union_only: {json.dumps(only_cat)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize speculative regime catalog and optional replay emission overlays."
    )
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=DEFAULT_CATALOG_DIR,
        help=f"Catalog directory (default: {DEFAULT_CATALOG_DIR})",
    )
    parser.add_argument(
        "--with-replay",
        nargs="*",
        metavar="PRESET",
        help=(
            "Run Layer 2 + snapshot replay for presets (requires Alpaca env). "
            "With no preset args, uses compare_speculative_regimes default triple."
        ),
    )
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument(
        "--strict-external-alignment",
        action="store_true",
        help="Skip presets that fail curated snapshot alignment when using --with-replay.",
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Omit preset × archetype matrix.",
    )
    args = parser.parse_args()

    cat_dir = args.catalog_dir.resolve()
    arch_path = cat_dir / "archetypes.json"
    links_path = cat_dir / "preset_links.json"

    if not arch_path.is_file():
        print(f"Missing {arch_path}", file=sys.stderr)
        sys.exit(1)
    if not links_path.is_file():
        print(f"Missing {links_path}", file=sys.stderr)
        sys.exit(1)

    catalog = load_archetypes_json(arch_path)
    links = load_preset_links_json(links_path)

    preset_registry_keys = _try_preset_registry_keys()
    if preset_registry_keys is None:
        print(
            "WARNING: replay PRESETS registry unavailable (optional deps?) — "
            "preset_key validation skipped.",
            file=sys.stderr,
        )

    for w in _validate_links_against_catalog(catalog, links, known_preset_keys=preset_registry_keys):
        print(f"WARNING: {w}", file=sys.stderr)

    _print_taxonomy_summary(catalog)
    _print_preset_link_table(links)
    if not args.skip_matrix:
        _print_cross_archetype_matrix(catalog, links)
    _print_emission_bridge(catalog)
    _print_taxonomy_diagnostics(catalog, links, preset_registry_keys=preset_registry_keys)

    if args.with_replay is not None:
        cmp_mod = _load_compare_module_cached()
        pk_list = list(args.with_replay) if args.with_replay else list(cmp_mod.DEFAULT_COMPARE_PRESETS)
        _run_replay_overlay(
            pk_list,
            feed=args.feed,
            strict_alignment=args.strict_external_alignment,
            catalog=catalog,
            links=links,
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
