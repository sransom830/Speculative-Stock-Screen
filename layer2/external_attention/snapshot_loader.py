"""
Historical external-attention snapshots — load, validate, convert to enrichment raw counts.

Curated schema (``schema_version`` 2) is replay-aligned: each file carries ``replay_date``,
``lookback_calendar_days``, and per-symbol observational counts for Reddit/news thematic mass.

Legacy schema (``schema_version`` 1) nests ``reddit_raw`` / ``news_raw`` under ``symbols`` dict.

No live APIs, sentiment, or Layer 3 logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from layer2.replay_runner import fetch_window_utc

from .models import NewsRawCounts, NewsThemeMass, RedditRawCounts

# When prior-window counts are omitted, derive deterministically from window totals.
_DEFAULT_PRIOR_RATIO = 0.52


def _parse_iso_date(s: str) -> date:
    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()


def clamp_nonneg_int(name: str, val: Any) -> int:
    x = int(val)
    if x < 0:
        raise ValueError(f"{name} must be >= 0")
    return x


def infer_prior_window(main_window: int, explicit_prior: Any | None) -> int:
    if explicit_prior is None:
        return max(0, int(round(main_window * _DEFAULT_PRIOR_RATIO)))
    return clamp_nonneg_int("prior_window", explicit_prior)


@dataclass(frozen=True)
class ExternalAttentionSnapshotRecord:
    """Single-symbol curated observables for one replay bundle."""

    symbol: str
    reddit_mentions: int
    reddit_mentions_prior: int
    reddit_comment_count: int
    reddit_comment_count_prior: int
    subreddit_count: int
    subreddit_list: tuple[str, ...]
    news_headline_count: int
    news_headline_count_prior: int
    thematic_topic_masses: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ExternalAttentionSnapshotDocument:
    schema_version: int
    replay_date: date
    lookback_calendar_days: int
    provider_slug: str
    symbols: tuple[ExternalAttentionSnapshotRecord, ...]
    notes: str | None


def _sorted_unique_subs(items: list[str]) -> tuple[str, ...]:
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    return tuple(sorted(set(cleaned), key=lambda s: (s.lower(), s)))


def _parse_topic_masses(rows: Any) -> tuple[tuple[str, int], ...]:
    if not isinstance(rows, list):
        raise ValueError("thematic_topic_masses must be a list")
    out: list[tuple[str, int]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"thematic_topic_masses[{i}] must be object")
        label = row.get("topic", row.get("label"))
        if label is None:
            raise ValueError(f"thematic_topic_masses[{i}] needs topic or label")
        mass = clamp_nonneg_int(f"thematic_topic_masses[{i}].mass", row.get("mass"))
        lab = str(label).strip()
        if not lab:
            raise ValueError(f"thematic_topic_masses[{i}] empty topic label")
        out.append((lab, mass))
    return tuple(sorted(out, key=lambda t: (t[0].lower(), t[0])))


def parse_curated_snapshot(payload: dict[str, Any]) -> ExternalAttentionSnapshotDocument:
    ver = int(payload.get("schema_version", 2))
    if ver < 2:
        raise ValueError("curated parser expects schema_version >= 2")

    replay_date = _parse_iso_date(str(payload["replay_date"]))
    lookback = clamp_nonneg_int("lookback_calendar_days", payload.get("lookback_calendar_days", 90))
    provider_slug = str(payload.get("provider_slug") or "historical_curated_v1")
    notes = payload.get("notes")
    notes_s = None if notes is None else str(notes)

    syms_raw = payload.get("symbols")
    if not isinstance(syms_raw, list) or not syms_raw:
        raise ValueError("symbols must be a non-empty array")

    records: list[ExternalAttentionSnapshotRecord] = []
    for i, blob in enumerate(syms_raw):
        if not isinstance(blob, dict):
            raise ValueError(f"symbols[{i}] must be object")
        sym_u = str(blob["symbol"]).strip().upper()
        if not sym_u:
            raise ValueError(f"symbols[{i}].symbol empty")

        rm = clamp_nonneg_int("reddit_mentions", blob["reddit_mentions"])
        rc = clamp_nonneg_int("reddit_comment_count", blob["reddit_comment_count"])
        scount = clamp_nonneg_int("subreddit_count", blob["subreddit_count"])
        subs_in = blob.get("subreddit_list", [])
        if not isinstance(subs_in, list):
            raise ValueError(f"symbols[{i}].subreddit_list must be array")
        subs = _sorted_unique_subs(subs_in)

        nh = clamp_nonneg_int("news_headline_count", blob["news_headline_count"])
        tpm = _parse_topic_masses(blob.get("thematic_topic_masses", []))

        rmp = infer_prior_window(rm, blob.get("reddit_mentions_prior"))
        rcp = infer_prior_window(rc, blob.get("reddit_comment_count_prior"))
        nhp = infer_prior_window(nh, blob.get("news_headline_count_prior"))

        if scount != len(subs):
            # Observability mismatch — loader warns via validate_snapshot_records_consistency
            pass

        records.append(
            ExternalAttentionSnapshotRecord(
                symbol=sym_u,
                reddit_mentions=rm,
                reddit_mentions_prior=rmp,
                reddit_comment_count=rc,
                reddit_comment_count_prior=rcp,
                subreddit_count=scount,
                subreddit_list=subs,
                news_headline_count=nh,
                news_headline_count_prior=nhp,
                thematic_topic_masses=tpm,
            )
        )

    records_sorted = tuple(sorted(records, key=lambda r: r.symbol))
    return ExternalAttentionSnapshotDocument(
        schema_version=ver,
        replay_date=replay_date,
        lookback_calendar_days=lookback,
        provider_slug=provider_slug,
        symbols=records_sorted,
        notes=notes_s,
    )


def curated_document_to_legacy_counts(
    doc: ExternalAttentionSnapshotDocument,
) -> dict[str, tuple[RedditRawCounts | None, NewsRawCounts | None]]:
    """Map curated rows → ``RedditRawCounts`` / ``NewsRawCounts`` tuples for enrichment."""
    out: dict[str, tuple[RedditRawCounts | None, NewsRawCounts | None]] = {}
    for rec in doc.symbols:
        r_raw = RedditRawCounts(
            mention_count_window=rec.reddit_mentions,
            mention_count_prior_window=rec.reddit_mentions_prior,
            comment_count_window=rec.reddit_comment_count,
            comment_count_prior_window=rec.reddit_comment_count_prior,
            distinct_subreddit_count=rec.subreddit_count,
        )
        masses = tuple(NewsThemeMass(label=lbl, mass=m) for lbl, m in rec.thematic_topic_masses)
        masses = tuple(sorted(masses, key=lambda mm: (mm.label.lower(), mm.label)))
        n_raw = NewsRawCounts(
            headline_count_window=rec.news_headline_count,
            headline_count_prior_window=rec.news_headline_count_prior,
            theme_masses=masses,
        )
        out[rec.symbol] = (r_raw, n_raw)
    return out


def validate_snapshot_records_consistency(doc: ExternalAttentionSnapshotDocument) -> list[str]:
    """Non-fatal observability warnings (deterministic ordering)."""
    warns: list[str] = []
    for rec in doc.symbols:
        if rec.subreddit_count != len(rec.subreddit_list):
            warns.append(
                f"{rec.symbol}: subreddit_count ({rec.subreddit_count}) != len(subreddit_list) ({len(rec.subreddit_list)})"
            )
    return sorted(warns)


def validate_replay_window_alignment(
    doc: ExternalAttentionSnapshotDocument,
    *,
    replay_day: date,
    lookback_calendar_days: int,
    fetch_start_utc: datetime,
    fetch_end_exclusive_utc: datetime,
) -> list[str]:
    """
    Ensure snapshot metadata matches the Layer 2 replay window parameters.

    Uses ``fetch_window_utc`` as the single source of truth for UTC bounds.
    """
    errs: list[str] = []
    if doc.replay_date != replay_day:
        errs.append(
            f"snapshot replay_date {doc.replay_date} != replay_day {replay_day}"
        )
    if doc.lookback_calendar_days != lookback_calendar_days:
        errs.append(
            f"snapshot lookback_calendar_days {doc.lookback_calendar_days} != "
            f"run lookback_calendar_days {lookback_calendar_days}"
        )
    exp_s, exp_e = fetch_window_utc(replay_day, lookback_calendar_days)
    if fetch_start_utc != exp_s:
        errs.append(
            f"fetch_start_utc {fetch_start_utc.isoformat()} != expected {exp_s.isoformat()}"
        )
    if fetch_end_exclusive_utc != exp_e:
        errs.append(
            f"fetch_end_exclusive_utc {fetch_end_exclusive_utc.isoformat()} != expected {exp_e.isoformat()}"
        )
    return errs


def _is_curated_payload(payload: dict[str, Any]) -> bool:
    if int(payload.get("schema_version", 0)) >= 2:
        return True
    sym = payload.get("symbols")
    if isinstance(sym, list):
        return True
    return False


def _parse_legacy_reddit_raw(obj: dict[str, Any]) -> RedditRawCounts:
    keys = (
        "mention_count_window",
        "mention_count_prior_window",
        "comment_count_window",
        "comment_count_prior_window",
        "distinct_subreddit_count",
    )
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"reddit_raw missing keys: {missing}")
    return RedditRawCounts(
        int(obj["mention_count_window"]),
        int(obj["mention_count_prior_window"]),
        int(obj["comment_count_window"]),
        int(obj["comment_count_prior_window"]),
        int(obj["distinct_subreddit_count"]),
    )


def _parse_legacy_news_raw(obj: dict[str, Any]) -> NewsRawCounts:
    if "headline_count_window" not in obj or "headline_count_prior_window" not in obj:
        raise ValueError("news_raw requires headline_count_window and headline_count_prior_window")
    themes_in = obj.get("theme_masses", [])
    if not isinstance(themes_in, list):
        raise ValueError("news_raw.theme_masses must be a list")
    masses: list[NewsThemeMass] = []
    for i, t in enumerate(themes_in):
        if not isinstance(t, dict):
            raise ValueError(f"theme_masses[{i}] must be object")
        masses.append(NewsThemeMass(str(t["label"]), int(t["mass"])))
    return NewsRawCounts(
        int(obj["headline_count_window"]),
        int(obj["headline_count_prior_window"]),
        tuple(sorted(masses, key=lambda m: (m.label.lower(), m.label))),
    )


def load_legacy_nested_snapshot(
    payload: dict[str, Any],
) -> tuple[str, dict[str, tuple[RedditRawCounts | None, NewsRawCounts | None]]]:
    """Schema v1: symbols dict → reddit_raw / news_raw blobs."""
    syms_in = payload.get("symbols")
    if not isinstance(syms_in, dict):
        raise ValueError("legacy snapshot.symbols must be object mapping symbol → blob")
    provider_slug = str(payload.get("provider_slug") or "historical_snapshot_v1")

    out: dict[str, tuple[RedditRawCounts | None, NewsRawCounts | None]] = {}
    for sym_k, blob in sorted(syms_in.items(), key=lambda kv: str(kv[0]).upper()):
        sym_u = str(sym_k).strip().upper()
        if not sym_u:
            continue
        if not isinstance(blob, dict):
            raise ValueError(f"symbols[{sym_k!r}] must be object")
        r_raw = None
        n_raw = None
        if "reddit_raw" in blob:
            r_raw = _parse_legacy_reddit_raw(blob["reddit_raw"])
        if "news_raw" in blob:
            n_raw = _parse_legacy_news_raw(blob["news_raw"])
        if r_raw is None and n_raw is None:
            raise ValueError(f"symbol {sym_u}: supply reddit_raw and/or news_raw")
        out[sym_u] = (r_raw, n_raw)
    return provider_slug, out


def load_external_attention_snapshot_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("snapshot root must be object")
    return payload


def load_external_snapshot_for_replay(
    path: Path | str,
) -> tuple[
    str,
    dict[str, tuple[RedditRawCounts | None, NewsRawCounts | None]],
    ExternalAttentionSnapshotDocument | None,
]:
    """
    Load snapshot file → ``(provider_slug, legacy_counts_map, curated_doc_or_none)``.

    Curated documents populate ``provider_slug`` from JSON; legacy leaves replay alignment fields absent
    (``curated_doc`` is ``None``).
    """
    payload = load_external_attention_snapshot_json(path)
    if _is_curated_payload(payload):
        doc = parse_curated_snapshot(payload)
        blob = curated_document_to_legacy_counts(doc)
        return doc.provider_slug, blob, doc

    slug, blob = load_legacy_nested_snapshot(payload)
    return slug, blob, None


def json_snapshot_preview(doc: ExternalAttentionSnapshotDocument) -> dict[str, Any]:
    """Stable diagnostic summary for CLI / logs."""
    return {
        "schema_version": doc.schema_version,
        "replay_date": doc.replay_date.isoformat(),
        "lookback_calendar_days": doc.lookback_calendar_days,
        "provider_slug": doc.provider_slug,
        "symbols": [r.symbol for r in doc.symbols],
        "notes": doc.notes,
    }
