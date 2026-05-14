#!/usr/bin/env python3
"""
Layer 2 ontology disagreement analysis — behavioral diagnostics only.

Loads truth-set validation runs (replay + expected states), focuses on FAIL rows,
compares factor/ecosystem attribution vs PASS symbols from the same event, prints
interpretable gap narratives and non-binding ontology refinement hints.

Does not modify scoring, weights, or thresholds. Not ML optimization.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from layer2.truth_set_validation import (  # noqa: E402
    attention_row,
    replay_cache_key,
    run_truth_set_validation,
)

FACTOR_KEYS = (
    "factor_relative_volume",
    "factor_dollar_volume_expansion",
    "factor_realized_vol_expansion",
    "factor_price_acceleration",
    "factor_momentum_persistence",
    "factor_participation_composite",
    "factor_acceleration_composite",
    "ecosystem_multiplier",
)


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _median_nonempty(vals: list[float]) -> float | None:
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return None
    return float(median(clean))


def pass_symbols_same_date(results: list[dict[str, Any]], *, stem: str, replay_date: str) -> list[str]:
    """PASS verdict symbols for one event stem + replay_date (deterministic sort)."""
    syms = []
    for r in results:
        if r["event_file_stem"] != stem:
            continue
        if str(r["replay_date"]) != replay_date:
            continue
        if r["verdict"] != "PASS":
            continue
        syms.append(str(r["symbol"]).upper())
    return sorted(set(syms))


def median_factors_event_pass_pool(
    results: list[dict[str, Any]],
    replay_cache: dict[tuple[Any, int, str], Any],
    *,
    stem: str,
) -> dict[str, float | None]:
    """Median factors across all PASS checks in this event (each row uses its own replay cache entry)."""
    pass_rows = sorted(
        [r for r in results if r["event_file_stem"] == stem and r["verdict"] == "PASS"],
        key=lambda x: (str(x["replay_date"]), str(x["symbol"]).upper()),
    )
    collected: dict[str, list[float]] = {k: [] for k in FACTOR_KEYS}
    for pr in pass_rows:
        key = replay_cache_key(pr)
        art = replay_cache.get(key)
        if art is None:
            continue
        row = attention_row(art.attention_df, str(pr["symbol"]).upper())
        if row is None:
            continue
        for k in FACTOR_KEYS:
            v = _safe_float(row.get(k))
            if v is not None:
                collected[k].append(v)
    return {k: _median_nonempty(collected[k]) for k in FACTOR_KEYS}


def median_factors_for_symbols(artifacts: Any, symbols_upper: list[str]) -> dict[str, float | None]:
    collected: dict[str, list[float]] = {k: [] for k in FACTOR_KEYS}
    for sym in symbols_upper:
        row = attention_row(artifacts.attention_df, sym)
        if row is None:
            continue
        for k in FACTOR_KEYS:
            v = _safe_float(row.get(k))
            if v is not None:
                collected[k].append(v)
    return {k: _median_nonempty(collected[k]) for k in FACTOR_KEYS}


def behavioral_interpretations(
    *,
    fail_factors: dict[str, float | None],
    pass_medians: dict[str, float | None],
    epsilon: float,
    observed_state: str,
    expected_states: list[str],
) -> list[str]:
    """Deterministic ordered diagnostics comparing FAIL row to PASS medians."""
    lines: list[str] = []

    def weaker(dim: str, label: str) -> None:
        fv = fail_factors.get(dim)
        pv = pass_medians.get(dim)
        if fv is None or pv is None:
            return
        if fv < pv - epsilon:
            lines.append(f"{label}: FAIL value {fv:.4f} < same-event PASS median {pv:.4f} (Δ≈{fv - pv:.4f})")

    weaker("factor_relative_volume", "Participation (relative_volume)")
    weaker("factor_dollar_volume_expansion", "Participation (dollar_volume_expansion)")
    weaker("factor_participation_composite", "Participation composite")

    weaker("factor_price_acceleration", "Acceleration (price_acceleration)")
    weaker("factor_momentum_persistence", "Acceleration (momentum_persistence)")
    weaker("factor_acceleration_composite", "Acceleration composite")

    weaker("factor_realized_vol_expansion", "Volatility / clustering signal (realized_vol_expansion)")
    weaker("ecosystem_multiplier", "Ecosystem amplification (multiplier)")

    exp_set = frozenset(expected_states)
    if "VOLATILITY_CLUSTER" in exp_set and observed_state == "NEUTRAL":
        rv_f = fail_factors.get("factor_realized_vol_expansion")
        rv_p = pass_medians.get("factor_realized_vol_expansion")
        if rv_f is not None and rv_p is not None and rv_f < rv_p:
            lines.append(
                "Regime mismatch note: truth-set expects volatility-cluster regime cadence but "
                f"observed NEUTRAL with weaker rv_exp ({rv_f:.4f}) vs PASS median ({rv_p:.4f})."
            )

    if "NARRATIVE_ACCELERATION" in exp_set and observed_state not in ("NARRATIVE_ACCELERATION",):
        acc_f = fail_factors.get("factor_acceleration_composite")
        acc_p = pass_medians.get("factor_acceleration_composite")
        if acc_f is not None and acc_p is not None and acc_f < acc_p - epsilon:
            lines.append(
                "Regime mismatch note: narrative acceleration label absent while acceleration composite "
                f"lags PASS median ({acc_f:.4f} vs {acc_p:.4f})."
            )

    return sorted(set(lines))


def ontology_refinement_suggestions(
    *,
    event_stem: str,
    interp_lines: list[str],
    fail_factors: dict[str, float | None],
    pass_medians: dict[str, float | None],
    epsilon: float,
    behavioral_tags: list[str],
) -> list[str]:
    """Non-binding, human-reviewed ontology hints (deterministic ordering)."""
    sug: set[str] = set()
    blob = " ".join(interp_lines).lower()
    tags_l = [str(t).lower() for t in behavioral_tags]

    em_f = fail_factors.get("ecosystem_multiplier")
    em_p = pass_medians.get("ecosystem_multiplier")
    eco_weak = (em_f is not None and em_p is not None and em_f < em_p - epsilon) or (
        "ecosystem amplification" in blob
    )

    acc_weak = "acceleration composite" in blob or "acceleration (" in blob
    part_weak = "participation" in blob
    vol_weak = "volatility / clustering" in blob or "rv_exp" in blob

    if event_stem == "crypto_reflexivity_2021" or any("crypto" in t for t in tags_l):
        if eco_weak:
            sug.add(
                "Diagnostic suggestion (manual review): crypto reflexivity episodes may need stronger "
                "ecosystem weighting or richer crypto-linked cohort tagging — do not auto-tune."
            )
        if vol_weak:
            sug.add(
                "Diagnostic suggestion (manual review): BTC-beta equities may need clearer volatility-cluster "
                "path vs neutral baseline — ontology gates only."
            )

    if event_stem == "ai_narrative_2024" or any("ai" in t for t in tags_l):
        if acc_weak:
            sug.add(
                "Diagnostic suggestion (manual review): AI narrative windows may require revisiting "
                "narrative acceleration thresholds vs cohort intersection rules — ontology only."
            )
        if eco_weak:
            sug.add(
                "Diagnostic suggestion (manual review): thematic concentration may warrant ecosystem "
                "multiplier calibration for semiconductor / AI cohorts — manual judgment."
            )

    if event_stem == "gme_january_2021" or any("meme" in t for t in tags_l):
        if part_weak:
            sug.add(
                "Diagnostic suggestion (manual review): meme battlefield peaks may need participation-factor "
                "emphasis or retail cohort fidelity checks — not score auto-edits."
            )
        if eco_weak:
            sug.add(
                "Diagnostic suggestion (manual review): correlated meme books may need battlefield ecosystem "
                "treatment review — diagnostics only."
            )

    if not sug and interp_lines:
        sug.add(
            "Diagnostic suggestion (manual review): disagreement drivers are mixed; inspect cohort labels "
            "(Layer 1 snapshot) and rule ordering before changing ontology."
        )

    return sorted(sug)


def format_source_cohorts(val: Any) -> str:
    if isinstance(val, list):
        return json.dumps(val)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze FAIL rows from Layer 2 truth-set validation (ontology diagnostics)."
    )
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
        help="Override lookback_calendar_days (must match validation run).",
    )
    parser.add_argument(
        "--feed-override",
        choices=("iex", "sip"),
        default=None,
        help="Override alpaca_feed from each event file.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Numeric gap threshold for weaker/stronger vs PASS median (default 0.05).",
    )
    args = parser.parse_args()
    epsilon = float(args.epsilon)
    if epsilon < 0:
        print("--epsilon must be non-negative.", file=sys.stderr)
        sys.exit(1)

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
    fails = [r for r in all_results if r["verdict"] == "FAIL"]
    fails = sorted(fails, key=lambda x: (x["event_file_stem"], x["replay_date"], x["symbol"]))

    print()
    print("=" * 72)
    print("Layer 2 validation disagreement analysis (ontology calibration)")
    print("=" * 72)
    print(f"events_dir: {events_dir}")
    print(f"total_checks: {len(all_results)} | FAIL: {len(fails)}")
    print(f"comparison_epsilon: {epsilon}")
    print()
    print("Outputs are interpretability diagnostics only — no scoring changes are applied.")

    if not fails:
        print()
        print("No FAIL cases in this validation run.")
        return

    for fr in fails:
        key = replay_cache_key(fr)
        art = replay_cache.get(key)
        if art is None:
            print(f"\nINTERNAL: missing replay cache for {key}", file=sys.stderr)
            continue

        sym_u = str(fr["symbol"]).upper()
        row = attention_row(art.attention_df, sym_u)
        print()
        print("#" * 72)
        print(
            f"# FAIL | {fr['event_file_stem']} | {fr['replay_date']} | {sym_u} "
            f"| observed={fr['observed_speculative_state']}"
        )
        print("#" * 72)
        print(f"expected_speculative_states: {fr['expected_speculative_states']}")
        print(f"observed_speculative_state:  {fr['observed_speculative_state']}")
        sc = fr.get("observed_score")
        print(f"speculative_attention_score: {sc if sc is None else float(sc):.4g}")
        print(f"behavioral_tags (truth-set): {fr['behavioral_tags']}")
        print(f"detail: {fr['detail']}")

        if row is None:
            print("No attention_df row for symbol (unexpected for FAIL). Skipping factor block.")
            continue

        print()
        print("--- factor attribution & ecosystem (FAIL row) ---")
        for k in FACTOR_KEYS:
            v = _safe_float(row.get(k))
            print(f"  {k}: {v if v is None else round(v, 6)}")
        print(f"  source_cohorts: {format_source_cohorts(row.get('source_cohorts'))}")

        pass_same_day = pass_symbols_same_date(all_results, stem=fr["event_file_stem"], replay_date=fr["replay_date"])
        baseline_label = "same replay_date PASS symbols"
        if pass_same_day:
            pass_medians = median_factors_for_symbols(art, pass_same_day)
        else:
            pass_medians = median_factors_event_pass_pool(all_results, replay_cache, stem=fr["event_file_stem"])
            baseline_label = "event-wide PASS pool (each PASS row scored on its own replay_date)"

        print()
        print(f"--- vs {baseline_label} ---")
        if pass_same_day:
            print(f"  PASS symbols (same date): {pass_same_day}")
        else:
            pass_pool = sorted(
                {
                    str(r["symbol"]).upper()
                    for r in all_results
                    if r["event_file_stem"] == fr["event_file_stem"] and r["verdict"] == "PASS"
                }
            )
            print(f"  PASS symbol pool (event): {pass_pool if pass_pool else '(none)'}")

        if pass_same_day:
            ref_syms = pass_same_day
        else:
            ref_syms = sorted(
                {
                    str(r["symbol"]).upper()
                    for r in all_results
                    if r["event_file_stem"] == fr["event_file_stem"] and r["verdict"] == "PASS"
                }
            )

        if not ref_syms:
            print("  (no PASS symbols in this event — median comparison omitted)")
        else:
            fail_factors = {k: _safe_float(row.get(k)) for k in FACTOR_KEYS}
            for k in FACTOR_KEYS:
                fm = fail_factors[k]
                pm = pass_medians[k]
                if fm is None or pm is None:
                    delta = None
                else:
                    delta = fm - pm
                d_s = "n/a" if delta is None else f"{delta:+.4f}"
                pm_s = "n/a" if pm is None else f"{pm:.4f}"
                fm_s = "n/a" if fm is None else f"{fm:.4f}"
                print(f"  {k}: FAIL={fm_s}  PASS_median={pm_s}  (FAIL−median={d_s})")

        fail_factors = {k: _safe_float(row.get(k)) for k in FACTOR_KEYS}
        interp = behavioral_interpretations(
            fail_factors=fail_factors,
            pass_medians=pass_medians,
            epsilon=epsilon,
            observed_state=str(fr["observed_speculative_state"]),
            expected_states=list(fr["expected_speculative_states"]),
        )
        print()
        print("--- behavioral interpretation diagnostics ---")
        if not interp:
            print("  (no automated gap vs PASS median exceeded threshold — inspect raw factors above)")
        for line in interp:
            print(f"  • {line}")

        sugg = ontology_refinement_suggestions(
            event_stem=str(fr["event_file_stem"]),
            interp_lines=interp,
            fail_factors=fail_factors,
            pass_medians=pass_medians,
            epsilon=epsilon,
            behavioral_tags=list(fr["behavioral_tags"]),
        )
        print()
        print("--- ontology refinement suggestions (diagnostic only; no auto-edits) ---")
        for s in sugg:
            print(f"  • {s}")


if __name__ == "__main__":
    main()
