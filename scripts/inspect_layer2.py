#!/usr/bin/env python3
"""
Layer 2 inspection / calibration cockpit.

Loads ``layer2_speculative_population`` and prints rankings, factor leaders,
and cohort/multiplier diagnostics (no recomputation — DB snapshot only).

Note: ``factor_dollar_volume_expansion`` is not persisted on the population table;
only relative volume, realized-vol expansion, and acceleration composite are stored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from layer2.fetch_population import fetch_layer2_speculative_population_df  # noqa: E402

# DB column -> engine naming used in calibration discussions
FACT_REL = "relative_volume_factor"
FACT_VOL = "volatility_factor"
FACT_ACC = "acceleration_factor"
STATE_COL = "attention_state"
SCORE_COL = "speculative_attention_score"
MULT_COL = "ecosystem_multiplier"


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _print_separator(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _print_top_ranked(df: pd.DataFrame, n: int) -> None:
    _print_separator(f"Top {n} by {SCORE_COL}")
    if df.empty or SCORE_COL not in df.columns:
        print("(no data)")
        return
    view = df.nlargest(n, SCORE_COL)
    cols = [c for c in [STATE_COL, SCORE_COL, MULT_COL, FACT_REL, FACT_VOL, FACT_ACC] if c in view.columns]
    if "symbol" in view.columns:
        out = view[["symbol"] + cols].copy()
    else:
        out = view[cols].copy()
    print(out.to_string(index=False))


def _print_detail_block(row: pd.Series) -> None:
    sym = row.get("symbol", "?")
    print(f"\n--- {sym} ---")
    print(f"speculative_attention_score: {row.get(SCORE_COL)}")
    print(f"speculative_attention_state: {row.get(STATE_COL)}")
    print(f"factor_relative_volume: {row.get(FACT_REL)}")
    print("factor_dollar_volume_expansion: (not stored in layer2_speculative_population)")
    print(f"factor_realized_vol_expansion: {row.get(FACT_VOL)}")
    print(f"factor_acceleration_composite: {row.get(FACT_ACC)}")
    print(f"ecosystem_multiplier: {row.get(MULT_COL)}")
    print(f"source_cohorts: {row.get('source_cohorts', [])}")
    if "last_updated" in row.index:
        print(f"last_updated: {row.get('last_updated')}")


def _print_factor_leaders(df: pd.DataFrame, n: int, title: str, col: str) -> None:
    _print_separator(title)
    if df.empty or col not in df.columns:
        print("(no data)")
        return
    work = df.copy()
    work["_v"] = _numeric(work, col)
    top = work.nlargest(n, "_v")[["symbol", col]].copy()
    print(top.to_string(index=False))


def _print_by_state(df: pd.DataFrame, n: int) -> None:
    _print_separator(f"Top {n} per {STATE_COL} (by {SCORE_COL})")
    if df.empty or STATE_COL not in df.columns or SCORE_COL not in df.columns:
        print("(no data)")
        return
    for state in sorted(df[STATE_COL].dropna().astype(str).unique()):
        sub = df.loc[df[STATE_COL].astype(str) == state]
        top = sub.nlargest(n, SCORE_COL)
        print(f"\n[{state}] ({len(sub)} names)")
        cols = [c for c in ["symbol", SCORE_COL, MULT_COL, FACT_REL, FACT_VOL, FACT_ACC] if c in top.columns]
        print(top[cols].to_string(index=False))


def _print_cohort_distribution(df: pd.DataFrame) -> None:
    _print_separator("Cohort tag frequency (exploded source_cohorts)")
    if df.empty or "source_cohorts" not in df.columns:
        print("(no data)")
        return
    exploded = df.explode("source_cohorts")
    exploded = exploded[exploded["source_cohorts"].notna() & (exploded["source_cohorts"] != "")]
    if exploded.empty:
        print("(no cohort tags)")
        return
    counts = exploded["source_cohorts"].value_counts().sort_values(ascending=False)
    print(counts.to_string())


def _print_multiplier_distribution(df: pd.DataFrame) -> None:
    _print_separator("Ecosystem multiplier distribution")
    if df.empty or MULT_COL not in df.columns:
        print("(no data)")
        return
    m = _numeric(df, MULT_COL).dropna()
    if m.empty:
        print("(no multiplier values)")
        return
    print(m.describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_string())
    mmax, mmin = float(m.max()), float(m.min())
    if mmax <= mmin + 1e-9:
        print("\n(all non-null multipliers are identical)")
        return
    try:
        cats = pd.qcut(m, q=5, duplicates="drop")
        print("\ncounts by quintile bucket:")
        print(cats.value_counts().sort_index().to_string())
    except ValueError:
        print("\nvalue counts (top 15):")
        print(m.round(4).value_counts().head(15).to_string())


def _print_state_distribution(df: pd.DataFrame) -> None:
    _print_separator("Attention state counts")
    if df.empty or STATE_COL not in df.columns:
        print("(no data)")
        return
    print(df[STATE_COL].astype(str).value_counts().sort_values(ascending=False).to_string())


def _print_factor_attribution_summary(df: pd.DataFrame, top_n: int) -> None:
    _print_separator(f"Factor means — full universe vs top {top_n} by score")
    if df.empty or SCORE_COL not in df.columns:
        print("(no data)")
        return
    top = df.nlargest(top_n, SCORE_COL)
    for name, col in [
        ("relative_volume (stored)", FACT_REL),
        ("realized_vol_expansion (stored)", FACT_VOL),
        ("acceleration_composite (stored)", FACT_ACC),
        ("ecosystem_multiplier", MULT_COL),
    ]:
        if col not in df.columns:
            continue
        full_m = _numeric(df, col).mean()
        top_m = _numeric(top, col).mean()
        print(f"{name}: universe_mean={full_m:.4f} top{top_n}_mean={top_m:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Layer 2 speculative population snapshot.")
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="How many rows for ranked leaderboards (default 25).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Comma-separated symbols to compare (e.g. MSTR,GME,PLTR). Case-insensitive.",
    )
    args = parser.parse_args()
    top_n = max(1, int(args.top))

    load_dotenv(PROJECT_ROOT / ".env")
    df = fetch_layer2_speculative_population_df()

    if df.empty:
        print("layer2_speculative_population: empty or unreachable.")
        return

    df = df.copy()
    if "symbol" not in df.columns:
        print("population table missing 'symbol' column.")
        return
    for c in (FACT_REL, FACT_VOL, FACT_ACC, MULT_COL, SCORE_COL):
        if c in df.columns:
            df[c] = _numeric(df, c)

    requested_order = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if requested_order:
        _print_separator("Comparison mode — requested symbols")
        print("universe rows:", len(df))
        for s in requested_order:
            row_df = df.loc[df["symbol"].astype(str).str.upper() == s]
            if row_df.empty:
                print(f"\n--- {s} ---\n(not in layer2_speculative_population)")
                continue
            _print_detail_block(row_df.iloc[0])

    _print_top_ranked(df, top_n)
    _print_state_distribution(df)
    _print_factor_attribution_summary(df, top_n)

    print("\n--- Per-symbol detail (top ranked) ---")
    leaders = df.nlargest(top_n, SCORE_COL) if SCORE_COL in df.columns else df.head(0)
    for _, row in leaders.iterrows():
        _print_detail_block(row)

    _print_factor_leaders(df, top_n, f"Top {top_n} by {FACT_REL} (relative volume)", FACT_REL)
    _print_factor_leaders(df, top_n, f"Top {top_n} by {FACT_VOL} (realized vol expansion)", FACT_VOL)
    _print_factor_leaders(df, top_n, f"Top {top_n} by {FACT_ACC} (acceleration composite)", FACT_ACC)
    _print_factor_leaders(df, top_n, f"Top {top_n} by {MULT_COL}", MULT_COL)

    _print_by_state(df, min(10, top_n))
    _print_cohort_distribution(df)
    _print_multiplier_distribution(df)

    print()
    print(
        "Note: factor_dollar_volume_expansion feeds the live score but is not written to "
        "layer2_speculative_population; extend persistence to store it for full attribution."
    )


if __name__ == "__main__":
    main()
