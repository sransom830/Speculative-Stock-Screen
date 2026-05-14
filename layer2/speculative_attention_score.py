"""
Market-native speculative attention (Layer 2).

Combines Layer 1 ecosystem rows with multi-day OHLCV (e.g. provider daily bars).
Does not classify fragility, confirmation, or direction — only attention concentration.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Sequence

import pandas as pd

from layer1.cohorts import cohort_labels_for_symbol

from .factors import (
    dollar_volume_expansion_score,
    ecosystem_multiplier,
    momentum_persistence_score,
    price_acceleration_score,
    realized_vol_expansion_score,
    relative_volume_score,
)
from .models import (
    NARRATIVE_COHORT_KEYS,
    SPECULATIVE_BATTLEFIELD_COHORT_KEYS,
    SpeculativeAttentionConfig,
    SpeculativeAttentionState,
)

_REQUIRED_LAYER1_COLUMNS = frozenset({"symbol"})


def _push_diagnostic(
    diagnostics_out: list[dict[str, Any]] | None,
    row: Mapping[str, Any],
) -> None:
    if diagnostics_out is not None:
        diagnostics_out.append(dict(row))


def _is_bad_numeric(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float):
        return math.isnan(x) or math.isinf(x)
    return False


def _factors_contain_bad(
    rel_v: float,
    dol_v: float,
    rv_e: float,
    p_acc: float,
    mom: float,
) -> bool:
    return any(_is_bad_numeric(v) for v in (rel_v, dol_v, rv_e, p_acc, mom))


def _normalize_weights(cfg: SpeculativeAttentionConfig) -> Mapping[str, float]:
    w = {
        "relative_volume": cfg.weight_relative_volume,
        "dollar_volume_expansion": cfg.weight_dollar_volume_expansion,
        "realized_vol_expansion": cfg.weight_realized_vol_expansion,
        "price_acceleration": cfg.weight_price_acceleration,
        "momentum_persistence": cfg.weight_momentum_persistence,
    }
    s = sum(w.values())
    if s <= 0:
        raise ValueError("SpeculativeAttentionConfig weights must sum to a positive value.")
    return {k: v / s for k, v in w.items()}


def _cohorts_for_symbol(
    symbol: str,
    layer1_by_symbol: Mapping[str, MutableMapping[str, Any]],
) -> list[str]:
    row = layer1_by_symbol.get(symbol)
    if row is None:
        return cohort_labels_for_symbol(symbol)
    raw = row.get("source_cohorts")
    if isinstance(raw, list) and len(raw) > 0:
        return [str(x) for x in raw]
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return cohort_labels_for_symbol(symbol)
    return cohort_labels_for_symbol(symbol)


def _classify_state(
    *,
    score: float,
    cohorts: Sequence[str],
    rel_vol: float,
    dollar_vol: float,
    rv_exp: float,
    price_acc: float,
    momentum: float,
    cfg: SpeculativeAttentionConfig,
) -> SpeculativeAttentionState:
    cset = set(cohorts)
    participation = 0.5 * (rel_vol + dollar_vol)
    acceleration = 0.5 * (price_acc + momentum)
    vol_tag = cfg.cohort_volatility_tag in cset
    battlefield = cset & SPECULATIVE_BATTLEFIELD_COHORT_KEYS
    liquidity_signal = max(participation, rv_exp)

    if (
        score >= cfg.score_retail_frenzy_min
        and "retail_meme" in cset
        and participation >= cfg.score_retail_frenzy_participation
    ):
        return SpeculativeAttentionState.RETAIL_FRENZY
    if rv_exp >= cfg.factor_vol_cluster_strong or (
        rv_exp >= cfg.factor_vol_cluster and vol_tag
    ):
        return SpeculativeAttentionState.VOLATILITY_CLUSTER
    if acceleration >= cfg.factor_narrative_accel and bool(cset & NARRATIVE_COHORT_KEYS):
        return SpeculativeAttentionState.NARRATIVE_ACCELERATION
    if battlefield:
        if (
            score >= cfg.score_high_speculation_ecosystem
            and liquidity_signal >= cfg.ecosystem_high_spec_liquidity_floor
        ):
            return SpeculativeAttentionState.HIGH_SPECULATION
    elif score >= cfg.score_high_speculation_generic:
        return SpeculativeAttentionState.HIGH_SPECULATION
    return SpeculativeAttentionState.NEUTRAL


def compute_speculative_attention(
    layer1_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    *,
    config: SpeculativeAttentionConfig | None = None,
    diagnostics_out: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Produce 0–100 ``speculative_attention_score`` and a primary ``speculative_attention_state``.

    Parameters
    ----------
    layer1_df :
        Canonical Layer 1 table (at least ``symbol``; ``source_cohorts`` optional).
    bars_df :
        Daily bars: ``symbol``, ``timestamp_utc``, ``open``, ``high``, ``low``, ``close``, ``volume``.
    diagnostics_out :
        If provided, appends one dict per ``layer1`` symbol attempt: ``status`` (``scored`` /
        ``skipped``), ``skip_reason``, bar counts, factors, cohorts — for calibration / replay
        debugging without changing scores.
    """
    if layer1_df.empty:
        return pd.DataFrame()
    missing = _REQUIRED_LAYER1_COLUMNS - frozenset(layer1_df.columns)
    if missing:
        raise ValueError(f"layer1_df missing columns: {sorted(missing)}")
    required_bars = {"timestamp_utc", "close", "volume"}

    cfg = config or SpeculativeAttentionConfig()
    w = _normalize_weights(cfg)

    layer1_by_symbol: dict[str, MutableMapping[str, Any]] = {
        str(r["symbol"]): dict(r)
        for _, r in layer1_df.iterrows()
    }
    allowed = set(layer1_by_symbol)
    need_bars = cfg.relative_volume_window + 1

    if bars_df.empty or "symbol" not in bars_df.columns:
        if diagnostics_out is not None:
            for sym in sorted(allowed):
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": False,
                        "bar_row_count": 0,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": "empty_or_invalid_bars_dataframe",
                        "factor_relative_volume": None,
                        "factor_dollar_volume_expansion": None,
                        "factor_realized_vol_expansion": None,
                        "factor_price_acceleration": None,
                        "factor_momentum_persistence": None,
                        "composite_pre_multiplier": None,
                        "ecosystem_multiplier": None,
                        "source_cohorts": _cohorts_for_symbol(sym, layer1_by_symbol),
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": None,
                    },
                )
        return pd.DataFrame()

    if not required_bars.issubset(bars_df.columns):
        raise ValueError(f"bars_df must include columns: {sorted(required_bars)}")

    symbols_in_bars = set(bars_df["symbol"].astype(str).unique())
    if diagnostics_out is not None:
        for sym in sorted(allowed):
            if sym not in symbols_in_bars:
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": False,
                        "bar_row_count": 0,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": "no_bar_rows_for_symbol_in_request",
                        "factor_relative_volume": None,
                        "factor_dollar_volume_expansion": None,
                        "factor_realized_vol_expansion": None,
                        "factor_price_acceleration": None,
                        "factor_momentum_persistence": None,
                        "composite_pre_multiplier": None,
                        "ecosystem_multiplier": None,
                        "source_cohorts": _cohorts_for_symbol(sym, layer1_by_symbol),
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": None,
                    },
                )

    rows: list[dict[str, Any]] = []
    for symbol, g in bars_df.groupby("symbol", sort=False):
        sym = str(symbol)
        raw_bar_count = int(len(g))
        if sym not in allowed:
            if diagnostics_out is not None:
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": False,
                        "bar_row_count": raw_bar_count,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": "symbol_not_in_layer1_snapshot",
                        "factor_relative_volume": None,
                        "factor_dollar_volume_expansion": None,
                        "factor_realized_vol_expansion": None,
                        "factor_price_acceleration": None,
                        "factor_momentum_persistence": None,
                        "composite_pre_multiplier": None,
                        "ecosystem_multiplier": None,
                        "source_cohorts": None,
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": None,
                    },
                )
            continue

        g2 = g.sort_values("timestamp_utc")
        close = g2["close"]
        vol = g2["volume"]
        n_close = int(len(close))

        if close.isna().all() or vol.isna().all():
            if diagnostics_out is not None:
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": True,
                        "bar_row_count": n_close,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": "missing_group_data_all_close_or_volume_nan",
                        "factor_relative_volume": None,
                        "factor_dollar_volume_expansion": None,
                        "factor_realized_vol_expansion": None,
                        "factor_price_acceleration": None,
                        "factor_momentum_persistence": None,
                        "composite_pre_multiplier": None,
                        "ecosystem_multiplier": None,
                        "source_cohorts": _cohorts_for_symbol(sym, layer1_by_symbol),
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": None,
                    },
                )
            continue

        if n_close < need_bars:
            if diagnostics_out is not None:
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": True,
                        "bar_row_count": n_close,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": f"insufficient_bars_got_{n_close}_need_{need_bars}",
                        "factor_relative_volume": None,
                        "factor_dollar_volume_expansion": None,
                        "factor_realized_vol_expansion": None,
                        "factor_price_acceleration": None,
                        "factor_momentum_persistence": None,
                        "composite_pre_multiplier": None,
                        "ecosystem_multiplier": None,
                        "source_cohorts": _cohorts_for_symbol(sym, layer1_by_symbol),
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": None,
                    },
                )
            continue

        exc_msg: str | None = None
        rel_v = dol_v = rv_e = p_acc = mom = float("nan")
        try:
            rel_v = relative_volume_score(vol, window=cfg.relative_volume_window)
            dol_v = dollar_volume_expansion_score(
                close,
                vol,
                short=cfg.dollar_volume_short,
                long=cfg.dollar_volume_long,
            )
            rv_e = realized_vol_expansion_score(
                close,
                short=cfg.rv_short,
                long=cfg.rv_long,
            )
            p_acc = price_acceleration_score(
                close,
                short=cfg.accel_short,
                long=cfg.accel_long,
            )
            mom = momentum_persistence_score(close, days=cfg.momentum_days)
        except Exception as e:
            exc_msg = f"{type(e).__name__}: {e}"
            if diagnostics_out is not None:
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": True,
                        "bar_row_count": n_close,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": "factor_computation_exception",
                        "factor_relative_volume": None,
                        "factor_dollar_volume_expansion": None,
                        "factor_realized_vol_expansion": None,
                        "factor_price_acceleration": None,
                        "factor_momentum_persistence": None,
                        "composite_pre_multiplier": None,
                        "ecosystem_multiplier": None,
                        "source_cohorts": _cohorts_for_symbol(sym, layer1_by_symbol),
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": exc_msg,
                    },
                )
            continue

        if _factors_contain_bad(rel_v, dol_v, rv_e, p_acc, mom):
            if diagnostics_out is not None:
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": True,
                        "bar_row_count": n_close,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": "nan_or_invalid_factor_output",
                        "factor_relative_volume": rel_v,
                        "factor_dollar_volume_expansion": dol_v,
                        "factor_realized_vol_expansion": rv_e,
                        "factor_price_acceleration": p_acc,
                        "factor_momentum_persistence": mom,
                        "composite_pre_multiplier": None,
                        "ecosystem_multiplier": None,
                        "source_cohorts": _cohorts_for_symbol(sym, layer1_by_symbol),
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": None,
                    },
                )
            continue

        composite = (
            w["relative_volume"] * rel_v
            + w["dollar_volume_expansion"] * dol_v
            + w["realized_vol_expansion"] * rv_e
            + w["price_acceleration"] * p_acc
            + w["momentum_persistence"] * mom
        )
        if _is_bad_numeric(composite):
            if diagnostics_out is not None:
                _push_diagnostic(
                    diagnostics_out,
                    {
                        "symbol": sym,
                        "entered_scoring_loop": True,
                        "bar_row_count": n_close,
                        "bars_required_min": need_bars,
                        "status": "skipped",
                        "skip_reason": "nan_composite_pre_multiplier",
                        "factor_relative_volume": rel_v,
                        "factor_dollar_volume_expansion": dol_v,
                        "factor_realized_vol_expansion": rv_e,
                        "factor_price_acceleration": p_acc,
                        "factor_momentum_persistence": mom,
                        "composite_pre_multiplier": composite,
                        "ecosystem_multiplier": None,
                        "source_cohorts": _cohorts_for_symbol(sym, layer1_by_symbol),
                        "speculative_attention_score": None,
                        "speculative_attention_state": None,
                        "exception_message": None,
                    },
                )
            continue

        cohorts = _cohorts_for_symbol(sym, layer1_by_symbol)
        em = ecosystem_multiplier(cohorts, cap=cfg.ecosystem_cap)
        score = float(max(0.0, min(100.0, composite * 100.0 * em)))
        state = _classify_state(
            score=score,
            cohorts=cohorts,
            rel_vol=rel_v,
            dollar_vol=dol_v,
            rv_exp=rv_e,
            price_acc=p_acc,
            momentum=mom,
            cfg=cfg,
        )
        participation_composite = 0.5 * (rel_v + dol_v)
        acceleration_composite = 0.5 * (p_acc + mom)

        if diagnostics_out is not None:
            _push_diagnostic(
                diagnostics_out,
                {
                    "symbol": sym,
                    "entered_scoring_loop": True,
                    "bar_row_count": n_close,
                    "bars_required_min": need_bars,
                    "status": "scored",
                    "skip_reason": None,
                    "factor_relative_volume": rel_v,
                    "factor_dollar_volume_expansion": dol_v,
                    "factor_realized_vol_expansion": rv_e,
                    "factor_price_acceleration": p_acc,
                    "factor_momentum_persistence": mom,
                    "composite_pre_multiplier": composite,
                    "ecosystem_multiplier": em,
                    "source_cohorts": cohorts,
                    "speculative_attention_score": score,
                    "speculative_attention_state": state.value,
                    "exception_message": None,
                },
            )

        rows.append(
            {
                "symbol": sym,
                "speculative_attention_score": score,
                "speculative_attention_state": state.value,
                "ecosystem_multiplier": em,
                "source_cohorts": cohorts,
                "factor_relative_volume": rel_v,
                "factor_dollar_volume_expansion": dol_v,
                "factor_realized_vol_expansion": rv_e,
                "factor_price_acceleration": p_acc,
                "factor_momentum_persistence": mom,
                "factor_participation_composite": participation_composite,
                "factor_acceleration_composite": acceleration_composite,
                "factor_volatility_composite": rv_e,
            }
        )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    keep = ["symbol", "company_name", "exchange", "asset_class"]
    meta = layer1_df[[c for c in keep if c in layer1_df.columns]]
    if not meta.empty:
        out = out.merge(meta, on="symbol", how="left")
    return out
