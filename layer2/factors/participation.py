"""Participation-style attention factors: liquidity surge vs its own baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def relative_volume_score(
    volume: pd.Series,
    *,
    window: int = 20,
) -> float:
    """
    Last bar volume vs trailing mean of prior ``window`` bars (excluding last).

    Maps ratio via ``clip(r / 3, 0, 1)`` so 3× prior average saturates.
    """
    v = volume.astype(float).values
    if len(v) < window + 1:
        return 0.0
    prior = v[-(window + 1) : -1]
    if prior.size == 0 or np.nanmean(prior) <= 0:
        return 0.0
    ratio = float(v[-1] / max(np.nanmean(prior), 1e-12))
    return _clip01(ratio / 3.0)


def dollar_volume_expansion_score(
    close: pd.Series,
    volume: pd.Series,
    *,
    short: int = 5,
    long: int = 20,
) -> float:
    """
    Recent mean dollar volume vs older mean within the same span (short vs long tail).

    Uses the last ``long`` rows: compares mean(C×V) over last ``short`` days vs
    mean over the prior ``long - short`` days. Expansion ratio clipped to [0,1]
    after ``ratio / 2`` (2× expansion saturates).
    """
    if len(close) < long or len(volume) < long:
        return 0.0
    c = close.astype(float).iloc[-long:].values
    vo = volume.astype(float).iloc[-long:].values
    dv = c * vo
    if short >= long:
        return 0.0
    recent = dv[-short:]
    older = dv[: long - short]
    m_r = float(np.nanmean(recent)) if recent.size else 0.0
    m_o = float(np.nanmean(older)) if older.size else 0.0
    if m_o <= 1e-9:
        return 0.0
    ratio = m_r / m_o
    if ratio <= 1.0:
        return 0.0
    return _clip01((ratio - 1.0) / 2.0)
