"""Realized volatility expansion (market-native, OHLC only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _log_returns(close: np.ndarray) -> np.ndarray:
    c = np.asarray(close, dtype=float)
    if c.size < 2:
        return np.array([])
    r = np.diff(np.log(c))
    return r[np.isfinite(r)]


def realized_vol_expansion_score(
    close: pd.Series,
    *,
    short: int = 5,
    long: int = 20,
) -> float:
    """
    Short-window realized vol vs longer baseline (std of log returns).

    Score = ``clip((short_std / long_std - 1) / 1.5, 0, 1)`` so moderate expansion maps upward.
    """
    if len(close) < long + 1:
        return 0.0
    arr = close.astype(float).iloc[-(long + 1) :].values
    full_r = _log_returns(arr)
    if full_r.size < long:
        return 0.0
    short_r = full_r[-short:] if short <= len(full_r) else full_r
    long_r = full_r[-long:]
    s_s = float(np.std(short_r, ddof=0)) if short_r.size else 0.0
    s_l = float(np.std(long_r, ddof=0)) if long_r.size else 0.0
    if s_l <= 1e-8:
        return 0.0
    ratio = s_s / s_l
    if ratio <= 1.0:
        return 0.0
    return float(max(0.0, min(1.0, (ratio - 1.0) / 2.0)))
