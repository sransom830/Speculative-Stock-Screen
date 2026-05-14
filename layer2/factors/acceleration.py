"""Price acceleration and short-horizon momentum persistence (not reversal logic)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def price_acceleration_score(
    close: pd.Series,
    *,
    short: int = 5,
    long: int = 20,
) -> float:
    """
    Magnitude of recent return vs slower trend (attention to fast moves).

    ``a = abs(r_short)``, ``b = abs(r_long) * (short/long)``; score uses excess
    ``clip((a - b) / 0.08, 0, 1)`` (8 pp excess saturates).
    """
    if len(close) < long + 1:
        return 0.0
    c = close.astype(float)
    c0 = float(c.iloc[-1])
    c_s = float(c.iloc[-short])
    c_l = float(c.iloc[-long])
    if c_s <= 0 or c_l <= 0:
        return 0.0
    r_short = c0 / c_s - 1.0
    r_long = c0 / c_l - 1.0
    a = abs(r_short)
    b = abs(r_long) * (float(short) / float(long))
    return float(max(0.0, min(1.0, (a - b) / 0.08)))


def momentum_persistence_score(
    close: pd.Series,
    *,
    days: int = 10,
) -> float:
    """
    Directional alignment of daily returns: mean of signs vs dominant direction.

    Value in ``[0, 0.5]`` rescaled to ``[0, 1]`` where 1 means all days aligned.
    """
    if len(close) < days + 1:
        return 0.0
    c = close.astype(float).iloc[-(days + 1) :].values
    r = np.diff(c) / np.maximum(c[:-1], 1e-12)
    if r.size == 0:
        return 0.0
    signs = np.sign(r)
    dominant = float(np.sign(np.nansum(r)))
    if dominant == 0.0:
        return 0.5
    frac = float(np.mean(signs == dominant))
    return float(max(0.0, min(1.0, (frac - 0.5) * 2.0)))
