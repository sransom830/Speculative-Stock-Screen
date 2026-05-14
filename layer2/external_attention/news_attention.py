"""
News observational enrichment — headline density + thematic structure only.

Uses thematic bucket masses (counts), not polarity embeddings. Provider adapters supply structured counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import (
    ExternalAttentionWindow,
    NewsAttentionFactors,
    NewsAttentionObservation,
    NewsRawCounts,
    NewsThemeMass,
    clamp_unit_interval,
)


@dataclass(frozen=True)
class NewsNormalizationPolicy:
    headline_density_reference_per_hour: float = 12.0
    headline_velocity_ratio_cap: float = 5.0
    cold_start_headline_reference: int = 24
    thematic_top_k: int = 3


def _window_hours(interval_start_utc: datetime, interval_end_utc: datetime) -> float:
    delta = interval_end_utc - interval_start_utc
    sec = delta.total_seconds()
    if sec <= 0:
        raise ValueError("window length must be positive.")
    return sec / 3600.0


def _headline_density_factor(count: int, *, hours: float, reference_per_hour: float) -> float:
    if reference_per_hour <= 0:
        raise ValueError("reference_per_hour must be positive.")
    density = count / hours
    return clamp_unit_interval(density / reference_per_hour)


def _headline_velocity_factor(recent: int, prior: int, *, ratio_cap: float, cold_ref: int) -> float:
    if ratio_cap <= 0:
        raise ValueError("ratio_cap must be positive.")
    if cold_ref < 1:
        raise ValueError("cold_ref must be >= 1.")
    if prior == 0:
        ratio = recent / float(cold_ref)
    else:
        ratio = (recent - prior) / float(prior)
    return clamp_unit_interval(ratio / ratio_cap)


def _sorted_theme_masses(theme_masses: tuple[NewsThemeMass, ...]) -> tuple[NewsThemeMass, ...]:
    return tuple(sorted(theme_masses, key=lambda m: (m.label.lower(), m.label)))


def _narrative_concentration_factor(theme_masses: tuple[NewsThemeMass, ...]) -> float:
    """
    Herfindahl-style concentration on thematic masses, scaled to ``[0, 1]``.

    Low concentration → ``0``; single-theme dominance → ``1``.
    """
    masses = _sorted_theme_masses(theme_masses)
    total = sum(m.mass for m in masses)
    if total <= 0:
        return 0.0
    shares = [m.mass / total for m in masses]
    n = len(shares)
    if n <= 1:
        return 1.0 if shares and shares[0] > 0 else 0.0
    hhi = sum(s * s for s in shares)
    h_min = 1.0 / n
    if hhi <= h_min:
        return 0.0
    return clamp_unit_interval((hhi - h_min) / (1.0 - h_min))


def _thematic_clustering_factor(theme_masses: tuple[NewsThemeMass, ...], *, top_k: int) -> float:
    """
    Share of thematic mass in the top ``k_eff`` buckets (``k_eff = min(top_k, n-1)``)
    versus uniform baseline ``k_eff/n`` so the denominator stays well-defined when ``n > 1``.

    Interpretable clustering signal without embedding/cluster ML.
    """
    masses = _sorted_theme_masses(theme_masses)
    total = sum(m.mass for m in masses)
    if total <= 0:
        return 0.0
    n = len(masses)
    if n == 1:
        return 1.0 if masses[0].mass > 0 else 0.0

    # Compare top-(k_eff) share to uniform baseline k_eff/n; require k_eff < n so baseline < 1.
    k_eff = max(1, min(top_k, n - 1))
    descending = sorted((m.mass for m in masses), reverse=True)
    top_mass = sum(descending[:k_eff])
    uniform_baseline = k_eff / float(n)
    share = top_mass / total
    excess = max(0.0, share - uniform_baseline)
    denom = 1.0 - uniform_baseline
    return clamp_unit_interval(excess / denom)


def compute_news_attention_observation(
    *,
    window: ExternalAttentionWindow,
    raw: NewsRawCounts,
    policy: NewsNormalizationPolicy | None = None,
    composite_weights: tuple[float, float, float, float] | None = None,
) -> NewsAttentionObservation:
    """
    Map headline/theme observables → normalized enrichment factors.

    Composite weights ``(density, velocity, concentration, clustering)`` sum to ``1.0``.
    """
    pol = policy or NewsNormalizationPolicy()
    hours = _window_hours(window.interval_start_utc, window.interval_end_utc)

    w_d, w_v, w_c, w_k = composite_weights or (0.28, 0.28, 0.22, 0.22)
    w_sum = w_d + w_v + w_c + w_k
    if w_sum <= 0:
        raise ValueError("composite weights must sum to a positive value.")
    w_d, w_v, w_c, w_k = (w_d / w_sum, w_v / w_sum, w_c / w_sum, w_k / w_sum)

    density = _headline_density_factor(
        raw.headline_count_window,
        hours=hours,
        reference_per_hour=pol.headline_density_reference_per_hour,
    )
    velocity = _headline_velocity_factor(
        raw.headline_count_window,
        raw.headline_count_prior_window,
        ratio_cap=pol.headline_velocity_ratio_cap,
        cold_ref=pol.cold_start_headline_reference,
    )
    concentration = _narrative_concentration_factor(raw.theme_masses)
    clustering = _thematic_clustering_factor(raw.theme_masses, top_k=pol.thematic_top_k)

    composite = clamp_unit_interval(w_d * density + w_v * velocity + w_c * concentration + w_k * clustering)

    factors = NewsAttentionFactors(
        news_attention_factor=composite,
        headline_density_factor=density,
        narrative_concentration_factor=concentration,
        thematic_clustering_factor=clustering,
    )
    return NewsAttentionObservation(window=window, raw=raw, factors=factors)
