"""Descriptive checks for changes in a metric's measurement denominator.

A historical linear association is a possible explanation, not a causal test.
Changing market volume may itself carry economic information: it must not be
confused with missing measurements. No statistical error-rate guarantee is made.
"""
from __future__ import annotations

from ._validation import number, series

MIN_PAIRS = 10
MOVE_Z = 2.0
GOOD_FIT_R = 0.30
SURVIVES_FRAC = 0.5


def _median(xs: list[float]) -> float:
    values = sorted(xs)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else values[middle - 1] / 2 + values[middle] / 2


def _robust_z(reference: list[float], x: float) -> float | None:
    if not reference:
        return None
    median = _median(reference)
    deviations = [number(abs(value - median), "reference deviation") for value in reference]
    scale = number(1.4826 * _median(deviations), "reference scale")
    return number((x - median) / scale, "standardized departure") if scale > 0 else None


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    n = len(xs)
    if n < 2:
        return None
    mx = number(sum(x / n for x in xs), "mean denominator")
    my = number(sum(y / n for y in ys), "mean metric")
    try:
        sxx = number(sum((x - mx) ** 2 for x in xs), "denominator variation")
        syy = number(sum((y - my) ** 2 for y in ys), "metric variation")
        if sxx <= 0:
            return None
        sxy = number(sum((x - mx) * (y - my) for x, y in zip(xs, ys)), "covariation")
        slope = number(sxy / sxx, "slope")
        correlation = (sxy / sxx ** 0.5) / syy ** 0.5 if syy > 0 else 0.0
        return number(my - slope * mx, "intercept"), slope, max(-1.0, min(1.0, correlation))
    except OverflowError as exc:
        raise ValueError("values are too large for denominator conditioning") from exc


def check(metric: list[float], sample_size: list[float]) -> dict:
    """Describe the latest paired reading without inferring causality.

    Both lists must have the same length and represent aligned observations.
    Legacy REAL/SAMPLING_ARTIFACT codes are descriptive compatibility labels;
    consult ``assessment`` and ``validity`` rather than treating them as proof.
    """
    metric = series(metric, "metric")
    sample_size = series(sample_size, "sample_size")
    if len(metric) != len(sample_size):
        raise ValueError("metric and sample_size must have the same length and alignment")
    if any(value <= 0 for value in sample_size):
        raise ValueError("sample_size must be positive; zero coverage is missing data, not a valid rate")
    n = len(metric)
    out = {
        "n_pairs": n,
        "validity": "descriptive",
        "guarantee": None,
        "is_causal": False,
        "method": "median-absolute-deviation departure and historical linear denominator association",
        "caveats": [
            "Linear association does not establish or rule out a sampling explanation.",
            "The latest observation is excluded from the historical fit.",
            "Serial dependence, sample composition and nonlinear relationships are not controlled.",
            "Market volume and measurement coverage are different quantities.",
            "REAL and SAMPLING_ARTIFACT are legacy descriptive labels, not statistical guarantees.",
        ],
    }
    if n < MIN_PAIRS:
        out.update(verdict="NOT_ENOUGH_HISTORY", assessment="insufficient_history",
                   reading=f"need {MIN_PAIRS} paired readings to check this, have {n}")
        return out

    m_ref, d_ref = metric[:-1], sample_size[:-1]
    m_now, d_now = metric[-1], sample_size[-1]
    delta = number(m_now - _median(m_ref), "metric departure")
    z_metric = _robust_z(m_ref, m_now)
    z_denom = _robust_z(d_ref, d_now)
    med_d = _median(d_ref)
    out.update({
        "metric_now": m_now,
        "sample_size_now": d_now,
        "metric_departure": None if z_metric is None else round(z_metric, 2),
        "metric_delta": delta,
        "zero_reference_spread": z_metric is None,
        "sample_size_departure": None if z_denom is None else round(z_denom, 2),
        "sample_size_change_pct": number(100 * ((d_now - med_d) / med_d), "denominator change"),
    })
    if (z_metric is None and delta == 0) or (z_metric is not None and abs(z_metric) < MOVE_Z):
        out.update(verdict="NO_MOVE", assessment="no_supported_departure",
                   reading="no departure at the descriptive threshold; this does not prove stability")
        return out
    if z_metric is None:
        out.update(verdict="UNCLEAR", assessment="departure_from_zero_spread",
                   reading="the latest value differs from a history with zero robust spread; a standardized effect cannot be estimated")
        return out

    fit = _fit(d_ref, m_ref)
    if fit is None:
        out.update(verdict="UNCLEAR", assessment="denominator_explanation_unidentified",
                   reading="the metric departed from history, but a constant denominator cannot identify a sampling relationship")
        return out
    intercept, slope, correlation = fit
    out["fit"] = {"slope": slope, "intercept": intercept, "correlation": round(correlation, 3)}
    out["denominator_extrapolation"] = not min(d_ref) <= d_now <= max(d_ref)
    if abs(correlation) < GOOD_FIT_R:
        out.update(verdict="UNCLEAR", assessment="weak_linear_association",
                   reading=f"a departure is present, but the weak linear association ({correlation:.2f}) cannot establish or exclude a sampling explanation")
        return out

    residuals = [number(y - (intercept + slope * x), "historical residual") for x, y in zip(d_ref, m_ref)]
    residual = number(m_now - (intercept + slope * d_now), "latest residual")
    z_residual = _robust_z(residuals, residual)
    out["departure_after_conditioning"] = None if z_residual is None else round(z_residual, 2)
    if z_residual is None:
        scale = max((abs(value) for value in residuals), default=0.0)
        explained = abs(residual) <= max(scale, 1e-9) * 10
    else:
        explained = not (abs(z_residual) >= abs(z_metric) * SURVIVES_FRAC and abs(z_residual) >= MOVE_Z)
    if explained:
        out.update(verdict="SAMPLING_ARTIFACT", assessment="consistent_with_denominator_association",
                   reading="the movement is consistent with the historical denominator relationship; this is not evidence of a causal sampling artifact")
    else:
        out.update(verdict="REAL", assessment="departure_survives_linear_conditioning",
                   reading="the departure remains after descriptive linear conditioning; this does not establish its cause or market importance")
    return out
