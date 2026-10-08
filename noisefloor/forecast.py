"""One-step adaptive empirical intervals with a reproducible forecast ledger.

Both public forecasting and historical scoring call the same prequential engine:
issue from the past, score when the next value arrives, then update. A clipped
coverage-feedback rule adapts historical absolute-error quantiles. These finite
intervals do NOT have arbitrary-shift, exchangeability-free or next-step coverage
guarantees. Coverage is measured and every miss is returned.

Adaptive conformal research motivates the feedback, but this bounded heuristic
is not an implementation of its coverage theorem. WIS is a proper interval
scoring rule; a favorable score does not establish trading profitability.
"""
from __future__ import annotations

from bisect import insort

from ._validation import number, probability, series

LEVELS = (0.50, 0.80, 0.95)
NOMINAL = 0.80
GAMMA0 = 0.05
MIN_HISTORY = 20
WARMUP = 10


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    position = (len(sorted_vals) - 1) * min(max(q, 0.0), 1.0)
    low = int(position)
    high = min(low + 1, len(sorted_vals) - 1)
    fraction = position - low
    return number(sorted_vals[low] * (1 - fraction) + sorted_vals[high] * fraction, "error quantile")


def interval_score(y: float, lo: float, hi: float, alpha: float) -> float:
    """Interval width plus tail penalties; alpha is the miscoverage level."""
    y, lo, hi = number(y, "y"), number(lo, "lo"), number(hi, "hi")
    alpha = probability(alpha, "alpha")
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    result = hi - lo
    if y < lo:
        result += (2.0 / alpha) * (lo - y)
    elif y > hi:
        result += (2.0 / alpha) * (y - hi)
    return number(result, "interval score")


def weighted_interval_score(y: float, median: float,
                            intervals: dict[float, tuple[float, float]]) -> float:
    """Proper WIS for supplied central intervals and median; lower is better."""
    y, median = number(y, "y"), number(median, "median")
    if not isinstance(intervals, dict):
        raise ValueError("intervals must map coverage levels to (lo, hi)")
    if not intervals:
        return number(abs(y - median), "absolute error")
    total = 0.5 * abs(y - median)
    for level, bounds in intervals.items():
        level = probability(level, "interval level")
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("each interval must have two bounds")
        alpha = 1.0 - level
        total += (alpha / 2.0) * interval_score(y, bounds[0], bounds[1], alpha)
    return number(total / (len(intervals) + 0.5), "weighted interval score")


def _bounds(point, pool, quantile):
    width = _quantile(pool, quantile)
    return [number(point - width, "lower forecast bound"), number(point + width, "upper forecast bound")]


def _prediction(point, pool, alpha, nominal):
    clipped = min(max(alpha, 0.01), 0.99)
    return {
        "forecast": point,
        "interval": _bounds(point, pool, 1.0 - clipped),
        "nominal_coverage": nominal,
        "adaptive_alpha": clipped,
        "error_history_size": len(pool),
    }


def _replay(values, nominal, *, collect):
    """Shared issue/observe/update loop; every prediction uses only its prefix."""
    pool = []
    alpha = 1.0 - nominal
    scored = 0
    records = []
    for index in range(1, len(values)):
        point, actual = values[index - 1], values[index]
        if len(pool) >= WARMUP:
            prediction = _prediction(point, pool, alpha, nominal)
            lo, hi = prediction["interval"]
            covered = lo <= actual <= hi
            if collect:
                adaptive, baseline = {}, {}
                for level in LEVELS:
                    level_alpha = min(max(alpha * (1.0 - level) / (1.0 - nominal), 0.01), 0.99)
                    adaptive[level] = _bounds(point, pool, 1.0 - level_alpha)
                    baseline[level] = _bounds(point, pool, level)
                records.append({
                    "index": index,
                    **prediction,
                    "actual": actual,
                    "covered": covered,
                    "missed_by": 0.0 if covered else min(abs(actual - lo), abs(actual - hi)),
                    "score": weighted_interval_score(actual, point, adaptive),
                    "baseline_score": weighted_interval_score(actual, point, baseline),
                })
            scored += 1
            gamma = GAMMA0 / (1.0 + 0.01 * scored) ** 0.5
            alpha = min(max(alpha + gamma * ((1.0 - nominal) - (0.0 if covered else 1.0)), 0.01), 0.99)
        insort(pool, number(abs(actual - point), "one-step absolute change"))
    prediction = _prediction(values[-1], pool, alpha, nominal) if len(pool) >= WARMUP else None
    return prediction, records


def _contract():
    return {
        "validity": "empirical_only",
        "guarantee": None,
        "method": "random-walk point forecast with clipped adaptive historical-error quantiles; shared prequential issue/score/update engine",
        "caveats": [
            "Nominal coverage is a target, not a next-observation or arbitrary-shift guarantee.",
            "Temporal dependence and changing regimes can impair coverage.",
            "Observations must be chronological and have consistent units and cadence.",
            "The horizon is one supplied observation; timestamps and market sessions are not inferred.",
        ],
    }


def next_value(values: list[float], *, nominal: float = NOMINAL) -> dict:
    """Issue the exact forecast scored by ``score`` when the next value arrives."""
    values = series(values, "values")
    nominal = probability(nominal, "nominal")
    if len(values) < WARMUP + 1:
        return {"ok": False, **_contract(),
                "reading": f"need {WARMUP + 1} readings to forecast, have {len(values)}"}
    prediction, _ = _replay(values, nominal, collect=False)
    point = prediction["forecast"]
    lo, hi = prediction["interval"]
    return {
        "ok": True,
        **prediction,
        **_contract(),
        "reading": f"one-step point estimate {point:.4g}, empirical interval [{lo:.4g}, {hi:.4g}], targeting {nominal:.0%} coverage; actual coverage must be measured",
    }


def score(values: list[float], *, nominal: float = NOMINAL) -> dict:
    """Replay issued forecasts and return every scored observation and miss."""
    values = series(values, "values")
    nominal = probability(nominal, "nominal")
    if len(values) < MIN_HISTORY:
        return {"ok": False, **_contract(),
                "reading": f"need {MIN_HISTORY} readings to score, have {len(values)}"}
    _, records = _replay(values, nominal, collect=True)
    misses = [record for record in records if not record["covered"]]
    n = len(records)
    empirical = (n - len(misses)) / n
    model_score = number(sum(record["score"] / n for record in records), "mean forecast score")
    baseline_score = number(sum(record["baseline_score"] / n for record in records), "mean baseline score")
    within_tolerance = abs(empirical - nominal) <= 0.05
    return {
        "ok": True,
        **_contract(),
        "n_forecasts": n,
        "nominal_coverage": nominal,
        "empirical_coverage": empirical,
        "calibrated": within_tolerance,  # compatibility alias; descriptive only
        "within_coverage_tolerance": within_tolerance,
        "calibration_criterion": "observed coverage within 0.05 of target; descriptive, not a significance test or certificate",
        "score": model_score,
        "baseline_score": baseline_score,
        "beats_baseline": model_score < baseline_score,
        "skill_vs_baseline": 1.0 - model_score / baseline_score if baseline_score > 0 else None,
        "n_misses": len(misses),
        "records": records,
        "misses": misses,
        "worst_misses": sorted(misses, key=lambda record: record["missed_by"], reverse=True)[:5],
        "records_complete": True,
        "reading": f"{n} issued forecasts replayed; observed coverage {empirical:.0%} against target {nominal:.0%}; {'within' if within_tolerance else 'outside'} descriptive five-point tolerance",
    }
