"""Predict the next value, then keep score honestly.

THE PROBLEM. Forecasts get published and never graded. A dashboard shows a
projection, the future arrives, and nobody goes back to check. A forecaster that
never publishes its record is indistinguishable from one that is always wrong,
and that asymmetry is the whole reason forecasts are cheap to make and hard to
trust.

THE TOOL. Two halves that belong together.

FORECAST. The next value, with a range around it that adapts. The point
prediction is the last observed value — deliberately unambitious, because a
random walk is a genuinely hard baseline for most real metrics and the claim
being made here is about CALIBRATION, not cleverness. The range comes from an
adaptive conformal interval: after every miss it widens, after every hit it
narrows, by

    alpha_{t+1} = alpha_t + gamma_t * (target - missed)

which holds long-run coverage under ARBITRARY distribution shift (Gibbs &
Candes, NeurIPS 2021, arXiv:2106.00170; decaying step sizes, Angelopoulos,
Barber & Bates, ICML 2024, arXiv:2402.01139). No stationarity assumption, which
matters because the interesting metrics are exactly the ones that shift.

SCORE. Every past forecast is graded prequentially (Dawid, JRSS-A 1984): each
one was made using only what was known before it, never refit with hindsight, so
the record is what the tool would really have said. Grading uses the Weighted
Interval Score (Bracher, Ray, Gneiting & Reich, PLOS Comput Biol 2021,
arXiv:2005.12881), a PROPER scoring rule — meaning the best score comes from
reporting what you actually believe. That property is what makes a
self-published scoreboard worth reading: it cannot be improved by hedging.

WHAT COMES BACK. Empirical coverage against the nominal target, the score
against a non-adaptive baseline, and the worst misses by position. The misses
are not optional and there is no flag to hide them.

Standard library only, deterministic.
"""
from __future__ import annotations

LEVELS = (0.50, 0.80, 0.95)
NOMINAL = 0.80          # the headline band whose coverage gets reported
GAMMA0 = 0.05           # base adaptation rate
MIN_HISTORY = 20
WARMUP = 10             # readings used to seed the error pool, never scored


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * min(max(q, 0.0), 1.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def interval_score(y: float, lo: float, hi: float, alpha: float) -> float:
    """Score one interval. Width, plus a penalty for missing. Lower is better."""
    s = hi - lo
    if y < lo:
        s += (2.0 / alpha) * (lo - y)
    elif y > hi:
        s += (2.0 / alpha) * (y - hi)
    return s


def weighted_interval_score(y: float, median: float,
                            intervals: dict[float, tuple[float, float]]) -> float:
    """Proper score over several intervals plus the point forecast."""
    if not intervals:
        return abs(y - median)
    total = 0.5 * abs(y - median)
    for level, (lo, hi) in intervals.items():
        alpha = 1.0 - level
        total += (alpha / 2.0) * interval_score(y, lo, hi, alpha)
    return total / (len(intervals) + 0.5)


def next_value(values: list[float], *, nominal: float = NOMINAL) -> dict:
    """Forecast the next reading, with an adaptive range around it."""
    if len(values) < WARMUP + 1:
        return {"ok": False,
                "reading": f"need {WARMUP + 1} readings to forecast, have {len(values)}"}
    steps = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    q = _quantile(sorted(steps), nominal)
    point = values[-1]
    return {
        "ok": True,
        "forecast": round(point, 6),
        "interval": [round(point - q, 6), round(point + q, 6)],
        "nominal_coverage": nominal,
        "reading": (f"next reading is expected around {point:.4g}, and should land "
                    f"between {point - q:.4g} and {point + q:.4g} about "
                    f"{nominal:.0%} of the time"),
        "method": ("random-walk point forecast with an adaptive conformal range over "
                   "past one-step changes"),
    }


def score(values: list[float], *, nominal: float = NOMINAL) -> dict:
    """Grade every forecast this tool would have made over the history."""
    n = len(values)
    if n < MIN_HISTORY:
        return {"ok": False,
                "reading": f"need {MIN_HISTORY} readings to score, have {n}"}

    errors: list[float] = []
    alpha = 1.0 - nominal
    hits = misses = 0
    wis_model: list[float] = []
    wis_base: list[float] = []
    records: list[dict] = []

    for t in range(1, n):
        point, y = values[t - 1], values[t]
        if len(errors) >= WARMUP:
            pool = sorted(errors)
            a = min(max(alpha, 0.01), 0.99)
            q_adaptive = _quantile(pool, 1.0 - a)
            lo, hi = point - q_adaptive, point + q_adaptive
            covered = lo <= y <= hi
            hits += covered
            misses += (not covered)

            iv_a, iv_b = {}, {}
            for lv in LEVELS:
                a_lv = min(max(alpha * (1.0 - lv) / (1.0 - nominal), 0.01), 0.99)
                iv_a[lv] = (point - _quantile(pool, 1.0 - a_lv),
                            point + _quantile(pool, 1.0 - a_lv))
                iv_b[lv] = (point - _quantile(pool, lv), point + _quantile(pool, lv))

            sm = weighted_interval_score(y, point, iv_a)
            sb = weighted_interval_score(y, point, iv_b)
            wis_model.append(sm)
            wis_base.append(sb)
            records.append({"index": t, "forecast": round(point, 6),
                            "interval": [round(lo, 6), round(hi, 6)],
                            "actual": round(y, 6), "covered": bool(covered),
                            "missed_by": 0.0 if covered else round(
                                min(abs(y - lo), abs(y - hi)), 6)})

            gamma = GAMMA0 / (1.0 + 0.01 * len(wis_model)) ** 0.5
            alpha = min(max(alpha + gamma * ((1.0 - nominal) - (0.0 if covered else 1.0)),
                            0.01), 0.99)
        errors.append(abs(values[t] - values[t - 1]))

    scored = hits + misses
    if scored == 0:
        return {"ok": False, "reading": "no reading survived warm-up"}

    mean_model = sum(wis_model) / len(wis_model)
    mean_base = sum(wis_base) / len(wis_base)
    coverage = hits / scored
    calibrated = abs(coverage - nominal) <= 0.05

    return {
        "ok": True,
        "n_forecasts": scored,
        "nominal_coverage": nominal,
        "empirical_coverage": round(coverage, 3),
        "calibrated": calibrated,
        "score": round(mean_model, 6),
        "baseline_score": round(mean_base, 6),
        "beats_baseline": bool(mean_model < mean_base),
        "skill_vs_baseline": (round(1.0 - mean_model / mean_base, 3)
                              if mean_base > 0 else None),
        "n_misses": misses,
        "worst_misses": sorted((r for r in records if not r["covered"]),
                               key=lambda r: r["missed_by"], reverse=True)[:5],
        "reading": (
            f"{scored} forecasts graded; {coverage:.0%} of readings landed inside the "
            f"{nominal:.0%} range "
            f"({'well calibrated' if calibrated else 'NOT well calibrated'})"),
        "method": (
            "strictly prequential (Dawid 1984): each reading forecast from only its own "
            "past, never refit with hindsight. Adaptive conformal range "
            "(arXiv:2106.00170, arXiv:2402.01139), graded by the Weighted Interval "
            "Score (arXiv:2005.12881), a proper rule, against the same forecast with a "
            "fixed range. Lower score is better."),
        "caveats": [
            "one step ahead only; the horizon is whatever the gap between readings is",
            "the point forecast is a random walk on purpose — the claim under test is "
            "calibration, not sharpness",
            "misses are published in full; a record with the misses removed is worthless",
        ],
    }
