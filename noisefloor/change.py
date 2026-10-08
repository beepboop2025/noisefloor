"""Descriptive rank-based change monitoring.

The Shiryaev-Roberts recursion R_t = (1 + R_{t-1}) * bet(p_t) compares
successive observations with preceding observations. It is a monitoring
statistic, NOT an e-value. Its value must not be supplied to e-BH.

Rank calibration requires exchangeable observations and appropriate sequential
rank validity. Financial observations may be serially dependent or changing in
volatility. Geometric weighting changes the calibration too. This implementation
therefore reports descriptive thresholds, not a universal false-alarm bound.
A quiet result is absence of a detected departure, not proof of no change.
"""
from __future__ import annotations

from ._validation import count, number, series

# A deterministic mixture of bets. Each is decreasing in p and integrates to 1,
# so applying one to a valid p-value yields evidence that is fair under "no
# change"; mixing over a grid keeps power against both sudden jumps and slow
# creep without having to guess which is coming.
BET_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

WATCH_A = 100.0      # descriptive watch threshold before direction adjustment
ALARM_A = 500.0      # descriptive alarm threshold before direction adjustment
WARMUP = 8           # readings needed before the detector will bet at all
STAT_CAP = 1e12      # numeric ceiling; an alarm resets long before this


def p_value(history: list[float], x: float, *, side: str = "up",
            weights: list[float] | None = None) -> float:
    """Rank tail fraction; p-value interpretation requires exchangeability.

    With weights, this is a weighted descriptive rank, not a calibrated p-value.
    Ties count as at least as extreme. We do not infer independence from ranks.
    """
    history = series(history, "history")
    x = number(x, "x")
    if side not in ("up", "down"):
        raise ValueError("side must be 'up' or 'down'")
    if weights is not None:
        weights = series(weights, "weights")
        if len(weights) != len(history):
            raise ValueError("weights must be parallel to history")
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be nonnegative")
        number(sum(weights), "sum of weights")
    return _rank(history, x, side, weights)


def _rank(history, x, side, weights):
    extreme = (lambda value: value >= x) if side == "up" else (lambda value: value <= x)
    if weights is None:
        return (sum(1 for value in history if extreme(value)) + 1) / (len(history) + 1)
    return (sum(weight for value, weight in zip(history, weights) if extreme(value)) + 1.0) / (
        sum(weights) + 1.0)


def decay_weights(n: int, half_life: float | None) -> list[float] | None:
    """Geometric weights, newest last and equal to 1. None disables weighting."""
    count(n, "n")
    if half_life is None:
        return None
    half_life = number(half_life, "half_life")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    if n == 0:
        return []
    return [0.5 ** ((n - 1 - i) / half_life) for i in range(n)]


def _bet(p: float) -> float:
    p = min(max(p, 1e-12), 1.0)
    return sum(e * p ** (e - 1.0) for e in BET_GRID) / len(BET_GRID)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else s[m - 1] / 2.0 + s[m] / 2.0


def _effect(reference: list[float], x: float) -> dict:
    """How far the reading sits from its past, in plain units and robust units."""
    if not reference:
        return {"direction": "flat", "delta": None, "robust_z": None,
                "percentile": None}
    med = _median(reference)
    mad = _median([number(abs(s - med), "reference deviation") for s in reference])
    scale = number(1.4826 * mad, "reference scale")
    delta = number(x - med, "latest departure")
    return {
        "direction": "up" if delta > 0 else "down" if delta < 0 else "flat",
        "delta": round(delta, 6),
        "robust_z": round(number(delta / scale, "standardized departure"), 2) if scale > 0 else None,
        "percentile": round(100.0 * sum(1 for s in reference if s < x) / len(reference), 1),
    }


def scan(values: list[float], *, two_sided: bool = True,
         half_life: float | None = None) -> dict:
    """Walk a metric's history and report where it changed.

    Returns the current state, the evidence behind it, which direction drove it,
    and the index of every past change point.
    """
    values = series(values, "values")
    if not isinstance(two_sided, bool):
        raise ValueError("two_sided must be a boolean")
    decay_weights(0, half_life)  # validate even when the history is empty
    sides = ("up", "down") if two_sided else ("up",)
    # Preserve historical thresholds. Doubling protects the direction budget
    # only when the underlying rank-process assumptions actually hold.
    watch_at, alarm_at = WATCH_A * len(sides), ALARM_A * len(sides)

    stat = {s: 0.0 for s in sides}
    reference: list[float] = []
    states, stats, alarms = [], [], []
    driver = None
    last_reference = []

    for i, x in enumerate(values):
        if i == len(values) - 1:
            last_reference = reference[:]
        if len(reference) >= WARMUP:
            w = decay_weights(len(reference), half_life)
            for s in sides:
                stat[s] = min((1.0 + stat[s]) * _bet(_rank(reference, x, s, w)), STAT_CAP)
        peak = max(stat.values())
        driver = max(stat, key=lambda s: stat[s]) if peak > 0 else None
        state = ("warming_up" if len(reference) < WARMUP else
                 "changed" if peak >= alarm_at else
                 "watch" if peak >= watch_at else "steady")
        states.append(state)
        stats.append(round(peak, 4))
        reference.append(x)
        if state == "changed":
            alarms.append(i)
            stat = {s: 0.0 for s in sides}
            reference = []      # the post-change world becomes the new normal

    effect = _effect(last_reference, values[-1]) if values else {}
    return {
        "n": len(values),
        "state": states[-1] if states else "no_data",
        "evidence": stats[-1] if stats else None,  # compatibility alias only
        "monitoring_statistic": stats[-1] if stats else None,
        "statistic_type": "shiryaev_roberts",
        "is_e_value": False,
        "validity": "descriptive",
        "weighted": half_life is not None,
        "assumptions": [
            "Observations are supplied in chronological order with consistent units.",
            "Rank calibration would require exchangeability and sequential validity; "
            "these are not established by this function.",
            "Serial dependence, drift, weighting and selected windows can alter false alarms.",
        ],
        "direction": driver,
        "effect": effect,
        "effect_reference_size": len(last_reference),
        "change_points": alarms,
        "states": states,
        "evidence_series": stats,
        "watch_at": watch_at,
        "changed_at": alarm_at,
        "readings_since_change": (len(values) - 1 - alarms[-1]) if alarms else None,
        "reading": _reading(states[-1] if states else "no_data", effect, driver,
                            alarms, len(values)),
        "guarantee": None,
        "interpretation": "A threshold crossing requests review; it is not a trade signal or a calibrated e-value.",
        "method": (
            "descriptive Shiryaev-Roberts monitoring: rank tail fraction against the "
            "metric's own past, mixture bet, R_t = (1+R_{t-1})*bet(p_t)"
            + (", two-sided with doubled descriptive thresholds"
               if two_sided else ", one-sided (upward only)")),
    }


def _reading(state: str, effect: dict, driver: str | None,
             change_points: list[int], n: int) -> str:
    if state == "no_data":
        return "no data"
    if state == "warming_up":
        # A detected change resets the reference, so the readings right after one
        # are "warming up". Saying only that would bury the change that just
        # happened — the single most important thing the caller needs to know.
        if change_points:
            since = n - 1 - change_points[-1]
            return (f"CHANGED {since} reading{'s' if since != 1 else ''} ago; the "
                    f"detector is re-arming and needs {WARMUP} readings of the new "
                    f"normal before it can judge again")
        return f"not enough history yet — needs {WARMUP} readings before it can judge"
    if state == "steady":
        return "no supported departure at the monitoring threshold; absence of an alarm is not proof of no change"
    z = effect.get("robust_z")
    where = f" ({driver})" if driver else ""
    size = f", {abs(z):.1f}x its usual spread" if z is not None else ""
    verb = "CHANGED" if state == "changed" else "possible change"
    return f"{verb}{where}{size}"
