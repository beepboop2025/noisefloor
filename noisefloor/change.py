"""Did this metric actually change, or is it noise?

THE PROBLEM. A number on a dashboard moves. Someone asks whether that is real.
The usual answers are a hard threshold ("alert above 500"), which nobody can
attach an error rate to, or a z-score against a rolling mean, which assumes a
stationary Gaussian world your metric does not live in and silently breaks the
moment the metric drifts.

THE TOOL. A conformal Shiryaev-Roberts detector. Each new reading is scored
against the metric's own past by a rank-based p-value, which needs no
distributional assumption at all. A decreasing bet turns that into evidence, and
the Shiryaev-Roberts recursion accumulates it:

    R_t = (1 + R_{t-1}) * bet(p_t)

The guarantee is the one that actually fits repeated monitoring:

    P( first false alarm within n readings ) <= n / A
    average readings between false alarms  >= A

So at A = 500 the detector cries wolf at most once every 500 readings on
average, stated per reading and valid however long it runs. Unlike a product
martingale, the additive term means a long calm stretch does not grind the
statistic to zero and leave it unable to react.

TWO-SIDED BY DEFAULT. A metric COLLAPSING is a change too, and an upward-only
detector is blind to half of what can go wrong: traffic vanishing, a logging
pipeline dying, conversions going to zero. Two detectors run, one per direction.
Merging them into a single number would be valid but halves the evidence at
every step and costs most of the power, so each runs at full strength and the
union over two is paid for by doubling both thresholds. The published guarantee
is therefore exactly the one-sided guarantee it replaces.

Standard library only, deterministic: the same series always gives the same
answer, and anyone can re-run it to check.
"""
from __future__ import annotations

# A deterministic mixture of bets. Each is decreasing in p and integrates to 1,
# so applying one to a valid p-value yields evidence that is fair under "no
# change"; mixing over a grid keeps power against both sudden jumps and slow
# creep without having to guess which is coming.
BET_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

WATCH_A = 100.0      # at most one false WATCH per 100 readings, on average
ALARM_A = 500.0      # at most one false ALARM per 500 readings, on average
WARMUP = 8           # readings needed before the detector will bet at all
STAT_CAP = 1e12      # numeric ceiling; an alarm resets long before this


def p_value(history: list[float], x: float, *, side: str = "up",
            weights: list[float] | None = None) -> float:
    """Rank-based p-value of x against its own past. No distribution assumed.

    p = (count at least as extreme + 1) / (n + 1). Ties count toward the
    numerator, which makes it conservative — validity is never bought with
    optimism, and no random tie-breaking is needed, so results are reproducible.

    weights (one per history entry, oldest first) let recent history count for
    more, which is how a drifting metric stops slowly redefining "normal"
    (nonexchangeable conformal prediction, Barber, Candes, Ramdas & Tibshirani,
    Ann. Statist. 2023, arXiv:2202.13415).
    """
    extreme = (lambda s: s >= x) if side == "up" else (lambda s: s <= x)
    if weights is None:
        return (sum(1 for s in history if extreme(s)) + 1) / (len(history) + 1)
    if len(weights) != len(history):
        raise ValueError("weights must be parallel to history")
    return (sum(w for s, w in zip(history, weights) if extreme(s)) + 1.0) / (
        sum(weights) + 1.0)


def decay_weights(n: int, half_life: float | None) -> list[float] | None:
    """Geometric weights, newest last and equal to 1. None disables weighting."""
    if not half_life or n <= 0:
        return None
    return [0.5 ** ((n - 1 - i) / half_life) for i in range(n)]


def _bet(p: float) -> float:
    p = min(max(p, 1e-12), 1.0)
    return sum(e * p ** (e - 1.0) for e in BET_GRID) / len(BET_GRID)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _effect(reference: list[float], x: float) -> dict:
    """How far the reading sits from its past, in plain units and robust units."""
    if not reference:
        return {"direction": "flat", "delta": None, "robust_z": None,
                "percentile": None}
    med = _median(reference)
    mad = _median([abs(s - med) for s in reference])
    scale = 1.4826 * mad          # comparable to a standard deviation, but robust
    delta = x - med
    return {
        "direction": "up" if delta > 0 else "down" if delta < 0 else "flat",
        "delta": round(delta, 6),
        "robust_z": round(delta / scale, 2) if scale > 0 else None,
        "percentile": round(100.0 * sum(1 for s in reference if s < x) / len(reference), 1),
    }


def scan(values: list[float], *, two_sided: bool = True,
         half_life: float | None = None) -> dict:
    """Walk a metric's history and report where it changed.

    Returns the current state, the evidence behind it, which direction drove it,
    and the index of every past change point.
    """
    sides = ("up", "down") if two_sided else ("up",)
    # union bound over the directions being watched, so the headline guarantee
    # below is the family-wise one rather than a per-direction one
    watch_at, alarm_at = WATCH_A * len(sides), ALARM_A * len(sides)

    stat = {s: 0.0 for s in sides}
    reference: list[float] = []
    states, stats, alarms = [], [], []
    driver = None

    for i, x in enumerate(values):
        if len(reference) >= WARMUP:
            w = decay_weights(len(reference), half_life)
            for s in sides:
                stat[s] = min((1.0 + stat[s]) * _bet(p_value(reference, x, side=s,
                                                             weights=w)), STAT_CAP)
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

    effect = _effect(values[:-1], values[-1]) if len(values) > 1 else {}
    return {
        "n": len(values),
        "state": states[-1] if states else "no_data",
        "evidence": stats[-1] if stats else None,
        "direction": driver,
        "effect": effect,
        "change_points": alarms,
        "states": states,
        "evidence_series": stats,
        "watch_at": watch_at,
        "changed_at": alarm_at,
        "readings_since_change": (len(values) - 1 - alarms[-1]) if alarms else None,
        "reading": _reading(states[-1] if states else "no_data", effect, driver,
                            alarms, len(values)),
        "guarantee": (
            f"under no change, at most one false 'changed' every {alarm_at:g} readings "
            f"on average, and P(first false flag within n readings) <= n/{alarm_at:g}; "
            f"valid however long this runs"),
        "method": (
            "conformal Shiryaev-Roberts detector: rank-based p-value against the "
            "metric's own past, mixture bet, R_t = (1+R_{t-1})*bet(p_t)"
            + (", two-sided with thresholds doubled to pay the union bound"
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
        return "no change: the latest reading is within what this metric normally does"
    z = effect.get("robust_z")
    where = f" ({driver})" if driver else ""
    size = f", {abs(z):.1f}x its usual spread" if z is not None else ""
    verb = "CHANGED" if state == "changed" else "possible change"
    return f"{verb}{where}{size}"
