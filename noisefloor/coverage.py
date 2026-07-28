"""Did the metric move, or did the thing you measured it with move?

THE PROBLEM, from a real incident. A censorship-measurement board watched an
index fall from 60.3 to 55.6 over three days — a clear, statistically detectable
move. Read naively it said the censorship had eased. But the number of
measurements underneath it had fallen from 353,676 to 208,933 over the same
window. The index was a ratio over a sample nobody controlled, and the sample
had thinned. The move was partly in the measuring instrument, not the world.

Almost every metric worth watching is a rate over a sample you do not control:
conversion over sessions, error rate over requests, satisfaction over responses,
click-through over impressions. When the denominator moves, the ratio moves for
reasons that have nothing to do with the thing you care about. This is the most
common way a dashboard tells a confident lie.

THE TOOL. Three questions in order:

  1. Did the metric move?        robust departure from its own past
  2. Did its denominator move?   the same measure on the sample size
  3. Does the move SURVIVE conditioning on the denominator?

Step 3 fits the metric against its denominator over the history and asks whether
the latest reading is still unusual once that relationship is accounted for. If
the departure largely disappears, the verdict is SAMPLING_ARTIFACT. If it
survives, the verdict is REAL — and it is now stronger evidence than the raw
move was, because the obvious confound has been ruled out.

Conditioning runs whenever the fit is informative, NOT only when the denominator
itself looks unusual. That distinction matters: in the incident above the index
correlated 0.70 with its own measurement count, so a perfectly ordinary-sized
dip in coverage still moved the published number. Gating on "did the denominator
do something strange" would have waved it straight through.

Deliberately conservative: the fit quality and sample count are published with
every verdict, and a weak fit yields UNCLEAR rather than licensing either
conclusion.

Standard library only, deterministic.
"""
from __future__ import annotations

MIN_PAIRS = 10          # below this a fit is not worth trusting
MOVE_Z = 2.0            # robust departure at which a series counts as having moved
GOOD_FIT_R = 0.30       # weaker correlation than this makes conditioning useless
SURVIVES_FRAC = 0.5     # keep this share of the raw departure and the move stands


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def _robust_z(reference: list[float], x: float) -> float | None:
    if not reference:
        return None
    med = _median(reference)
    scale = 1.4826 * _median([abs(s - med) for s in reference])
    return (x - med) / scale if scale > 0 else None


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Least squares y = a + b*x, plus correlation. None if x never varies."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    r = sxy / ((sxx * syy) ** 0.5) if syy > 0 else 0.0
    return (my - b * mx, b, r)


def check(metric: list[float], sample_size: list[float]) -> dict:
    """Judge the latest reading. Both lists are parallel, oldest first."""
    n = min(len(metric), len(sample_size))
    metric, sample_size = metric[:n], sample_size[:n]
    if n < MIN_PAIRS:
        return {"verdict": "NOT_ENOUGH_HISTORY", "n_pairs": n,
                "reading": f"need {MIN_PAIRS} paired readings to check this, have {n}"}

    m_ref, d_ref = metric[:-1], sample_size[:-1]
    m_now, d_now = metric[-1], sample_size[-1]
    z_metric = _robust_z(m_ref, m_now)
    z_denom = _robust_z(d_ref, d_now)
    med_d = _median(d_ref)

    out = {
        "n_pairs": n,
        "metric_now": round(m_now, 6),
        "sample_size_now": round(d_now, 6),
        "metric_departure": None if z_metric is None else round(z_metric, 2),
        "sample_size_departure": None if z_denom is None else round(z_denom, 2),
        "sample_size_change_pct": (round(100.0 * (d_now - med_d) / abs(med_d), 1)
                                   if med_d else None),
    }

    if z_metric is None or abs(z_metric) < MOVE_Z:
        out["verdict"] = "NO_MOVE"
        out["reading"] = "the metric has not departed from its own history"
        return out

    fit = _fit(d_ref, m_ref)
    if fit is None:
        out["verdict"] = "UNCLEAR"
        out["reading"] = "the sample size never varies here, so it cannot be ruled in or out"
        return out
    a, b, r = fit
    out["fit"] = {"slope": round(b, 8), "intercept": round(a, 6),
                  "correlation": round(r, 3)}

    if abs(r) < GOOD_FIT_R:
        out["verdict"] = "REAL"
        out["reading"] = (f"the move is real: sample size does not track this metric "
                          f"historically (correlation {r:.2f}), so it cannot explain it")
        return out

    resid_ref = [y - (a + b * x) for x, y in zip(d_ref, m_ref)]
    resid_now = m_now - (a + b * d_now)
    z_resid = _robust_z(resid_ref, resid_now)
    out["departure_after_conditioning"] = None if z_resid is None else round(z_resid, 2)

    if z_resid is None:
        spread = max((abs(v) for v in resid_ref), default=0.0)
        if abs(resid_now) <= max(spread, 1e-9) * 10:
            out["verdict"] = "SAMPLING_ARTIFACT"
            out["reading"] = (
                f"the metric is a near-exact function of its sample size on this history "
                f"(correlation {r:.2f}) and the latest reading sits on that same "
                f"relationship: the move is the sample, not the world")
        else:
            out["verdict"] = "REAL"
            out["reading"] = (f"sample size explained this metric exactly until now "
                              f"(correlation {r:.2f}); the latest reading breaks that "
                              f"relationship, which is itself the finding")
    elif abs(z_resid) >= abs(z_metric) * SURVIVES_FRAC and abs(z_resid) >= MOVE_Z:
        out["verdict"] = "REAL"
        out["reading"] = (
            f"the move survives once sample size is accounted for "
            f"({z_metric:.1f} -> {z_resid:.1f}); the sample tracks this metric "
            f"(correlation {r:.2f}) and absorbs part of the move, but not the bulk")
    else:
        out["verdict"] = "SAMPLING_ARTIFACT"
        out["reading"] = (
            f"your sample size changed by {out['sample_size_change_pct']}% at the same "
            f"time and it tracks this metric historically (correlation {r:.2f}). "
            f"Accounting for it shrinks the move from {z_metric:.1f} to {z_resid:.1f}. "
            f"This is not evidence that the underlying thing changed.")

    out["method"] = (
        "robust (median-absolute-deviation) departure of the metric and of its sample "
        "size, then least squares metric ~ a + b*sample_size over the history with the "
        "latest residual scored against past residuals. Conditioning runs whenever the "
        f"fit is informative (|correlation| >= {GOOD_FIT_R}), not only when the sample "
        "size itself looks unusual.")
    out["caveats"] = [
        "rules out the SAMPLE-SIZE confound specifically, not every confound",
        "the fit is linear and history is finite; correlation and pair count are "
        "published with every verdict so the call can be judged",
    ]
    return out
