"""Bernoulli confidence sequences under explicit sampling assumptions.

A beta-binomial mixture likelihood ratio supplies a confidence sequence for a
constant Bernoulli success probability. The two-arm comparison splits alpha
between arms. Monitoring cumulative counts is allowed under this model;
changing success probabilities, arbitrary dependent trading outcomes, selected
windows and non-Bernoulli data do not inherit that guarantee.

References: https://arxiv.org/abs/2210.01948 and
https://arxiv.org/abs/1906.07801. Numeric simulation tests are diagnostic checks,
not proof that an arbitrary financial strategy has the assumed data model.
"""
from __future__ import annotations

import math

from ._validation import count, number, probability

DEFAULT_ALPHA = 0.05

# Beta prior for the mixture below. Beta(1,1) is uniform on [0,1] — it assumes
# nothing about the rate. The guarantee holds for ANY proper prior, so this
# choice affects power only, never validity.
PRIOR_A = 1.0
PRIOR_B = 1.0


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _counts(successes, n):
    count(successes, "successes")
    count(n, "n")
    if successes > n:
        raise ValueError("successes must not exceed n")


def _marginal(successes, n, prior_a, prior_b):
    prior_a = number(prior_a, "prior_a")
    prior_b = number(prior_b, "prior_b")
    if prior_a <= 0 or prior_b <= 0:
        raise ValueError("prior parameters must be positive")
    try:
        return number(_log_beta(prior_a + successes, prior_b + n - successes) -
                      _log_beta(prior_a, prior_b), "log marginal likelihood")
    except OverflowError as exc:
        raise ValueError("counts or prior parameters exceed numerical range") from exc


def _log_value(successes, n, p0, marginal):
    if p0 == 0:
        return math.inf if successes else marginal
    if p0 == 1:
        return math.inf if n - successes else marginal
    return marginal - successes * math.log(p0) - (n - successes) * math.log1p(-p0)


def log_evalue(successes: int, n: int, p0: float, *,
               prior_a: float = PRIOR_A, prior_b: float = PRIOR_B) -> float:
    """Log beta-binomial mixture evidence against a specified Bernoulli rate.

    Validity requires Bernoulli trials with fixed conditional null probability,
    a prior fixed before observation, and cumulative rather than selected data.
    Positive infinity occurs only for observations impossible under p0=0 or 1.
    """
    _counts(successes, n)
    p0 = number(p0, "p0")
    if not 0 <= p0 <= 1:
        raise ValueError("p0 must be in [0,1]")
    return _log_value(successes, n, p0, _marginal(successes, n, prior_a, prior_b))


def confidence_sequence(successes: int, n: int, *, alpha: float = DEFAULT_ALPHA,
                        prior_a: float = PRIOR_A, prior_b: float = PRIOR_B,
                        tol: float = 1e-6) -> tuple[float, float]:
    """Confidence sequence for a constant Bernoulli rate under the stated model."""
    _counts(successes, n)
    alpha = probability(alpha, "alpha")
    tol = number(tol, "tol")
    if not 0 < tol < 0.5:
        raise ValueError("tol must be in (0,0.5)")
    marginal = _marginal(successes, n, prior_a, prior_b)
    if n == 0:
        return (0.0, 1.0)
    threshold = -math.log(alpha)
    centre = successes / n

    def evidence(p0):
        return _log_value(successes, n, p0, marginal)

    if evidence(centre) >= threshold:
        return (0.0, 1.0)  # conservative fallback for numerical loss of precision

    def boundary(excluded, included):
        for _ in range(60):
            middle = excluded / 2 + included / 2
            if evidence(middle) >= threshold:
                excluded = middle
            else:
                included = middle
        return excluded  # round outward to preserve interval coverage

    lower = 0.0 if evidence(0.0) < threshold else boundary(0.0, centre)
    upper = 1.0 if evidence(1.0) < threshold else boundary(1.0, centre)
    return max(0.0, lower), min(1.0, upper)


def compare(a_successes: int, a_n: int, b_successes: int, b_n: int, *,
            alpha: float = DEFAULT_ALPHA,
            labels: tuple[str, str] = ("A", "B")) -> dict:
    """Can you call a winner yet? Safe to run after every observation.

    Returns the verdict, both intervals, and how much more data a still-
    undecided test would plausibly need.
    """
    _counts(a_successes, a_n)
    _counts(b_successes, b_n)
    alpha = probability(alpha, "alpha")
    reserved = {"decided", "winner", "reading", "alpha", "safe_to_peek", "method",
                "guarantee", "assumptions", "validity", "observed_difference_pp",
                "what_would_decide_it"}
    if (not isinstance(labels, (list, tuple)) or len(labels) != 2 or
            any(not isinstance(label, str) or not label or label in reserved for label in labels) or
            labels[0] == labels[1]):
        raise ValueError("labels must be two distinct non-empty names that do not replace result fields")

    # alpha split across the two arms so the pair carries the stated alpha
    per_arm = alpha / 2.0
    a_lo, a_hi = confidence_sequence(a_successes, a_n, alpha=per_arm)
    b_lo, b_hi = confidence_sequence(b_successes, b_n, alpha=per_arm)

    a_rate = a_successes / a_n if a_n else None
    b_rate = b_successes / b_n if b_n else None

    if a_n == 0 or b_n == 0:
        decided, winner = False, None
        reading = "no data yet in one arm"
    elif a_lo > b_hi:
        decided, winner = True, labels[0]
        reading = f"{labels[0]} wins: its interval sits entirely above {labels[1]}'s"
    elif b_lo > a_hi:
        decided, winner = True, labels[1]
        reading = f"{labels[1]} wins: its interval sits entirely above {labels[0]}'s"
    else:
        decided, winner = False, None
        reading = ("no winner yet — the intervals still overlap, so the observed "
                   "difference has not cleared this test under its assumptions")

    out = {
        "decided": decided,
        "winner": winner,
        "reading": reading,
        "alpha": alpha,
        labels[0]: {"successes": a_successes, "n": a_n,
                    "rate": None if a_rate is None else round(a_rate, 5),
                    "interval": [a_lo, a_hi]},
        labels[1]: {"successes": b_successes, "n": b_n,
                    "rate": None if b_rate is None else round(b_rate, 5),
                    "interval": [b_lo, b_hi]},
        "safe_to_peek": True,
        "validity": "conditional_on_bernoulli_model",
        "guarantee": f"Under the stated Bernoulli sampling assumptions, the chance of either confidence sequence ever excluding its true constant rate is at most {alpha:g}.",
        "assumptions": [
            "Each arm consists of Bernoulli observations with a constant conditional success probability.",
            "Counts are cumulative and observations are not selected or reused across windows.",
            "Adaptive allocation must be predictable from past information.",
            "The prior and hypothesis definitions are fixed before examining outcomes.",
            "Changing market regimes or arbitrary dependent trading outcomes are not covered.",
        ],
        "method": "beta-binomial mixture confidence sequences with alpha split between arms; Ville's inequality, arXiv:2210.01948; guarantees are conditional on the stated sampling model",

    }

    if not decided and a_n and b_n:
        out["observed_difference_pp"] = round(100.0 * (b_rate - a_rate), 2)
        out["what_would_decide_it"] = (
            "either a larger true difference or more data — the intervals "
            "narrow roughly with the square root of sample size")
    return out


def naive_z_test(a_successes: int, a_n: int, b_successes: int, b_n: int) -> float | None:
    """Two-proportion z-test p-value: the thing you should NOT peek at.

    Provided so the cost of peeking can be measured rather than asserted — see
    the calibration tests. Its nominal interpretation requires a valid sampling
    model and a single pre-committed analysis, with adequate expected counts.
    """
    _counts(a_successes, a_n)
    _counts(b_successes, b_n)
    if a_n == 0 or b_n == 0:
        return None
    p1, p2 = a_successes / a_n, b_successes / b_n
    pool = (a_successes + b_successes) / (a_n + b_n)
    se = math.sqrt(pool * (1 - pool) * (1 / a_n + 1 / b_n))
    if se <= 0:
        return None
    z = (p2 - p1) / se
    # two-sided p-value via the normal CDF
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
