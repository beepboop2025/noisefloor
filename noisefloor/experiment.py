"""Peek-safe A/B testing: check your experiment as often as you like.

THE PROBLEM. You ship a change, then you look at the results. Then you look
again an hour later. Then again tomorrow. You stop when it looks like a win.

That procedure lies to you, and not by a little. A t-test or a two-proportion
z-test is only valid if you decide the sample size in advance, run to exactly
that size, and look ONCE. Every extra look is another chance for noise to
wander across the threshold, and stopping the moment it does is the statistical
equivalent of flipping a coin until you are ahead and then declaring the coin
biased. With continuous monitoring the false-positive rate of a nominally 5%
test climbs toward certainty rather than staying at 5%.

Nobody stops peeking, because peeking is the rational thing to do when a bad
variant is costing money. So the fix is not discipline, it is different maths.

THE TOOL. A confidence sequence: an interval that is valid at EVERY time
simultaneously, not at one pre-chosen time. Formally, for a sequence of
intervals CI_t,

    P( there exists any t at all where the true rate falls outside CI_t ) <= alpha

Because the guarantee is over all times at once, you may look after every single
observation, stop whenever you like for any reason, and the coverage still
holds. Optional stopping is free.

The construction is a beta-binomial mixture test martingale. A mixture of
likelihood ratios is a non-negative martingale under the null, so Ville's
inequality bounds the chance it EVER crosses 1/alpha by alpha — the interval is
simply every rate that evidence has not yet crossed that line against. Because
it is exact for Bernoulli outcomes rather than a worst-case bound for anything
in [0,1], it is far tighter than a range-based boundary: Hoeffding charges you
for a variance of 1/4 however rare the event, and conversion rates are rarely
near 1/2. For the general theory see Ramdas, Grünwald, Vovk & Shafer,
"Game-theoretic statistics and safe anytime-valid inference", Statistical
Science 2023 (arXiv:2210.01948), and Grünwald, de Heide & Koolen, "Safe
Testing", JRSS-B 2024 (arXiv:1906.07801).

This covers conversion rates, click-through, retention, pass rates and any
other proportion. A winner is declared only when the two arms' intervals are
disjoint, with alpha split across the arms so the pair carries alpha.

MEASURED BEHAVIOUR (reproduce with tests/test_calibration.py). Four hundred A/A
tests — both arms on an identical 10% rate, so every "winner" is false — peeked
at every 20 observations out to 3,000:

    two-proportion z-test, peeked     40.2% false winners
    noisefloor confidence sequence     0.0% false winners

WHAT THIS COSTS. Anytime validity is not free. At any fixed sample size the
interval is wider than a fixed-horizon one, so calling the same effect takes
roughly twice the data: 10% versus 13% resolves at a median of about 14,300
observations here, against roughly 7,000 for a correctly run one-look test. You
are buying the right to stop early, to stop late, and to stop for any reason at
all. For most teams that is a bargain, because the realistic alternative is not
a clean one-look test — it is a one-look test being peeked at, which is the 40%
column above.

Everything here is standard library only and deterministic.
"""
from __future__ import annotations

import math

DEFAULT_ALPHA = 0.05

# Beta prior for the mixture below. Beta(1,1) is uniform on [0,1] — it assumes
# nothing about the rate. The guarantee holds for ANY proper prior, so this
# choice affects power only, never validity.
PRIOR_A = 1.0
PRIOR_B = 1.0


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_evalue(successes: int, n: int, p0: float, *,
               prior_a: float = PRIOR_A, prior_b: float = PRIOR_B) -> float:
    """log of the evidence against the rate being exactly p0.

    A beta-binomial mixture of likelihood ratios. Because a mixture of
    likelihood ratios is a test martingale, Ville's inequality gives
    P(ever exceeding 1/alpha) <= alpha, which is what makes peeking free.

    Being exact for Bernoulli data rather than a worst-case bound on [0,1] is
    what buys the power: a range-based bound like Hoeffding's charges you for
    a variance of 1/4 no matter how rare the event, and conversion rates are
    usually nowhere near 1/2.
    """
    s, f = successes, n - successes
    if p0 <= 0.0:
        return math.inf if s > 0 else 0.0
    if p0 >= 1.0:
        return math.inf if f > 0 else 0.0
    log_marginal = _log_beta(prior_a + s, prior_b + f) - _log_beta(prior_a, prior_b)
    log_null = s * math.log(p0) + f * math.log1p(-p0)
    return log_marginal - log_null


def confidence_sequence(successes: int, n: int, *, alpha: float = DEFAULT_ALPHA,
                        prior_a: float = PRIOR_A, prior_b: float = PRIOR_B,
                        tol: float = 1e-6) -> tuple[float, float]:
    """Anytime-valid interval for a rate, from successes out of n.

    The interval is every rate the evidence has NOT ruled out: all p0 with
    log_evalue(p0) < log(1/alpha). Valid at every n simultaneously, so you may
    call this after each observation and stop whenever you like.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")

    threshold = math.log(1.0 / alpha)
    centre = min(max(successes / n, tol), 1.0 - tol)
    if log_evalue(successes, n, centre, prior_a=prior_a, prior_b=prior_b) >= threshold:
        # even the best-supported rate is "ruled out" — only possible when the
        # mixture has not yet accumulated evidence; report total ignorance
        return (0.0, 1.0)

    def _bisect(lo: float, hi: float) -> float:
        # invariant: lo is excluded, hi is included
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if log_evalue(successes, n, mid, prior_a=prior_a,
                          prior_b=prior_b) >= threshold:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lower = 0.0 if log_evalue(successes, n, tol, prior_a=prior_a,
                              prior_b=prior_b) < threshold else _bisect(0.0, centre)
    upper = 1.0 if log_evalue(successes, n, 1.0 - tol, prior_a=prior_a,
                              prior_b=prior_b) < threshold else _bisect(1.0, centre)
    return (max(0.0, lower), min(1.0, upper))


def compare(a_successes: int, a_n: int, b_successes: int, b_n: int, *,
            alpha: float = DEFAULT_ALPHA,
            labels: tuple[str, str] = ("A", "B")) -> dict:
    """Can you call a winner yet? Safe to run after every observation.

    Returns the verdict, both intervals, and how much more data a still-
    undecided test would plausibly need.
    """
    if a_n < 0 or b_n < 0:
        raise ValueError("sample sizes cannot be negative")
    if not (0 <= a_successes <= a_n) or not (0 <= b_successes <= b_n):
        raise ValueError("successes must be between 0 and n")

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
                   "difference is within what chance explains")

    out = {
        "decided": decided,
        "winner": winner,
        "reading": reading,
        "alpha": alpha,
        labels[0]: {"successes": a_successes, "n": a_n,
                    "rate": None if a_rate is None else round(a_rate, 5),
                    "interval": [round(a_lo, 5), round(a_hi, 5)]},
        labels[1]: {"successes": b_successes, "n": b_n,
                    "rate": None if b_rate is None else round(b_rate, 5),
                    "interval": [round(b_lo, 5), round(b_hi, 5)]},
        "safe_to_peek": True,
        "method": (
            "anytime-valid confidence sequences from a beta-binomial mixture test "
            "martingale (Ville's inequality; see arXiv:2210.01948 for the general "
            "theory). A winner is called only when the two intervals are disjoint, "
            "with alpha split across the arms. Valid at every sample size "
            "simultaneously, so checking after every observation and stopping "
            "whenever you like does not inflate the error rate. Measured on A/A "
            "tests: a peeked z-test produced 40.2% false winners, this produced 0%."),
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
    peeking_damage(). Valid only at a single pre-committed sample size.
    """
    if a_n <= 0 or b_n <= 0:
        return None
    p1, p2 = a_successes / a_n, b_successes / b_n
    pool = (a_successes + b_successes) / (a_n + b_n)
    se = math.sqrt(pool * (1 - pool) * (1 / a_n + 1 / b_n))
    if se <= 0:
        return None
    z = (p2 - p1) / se
    # two-sided p-value via the normal CDF
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
