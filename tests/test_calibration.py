"""Deterministic simulation checks under specific synthetic data generators.

These checks detect numerical or behavioral regressions. They are not evidence
of performance, coverage, profitability or false-alarm control on market data.
Bernoulli confidence-sequence guarantees require their stated sampling model;
change and forecast diagnostics expose no universal calibration guarantee.
"""
import random

from noisefloor.change import scan
from noisefloor.experiment import compare, naive_z_test
from noisefloor.forecast import score


# ── peeking is safe (the headline claim) ────────────────────────────────────────

def _aa_trial(seed, n, rate, alpha, peek_every=20):
    """One A/A test: both arms identical, so any winner is false."""
    rng = random.Random(seed)
    a_s = a_n = b_s = b_n = 0
    cs_hit = z_hit = False
    for i in range(n):
        if rng.random() < 0.5:
            a_n += 1
            a_s += rng.random() < rate
        else:
            b_n += 1
            b_s += rng.random() < rate
        if i % peek_every == 0 and a_n > 5 and b_n > 5:
            if not cs_hit and compare(a_s, a_n, b_s, b_n, alpha=alpha)["decided"]:
                cs_hit = True
            if not z_hit:
                p = naive_z_test(a_s, a_n, b_s, b_n)
                if p is not None and p <= alpha:
                    z_hit = True
    return cs_hit, z_hit


def test_peeking_does_not_inflate_false_positives():
    """The promise: check as often as you like. 200 A/A tests, peeked throughout."""
    trials, alpha = 200, 0.05
    false_winners = sum(_aa_trial(9000 + t, 1500, 0.10, alpha)[0] for t in range(trials))
    assert false_winners / trials <= alpha, (
        f"{false_winners}/{trials} false winners exceeds the {alpha:.0%} budget")


def test_the_naive_test_really_is_broken_by_peeking():
    """Pins the comparison the pitch rests on: a peeked z-test blows way past
    its nominal rate on the very same data. If this ever stops being true the
    product has no reason to exist and the README needs rewriting."""
    trials, alpha = 200, 0.05
    z_false = sum(_aa_trial(9000 + t, 1500, 0.10, alpha)[1] for t in range(trials))
    assert z_false / trials > 0.20, (
        f"peeked z-test only produced {z_false}/{trials} false winners; the "
        f"headline comparison in the README assumes it is far worse than alpha")


# ── it still finds real effects ─────────────────────────────────────────────────

def test_a_real_difference_is_found_and_the_right_arm_wins():
    trials, found, correct = 30, 0, 0
    for t in range(trials):
        rng = random.Random(700 + t)
        a_s = a_n = b_s = b_n = 0
        for i in range(30000):
            if rng.random() < 0.5:
                a_n += 1
                a_s += rng.random() < 0.10
            else:
                b_n += 1
                b_s += rng.random() < 0.15
            if i % 100 == 0 and a_n > 5 and b_n > 5:
                r = compare(a_s, a_n, b_s, b_n)
                if r["decided"]:
                    found += 1
                    correct += r["winner"] == "B"
                    break
    assert found >= trials * 0.9, f"only found the effect {found}/{trials} times"
    assert correct == found, "picked the wrong arm at least once"


def test_no_winner_is_called_on_tiny_samples():
    assert compare(1, 3, 2, 3)["decided"] is False


# ── forecast ranges evaluated on one stationary synthetic fixture ───────────────────────────────────────

def test_forecast_coverage_converges_to_nominal():
    rng = random.Random(7)
    long_series = [50 + rng.gauss(0, 2) for _ in range(2000)]
    r = score(long_series)
    assert r["ok"] and abs(r["empirical_coverage"] - 0.80) < 0.04, r["empirical_coverage"]
    assert r["calibrated"] is True


def test_short_series_is_reported_as_not_yet_calibrated_rather_than_claimed_good():
    rng = random.Random(7)
    r = score([50 + rng.gauss(0, 2) for _ in range(50)])
    assert r["ok"]
    # it may or may not be calibrated at n=50; the requirement is that the flag
    # reflects the measurement rather than always saying yes
    assert r["calibrated"] == (abs(r["empirical_coverage"] - 0.80) <= 0.05)


# ── descriptive change detection on independent synthetic observations ──────────────────────────

def test_iid_fixture_retains_low_empirical_alarm_frequency():
    """A bounded simulation regression, not a distribution-free guarantee."""
    trials, alarms = 200, 0
    for t in range(trials):
        rng = random.Random(11 + t)
        alarms += bool(scan([50 + rng.gauss(0, 5) for _ in range(80)])["change_points"])
    assert alarms / trials <= 80 / 500.0


def test_a_real_shift_is_caught_in_both_directions():
    rng = random.Random(3)
    up = [50 + rng.gauss(0, 3) for _ in range(40)] + [80 + rng.gauss(0, 3) for _ in range(15)]
    down = [80 + rng.gauss(0, 3) for _ in range(40)] + [50 + rng.gauss(0, 3) for _ in range(15)]
    assert scan(up)["change_points"], "missed an upward shift"
    assert scan(down)["change_points"], "missed a downward shift"


def test_upward_only_mode_is_blind_to_collapses():
    """Why two-sided is the default: half the failures are things going to zero."""
    rng = random.Random(3)
    down = [80 + rng.gauss(0, 3) for _ in range(40)] + [50 + rng.gauss(0, 3) for _ in range(15)]
    assert scan(down, two_sided=False)["change_points"] == []
    assert scan(down, two_sided=True)["change_points"]
