"""Regression tests for inference boundaries and exact forecast replay."""
import math
import random

import pytest

from noisefloor import change, coverage, experiment, forecast, multiple


def test_raw_monitoring_score_cannot_implicitly_enable_fdr_selection():
    result = change.scan([float(i) for i in range(40)])
    board = multiple.select({"metric": result["evidence"]})
    assert result["is_e_value"] is False
    assert result["statistic_type"] == "shiryaev_roberts"
    assert result["guarantee"] is None
    assert board["selected"] == []
    assert board["guarantee"] is None
    assert board["board_evidence"] is None
    assert board["validity"] == "unverified_inputs"


def test_conditional_ebh_requires_explicit_boolean_confirmation():
    result = multiple.select({"a": 100.0, "b": 1.0}, valid_evalues=True)
    assert result["selected"] == ["a"]
    assert result["threshold"] == 20
    assert result["validity"] == "conditional_on_valid_evalues"
    assert result["guarantee"].startswith("If every supplied")
    with pytest.raises(ValueError):
        multiple.select({"a": 100.0}, valid_evalues="true")


def test_all_change_modes_publish_descriptive_assumptions():
    for half_life in (None, 5.0):
        for two_sided in (False, True):
            result = change.scan([float(i % 3) for i in range(30)],
                                 half_life=half_life, two_sided=two_sided)
            assert result["guarantee"] is None
            assert result["validity"] == "descriptive"
            assert result["assumptions"]
            assert "n/1000" not in str(result)


def test_change_effect_uses_the_current_regime_reference_after_reset():
    values = [0.0] * 30 + [float(i) for i in range(100, 130)]
    alarm = change.scan(values)["change_points"][0]
    result = change.scan(values[:alarm + 2])
    assert result["state"] == "warming_up"
    assert result["effect_reference_size"] == 0
    assert result["effect"]["robust_z"] is None


def _volatility_shift():
    rng = random.Random(193)
    values = [0.0]
    for index in range(140):
        values.append(values[-1] + rng.gauss(0, 1 if index < 70 else 10))
    return values


def test_every_scored_forecast_equals_what_public_interface_issued():
    values = _volatility_shift()
    report = forecast.score(values)
    for record in report["records"]:
        issued = forecast.next_value(values[:record["index"]])
        for key in ("forecast", "interval", "nominal_coverage", "adaptive_alpha"):
            assert record[key] == issued[key]
        assert record["covered"] == (issued["interval"][0] <= record["actual"] <= issued["interval"][1])
    assert report["n_misses"] == len(report["misses"])
    assert report["empirical_coverage"] == 1 - len(report["misses"]) / report["n_forecasts"]
    assert report["records_complete"] is True


def test_future_values_cannot_change_prior_forecasts_or_scores():
    prefix = _volatility_shift()[:75]
    before = forecast.score(prefix)
    extended = forecast.score(prefix + [1000.0, -4000.0, 2.0])
    assert extended["records"][:len(before["records"])] == before["records"]


def test_adverse_sequence_publishes_every_miss_without_a_coverage_claim():
    report = forecast.score([float(i * i) for i in range(300)])
    assert report["n_forecasts"] == report["n_misses"] == 289
    assert len(report["misses"]) == 289
    assert len(report["worst_misses"]) == 5
    assert report["empirical_coverage"] == 0
    assert report["validity"] == "empirical_only"
    assert report["guarantee"] is None
    assert report["calibrated"] is False


def test_constant_history_jump_is_not_no_move():
    result = coverage.check([1.0] * 20 + [1000.0], [100.0] * 21)
    assert result["verdict"] == "UNCLEAR"
    assert result["assessment"] == "departure_from_zero_spread"
    assert result["metric_delta"] == 999
    assert result["metric_departure"] is None
    assert result["is_causal"] is False


def test_mismatched_arrays_cannot_silently_drop_the_latest_move():
    with pytest.raises(ValueError, match="same length"):
        coverage.check(list(range(20)) + [1000], list(range(10)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_public_numeric_series_reject_invalid_values(bad):
    functions = [
        lambda: change.scan([1.0, bad]),
        lambda: change.p_value([1.0], bad),
        lambda: coverage.check([1.0, bad], [100.0, 100.0]),
        lambda: forecast.next_value([1.0, bad]),
        lambda: forecast.score([1.0, bad]),
        lambda: multiple.select({"x": bad}),
        lambda: multiple.merge([bad]),
    ]
    for function in functions:
        with pytest.raises(ValueError):
            function()


@pytest.mark.parametrize("bad", [0, 1, -1, 1.1, True, float("nan"), float("inf")])
def test_probabilities_are_validated_even_without_history(bad):
    for function in (
        lambda: forecast.next_value([], nominal=bad),
        lambda: forecast.score([], nominal=bad),
        lambda: multiple.select({}, alpha=bad),
        lambda: experiment.compare(0, 0, 0, 0, alpha=bad),
        lambda: experiment.confidence_sequence(0, 0, alpha=bad),
    ):
        with pytest.raises(ValueError):
            function()


@pytest.mark.parametrize("bad", [-1, 1.5, True, float("nan"), "3"])
def test_bernoulli_counts_cannot_be_coerced(bad):
    for function in (
        lambda: experiment.compare(bad, 10, 1, 10),
        lambda: experiment.confidence_sequence(bad, 10),
        lambda: experiment.log_evalue(bad, 10, 0.5),
        lambda: experiment.naive_z_test(bad, 10, 1, 10),
    ):
        with pytest.raises(ValueError):
            function()


def test_bernoulli_contract_and_exact_boundary_likelihood():
    assert experiment.log_evalue(0, 10, 0) == pytest.approx(-math.log(11))
    assert experiment.log_evalue(10, 10, 1) == pytest.approx(-math.log(11))
    result = experiment.compare(1, 10, 2, 10)
    assert result["validity"] == "conditional_on_bernoulli_model"
    assert result["assumptions"]
    for labels in (("A", "A"), ("alpha", "B"), ("A", "")):
        with pytest.raises(ValueError):
            experiment.compare(1, 10, 2, 10, labels=labels)


def test_invalid_rank_and_interval_parameters_are_rejected():
    for half_life in (0, -1, True, math.inf):
        with pytest.raises(ValueError):
            change.scan([], half_life=half_life)
    with pytest.raises(ValueError):
        change.p_value([1, 2], 3, weights=[-1, 1])
    with pytest.raises(ValueError):
        change.p_value([1, 2], 3, side="left")
    with pytest.raises(ValueError):
        change.scan([1, 2], two_sided="false")
    with pytest.raises(ValueError):
        coverage.check([1], [0])
    with pytest.raises(ValueError):
        multiple.select({"x": -1})
    with pytest.raises(ValueError):
        forecast.interval_score(1, 2, 0, .2)
    with pytest.raises(ValueError):
        forecast.weighted_interval_score(1, 1, {1: (0, 2)})
