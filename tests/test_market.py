"""Semantic market safety and useful-detection tests, not performance claims."""
import copy
import math
from datetime import datetime, timedelta, timezone

import pytest

from noisefloor import market


def stamp(day):
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)).isoformat()


def series(values, **kwargs):
    item = {"id": "spread", "kind": "spread", "unit": "basis_points",
            "source": {"id": "fixture", "rights": "permitted"},
            "max_age_seconds": 86400, "max_gap_seconds": 86400,
            "observations": [{"observed_at": stamp(i), "available_at": stamp(i), "value": v}
                             for i, v in enumerate(values)]}
    item.update(kwargs)
    return item


def analyze(values, **kwargs):
    return market.assess([series(values, **kwargs)], as_of=stamp(len(values)-1))["series"][0]


BASELINE = [10., 11., 9., 10., 12., 8.] * 5


def test_persistent_shift_receives_review_with_effect_and_source():
    result = analyze(BASELINE + [30., 31., 30.])
    assert result["status"] == "review"
    assert "persistent_departure" in result["reasons"]
    assert result["diagnostics"]["median_change"] == 20
    assert result["source"]["id"] == "fixture"
    assert result["validity"] == "descriptive"
    assert result["guarantee"] is None and result["is_e_value"] is False


def test_isolated_spike_is_not_called_persistent_or_proved_noise():
    result = analyze(BASELINE + [10., 10., 100.])
    assert result["status"] == "watch"
    assert "latest_departure" in result["reasons"]
    assert "persistent_departure" not in result["reasons"]


def test_regular_price_trend_is_analyzed_as_returns_not_price_level():
    result = analyze([100 * math.exp(.01*i) for i in range(34)], kind="price", unit="USD")
    assert result["transform"] == "log_return"
    assert abs(result["diagnostics"]["median_change"]) < 1e-12
    assert result["analyzed_points"] == 33


@pytest.mark.parametrize("daily_log_return", [.001, .03, -.001, -.03])
@pytest.mark.parametrize("price_scale", [100.0, 1e-200])
def test_constant_geometric_returns_do_not_alert_on_log_rounding(daily_log_return, price_scale):
    result = analyze([price_scale * math.exp(daily_log_return*i) for i in range(34)],
                     kind="price", unit="USD")
    assert result["status"] == "no_supported_departure"
    assert result["reasons"] == []
    assert result["diagnostics"]["representation_tolerance"] > 0
    assert result["diagnostics"]["monitor"]["state"] == "steady"


@pytest.mark.parametrize("direction", [1, -1])
def test_tiny_direct_measurement_shift_retains_direction_and_effect(direction):
    result = analyze([0.0] * 30 + [direction * 1e-200] * 3)
    assert result["status"] == "review"
    assert "persistent_departure" in result["reasons"]
    assert result["diagnostics"]["persistence_points"] == 3
    assert result["diagnostics"]["median_change"] == direction * 1e-200
    assert result["diagnostics"]["latest_change"] == direction * 1e-200
    assert result["diagnostics"]["representation_tolerance"] == 0


def test_constant_baseline_jump_cannot_be_quiet():
    result = analyze([1.] * 30 + [100.] * 3)
    assert result["status"] == "review"
    assert result["diagnostics"]["robust_departure"] is None
    assert "constant_baseline_changed" in result["reasons"]


def test_null_is_not_filled_and_requires_new_contiguous_history():
    result = analyze(BASELINE + [None, 1., 2., 3.])
    assert result["status"] == "insufficient_visibility"
    assert result["contiguous_points"] == 3
    assert result["missing_points"] == 1


def test_latest_null_has_specific_visibility_gap():
    result = analyze(BASELINE + [10., 10., None])
    assert "latest_value_missing" in result["reasons"]
    assert result["diagnostics"] is None


def test_gap_never_creates_cross_gap_price_return():
    item = series([100.] * 35, kind="price")
    for p in item["observations"][-3:]:
        p["observed_at"] = stamp(100 + item["observations"].index(p))
        p["available_at"] = p["observed_at"]
    r = market.assess([item], as_of=stamp(134))["series"][0]
    assert r["contiguous_points"] == 2
    assert r["status"] == "insufficient_visibility"


def test_null_at_gap_does_not_reenter_numeric_history():
    item = series([100.] * 34 + [None], kind="price")
    item["observations"][-1].update(observed_at=stamp(100), available_at=stamp(100))
    r = market.assess([item], as_of=stamp(100))["series"][0]
    assert r["contiguous_points"] == 0
    assert "latest_value_missing" in r["reasons"]


@pytest.mark.parametrize("rights", ["unknown", "restricted"])
def test_rights_gap_is_not_assessed(rights):
    result = analyze(BASELINE + [20.] * 3, source={"id": "licensed-feed", "rights": rights})
    assert result["status"] == "insufficient_visibility"
    assert result["diagnostics"] is None


def test_staleness_uses_observation_clock_not_response_clock():
    item = series(BASELINE + [10.] * 3)
    item["observations"][-1]["available_at"] = stamp(40)
    result = market.assess([item], as_of=stamp(40))["series"][0]
    assert "stale_observation" in result["reasons"]
    assert result["age_seconds"] == 8 * 86400


def test_no_future_observation_or_future_knowledge_enters_diagnostic():
    item = series(BASELINE + [20.] * 3)
    result = market.assess([item], as_of=stamp(31))["series"][0]
    assert "future_observation" in result["reasons"]
    item["observations"][-1]["available_at"] = stamp(40)
    result = market.assess([item], as_of=stamp(32))["series"][0]
    assert "not_yet_available" in result["reasons"]


def test_missing_availability_retains_uncertainty():
    item = series(BASELINE + [10.] * 3)
    del item["observations"][0]["available_at"]
    result = market.assess([item], as_of=stamp(32))["series"][0]
    assert not result["availability_timestamps_complete"]
    assert not result["as_published_history_verified"]


def test_measurement_drop_is_a_review_not_proof_market_move_is_fake():
    item = series(BASELINE + [20.] * 3)
    for i, p in enumerate(item["observations"]):
        p["sample_size"] = 100 if i < 30 else 20
    result = market.assess([item], as_of=stamp(32))["series"][0]
    assert result["diagnostics"]["coverage"]["relative_change"] == -.8
    assert "review_measurement_coverage" in result["reasons"]
    assert "persistent_departure" in result["reasons"]


def test_incomplete_coverage_is_visible_even_when_values_are_complete():
    item = series(BASELINE + [10.] * 3)
    item["observations"][-1]["sample_size"] = 100
    result = market.assess([item], as_of=stamp(32))["series"][0]
    assert "measurement_coverage_incomplete" in result["warnings"]
    assert result["status"] == "review"


def test_correlated_series_are_not_independent_confirmation():
    a = series(BASELINE + [20.] * 3, id="a")
    b = series([x * 2 for x in BASELINE + [20.] * 3], id="b")
    result = market.assess([a, b], as_of=stamp(32))
    assert result["related_movements"][0]["correlation"] == 1
    assert result["related_movements"][0]["independent_confirmation"] is False


@pytest.mark.parametrize("magnitude", [1e200, 1e-200])
@pytest.mark.parametrize("direction", [1, -1])
def test_correlation_is_stable_across_large_and_small_units(magnitude, direction):
    values = [value * magnitude for value in BASELINE + [20.] * 3]
    a = series(values, id="a")
    b = series([direction * 2 * value for value in values], id="b")
    result = market.assess([a, b], as_of=stamp(32))
    relation = result["related_movements"][0]
    assert relation["correlation"] == direction
    assert relation["aligned_points"] == 33
    assert relation["independent_confirmation"] is False


@pytest.mark.parametrize("clock_style", ["z", "fractional_utc"])
def test_equivalent_timestamp_instants_align_without_changing_source_clocks(clock_style):
    a = series(BASELINE + [20.] * 3, id="a")
    b = series([2 * value for value in BASELINE + [20.] * 3], id="b")
    for point in b["observations"]:
        original = point["observed_at"]
        point["observed_at"] = (original.replace("+00:00", "Z") if clock_style == "z" else
                                datetime.fromisoformat(original).isoformat(timespec="microseconds"))
        point["available_at"] = point["observed_at"]
    result = market.assess([a, b], as_of=stamp(32))
    assert result["related_movements"][0]["aligned_points"] == 33
    assert result["related_movements"][0]["correlation"] == 1
    assert result["series"][1]["observed_at"] == b["observations"][-1]["observed_at"]


def test_partial_and_empty_coverage_never_look_complete():
    a = series(BASELINE + [20.] * 3, id="a")
    b = series([], id="b")
    result = market.assess([a, b], as_of=stamp(32))
    assert result["status"] == "partial"
    assert result["visibility_gaps"] == ["b"]
    assert market.assess([], as_of=stamp(32))["status"] == "insufficient_visibility"


def test_deterministic_receipt_input_immutability_and_policy_identity():
    items = [series(BASELINE + [20.] * 3)]
    snapshot = copy.deepcopy(items)
    a = market.assess(items, as_of=stamp(32))
    b = market.assess(items, as_of=stamp(32))
    assert a == b and items == snapshot
    c = market.assess(items, as_of=stamp(32), policy={"move_threshold": 10})
    assert a["request_sha256"] != c["request_sha256"]
    assert a["policy_sha256"] != c["policy_sha256"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "10"])
def test_invalid_numeric_input_rejected(value):
    with pytest.raises(ValueError):
        analyze(BASELINE + [value] * 3)


def test_duplicate_ids_and_clocks_rejected():
    item = series(BASELINE + [20.] * 3)
    with pytest.raises(ValueError, match="ids"):
        market.assess([item, item], as_of=stamp(32))
    item["observations"][-1]["observed_at"] = stamp(31)
    with pytest.raises(ValueError, match="increasing"):
        market.assess([item], as_of=stamp(32))


def test_unknown_policy_timezone_and_price_are_rejected():
    with pytest.raises(ValueError):
        market.assess([], as_of="2026-01-01T00:00:00")
    with pytest.raises(ValueError):
        market.assess([], as_of=stamp(32), policy={"magic_probability": .99})
    with pytest.raises(ValueError):
        analyze([1.] * 33 + [0.], kind="price")
