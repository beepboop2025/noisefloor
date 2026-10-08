"""Behaviour and edge cases for each check."""
import json
import math
import subprocess
import sys

import pytest

from noisefloor import change, coverage, experiment, forecast, multiple


# ── experiment ──────────────────────────────────────────────────────────────────

def test_interval_narrows_as_data_arrives():
    widths = []
    for n in (100, 1000, 10000):
        lo, hi = experiment.confidence_sequence(n // 10, n)
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]


def test_interval_contains_the_observed_rate():
    for s, n in ((10, 100), (1, 1000), (999, 1000), (0, 30), (30, 30)):
        lo, hi = experiment.confidence_sequence(s, n)
        assert lo <= s / n <= hi, (s, n, lo, hi)


def test_interval_stays_inside_zero_one():
    for s, n in ((0, 10), (10, 10), (1, 5)):
        lo, hi = experiment.confidence_sequence(s, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_no_data_means_total_ignorance_not_a_point_estimate():
    assert experiment.confidence_sequence(0, 0) == (0.0, 1.0)


def test_compare_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        experiment.compare(10, 5, 1, 10)          # more successes than trials
    with pytest.raises(ValueError):
        experiment.compare(1, -1, 1, 10)          # negative n
    with pytest.raises(ValueError):
        experiment.confidence_sequence(1, 10, alpha=0)


def test_compare_is_symmetric_in_its_arms():
    a = experiment.compare(500, 5000, 750, 5000)
    b = experiment.compare(750, 5000, 500, 5000, labels=("B", "A"))
    assert a["winner"] == b["winner"] == "B"


def test_custom_labels_are_used_throughout():
    r = experiment.compare(500, 5000, 750, 5000, labels=("control", "variant"))
    assert "control" in r and "variant" in r
    assert r["winner"] == "variant"


def test_result_always_states_that_peeking_is_safe():
    r = experiment.compare(5, 50, 8, 50)
    assert r["safe_to_peek"] is True and "arXiv" in r["method"]


def test_evidence_is_zero_against_the_observed_rate_direction():
    """The e-value should be smallest near the observed rate, largest far away."""
    near = experiment.log_evalue(100, 1000, 0.10)
    far = experiment.log_evalue(100, 1000, 0.50)
    assert far > near


def test_impossible_rates_are_ruled_out_immediately():
    assert experiment.log_evalue(5, 10, 0.0) == math.inf
    assert experiment.log_evalue(5, 10, 1.0) == math.inf
    assert experiment.log_evalue(0, 10, 0.0) == pytest.approx(-math.log(11))


# ── change ──────────────────────────────────────────────────────────────────────

def test_flat_series_never_changes():
    r = change.scan([42.0] * 60)
    assert r["change_points"] == [] and r["state"] in ("steady", "warming_up")


def test_effect_reports_direction_and_size():
    r = change.scan([50.0 + (i % 3) * 0.1 for i in range(30)] + [95.0])
    assert r["effect"]["direction"] == "up"
    assert r["effect"]["robust_z"] > 3
    assert 0 <= r["effect"]["percentile"] <= 100


def test_change_after_a_detection_is_surfaced_not_buried():
    """After a change the detector re-arms; the reading must still say a change
    happened rather than only reporting that it is warming up."""
    import random
    rng = random.Random(4)
    s = [50 + rng.gauss(0, 2) for _ in range(40)] + [75 + rng.gauss(0, 2) for _ in range(12)]
    r = change.scan(s)
    assert "CHANGED" in r["reading"]
    assert r["readings_since_change"] is not None


def test_two_sided_thresholds_pay_the_union_bound():
    two = change.scan([float(i % 5) for i in range(30)])
    one = change.scan([float(i % 5) for i in range(30)], two_sided=False)
    assert two["changed_at"] == one["changed_at"] * 2


def test_weights_must_match_history():
    with pytest.raises(ValueError):
        change.p_value([1.0, 2.0], 3.0, weights=[1.0])


def test_decay_weights_favour_recent_readings():
    w = change.decay_weights(5, 2.0)
    assert w[-1] == 1.0 and w[0] < w[-1]
    assert change.decay_weights(5, None) is None


def test_pvalue_is_conservative_and_never_zero():
    assert change.p_value([1.0, 2.0, 3.0], 99.0) == pytest.approx(1 / 4)
    assert change.p_value([1.0, 2.0, 3.0], -99.0, side="down") == pytest.approx(1 / 4)


def test_scan_is_deterministic():
    s = [float((i * 7) % 13) for i in range(60)]
    assert change.scan(s) == change.scan(s)


# ── coverage ────────────────────────────────────────────────────────────────────

def test_short_history_refuses_to_judge():
    assert coverage.check([1.0] * 5, [10.0] * 5)["verdict"] == "NOT_ENOUGH_HISTORY"


def test_pure_sampling_artifact_is_caught():
    den = [1000.0 + 40 * ((i * 7) % 11) for i in range(40)]
    met = [0.05 * d for d in den]
    den.append(300.0)
    met.append(0.05 * 300.0)
    r = coverage.check(met, den)
    assert r["verdict"] == "SAMPLING_ARTIFACT"
    assert "not evidence" in r["reading"] or "not the world" in r["reading"]


def test_weak_denominator_association_does_not_prove_a_real_move():
    den = [1000.0 + (i % 4) for i in range(40)] + [1001.0]
    met = [50.0 + (i % 3) * 0.1 for i in range(40)] + [95.0]
    assert coverage.check(met, den)["verdict"] == "UNCLEAR"


def test_flat_metric_is_no_move():
    assert coverage.check([50.0 + (i % 3) for i in range(30)],
                          [1000.0] * 30)["verdict"] == "NO_MOVE"


def test_verdicts_always_carry_a_plain_reading():
    r = coverage.check([50.0 + (i % 3) for i in range(30)], [1000.0] * 30)
    assert r["reading"] and r["verdict"]


# ── forecast ────────────────────────────────────────────────────────────────────

def test_forecast_needs_history():
    assert forecast.next_value([1.0, 2.0])["ok"] is False
    assert forecast.score([1.0, 2.0])["ok"] is False


def test_forecast_interval_brackets_the_point():
    r = forecast.next_value([50.0 + (i % 5) for i in range(40)])
    assert r["interval"][0] <= r["forecast"] <= r["interval"][1]


def test_score_publishes_misses():
    import random
    rng = random.Random(3)
    s = [50 + rng.gauss(0, 2) for _ in range(120)]
    s[100] = 500.0
    r = forecast.score(s)
    assert r["n_misses"] >= 1 and r["worst_misses"]
    assert r["worst_misses"][0]["covered"] is False
    by = [m["missed_by"] for m in r["worst_misses"]]
    assert by == sorted(by, reverse=True)


def test_score_can_report_its_own_method_losing():
    import random
    rng = random.Random(17)
    r = forecast.score([rng.gauss(10, 2) for _ in range(300)])
    assert isinstance(r["beats_baseline"], bool)
    assert r["skill_vs_baseline"] is not None


def test_proper_score_prefers_the_honest_forecast():
    y = 10.0
    honest = forecast.weighted_interval_score(y, 10.0, {0.8: (9.0, 11.0)})
    biased = forecast.weighted_interval_score(y, 16.0, {0.8: (15.0, 17.0)})
    assert honest < biased


def test_interval_score_penalises_missing_a_tighter_interval_more():
    assert (forecast.interval_score(20.0, 9.0, 11.0, 0.05)
            > forecast.interval_score(20.0, 9.0, 11.0, 0.50))


# ── multiple ────────────────────────────────────────────────────────────────────

def test_ordinary_evidence_selects_nothing():
    assert multiple.select({f"m{i}": 1.0 for i in range(20)})["selected"] == []


def test_threshold_follows_the_published_formula():
    ev = {f"m{i}": 1.0 for i in range(9)}
    ev["big"] = 101.0
    r = multiple.select(ev, alpha=0.1, valid_evalues=True)
    assert r["selected"] == ["big"] and r["threshold"] == 100.0


def test_watching_more_metrics_raises_the_bar():
    alone = multiple.select({"x": 500.0}, valid_evalues=True)
    crowded = multiple.select({"x": 500.0, **{f"m{i}": 1.0 for i in range(30)}}, valid_evalues=True)
    assert crowded["threshold"] > alone["threshold"]


def test_merge_is_the_mean_not_a_product():
    assert multiple.merge([1.0, 3.0]) == 2.0
    assert multiple.merge([]) == 1.0


def test_empty_board_is_not_an_error():
    r = multiple.select({})
    assert r["selected"] == [] and r["n_watched"] == 0


def test_alpha_must_be_a_probability():
    with pytest.raises(ValueError):
        multiple.select({"a": 1.0}, alpha=1.5)


# ── MCP server ──────────────────────────────────────────────────────────────────

def _rpc(*messages):
    payload = "\n".join(json.dumps(m) for m in messages)
    out = subprocess.run([sys.executable, "-m", "noisefloor.mcp_server"],
                         input=payload, capture_output=True, text=True, timeout=120)
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def test_server_initializes_and_lists_every_tool():
    replies = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                   {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert replies[0]["result"]["serverInfo"]["name"] == "noisefloor"
    names = {t["name"] for t in replies[1]["result"]["tools"]}
    assert names == {"ab_test", "did_it_change", "real_or_sampling",
                     "forecast_next", "score_forecasts", "which_metrics_matter",
                     "market_assessment", "narrative_triage"}


def test_server_runs_a_tool_and_returns_json():
    replies = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "ab_test",
                               "arguments": {"a_successes": 500, "a_total": 5000,
                                             "b_successes": 750, "b_total": 5000}}})
    payload = json.loads(replies[0]["result"]["content"][0]["text"])
    assert payload["winner"] == "B" and payload["decided"] is True


def test_server_fails_loud_on_bad_input():
    replies = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "did_it_change",
                               "arguments": {"values": "not a list"}}})
    assert replies[0]["result"]["isError"] is True


def test_server_rejects_unknown_tools_and_methods():
    replies = _rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "nope", "arguments": {}}},
                   {"jsonrpc": "2.0", "id": 2, "method": "does/not/exist"})
    assert replies[0]["error"]["code"] == -32602
    assert replies[1]["error"]["code"] == -32601


def test_server_survives_malformed_json():
    out = subprocess.run([sys.executable, "-m", "noisefloor.mcp_server"],
                         input='{bad json\n{"jsonrpc":"2.0","id":9,"method":"ping"}',
                         capture_output=True, text=True, timeout=120)
    replies = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    assert replies[0]["error"]["code"] == -32700
    assert replies[1]["result"] == {}


def test_package_has_no_third_party_imports():
    """Zero dependencies is the selling point; guard it."""
    import pathlib
    import re
    banned = re.compile(r"^\s*(?:import|from)\s+(numpy|scipy|pandas|sklearn)\b", re.M)
    for path in pathlib.Path("noisefloor").glob("*.py"):
        assert not banned.search(path.read_text(encoding="utf-8")), path


# ── registry manifest ───────────────────────────────────────────────────────────
# The v0.1.0 release published fine to PyPI but was rejected by the MCP Registry
# for a 166-character description against a 100-character limit. The limits are
# only discoverable from the schema, so they are asserted here rather than
# rediscovered one failed release at a time.

def test_server_json_respects_registry_limits():
    import pathlib
    sj = json.loads(pathlib.Path("server.json").read_text())
    assert 1 <= len(sj["description"]) <= 100, len(sj["description"])
    assert 1 <= len(sj["title"]) <= 100, len(sj["title"])
    assert 3 <= len(sj["name"]) <= 200
    import re
    assert re.match(r"^[a-zA-Z0-9.-]+/[a-zA-Z0-9._-]+$", sj["name"]), sj["name"]


def test_registry_manifest_advertises_the_live_http_remote():
    import pathlib
    sj = json.loads(pathlib.Path("server.json").read_text())
    assert sj["remotes"] == [{
        "type": "streamable-http",
        "url": "https://api.seiche.info/noisefloor/mcp",
    }]


def test_versions_agree_across_the_package():
    import pathlib
    import re
    sj = json.loads(pathlib.Path("server.json").read_text())
    pyproject = pathlib.Path("pyproject.toml").read_text()
    init = pathlib.Path("noisefloor/__init__.py").read_text()
    from noisefloor.mcp_server import SERVER_VERSION
    versions = {
        "server.json": sj["version"],
        "packages[]": {p["version"] for p in sj.get("packages", [])}.pop(),
        "pyproject": re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1),
        "__init__": re.search(r'__version__ = "([^"]+)"', init).group(1),
        "mcp_server": SERVER_VERSION,
    }
    assert len(set(versions.values())) == 1, versions


def test_readme_carries_the_registry_verification_marker():
    """The MCP Registry validates a PyPI package by finding an EXACT marker in
    the rendered README: 'mcp-name: <server name>'. The bare name is not enough —
    v0.1.1 had the name and was still rejected. Removing or reformatting this
    line breaks publishing with an error that names no file."""
    import pathlib
    sj = json.loads(pathlib.Path("server.json").read_text())
    marker = f"mcp-name: {sj['name']}"
    assert marker in pathlib.Path("README.md").read_text(), marker
