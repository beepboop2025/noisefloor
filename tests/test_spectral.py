"""Known spectra, adversarial observation panels and transport parity."""
import copy
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from noisefloor import dyson, spectral
from noisefloor.adapters import from_financial_evidence
from noisefloor._linalg import eigh
from noisefloor.mcp_server import call_tool, handle, openapi


def panel(columns=None, *, kind="return", points=120):
    columns = columns or [[math.sin(2 * math.pi * frequency * i / points) for i in range(points)]
                          for frequency in (1, 3, 7, 11)]
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clocks = [(start + timedelta(days=i)).isoformat() for i in range(len(columns[0]) + 1)]
    return {"as_of": clocks[-1], "policy": {"window_points": len(columns[0]), "max_windows": 1},
            "series": [{"id": f"series-{k}", "kind": kind, "unit": "fraction",
                        "source": {"id": "synthetic", "rights": "permitted"},
                        "max_age_seconds": 86400, "max_gap_seconds": 86400,
                        "observations": [{"observed_at": stamp, "available_at": stamp, "value": value}
                                         for stamp, value in zip(clocks, [0.0] + values)]}
                       for k, values in enumerate(columns)]}


def test_orthogonal_panel_has_identity_spectrum_and_full_effective_rank():
    report = spectral.assess(**panel())
    latest = report["latest"]
    assert report["status"] == "assessed"
    assert latest["eigenvalues"] == pytest.approx([1] * 4, abs=1e-12)
    assert latest["leading_variance_share"] == pytest.approx(0.25)
    assert latest["effective_rank"] == pytest.approx(4)
    assert latest["participation_ratio"] == pytest.approx(4)
    assert latest["reference"]["above_upper_count"] == 0
    assert latest["reference"]["finite_sample_calibrated"] is False
    assert latest["reference"]["q"] == 4 / 119
    assert report["execution_authority"] is False


def test_duplicate_and_hedged_series_expose_one_shared_dimension():
    values = [math.sin(i) for i in range(100)]
    report = spectral.assess(**panel([values, values, [-x for x in values], values]))
    latest = report["latest"]
    assert latest["eigenvalues"] == pytest.approx([4, 0, 0, 0], abs=1e-10)
    assert latest["reference"]["above_upper_count"] == 1
    assert latest["reference"]["below_lower_count"] == 3
    assert latest["participation_ratio"] == pytest.approx(1)
    assert latest["effective_rank"] == pytest.approx(1)
    assert latest["leading_loadings"]["series-0"] * latest["leading_loadings"]["series-2"] < 0
    # Redundant near-zero modes survive: never manufacture an invertible risk matrix.
    cleaned, _ = eigh(latest["shrinkage_candidate"]["correlation"])
    assert cleaned[-1] == pytest.approx(0, abs=1e-10)


def test_cleaning_retains_psd_and_unit_diagonal_for_noisy_factor_panel():
    rng = random.Random(9)
    common = [rng.gauss(0, 1) for _ in range(200)]
    columns = [[0.4*x + rng.gauss(0, 1) for x in common] for _ in range(12)]
    latest = spectral.assess(**panel(columns))["latest"]
    cleaned = latest["shrinkage_candidate"]["correlation"]
    values, _ = eigh(cleaned)
    assert min(values) >= -1e-10
    assert sum(values) == pytest.approx(12)
    assert all(cleaned[i][i] == pytest.approx(1) for i in range(12))
    assert all(cleaned[i][j] == pytest.approx(cleaned[j][i]) for i in range(12) for j in range(12))


def test_scale_and_order_do_not_change_correlations():
    original = panel()
    changed = copy.deepcopy(original)
    for series, factor in zip(changed["series"], [1e-150, 1e150, 1e-25, 1e25]):
        for point in series["observations"]:
            point["value"] *= factor
    changed["series"].reverse()
    first, second = spectral.assess(**original), spectral.assess(**changed)
    assert second["latest"]["eigenvalues"] == pytest.approx(first["latest"]["eigenvalues"])
    assert second["latest"]["series_ids"] == first["latest"]["series_ids"]
    assert original == panel()  # input was never modified


@pytest.mark.parametrize("problem", ["stale", "restricted", "future", "unavailable", "latest_missing"])
def test_ineligible_member_blocks_entire_requested_panel(problem):
    request = panel()
    series = request["series"][0]
    last = series["observations"][-1]
    if problem == "stale":
        request["as_of"] = "2026-12-31T00:00:00Z"
    elif problem == "restricted":
        series["source"]["rights"] = "restricted"
    elif problem == "future":
        last["observed_at"] = last["available_at"] = "2026-12-31T00:00:00Z"
    elif problem == "unavailable":
        last["available_at"] = "2026-12-31T00:00:00Z"
    else:
        last["value"] = None
    report = spectral.assess(**request)
    assert report["status"] == "insufficient_visibility"
    assert report["windows"] == []
    assert report["latest"] is None
    assert report["coverage"]["requested_series"] == 4


def test_explicit_missingness_and_unequal_intervals_never_become_pairwise_deletion():
    for missing_value in (True, False):
        request = panel()
        if missing_value:
            request["series"][0]["observations"][-12]["value"] = None
        else:
            request["series"][0]["observations"].pop(-12)
            request["series"][0]["max_gap_seconds"] = 3 * 86400
        request["policy"]["window_points"] = 20
        report = spectral.assess(**request)
        assert report["status"] == "insufficient_visibility"
        assert report["alignment"]["aligned_points"] < 20


def test_latest_unaligned_interval_does_not_silently_use_old_data():
    request = panel()
    request["series"][0]["observations"].pop()
    request["policy"]["window_points"] = 20
    report = spectral.assess(**request)
    assert report["status"] == "insufficient_visibility"
    assert report["alignment"]["aligned_points"] == 0


def test_equivalent_clock_spellings_align():
    request = panel()
    for point in request["series"][0]["observations"]:
        point["observed_at"] = point["observed_at"].replace("+00:00", ".000000Z")
    assert spectral.assess(**request)["status"] == "assessed"


def test_constant_and_geometric_price_series_do_not_amplify_roundoff():
    request = panel([[0.25] * 60, [math.sin(i) for i in range(60)]])
    assert spectral.assess(**request)["latest"]["status"] == "insufficient_visibility"
    request = panel(points=60)
    for item in request["series"]:
        item["kind"] = "price"
        for i, point in enumerate(item["observations"]):
            point["value"] = 100 * math.exp(0.001 * i)
    result = spectral.assess(**request)
    assert result["status"] == "insufficient_visibility"


def test_rate_levels_are_differenced_and_returns_keep_their_values():
    request = panel()
    expected = spectral.assess(**request)["latest"]["eigenvalues"]
    for item in request["series"]:
        item["kind"] = "rate"
        total = 100.0
        for point in item["observations"]:
            total += point["value"]
            point["value"] = total
    report = spectral.assess(**request)
    assert all(item["transform"] == "first_difference" for item in report["series"])
    assert report["latest"]["eigenvalues"] == pytest.approx(expected, abs=1e-10)


def test_changing_measurement_coverage_is_disclosed_without_claiming_adjustment():
    request = panel()
    for i, point in enumerate(request["series"][0]["observations"]):
        point["sample_size"] = 100 if i < 60 else 10
    report = spectral.assess(**request)
    first = report["series"][0]
    assert first["measurement_coverage"]["status"] == "variable"
    assert first["measurement_coverage"]["sampling_adjustment_applied"] is False
    assert "review_measurement_coverage" in first["warnings"]


def test_rolling_concentration_rises_and_past_windows_ignore_later_values():
    rng = random.Random(21)
    common = [rng.gauss(0, 1) for _ in range(60)]
    columns = [[rng.gauss(0, 1) for _ in range(60)] + common for _ in range(6)]
    request = panel(columns)
    request["policy"] = {"window_points": 60, "step_points": 60, "max_windows": 2}
    report = spectral.assess(**request)
    assert report["dynamics"][0]["leading_variance_share_change"] > 0.5
    assert report["dynamics"][0]["calibrated_regime_change"] is False
    prefix = copy.deepcopy(request)
    for item in prefix["series"]:
        item["observations"] = item["observations"][:61]
    prefix["as_of"] = prefix["series"][0]["observations"][-1]["observed_at"]
    assert spectral.assess(**prefix)["latest"] == report["windows"][0]


def test_degenerate_modes_do_not_claim_vector_rotation():
    columns = [[math.sin(2*math.pi*k*i/60) for i in range(120)] for k in (1, 3, 7, 11)]
    request = panel(columns)
    request["policy"] = {"window_points": 60, "step_points": 60, "max_windows": 2}
    change = spectral.assess(**request)["dynamics"][0]
    assert change["leading_vector_squared_overlap"] is None
    assert change["leading_vector_comparison"] == "unresolved_degenerate_mode"


def test_too_many_series_for_the_reference_and_duplicate_ids():
    request = panel(points=20)
    request["series"] = [copy.deepcopy(request["series"][i % 4]) for i in range(22)]
    for i, item in enumerate(request["series"]):
        item["id"] = str(i)
    assert spectral.assess(**request)["status"] == "insufficient_visibility"
    request["series"][0]["id"] = request["series"][1]["id"]
    with pytest.raises(ValueError, match="unique"):
        spectral.assess(**request)


@pytest.mark.parametrize("dimension", [2, 8, 32])
def test_eigensolver_reconstructs_symmetric_matrices(dimension):
    rng = random.Random(113)
    matrix = [[0.0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i+1):
            matrix[i][j] = matrix[j][i] = rng.gauss(0, 1)
    values, vectors = eigh(matrix)
    for i in range(dimension):
        for j in range(dimension):
            rebuilt = math.fsum(vectors[i][k] * values[k] * vectors[j][k] for k in range(dimension))
            assert rebuilt == pytest.approx(matrix[i][j], abs=2e-12)
            inner = math.fsum(vectors[k][i] * vectors[k][j] for k in range(dimension))
            assert inner == pytest.approx(float(i == j), abs=2e-12)
    assert sum(values) == pytest.approx(sum(matrix[i][i] for i in range(dimension)))


def test_dyson_is_seeded_synthetic_and_does_not_mutate_global_rng():
    state = random.getstate()
    first = dyson.simulate(dimension=5, steps=12, seed=11)
    assert random.getstate() == state
    assert first == dyson.simulate(dimension=5, steps=12, seed=11)
    assert first != dyson.simulate(dimension=5, steps=12, seed=12)
    assert first["status"] == "synthetic"
    assert first["market_calibrated"] is False
    assert first["is_covariance_model"] is False
    assert first["execution_authority"] is False
    assert first["path"][0]["eigenvalues"] == [0] * 5
    for point in first["path"][1:]:
        assert all(a > b for a, b in zip(point["eigenvalues"], point["eigenvalues"][1:]))
    assert any(value < 0 for value in first["path"][-1]["eigenvalues"])


def test_dyson_brownian_scaling_and_prefix_replay():
    first = dyson.simulate(dimension=4, steps=8, dt=0.01, seed=3)
    scaled = dyson.simulate(dimension=4, steps=8, dt=0.04, seed=3)
    assert scaled["path"][-1]["eigenvalues"] == pytest.approx([2*x for x in first["path"][-1]["eigenvalues"]])
    assert dyson.simulate(dimension=4, steps=4, dt=0.01, seed=3)["path"] == first["path"][:5]


@pytest.mark.parametrize("arguments", [{"dimension": True}, {"dimension": 17}, {"steps": 101},
                                       {"seed": -1}, {"dt": float("nan")}, {"dt": 0}, {"buy": True}])
def test_dyson_rejects_invalid_or_excessive_requests(arguments):
    with pytest.raises(ValueError):
        call_tool("dyson_reference", arguments)


@pytest.mark.parametrize("command,name,path,payload", [
    ("spectral", "spectral_assessment", "/v1/spectral/assess", panel()),
    ("dyson", "dyson_reference", "/v1/research/dyson", {"seed": 7, "steps": 3}),
])
def test_new_tools_match_python_cli_mcp_and_discovery(command, name, path, payload):
    expected = spectral.assess(**payload) if command == "spectral" else dyson.simulate(**payload)
    assert call_tool(name, payload) == expected
    reply = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": name, "arguments": payload}})
    assert reply["result"]["structuredContent"] == expected
    result = subprocess.run([sys.executable, "-m", "noisefloor.cli", command], input=json.dumps(payload),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    assert openapi()["paths"][path]["post"]["operationId"] == name


@pytest.mark.parametrize("product,dataset", [("LiquiLens", "institution_history"),
                                           ("Seiche", "money_market_history"),
                                           ("Undertow", "market_liquidity")])
def test_financial_evidence_panels_retain_native_identity_and_review_holds(product, dataset):
    request = panel(points=60)
    rows = [{"product": product, "dataset": dataset, "entity_id": item["id"],
             "entity_name": item["id"], "metric": "synthetic-return", "unit": "fraction",
             "source_url": "https://example.invalid/source", "value": point["value"],
             "as_of": point["observed_at"], "knowledge_time": point["available_at"],
             "rights_status": "allowed", "source_status": "FRESH", "availability": "AVAILABLE",
             "transport_status": "complete"}
            for item in request["series"] for point in item["observations"]]
    table = {"schema": "financial-evidence.agent-result.v1", "results": rows,
             "returned_rows": len(rows), "total_rows": len(rows), "offset": 0,
             "next_offset": None, "transport_status": "complete"}
    adapted = from_financial_evidence(table, kind="return", max_age_seconds=86400, max_gap_seconds=86400)
    report = spectral.assess(adapted["series"], as_of=request["as_of"], policy=request["policy"])
    assert report["status"] == "assessed"
    assert len(report["series"]) == 4
    assert all(identity["product"] == product for identity in adapted["provenance"]["identities"])
    assert report["latest"]["eigenvalues"] == pytest.approx([1] * 4)
    table["results"][-1]["source_status"] = "SOURCE_REVIEW_HOLD"
    blocked = from_financial_evidence(table, kind="return", max_age_seconds=86400, max_gap_seconds=86400)
    report = spectral.assess(blocked["series"], as_of=request["as_of"], policy=request["policy"])
    assert report["status"] == "insufficient_visibility"
    assert report["windows"] == []


@pytest.mark.parametrize("product", ["liquilens", "seiche", "undertow", "riptide", "trading_agents", "palimpsest"])
def test_product_examples_remain_executable_synthetic_research(product):
    from examples.spectral_products import request_for
    request = request_for(product)
    assert all(item["source"]["id"].startswith("synthetic:") for item in request["series"])
    result = spectral.assess(**request)
    assert result["status"] == "assessed"
    assert result["maturity"] == "research"
    assert result["dynamics"][0]["leading_variance_share_change"] > 0.4
