"""Validate real branch results with a standards validator, without coercion."""
import copy
import json
import math
import random
import subprocess
import sys
import urllib.request

import pytest
from jsonschema import Draft202012Validator, ValidationError

from noisefloor.mcp_server import REST_TOOLS, TOOLS, call_tool, capabilities, handle, openapi
from noisefloor.output_schemas import ERROR_SCHEMA, SUCCESS_SCHEMAS, output_schema, success_schema
from test_http import base_url, _post
from test_market import BASELINE, series, stamp
from test_narrative import event, NOW
from test_spectral import panel


def cases():
    """Reuse semantic fixtures, retaining unavailable branches as valid results."""
    rows = []

    def add(label, name, arguments):
        rows.append((label, name, copy.deepcopy(arguments)))

    for label, a, an, b, bn in [("empty-arm", 0, 0, 1, 2), ("undecided", 5, 10, 5, 10),
                               ("a-wins", 1000, 1000, 0, 1000), ("b-wins", 0, 1000, 1000, 1000)]:
        add(label, "ab_test", dict(a_successes=a, a_total=an, b_successes=b, b_total=bn))
    add("dynamic-error-label", "ab_test", dict(a_successes=1, a_total=10, b_successes=2, b_total=10,
                                               a_label="error", b_label="arbitrary arm / 2"))
    for label, values, extra in [("warmup", [1], {}), ("steady", [1, 2, 3] * 10, {}),
                                 ("shift", [0] * 30 + [100] * 5, {}),
                                 ("watch", list(range(100)) + [10000, 10001], {}),
                                 ("changed", list(range(100)) + list(range(10000, 10005)), {}),
                                 ("weighted-one-sided", [1, 2, 3] * 10 + [20], {"half_life": 4, "two_sided": False})]:
        add(label, "did_it_change", {"values": values, **extra})
    denominator = list(range(100, 130))
    for label, metric, counts in [
        ("short", [1], [10]), ("no-move", [1] * 10, [10] * 10),
        ("zero-spread", [1] * 20 + [100], [10] * 21),
        ("constant-denominator", list(range(20)) + [1000], [10] * 21),
        ("explained", [2 * x for x in denominator] + [1000], denominator + [500]),
        ("unexplained", [2 * x + (i % 3) for i, x in enumerate(denominator)] + [1000], denominator + [115]),
        ("weak-fit", [10 + i % 3 for i in range(30)] + [1000], denominator + [115]),
    ]:
        add(label, "real_or_sampling", {"values": metric, "sample_sizes": counts})
    for name in ["forecast_next", "score_forecasts"]:
        for label, values in [("short", [1]), ("constant", [1] * 40),
                              ("variable", [math.sin(i) for i in range(60)]),
                              ("misses", [0] * 30 + [1000 * i for i in range(1, 31)])]:
            add(label, name, {"values": values})
    for label, evidence, valid in [("unverified", {"x": 1000}, False),
                                    ("selected", {"x": 1000, "y": 1}, True),
                                    ("none-selected", {"x": 1}, True)]:
        add(label, "which_metrics_matter", {"evidence": evidence, "valid_evalues": valid})

    add("empty", "market_assessment", {"as_of": stamp(40), "series": []})
    for label, values in [("steady", BASELINE + [10, 11, 9]), ("review", BASELINE + [30, 31, 30]),
                           ("watch", BASELINE + [10, 10, 100]), ("zero-spread", [1] * 30 + [100] * 3),
                           ("short", [1, 2]), ("missing", BASELINE + [None]), ("no-observations", [])]:
        add(label, "market_assessment", {"as_of": stamp(len(values) - 1), "series": [series(values)]})
    price = series([100 * math.exp(.01 * i) for i in range(34)], kind="price", unit="USD")
    add("price", "market_assessment", {"as_of": stamp(33), "series": [price]})
    for label in ["restricted", "unknown", "stale", "future", "unavailable", "early-availability", "coverage", "incomplete-coverage", "calendar-gap"]:
        item = series(BASELINE + [30, 31, 30])
        cutoff = stamp(32)
        if label in ["restricted", "unknown"]:
            item["source"]["rights"] = label
        elif label == "stale":
            cutoff = stamp(80)
        elif label == "future":
            cutoff = stamp(20)
        elif label == "unavailable":
            item["observations"][-1]["available_at"] = stamp(100)
        elif label == "early-availability":
            item["observations"][-1]["available_at"] = stamp(0)
        elif label == "calendar-gap":
            item["max_gap_seconds"] = 3600
        else:
            for i, point in enumerate(item["observations"]):
                point["sample_size"] = 100 if i < 30 else 10
            if label == "incomplete-coverage":
                item["observations"][-1]["sample_size"] = None
        add(label, "market_assessment", {"as_of": cutoff, "series": [item]})
    first = series(BASELINE + [30, 31, 30])
    second = copy.deepcopy(first)
    second["id"] = "second"
    add("related", "market_assessment", {"as_of": stamp(32), "series": [first, second]})
    second["source"]["rights"] = "restricted"
    add("partial", "market_assessment", {"as_of": stamp(32), "series": [first, second]})

    add("empty", "narrative_triage", {"as_of": NOW, "events": []})
    add("repeat-conflict-deferred", "narrative_triage", {"as_of": NOW, "policy": {"max_items": 1}, "events": [
        event(event_key="release"), event("b", event_key="release"),
        event("c", title="Bank A reports quarterly deposit growth of 50%", event_key="release"),
    ]})
    add("excluded-unmapped", "narrative_triage", {"as_of": NOW, "focus": ["bank-a"], "events": [
        event("unmapped", entities=[]), event("off-focus", entities=["bank-b"]),
        event("future", published_at="2027-01-01T00:00:00Z", available_at="2027-01-01T00:00:00Z"),
        event("stale", published_at="2020-01-01T00:00:00Z", available_at="2020-01-01T00:00:00Z"),
    ]})
    no_available = event()
    no_available.pop("available_at")
    add("availability-omitted", "narrative_triage", {"as_of": NOW, "events": [no_available]})

    add("empty", "spectral_assessment", {"as_of": NOW, "series": []})
    for label in ["assessed", "restricted", "one-member", "too-short", "constant", "unmatched", "coverage", "degenerate-dynamics"]:
        request = panel(points=40)
        request["policy"] = {"window_points": 20, "step_points": 20, "max_windows": 2}
        if label == "restricted":
            request["series"][0]["source"]["rights"] = "restricted"
        elif label == "one-member":
            request["series"] = request["series"][:1]
        elif label == "too-short":
            for item in request["series"]:
                item["observations"] = item["observations"][-10:]
        elif label == "constant":
            for point in request["series"][0]["observations"]:
                point["value"] = 1
        elif label == "unmatched":
            request["series"][0]["observations"].pop(-2)
        elif label == "coverage":
            for i, point in enumerate(request["series"][0]["observations"]):
                point["sample_size"] = 10 + i
        elif label == "degenerate-dynamics":
            request = panel([[math.sin(2 * math.pi * k * i / 60) for i in range(120)] for k in (1, 3, 7, 11)])
            request["policy"] = {"window_points": 60, "step_points": 60, "max_windows": 2}
        add(label, "spectral_assessment", request)
    rng = random.Random(8)
    factor = [rng.gauss(0, 1) for _ in range(60)]
    request = panel([[x + rng.gauss(0, .1) for x in factor] for _ in range(3)])
    request["policy"] = {"window_points": 30, "step_points": 30, "max_windows": 2}
    add("resolved-dynamics", "spectral_assessment", request)
    request = panel([[0] * 20 + [math.sin(i) for i in range(20)],
                     [0] * 20 + [math.cos(i) for i in range(20)]])
    request["policy"] = {"window_points": 20, "step_points": 20, "max_windows": 2}
    add("partial-windows", "spectral_assessment", request)
    add("default", "dyson_reference", {})
    add("small", "dyson_reference", {"dimension": 2, "steps": 1, "seed": 42})
    return rows


CASES = cases()


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_schema_is_valid_and_covers_existing_tool_error(name):
    schema = output_schema(name)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator.check_schema(success_schema(name))
    assert schema["type"] == "object" and schema["properties"]
    result = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": name, "arguments": {"unexpected": True}}})["result"]
    assert result["isError"] is True
    Draft202012Validator(schema).validate(result["structuredContent"])
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


@pytest.mark.parametrize("label,name,arguments", CASES, ids=[name + ":" + label for label, name, _ in CASES])
def test_actual_result_matches_typed_contract(label, name, arguments):
    before = copy.deepcopy(arguments)
    expected = call_tool(name, arguments)
    result = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": name, "arguments": arguments}})["result"]
    assert not result.get("isError")
    assert arguments == before
    assert result["structuredContent"] == expected == json.loads(result["content"][0]["text"])
    Draft202012Validator(success_schema(name)).validate(expected)
    Draft202012Validator(output_schema(name)).validate(expected)
    if name == "ab_test" and label == "dynamic-error-label":
        assert isinstance(expected["error"], dict)
        assert "A" not in expected and "B" not in expected
    if name == "did_it_change" and label in ("watch", "changed"):
        assert expected["state"] == label
    if name == "spectral_assessment" and label == "degenerate-dynamics":
        assert expected["dynamics"][0]["leading_vector_squared_overlap"] is None
    if name == "spectral_assessment" and label == "partial-windows":
        assert expected["status"] == "partial"


def test_named_contracts_reject_wrong_types_missing_success_fields_and_invented_guarantees():
    for name, arguments, key, invalid in [
        ("did_it_change", {"values": [1]}, "is_e_value", True),
        ("forecast_next", {"values": list(range(20))}, "forecast", "19"),
        ("market_assessment", {"series": [], "as_of": NOW}, "execution_authority", True),
    ]:
        result = call_tool(name, arguments)
        result[key] = invalid
        with pytest.raises(ValidationError):
            Draft202012Validator(output_schema(name)).validate(result)
    for name in ["forecast_next", "score_forecasts"]:
        result = call_tool(name, {"values": [1]})
        result["ok"] = True
        with pytest.raises(ValidationError):
            Draft202012Validator(output_schema(name)).validate(result)
    with pytest.raises(ValidationError):
        Draft202012Validator(output_schema("did_it_change")).validate({"error": {"message": "wrong shape"}})


def test_empty_evidence_is_a_tool_error_not_an_empty_success_board():
    for valid in [False, True]:
        result = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "which_metrics_matter", "arguments": {"evidence": {}, "valid_evalues": valid}}})["result"]
        assert result["isError"] is True
        Draft202012Validator(output_schema("which_metrics_matter")).validate(result["structuredContent"])


def test_discovery_contracts_agree_and_are_independently_owned():
    listing = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    advertised = {tool["name"]: tool["outputSchema"] for tool in listing}
    assert set(advertised) == set(SUCCESS_SCHEMAS) == set(TOOLS)
    for tool in capabilities()["tools"]:
        assert tool["output_schema"] == advertised[tool["name"]]
    doc = openapi()
    for path, name in REST_TOOLS.items():
        responses = doc["paths"][path]["post"]["responses"]
        assert responses["200"]["content"]["application/json"]["schema"] == success_schema(name)
        for status in ["400", "413", "422"]:
            assert responses[status]["content"]["application/json"]["schema"] == ERROR_SCHEMA
    listing[0]["outputSchema"]["properties"].clear()
    doc["paths"][next(iter(REST_TOOLS))]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["properties"].clear()
    assert output_schema(listing[0]["name"])["properties"]
    assert openapi()["paths"][next(iter(REST_TOOLS))]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["properties"]


def test_stdio_discovery_and_all_tool_results_validate():
    selected = {name: arguments for _, name, arguments in CASES}
    messages = [{"jsonrpc": "2.0", "id": 0, "method": "tools/list"}]
    messages += [{"jsonrpc": "2.0", "id": i + 1, "method": "tools/call", "params": {"name": name, "arguments": args}}
                 for i, (name, args) in enumerate(selected.items())]
    run = subprocess.run([sys.executable, "-m", "noisefloor.mcp_server"],
                         input="\n".join(map(json.dumps, messages)) + "\n", text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stderr
    replies = [json.loads(line) for line in run.stdout.splitlines()]
    schemas = {tool["name"]: tool["outputSchema"] for tool in replies[0]["result"]["tools"]}
    for (name, args), reply in zip(selected.items(), replies[1:]):
        payload = reply["result"]["structuredContent"]
        Draft202012Validator(schemas[name]).validate(payload)
        assert payload == call_tool(name, args)


def test_http_discovery_rest_and_mcp_use_same_typed_contract(base_url):
    status, listing = _post(base_url, {"jsonrpc": "2.0", "id": 0, "method": "tools/list"})
    assert status == 200
    advertised = {t["name"]: t["outputSchema"] for t in listing["result"]["tools"]}
    with urllib.request.urlopen(base_url + "/openapi.json") as response:
        doc = json.load(response)
    for path, name in REST_TOOLS.items():
        args = next(args for _, tool, args in CASES if tool == name)
        status, result = _post(base_url, args, path)
        assert status == 200
        Draft202012Validator(doc["paths"][path]["post"]["responses"]["200"]["content"]["application/json"]["schema"]).validate(result)
        status, reply = _post(base_url, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})
        assert status == 200 and reply["result"]["structuredContent"] == result
        Draft202012Validator(advertised[name]).validate(result)
        status, result = _post(base_url, {"unexpected": True}, path)
        assert status == 422
        Draft202012Validator(doc["paths"][path]["post"]["responses"]["422"]["content"]["application/json"]["schema"]).validate(result)
