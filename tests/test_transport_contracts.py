"""The same offline assessment contract across Python, CLI, REST and MCP."""
import copy
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from noisefloor.mcp_server import TOOLS, call_tool, capabilities, handle, openapi
from noisefloor.schemas import MARKET_REQUEST, MAX_BATCH, MAX_SERIES_POINTS, loads, validate


def market_input():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    return {"as_of": "2026-10-08T00:00:00Z", "policy": {"baseline_points": 10, "recent_points": 2},
            "series": [{"id": "test-spread", "kind": "spread", "unit": "bp",
                        "source": {"id": "test-source", "rights": "permitted"},
                        "max_age_seconds": 60 * 86400,
                        "observations": [{"observed_at": (start + timedelta(days=i)).isoformat(),
                                          "value": 10 + (i % 3)} for i in range(24)]}]}


def narrative_input():
    return {"as_of": "2026-10-08T00:00:00Z", "focus": ["BANK-A"], "events": [
        {"id": "filing-1", "title": "BANK-A files quarterly earnings",
         "published_at": "2026-10-07T15:00:00Z", "entities": ["BANK-A"],
         "source": {"id": "bank-a", "kind": "primary", "url": "https://example.com/filing"}},
        {"id": "headline-1", "title": "BANK-A files quarterly earnings",
         "published_at": "2026-10-07T16:00:00Z", "entities": ["BANK-A"],
         "source": {"id": "wire-a", "kind": "reporting", "family": "wire-a"}},
    ]}


@pytest.mark.parametrize("name,payload", [("market_assessment", market_input), ("narrative_triage", narrative_input)])
def test_python_and_mcp_share_exact_results(name, payload):
    data = payload()
    expected = call_tool(name, data)
    reply = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": name, "arguments": data}})
    assert reply["result"]["structuredContent"] == expected
    assert json.loads(reply["result"]["content"][0]["text"]) == expected


@pytest.mark.parametrize("command,name,payload", [("market", "market_assessment", market_input),
                                                   ("narrative", "narrative_triage", narrative_input)])
def test_cli_stdin_and_file_match_python(command, name, payload, tmp_path):
    raw = json.dumps(payload())
    expected = call_tool(name, payload())
    stdin_run = subprocess.run([sys.executable, "-m", "noisefloor.cli", command],
                               input=raw, text=True, capture_output=True, timeout=15)
    assert stdin_run.returncode == 0, stdin_run.stdout + stdin_run.stderr
    assert json.loads(stdin_run.stdout) == expected
    path = tmp_path / "input.json"
    path.write_text(raw)
    file_run = subprocess.run([sys.executable, "-m", "noisefloor.cli", command, "--input", str(path)],
                              text=True, capture_output=True, timeout=15)
    assert file_run.returncode == 0, file_run.stdout + file_run.stderr
    assert json.loads(file_run.stdout) == expected


def test_capabilities_are_stable_and_do_not_share_mutable_schema():
    first = capabilities()
    assert len(first["tools"]) == 8
    first["tools"][0]["input_schema"]["required"].append("injected")
    assert "injected" not in capabilities()["tools"][0]["input_schema"]["required"]
    assert capabilities() == capabilities()
    assert capabilities()["posture"]["fetches_sources"] is False
    assert capabilities()["posture"]["executes_trades"] is False
    result = subprocess.run([sys.executable, "-m", "noisefloor.cli", "capabilities"],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0
    assert json.loads(result.stdout) == capabilities()


def test_openapi_uses_the_same_owned_schemas():
    doc = openapi()
    assert doc["openapi"] == "3.1.0"
    for path, name in [("/v1/market/assess", "market_assessment"), ("/v1/narrative/triage", "narrative_triage")]:
        assert doc["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"] == TOOLS[name][1]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf"), True])
def test_numeric_validation_rejects_nonfinite_and_boolean(invalid):
    with pytest.raises(ValueError):
        call_tool("did_it_change", {"values": [1, invalid]})
    payload = market_input()
    payload["series"][0]["observations"][0]["value"] = invalid
    with pytest.raises(ValueError):
        validate(payload, MARKET_REQUEST)


@pytest.mark.parametrize("field,value", [("a_total", True), ("a_total", 10.5), ("a_total", "10"), ("a_total", -1)])
def test_count_validation_never_coerces(field, value):
    request = {"a_successes": 1, "a_total": 10, "b_successes": 2, "b_total": 10}
    request[field] = value
    with pytest.raises(ValueError):
        call_tool("ab_test", request)


def test_rejects_extra_fields_and_oversized_histories():
    with pytest.raises(ValueError):
        call_tool("did_it_change", {"values": [1], "trade": True})
    with pytest.raises(ValueError):
        call_tool("did_it_change", {"values": [1] * (MAX_SERIES_POINTS + 1)})
    request = market_input()
    request["series"][0]["source"]["secret"] = "not-allowed"
    with pytest.raises(ValueError):
        validate(request, MARKET_REQUEST)


@pytest.mark.parametrize("clock", ["2026-10-08", "2026-10-08T01:00:00", "2026-10-08T01:00:00+01:00", "bad"])
def test_clock_requires_explicit_utc(clock):
    request = market_input()
    request["as_of"] = clock
    with pytest.raises(ValueError):
        validate(request, MARKET_REQUEST)


@pytest.mark.parametrize("raw", ['{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}'])
def test_json_rejects_nonstandard_numbers_and_duplicate_keys(raw):
    with pytest.raises(ValueError):
        loads(raw)


@pytest.mark.parametrize("message", [None, [], "no", 1, {"method": "ping"}, {"jsonrpc": "2.0", "id": True, "method": "ping"}])
def test_malformed_envelopes_return_errors(message):
    assert handle(message)["error"]["code"] == -32600


@pytest.mark.parametrize("params", [[], "bad", None, 7])
def test_bad_params_fail_without_crashing(params):
    result = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params})
    assert result["error"]["code"] == -32602


def test_stdio_recovers_after_structural_and_validation_errors():
    messages = [None, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "did_it_change", "arguments": {"values": [True]}}},
                {"jsonrpc": "2.0", "id": 3, "method": "ping"}]
    result = subprocess.run([sys.executable, "-m", "noisefloor.mcp_server"],
                            input="\n".join(json.dumps(x) for x in messages), text=True,
                            capture_output=True, timeout=15)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert result.returncode == 0
    assert replies[0]["error"]["code"] == -32600
    assert replies[1]["error"]["code"] == -32602
    assert replies[2]["result"]["isError"] is True
    assert replies[3]["result"] == {}


def test_notification_does_not_run_tool(monkeypatch):
    def fail(*args):
        raise AssertionError("notification must not invoke computation")
    monkeypatch.setattr("noisefloor.mcp_server.call_tool", fail)
    assert handle({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "market_assessment"}}) is None


def test_cli_invalid_input_is_json_and_nonzero():
    result = subprocess.run([sys.executable, "-m", "noisefloor.cli", "market"],
                            input='{"series":[],"as_of":NaN}', text=True, capture_output=True, timeout=15)
    assert result.returncode == 2
    assert "error" in json.loads(result.stdout)
