"""The streamable HTTP transport serves exactly what stdio serves."""
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

from noisefloor.http_server import _Handler, serve
from noisefloor.mcp_server import SERVER_VERSION, TOOLS


@pytest.fixture(scope="module")
def base_url():
    server = serve("127.0.0.1", 0)  # port 0: the OS picks a free one
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _post(base_url, payload, path="/mcp"):
    req = urllib.request.Request(
        base_url + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "application/json, text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()
            return resp.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, json.loads(body) if body else None


def test_initialize_over_http(base_url):
    status, reply = _post(base_url, {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "0"}}})
    assert status == 200
    assert reply["result"]["serverInfo"] == {
        "name": "noisefloor", "version": SERVER_VERSION,
        "title": "noisefloor — is this number real?",
        "websiteUrl": "https://github.com/beepboop2025/noisefloor"}


def test_notification_gets_202_and_no_body(base_url):
    status, reply = _post(base_url, {"jsonrpc": "2.0",
                                     "method": "notifications/initialized"})
    assert status == 202
    assert reply is None


def test_tools_list_matches_the_registry(base_url):
    status, reply = _post(base_url, {"jsonrpc": "2.0", "id": 2,
                                     "method": "tools/list"})
    assert status == 200
    assert [t["name"] for t in reply["result"]["tools"]] == sorted(TOOLS)


def test_a_real_tool_call_computes(base_url):
    status, reply = _post(base_url, {
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "did_it_change",
                   "arguments": {"values": [10, 11, 10, 12, 11, 10, 11, 12]}}})
    assert status == 200
    payload = json.loads(reply["result"]["content"][0]["text"])
    assert payload["n"] == 8 and "state" in payload


def test_real_tool_call_emits_privacy_safe_activation(base_url, capfd):
    marker = 987654321.123456
    capfd.readouterr()
    status, _reply = _post(base_url, {
        "jsonrpc": "2.0", "id": 30, "method": "tools/call",
        "params": {"name": "did_it_change",
                   "arguments": {"values": [marker, marker + 1]}}})

    assert status == 200
    captured = capfd.readouterr()
    assert (
        "mcp_activation product=noisefloor surface=public "
        "tool=did_it_change outcome=success origin=direct"
    ) in captured.err
    assert str(marker) not in captured.err


def test_hosted_command_disables_raw_request_access_logging(capfd):
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'noisefloor-mcp-http = "noisefloor.http_server:main"' in pyproject
    secret = "credential-shaped-query-must-not-reach-journal"
    handler = object.__new__(_Handler)

    capfd.readouterr()
    handler.log_message(
        '"%s" %s %s',
        f"POST /mcp?api_key={secret} HTTP/1.1",
        "404",
        "-",
    )

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_discovery_does_not_emit_activation(base_url, capfd):
    capfd.readouterr()
    status, _reply = _post(base_url, {
        "jsonrpc": "2.0", "id": 31, "method": "tools/list"})
    assert status == 200
    assert "mcp_activation" not in capfd.readouterr().err


def test_batch_of_two_returns_two_replies(base_url):
    status, reply = _post(base_url, [
        {"jsonrpc": "2.0", "id": 10, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 11, "method": "ping"}])
    assert status == 200
    assert [r["id"] for r in reply] == [10, 11]


def test_malformed_json_is_a_parse_error_not_a_crash(base_url):
    req = urllib.request.Request(base_url + "/mcp", data=b"{nope",
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req, timeout=10)
    assert exc_info.value.code == 400
    assert json.load(exc_info.value)["error"]["code"] == -32700


def test_get_is_refused_but_healthz_answers(base_url):
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(base_url + "/mcp", timeout=10)
    assert exc_info.value.code == 405
    with urllib.request.urlopen(base_url + "/healthz", timeout=10) as resp:
        assert json.load(resp)["version"] == SERVER_VERSION


def test_cors_preflight(base_url):
    req = urllib.request.Request(base_url + "/mcp", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"


def test_unknown_post_route_is_not_treated_as_mcp(base_url):
    status, body = _post(base_url, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, path="/unknown")
    assert status == 404
    assert body == {"error": "route not found"}


def test_discovery_routes_expose_current_contracts(base_url):
    from noisefloor.mcp_server import capabilities, openapi
    for path, expected in [("/v1/capabilities", capabilities()), ("/openapi.json", openapi())]:
        with urllib.request.urlopen(base_url + path, timeout=10) as response:
            assert json.load(response) == expected


def test_prefixed_hosted_mcp_route_survives(base_url):
    status, reply = _post(base_url, {"jsonrpc": "2.0", "id": 1, "method": "ping"}, path="/noisefloor/mcp")
    assert status == 200 and reply["result"] == {}


@pytest.mark.parametrize("path,tool,key", [("/v1/market/assess", "market_assessment", "series"),
                                          ("/v1/narrative/triage", "narrative_triage", "events")])
def test_rest_matches_shared_python_and_mcp(base_url, path, tool, key):
    from noisefloor.mcp_server import call_tool, handle
    payload = {key: [], "as_of": "2026-10-08T00:00:00Z"}
    expected = call_tool(tool, payload)
    status, rest = _post(base_url, payload, path=path)
    mcp = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": tool, "arguments": payload}})
    assert status == 200
    assert rest == expected == mcp["result"]["structuredContent"]


def test_rest_validation_returns_422_and_preserves_data_privacy(base_url, capfd):
    marker = "headline-private-marker-do-not-log"
    capfd.readouterr()
    status, reply = _post(base_url, {"events": [{"title": marker}], "as_of": "2026-10-08T00:00:00Z"},
                          path="/v1/narrative/triage")
    assert status == 422 and "error" in reply
    log = capfd.readouterr().err
    assert marker not in log
    assert "tool=narrative_triage outcome=error" in log


@pytest.mark.parametrize("path,name,payload", [
    ("/v1/spectral/assess", "spectral_assessment", {"series": [], "as_of": "2026-10-09T00:00:00Z"}),
    ("/v1/research/dyson", "dyson_reference", {"dimension": 3, "steps": 5, "seed": 6116}),
])
def test_spectral_rest_and_mcp_share_results(base_url, path, name, payload):
    from noisefloor.mcp_server import call_tool
    expected = call_tool(name, payload)
    status, actual = _post(base_url, payload, path=path)
    assert status == 200
    assert actual == expected
    status, reply = _post(base_url, {"jsonrpc": "2.0", "id": 61, "method": "tools/call",
                                   "params": {"name": name, "arguments": payload}})
    assert status == 200
    assert reply["result"]["structuredContent"] == expected


def test_full_product_panel_works_over_rest(base_url):
    from examples.spectral_products import request_for
    from noisefloor import spectral
    request = request_for("seiche")
    status, actual = _post(base_url, request, path="/v1/spectral/assess")
    assert status == 200
    assert actual == spectral.assess(**request)
    assert actual["status"] == "assessed"


def test_research_routes_reject_bad_input_without_logging_observations(base_url, capfd):
    from examples.spectral_products import request_for
    request = request_for("liquilens")
    marker = "private-institution-never-log"
    request["series"][0]["id"] = marker
    request["series"][0]["observations"][0]["value"] = True
    capfd.readouterr()
    status, reply = _post(base_url, request, path="/v1/spectral/assess")
    assert status == 422 and "error" in reply
    assert marker not in capfd.readouterr().err


def test_rpc_malformed_batch_member_does_not_prevent_following_ping(base_url):
    status, reply = _post(base_url, [None, {"jsonrpc": "2.0", "id": 2, "method": "ping"}])
    assert status == 200
    assert reply[0]["error"]["code"] == -32600
    assert reply[1]["result"] == {}


def test_http_rejects_overlong_rpc_batch(base_url):
    from noisefloor.schemas import MAX_BATCH
    status, reply = _post(base_url, [{"jsonrpc": "2.0", "id": 1, "method": "ping"}] * (MAX_BATCH + 1))
    assert status == 400
    assert reply["error"]["code"] == -32600


def test_http_rejects_nonfinite_json(base_url):
    req = urllib.request.Request(base_url + "/mcp", data=b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"did_it_change","arguments":{"values":[NaN]}}}',
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req, timeout=10)
    assert error.value.code == 400
    assert json.load(error.value)["error"]["code"] == -32700


def test_http_requires_json_content_type(base_url):
    req = urllib.request.Request(base_url + "/mcp", data=b'{}', headers={"Content-Type": "text/plain"})
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req, timeout=10)
    assert error.value.code == 415


def test_host_saturation_rejects_without_spawning_and_recovers(monkeypatch):
    """Four active computations fill the host; fifth gets 503, then capacity returns."""
    from concurrent.futures import ThreadPoolExecutor
    from noisefloor.mcp_server import MAX_HTTP_CONNECTIONS, capabilities

    release = threading.Event()
    condition = threading.Condition()
    entered = 0

    def held_tool(name, arguments):
        nonlocal entered
        with condition:
            entered += 1
            condition.notify_all()
        assert release.wait(10), "test did not release held computation"
        return {"status": "finished"}

    monkeypatch.setattr("noisefloor.http_server.call_tool", held_tool)
    server = serve("127.0.0.1", 0)
    host_thread = threading.Thread(target=server.serve_forever, daemon=True)
    host_thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    assert MAX_HTTP_CONNECTIONS == 4
    assert capabilities()["limits"]["max_http_connections_per_process"] == 4
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_post, url, {}, "/v1/market/assess") for _ in range(4)]
            try:
                with condition:
                    assert condition.wait_for(lambda: entered == 4, timeout=5)
                request = urllib.request.Request(url + "/healthz")
                with pytest.raises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request, timeout=3)
                assert error.value.code == 503
                assert error.value.headers["Retry-After"] == "1"
                assert json.load(error.value) == {"error": "server_busy", "retryable": True}
                assert entered == 4  # excess connection never invokes the tool
            finally:
                release.set()
            assert [future.result(timeout=5) for future in futures] == [(200, {"status": "finished"})] * 4
        with urllib.request.urlopen(url + "/healthz", timeout=3) as response:
            assert response.status == 200
            assert json.load(response)["version"] == SERVER_VERSION
        status, body = _post(url, {}, path="/v1/market/assess")
        assert status == 200 and body == {"status": "finished"}
        assert entered == 5
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        host_thread.join(timeout=3)
