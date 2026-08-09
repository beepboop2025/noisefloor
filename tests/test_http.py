"""The streamable HTTP transport serves exactly what stdio serves."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from noisefloor.http_server import serve
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
