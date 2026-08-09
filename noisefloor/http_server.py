"""noisefloor MCP server over streamable HTTP — the same six tools, hosted.

The stdio transport in mcp_server serves one local assistant. Registries and
gateways (Smithery among them) instead expect a public HTTPS endpoint speaking
the streamable HTTP transport: JSON-RPC 2.0 in a POST body, one response out.
This wraps the exact same handle() in that shape, statelessly — no sessions,
no server-initiated streams, standard library only.

Run:  python -m noisefloor.http_server [--host 127.0.0.1] [--port 8792]

Meant to sit behind a reverse proxy that terminates TLS; it binds loopback by
default for exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .mcp_server import SERVER_NAME, SERVER_VERSION, TOOLS, handle

MAX_BODY_BYTES = 10 * 1024 * 1024  # a million-point series is ~8 MB of JSON

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers":
        "Content-Type, Accept, Authorization, Mcp-Session-Id, MCP-Protocol-Version",
}


def _log_mcp_activation(msg: dict, response: dict, origin: str) -> None:
    """Log an allowlisted activation without retaining request contents."""
    if msg.get("method") != "tools/call":
        return
    params = msg.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    tool = name if name in TOOLS else "unknown"
    result = response.get("result") if isinstance(response, dict) else None
    failed = (isinstance(response, dict) and "error" in response) or (
        isinstance(result, dict) and result.get("isError") is True)
    outcome = "error" if failed else "success"
    print(
        f"mcp_activation product=noisefloor surface=public "
        f"tool={tool} outcome={outcome} origin={origin}",
        file=sys.stderr,
        flush=True,
    )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self, status: int, body: bytes = b"",
               content_type: str = "application/json") -> None:
        self.send_response(status)
        for k, v in _CORS.items():
            self.send_header(k, v)
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _reply_json(self, status: int, payload) -> None:
        self._reply(status, json.dumps(payload, ensure_ascii=False).encode())

    def do_OPTIONS(self) -> None:  # CORS preflight
        self._reply(204)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("healthz") or self.path in ("/health", ""):
            self._reply_json(200, {"name": SERVER_NAME, "version": SERVER_VERSION,
                                   "transport": "streamable-http"})
            return
        # no server-initiated streams: the spec's answer for a GET is 405
        self._reply_json(405, {"error": "GET not supported; POST JSON-RPC messages"})

    def do_DELETE(self) -> None:
        # stateless — there is no session to terminate
        self._reply_json(405, {"error": "sessions are not used by this server"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._reply_json(400, {"jsonrpc": "2.0", "id": None, "error": {
                "code": -32600, "message": f"body required, at most {MAX_BODY_BYTES} bytes"}})
            return
        try:
            msg = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._reply_json(400, {"jsonrpc": "2.0", "id": None, "error": {
                "code": -32700, "message": "parse error"}})
            return

        batch = isinstance(msg, list)
        msgs = msg if batch else [msg]
        if not all(isinstance(m, dict) for m in msgs) or not msgs:
            self._reply_json(400, {"jsonrpc": "2.0", "id": None, "error": {
                "code": -32600, "message": "expected a JSON-RPC message or batch"}})
            return

        replies = []
        for message in msgs:
            reply = handle(message)
            if reply is not None:
                origin = ("edge" if self.headers.get("X-Forwarded-For")
                          else "direct")
                _log_mcp_activation(message, reply, origin)
                replies.append(reply)
        if not replies:                      # notifications only
            self._reply(202)
        elif batch:
            self._reply_json(200, replies)
        else:
            self._reply_json(200, replies[0])

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", file=sys.stderr)


def serve(host: str, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8792")))
    args = ap.parse_args()
    server = serve(args.host, args.port)
    print(f"{SERVER_NAME} {SERVER_VERSION} streamable HTTP on "
          f"http://{args.host}:{server.server_address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
