"""Stateless MCP and REST transport; bind loopback behind a TLS reverse proxy."""
from __future__ import annotations

import argparse
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .mcp_server import (
    MAX_HTTP_CONNECTIONS, REST_TOOLS, SERVER_NAME, SERVER_VERSION, TOOLS,
    call_tool, capabilities, handle, openapi,
)
from .schemas import MAX_BATCH, MAX_BODY_BYTES, dumps, loads
from .identity import implementation_sha256

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Accept, Authorization, Mcp-Session-Id, MCP-Protocol-Version",
}


def _log_activation(tool, failed, origin, surface="public"):
    name = tool if isinstance(tool, str) and tool in TOOLS else "unknown"
    print(f"mcp_activation product=noisefloor surface={surface} tool={name} "
          f"outcome={'error' if failed else 'success'} origin={origin}", file=sys.stderr, flush=True)


def _log_mcp_activation(msg, response, origin):
    if not isinstance(msg, dict) or msg.get("method") != "tools/call":
        return
    params = msg.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    result = response.get("result") if isinstance(response, dict) else None
    failed = (isinstance(response, dict) and "error" in response) or (
        isinstance(result, dict) and result.get("isError") is True)
    _log_activation(name, failed, origin)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 15

    def _path(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"
        # Preserve the published reverse-proxy prefix as well as direct hosting.
        if path.startswith("/noisefloor/"):
            path = path[len("/noisefloor"):]
        return path

    def _reply(self, status, body=b"", content_type="application/json"):
        self.send_response(status)
        for key, value in _CORS.items():
            self.send_header(key, value)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if body:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _reply_json(self, status, payload):
        self._reply(status, dumps(payload).encode("utf-8"))

    def _reject(self, status, message):
        self.close_connection = True  # unread request bytes cannot become another request
        self._reply_json(status, {"error": message})

    def do_OPTIONS(self):
        if self._path() not in {"/mcp", "/health", "/healthz", "/v1/capabilities", "/openapi.json", *REST_TOOLS}:
            self._reject(404, "route not found")
            return
        self._reply(204)

    def do_GET(self):
        path = self._path()
        if path in ("/healthz", "/health"):
            self._reply_json(200, {"name": SERVER_NAME, "version": SERVER_VERSION, "transport": "streamable-http",
                                   "implementation_sha256": implementation_sha256()})
        elif path == "/v1/capabilities":
            self._reply_json(200, capabilities())
        elif path == "/openapi.json":
            self._reply_json(200, openapi())
        elif path == "/mcp" or path in REST_TOOLS:
            self._reject(405, "POST required")
        else:
            self._reject(404, "route not found")

    def do_DELETE(self):
        self._reject(405 if self._path() == "/mcp" else 404,
                     "sessions are not used" if self._path() == "/mcp" else "route not found")

    def do_POST(self):
        path = self._path()
        if path != "/mcp" and path not in REST_TOOLS:
            self._reject(404, "route not found")
            return
        if self.headers.get_content_type() != "application/json":
            self._reject(415, "Content-Type must be application/json")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1:
            self._reject(400, "one Content-Length required; transfer encoding is unsupported")
            return
        try:
            length = int(lengths[0])
        except ValueError:
            length = 0
        if length <= 0:
            self._reject(400, "nonempty request body required")
            return
        if length > MAX_BODY_BYTES:
            self._reject(413, "request body exceeds the byte limit")
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete body")
            message = loads(raw)
        except (ValueError, TimeoutError, OSError):
            self.close_connection = True
            if path == "/mcp":
                self._reply_json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            else:
                self._reply_json(400, {"error": "invalid JSON body"})
            return
        origin = "edge" if self.headers.get("X-Forwarded-For") else "direct"
        if path in REST_TOOLS:
            tool = REST_TOOLS[path]
            try:
                payload = call_tool(tool, message)
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                _log_activation(tool, True, origin, surface="rest")
                self._reply_json(422, {"error": str(exc)})
                return
            _log_activation(tool, False, origin, surface="rest")
            self._reply_json(200, payload)
            return
        batch = isinstance(message, list)
        messages = message if batch else [message]
        if not messages or len(messages) > MAX_BATCH:
            self._reply_json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "invalid batch size"}})
            return
        replies = []
        for item in messages:
            reply = handle(item)
            if reply is not None:
                _log_mcp_activation(item, reply, origin)
                replies.append(reply)
        if not replies:
            self._reply(202)
        else:
            self._reply_json(200, replies if batch else replies[0])

    def log_message(self, _fmt, *_args):
        # Never log raw targets, queries, headers, observations or headlines.
        return


class _BoundedHTTPServer(ThreadingHTTPServer):
    """Bound active connections before creating worker threads.

    HTTP/1.1 requests run sequentially on each connection, so this also bounds
    in-flight computation. Idle connections hold a slot until the handler's
    timeout; TLS and per-client quotas remain the reverse proxy's responsibility.
    """
    daemon_threads = True
    request_queue_size = MAX_HTTP_CONNECTIONS * 2

    def __init__(self, address, handler):
        self._connection_slots = threading.BoundedSemaphore(MAX_HTTP_CONNECTIONS)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._connection_slots.acquire(blocking=False):
            body = b'{"error":"server_busy","retryable":true}'
            response = (b"HTTP/1.1 503 Service Unavailable\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Connection: close\r\nRetry-After: 1\r\n"
                        b"Cache-Control: no-store\r\n"
                        b"Access-Control-Allow-Origin: *\r\n"
                        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body)
            try:
                request.settimeout(1)
                request.sendall(response)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connection_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


def serve(host, port):
    return _BoundedHTTPServer((host, port), _Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8792")))
    args = parser.parse_args()
    server = serve(args.host, args.port)
    print(f"{SERVER_NAME} {SERVER_VERSION} HTTP on http://{args.host}:{server.server_address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
