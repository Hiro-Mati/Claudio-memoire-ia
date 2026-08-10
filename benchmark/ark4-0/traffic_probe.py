#!/usr/bin/env python3
"""Small local HTTP endpoint used to prove kubevpn traffic interception."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_SENSITIVE_HEADER_FRAGMENTS = (
    "authorization",
    "cookie",
    "api-key",
    "apikey",
    "secret",
    "token",
)


def _safe_headers(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in headers.items():
        lowered = str(key).lower()
        result[lowered] = (
            "<redacted>"
            if any(fragment in lowered for fragment in _SENSITIVE_HEADER_FRAGMENTS)
            else str(value)
        )
    return result


def _summarize_body(body: bytes) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "length": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    if not body:
        return summary
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        summary["format"] = "opaque"
        return summary
    summary["format"] = "json"
    if isinstance(value, dict):
        summary["top_level_keys"] = sorted(str(key) for key in value)
    elif isinstance(value, list):
        summary["item_count"] = len(value)
    else:
        summary["json_type"] = type(value).__name__
    return summary


class TrafficProbeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        log_file: Path,
        response_status: int,
        marker: str,
    ) -> None:
        super().__init__(server_address, TrafficProbeHandler)
        self.log_file = log_file
        self.response_status = response_status
        self.marker = marker

    def record(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)


class TrafficProbeHandler(BaseHTTPRequestHandler):
    server: TrafficProbeServer

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def do_PUT(self) -> None:  # noqa: N802
        self._handle()

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._handle()

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _handle(self) -> None:
        content_length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(content_length) if content_length else b""
        is_health = self.path.split("?", 1)[0] in {"/healthz", "/readyz"}
        status = 200 if is_health else self.server.response_status
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "marker": self.server.marker,
            "client": self.client_address[0],
            "method": self.command,
            "path": self.path,
            "headers": _safe_headers(self.headers),
            "body": _summarize_body(body),
            "response_status": status,
        }
        self.server.record(event)
        response = json.dumps(
            {
                "status": "ok" if is_health else "intercepted",
                "code": self.server.marker,
                "message": "request reached the local Ark4 kubevpn traffic probe",
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--response-status", default=503, type=int)
    parser.add_argument("--marker", default="ARK4_LOCAL_TRAFFIC_PROBE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = TrafficProbeServer(
        (args.host, args.port),
        log_file=Path(args.log_file).expanduser().resolve(),
        response_status=args.response_status,
        marker=args.marker,
    )
    print(
        f"[ark4-traffic-probe] listening at http://{args.host}:{args.port}; log={server.log_file}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
