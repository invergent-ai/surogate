"""HTTP surface for the pool router — the product's single API endpoint.

Serves an OpenAI-compatible ``/v1/chat/completions`` and ``/v1/models`` backed
by :class:`ultra.router_endpoint.PoolRouter`, so any agent loop that speaks the
OpenAI protocol (Terminus 2 via LiteLLM, mini-swe-agent, Claude Code, …) can
drive the open-weight pool as if it were one model.

Run:
    python -m ultra.router_server --port 8022 \
        --binding director/manifests/fugu_clean_v1/grpo_pilot_train/current_pool_binding_ow2.json
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .router_endpoint import DEFAULT_BINDING, PoolRouter

SERVED_NAME = "fugu-open"


class _Handler(BaseHTTPRequestHandler):
    router: PoolRouter  # set by serve()

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, self.router.models_payload(SERVED_NAME))
        else:
            self._send(404, {"error": {"message": f"unknown path {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            self._send(400, {"error": {"message": "invalid JSON body"}})
            return
        if payload.get("stream"):
            self._send(400, {"error": {"message": "streaming not supported yet"}})
            return
        try:
            result = self.router.forward(payload)
            wid = result.get("_routing", {}).get("worker_id")
            print(f"routed turn -> worker {wid}", flush=True)
            self._send(200, result)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            self._send(exc.code, {"error": {"message": detail}})
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the server
            self._send(502, {"error": {"message": f"{type(exc).__name__}: {exc}"}})

    def log_message(self, fmt: str, *args) -> None:  # quiet default logging
        pass


def serve(port: int, binding_path: str, heavy: bool = False) -> None:
    _Handler.router = PoolRouter(binding_path, heavy=heavy)
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    default = _Handler.router.slots[_Handler.router.default_worker_id]
    print(
        f"fugu-open router on :{port} | default worker {default.worker_id} "
        f"({default.model}) | binding {binding_path}",
        flush=True,
    )
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--binding", default=str(DEFAULT_BINDING))
    parser.add_argument(
        "--heavy",
        action="store_true",
        help="conductor-planned workflows with isolated worker loops per step",
    )
    args = parser.parse_args()
    serve(args.port, args.binding, args.heavy)


if __name__ == "__main__":
    main()
