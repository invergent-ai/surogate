"""Teacher scoring must retain input-token alignment, not the most likely token."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

SPEC = importlib.util.spec_from_file_location(
    "teacher_proxy", Path(__file__).resolve().parents[2] / "examples/turnopd/teacher_proxy.py"
)
proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy)


def test_scores_actual_prompt_token_not_first_candidate():
    response = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {"99": {"logprob": -0.1}, "22": {"logprob": -2.0}},
                    {"88": {"logprob": -0.2}, "33": {"logprob": -3.0}},
                ]
            }
        ]
    }
    scores = proxy.scored_prompt([11, 22, 33], response)
    # This is the shape the orchestrator consumes: one entry per prompt ID,
    # including the null first-token position.
    actual = [0.0 if item is None else next(iter(item.values()))["logprob"] for item in scores]
    assert actual == [0.0, -2.0, -3.0]


@pytest.mark.parametrize("entries", [None, [None], [None, {"99": {"logprob": -0.1}}]])
def test_refuses_missing_or_misaligned_scores(entries):
    with pytest.raises(ValueError, match="Teacher"):
        proxy.scored_prompt([11, 22], {"choices": [{"prompt_logprobs": entries}]})


def test_http_translation_and_teacher_auth():
    calls = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append((self.path, self.headers.get("Authorization"), None))
            self.respond({"object": "list", "data": [{"id": "teacher"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, self.headers.get("Authorization"), body))
            self.respond({"choices": [{"prompt_logprobs": [None, {"22": {"logprob": -2.0}}]}]})

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    frontend = ThreadingHTTPServer(
        ("127.0.0.1", 0), proxy.handler_for(f"http://127.0.0.1:{upstream.server_port}/v1", "teacher-key")
    )
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (upstream, frontend)]
    for thread in threads:
        thread.start()
    try:
        base = f"http://127.0.0.1:{frontend.server_port}"
        with urlopen(base + "/v1/models", timeout=5) as response:
            assert json.loads(response.read())["data"][0]["id"] == "teacher"
        with urlopen(base + "/health", timeout=5) as response:
            assert json.loads(response.read()) == {"status": "ok"}
        request = Request(
            base + "/v1/chat/completions/tokens",
            data=json.dumps({"model": "teacher", "tokens": [11, 22], "prompt_logprobs": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result["prompt_logprobs"] == [None, {"22": {"logprob": -2.0}}]
        path, auth, body = calls[-1]
        assert path == "/v1/completions"
        assert auth == "Bearer teacher-key"
        assert body["prompt"] == [11, 22]
        assert body["prompt_logprobs"] == 1
        assert body["max_tokens"] == 1
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/unsupported", timeout=5)
        assert error.value.code == 404
    finally:
        for server in (frontend, upstream):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5)
