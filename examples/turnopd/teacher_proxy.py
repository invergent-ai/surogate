"""Adapt a vLLM prompt-scoring server to the GRPO teacher token endpoint.

The proxy runs on CPU. The upstream teacher must use the student's tokenizer.
Run the upstream with --max-logprobs 1; see README.md for the complete launch.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def scored_prompt(tokens, response):
    """Keep only the actual input token's logprob, preserving token alignment."""
    entries = response["choices"][0].get("prompt_logprobs")
    if not entries or len(entries) != len(tokens):
        raise ValueError("Teacher must return one prompt_logprobs entry per input token")
    result = [None]  # First token has no preceding context.
    for token, entry in zip(tokens[1:], entries[1:]):
        key = str(token)
        if not entry or key not in entry or entry[key].get("logprob") is None:
            raise ValueError("Teacher omitted an input token's log probability")
        result.append({key: entry[key]})
    return result


def handler_for(upstream, api_key):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def upstream_request(self, path, body=None):
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = Request(
                upstream + path, headers=headers, data=json.dumps(body).encode() if body is not None else None
            )
            with urlopen(request, timeout=1200) as response:
                return json.loads(response.read())

        def do_GET(self):
            if self.path not in {"/health", "/v1/models"}:
                self.send_json(404, {"error": "unknown endpoint"})
                return
            try:
                models = self.upstream_request("/models")
                self.send_json(200, models if self.path == "/v1/models" else {"status": "ok"})
            except (HTTPError, URLError) as error:
                self.send_json(502, {"error": str(error)})

        def do_POST(self):
            if self.path != "/v1/chat/completions/tokens":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                tokens = body.get("tokens")
                if not isinstance(tokens, list) or len(tokens) < 2 or any(type(t) is not int or t < 0 for t in tokens):
                    raise ValueError("tokens must contain at least two nonnegative integer IDs")
                response = self.upstream_request(
                    "/completions",
                    {
                        "model": body["model"],
                        "prompt": tokens,
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "prompt_logprobs": 1,
                    },
                )
                self.send_json(
                    200,
                    {
                        "id": response.get("id", "teacher-score"),
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": body["model"],
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
                        ],
                        "prompt_logprobs": scored_prompt(tokens, response),
                    },
                )
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, {"error": str(error)})
            except (HTTPError, URLError) as error:
                self.send_json(502, {"error": str(error)})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), handler_for(args.upstream.rstrip("/"), os.environ.get("TEACHER_API_KEY"))
    )
    print(f"Teacher proxy listening on http://127.0.0.1:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
