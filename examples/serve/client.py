"""Small HTTP examples using only the Python standard library; see README.md."""

import argparse
import base64
import json
import mimetypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


class Client:
    def __init__(self, base_url, api_key):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def request(self, path, body=None, *, method=None, stream=False):
        data = json.dumps(body).encode() if body is not None else None
        request = Request(self.base_url + path, data=data, headers=self.headers, method=method)
        with urlopen(request, timeout=300) as response:
            if stream:
                # Print SSE events, including reasoning deltas, usage, and [DONE].
                for line in response:
                    print(line.decode(), end="", flush=True)
                return None
            text = response.read().decode()
            if "application/json" in response.headers.get("Content-Type", ""):
                return json.loads(text)
            return text


def media_url(path):
    if path.startswith(("https://", "http://", "data:")):
        return path
    # Encode client-local files so the server need not share our filesystem.
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def run(args):
    client = Client(args.base_url, os.environ.get("SUROGATE_API_KEY"))
    messages = [{"role": "user", "content": "Explain what a LoRA adapter learns in one sentence."}]
    chat = {"model": args.model, "messages": messages, "max_tokens": 256}
    command = args.command
    if command in {"chat", "stream", "reasoning"}:
        if command == "reasoning":
            chat.update(chat_template_kwargs={"enable_thinking": True}, max_tokens=1024)
        else:
            chat["chat_template_kwargs"] = {"enable_thinking": False}
        if command == "stream":
            chat.update(stream=True, stream_options={"include_usage": True})
            return client.request("/v1/chat/completions", chat, stream=True)
        chat.update(
            logprobs=True,
            return_token_ids=True,
            seed=1234,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            min_p=0.02,
            repetition_penalty=1.05,
        )
        return client.request("/v1/chat/completions", chat)
    if command == "completions":
        return client.request(
            "/v1/completions",
            {
                "model": args.model,
                "prompt": "The capital of France is",
                "max_tokens": 32,
                "temperature": 0,
                "stop": ["\n"],
            },
        )
    if command == "tokens":
        tokenized = client.request(
            "/tokenize",
            {
                "model": args.model,
                "messages": messages,
                "with_token_strings": True,
            },
        )
        return {
            "tokenized": tokenized,
            "completion": client.request(
                "/v1/chat/completions/tokens",
                chat
                | {
                    "tokens": tokenized["tokens"],
                    "return_token_ids": True,
                    "logprobs": True,
                },
            ),
        }
    if command == "responses":
        first = client.request(
            "/v1/responses",
            {
                "model": args.model,
                "input": "Remember this word: otter.",
                "max_output_tokens": 256,
                "store": True,
            },
        )
        second = client.request(
            "/v1/responses",
            {
                "model": args.model,
                "input": "Which word did I ask you to remember?",
                "previous_response_id": first["id"],
                "max_output_tokens": 256,
                "store": True,
            },
        )
        result = {
            "first": first,
            "followup": second,
            "stored": client.request("/v1/responses/" + second["id"]),
            "input_items": client.request("/v1/responses/" + second["id"] + "/input_items"),
            "input_tokens": client.request(
                "/v1/responses/input_tokens",
                {
                    "model": args.model,
                    "input": "Count this input.",
                },
            ),
        }
        for response in (second, first):
            client.request("/v1/responses/" + response["id"], method="DELETE")
        return result
    if command == "anthropic":
        body = {"model": args.model, "system": "Be concise.", "messages": messages, "max_tokens": 256}
        return {
            "count": client.request("/v1/messages/count_tokens", body),
            "message": client.request("/v1/messages", body),
        }
    if command == "tools":
        # This demo executes one local, deterministic function. No external actions.
        chat["messages"] = [{"role": "user", "content": "Use the tool to convert 5 kilometers to meters."}]
        chat["chat_template_kwargs"] = {"enable_thinking": False}
        chat["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "to_meters",
                    "description": "Convert kilometers to meters.",
                    "parameters": {
                        "type": "object",
                        "properties": {"kilometers": {"type": "number"}},
                        "required": ["kilometers"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        chat["tool_choice"] = "auto"
        response = client.request("/v1/chat/completions", chat)
        message = response["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            return response  # Automatic choice may produce a direct answer.
        chat["messages"].append(message)
        for call in calls:
            function = call["function"]
            values = json.loads(function["arguments"])
            if (
                function["name"] != "to_meters"
                or set(values) != {"kilometers"}
                or type(values["kilometers"]) not in (int, float)
            ):
                raise ValueError("Model returned invalid tool arguments")
            chat["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps({"meters": values["kilometers"] * 1000}),
                }
            )
        chat["tool_choice"] = "none"
        return client.request("/v1/chat/completions", chat)
    if command in {"image", "video"}:
        if not args.media:
            raise ValueError("--media needs a local file or URL")
        field = "image_url" if command == "image" else "video_url"
        chat["messages"] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this media briefly."},
                    {"type": field, field: {"url": media_url(args.media)}},
                ],
            }
        ]
        return client.request("/v1/chat/completions", chat)
    if command == "embeddings":
        return client.request(
            "/v1/embeddings",
            {
                "model": args.model,
                "input": ["A cat on a mat.", "A kitten resting on a rug."],
            },
        )
    if command == "concurrent":
        # Identical prefixes exercise prompt reuse; requests run concurrently.
        def complete(index):
            return client.request("/v1/chat/completions", chat | {"seed": index})

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(complete, range(8)))
        return {"responses": responses, "cache": client.request("/kv_stats")}
    if command == "status":
        return {path: client.request(path) for path in ("/health", "/v1/models", "/kv_stats", "/metrics")}
    target = "?model=" + quote(args.model, safe="")
    if command == "sleep":
        return client.request("/sleep" + target, {}, method="POST")
    if command == "wake":
        return client.request("/wake_up" + target, {}, method="POST")
    if command == "load-lora":
        if not args.adapter:
            raise ValueError("--adapter needs a PEFT adapter directory on the server")
        return client.request(
            "/v1/load_lora_adapter" + target,
            {
                "lora_name": args.adapter_name,
                "lora_path": args.adapter,
            },
        )
    if command == "unload-lora":
        return client.request("/v1/unload_lora_adapter" + target, {"lora_name": args.adapter_name})
    raise ValueError(f"Unknown command: {command}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "chat",
            "stream",
            "reasoning",
            "completions",
            "tokens",
            "responses",
            "anthropic",
            "tools",
            "image",
            "video",
            "embeddings",
            "concurrent",
            "status",
            "sleep",
            "wake",
            "load-lora",
            "unload-lora",
        ],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Server root URL (omit /v1)")
    parser.add_argument("--model", default="demo")
    parser.add_argument("--media", help="Client-local image/video file, HTTP URL, or data URI")
    parser.add_argument("--adapter", help="Server-local PEFT directory")
    parser.add_argument("--adapter-name", default="tuned")
    args = parser.parse_args()
    try:
        result = run(args)
    except HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode()}", file=sys.stderr)
        raise SystemExit(1) from error
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
