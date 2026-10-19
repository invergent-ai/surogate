#!/usr/bin/env python3
"""Replay vLLM's max_tokens answers (csrc/src/testing/serve/fixtures/serve/vllm_max_tokens/parity.jsonl) against a
running surogate-engine and compare.

The engine must serve the checkpoint the fixture was made with (Surogate3.7-35B-A3B: same tokenizer and chat
template), at the fixture's max_model_len, without --thinking/--no-thinking (vLLM rendered the template's default),
with --served-model-name m and a --default-max-tokens of at least the context (the rows taken here are the ones
whose vLLM had no generation-config default, i.e. "the whole rest of the context"):

    surogate-engine ART --served-model-name m --max-model-len 4096 --default-max-tokens 262144 ...
    python3 vllm_max_tokens_smoke.py --port 18501 --max-model-len 4096 parity.jsonl

Per row:
  * vLLM refused it: the engine must answer with the same status and byte-identical body, for stream=true as well;
  * vLLM accepted it with max_tokens N <= --exact-limit: sent with ignore_eos, the engine must generate exactly N
    tokens (finish_reason "length") and count the prompt as vLLM did;
  * vLLM accepted it with a larger N: sent streaming, the engine must answer 200 with an event stream (the
    connection is closed after the first event);
  * /v1/completions without a max_tokens (vLLM: 16, the engine: its default capped to the rest -- the documented
    exception) is checked against the engine's rule: 200 when the prompt leaves room, else 400.
Token-id prompts are skipped (the engine's /v1/completions takes text only).
"""

import argparse
import http.client
import json
import sys
import time


def content(k):
    return "a" + " a" * (k - 1)


def request_for(row):
    body = dict(row["body"], model="m")
    if row["endpoint"] == "chat":
        return "/v1/chat/completions", dict(body, messages=[{"role": "user", "content": content(row["k"])}])
    return "/v1/completions", dict(body, prompt=content(row["k"]))


def send(port, path, body, first_event_only=False, timeout=900):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if first_event_only and response.status == 200:
        line = response.fp.readline()
        connection.close()
        return response.status, response.getheader("Content-Type", ""), line.decode(errors="replace")
    data = response.read().decode()
    connection.close()
    return response.status, response.getheader("Content-Type", ""), data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--exact-limit", type=int, default=32)
    parser.add_argument("--prompts", default="", help="comma-separated prompt_tokens to keep (default all)")
    parser.add_argument("--accepted-large", type=int, default=1000000,
                        help="at most this many accepted rows with N > exact-limit")
    parser.add_argument("--vllm-port", type=int, default=0,
                        help="a vLLM render server at the same max_model_len: also compare /v1/responses live")
    args = parser.parse_args()
    keep = {int(p) for p in args.prompts.split(",") if p}

    rows = [json.loads(line) for line in open(args.fixture)]
    rows = [r for r in rows if r["max_model_len"] == args.max_model_len and r["default_max_tokens"] is None
            and r["prompt"] == "text" and (not keep or r["prompt_tokens"] in keep)]
    counts = {"refused_same": 0, "exact_budget": 0, "accepted_stream": 0, "exception_rule": 0}
    failures = []
    large = 0
    started = time.time()
    for row in rows:
        path, body = request_for(row)
        vllm = row["vllm"]
        b = row["body"]
        absent = all(b.get(f) is None for f in ("max_tokens", "max_completion_tokens"))
        tag = f"row {row['id']} {row['endpoint']} P={row['prompt_tokens']} {json.dumps(b)}"
        if row["endpoint"] == "completions" and absent:
            rest = row["max_model_len"] - row["prompt_tokens"]
            fits = rest >= 1 and b.get("min_tokens", 0) <= rest
            if fits:
                status, _, _ = send(args.port, path, dict(body, stream=True), first_event_only=True)
            else:
                status, _, _ = send(args.port, path, body)
            if status != (200 if fits else 400):
                failures.append(f"{tag}: engine rule expects {200 if fits else 400}, got {status}")
            else:
                counts["exception_rule"] += 1
            continue
        if vllm["status"] != 200:
            status, ctype, data = send(args.port, path, body)
            if status != vllm["status"] or data != vllm["http_body"] or "application/json" not in ctype:
                failures.append(f"{tag}:\n  engine {status} {ctype} {data}\n  vllm   {vllm['status']} {vllm['http_body']}")
            else:
                counts["refused_same"] += 1
            continue
        n = vllm["max_tokens"]
        if n <= args.exact_limit:
            exact = dict(body, ignore_eos=True, stream=False)
            status, _, data = send(args.port, path, exact)
            ok = status == 200
            if ok:
                reply = json.loads(data)
                usage = reply["usage"]
                finish = reply["choices"][0]["finish_reason"]
                ok = usage["completion_tokens"] == n and usage["prompt_tokens"] == row["prompt_tokens"] and \
                    finish == "length"
            if not ok:
                failures.append(f"{tag}: vLLM budget {n}, engine {status} {data[:400]}")
            else:
                counts["exact_budget"] += 1
            continue
        if large >= args.accepted_large:
            continue
        large += 1
        status, ctype, first = send(args.port, path, dict(body, stream=True), first_event_only=True)
        if status != 200 or "text/event-stream" not in ctype or not first.startswith("data:"):
            failures.append(f"{tag}: vLLM accepts (max_tokens {n}), engine {status} {ctype} {first[:300]}")
        else:
            counts["accepted_stream"] += 1
    if args.vllm_port:
        # The Responses API names its field max_output_tokens; the engine refuses below 16 as OpenAI does.
        C = args.max_model_len
        for max_output in (C + 1, C, C - 1, C - 6, C - 64, 16):
            for stream in (False, True):
                body = {"model": "m", "input": content(40), "max_output_tokens": max_output, "stream": stream}
                v_status, _, v_body = send(args.vllm_port, "/v1/responses/render", body)
                tag = f"responses max_output_tokens={max_output} stream={stream}"
                if v_status != 200:
                    status, ctype, data = send(args.port, "/v1/responses", body)
                    if status != v_status or data != v_body:
                        failures.append(f"{tag}:\n  engine {status} {data}\n  vllm   {v_status} {v_body}")
                    else:
                        counts["refused_same"] += 1
                else:
                    status, _, first = send(args.port, "/v1/responses", dict(body, stream=True), first_event_only=True)
                    if status != 200:
                        failures.append(f"{tag}: vLLM accepts, engine {status} {first[:300]}")
                    else:
                        counts["accepted_stream"] += 1
    print(json.dumps({"rows": len(rows), **counts, "failures": len(failures),
                      "seconds": round(time.time() - started, 1)}))
    for failure in failures[:50]:
        print("FAIL", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
