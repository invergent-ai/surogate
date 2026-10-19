#!/usr/bin/env python3
"""End-to-end checks for --kv-admission (DESIGN-kv-admission.md) against running engines.

  identity    the same requests, one at a time, on two engines (e.g. --kv-admission reserve and
              ondemand, or ondemand with SUROGATE_SERVE_KV_FORCE_PREEMPT): byte-identical
              content / reasoning / tool calls per request, and where the first difference is.
  generate    the same, against one engine, answers recorded to --out; compare two recordings
              with `compare --base A.jsonl --test B.jsonl` (one GPU, engines one after another).
  load        many requests at once against one engine (a corpus or synthetic long prompts, no
              max_tokens unless given): every request answers, nothing errors or hangs; latency,
              throughput and the engine's KV counters (/metrics) before and after.

Requests are JSONL, one {"id", "category", "body"} per line, body = a Chat Completions request
(scripts/serve_prompt_parity.py writes and reads the same format).

  python scripts/kv_admission_e2e.py identity --base http://127.0.0.1:8470 --test http://127.0.0.1:8471 \\
      --requests requests.jsonl --n 64 --max-tokens 512 --temperature 0 --out identity.json
  python scripts/kv_admission_e2e.py load --url http://127.0.0.1:8471 --requests requests.jsonl \\
      --categories bfcl_live_simple,bfcl_multi_turn --n 300 --concurrency 32 --out load.json
  python scripts/kv_admission_e2e.py load --url http://127.0.0.1:8471 --synthetic 24 \\
      --prompt-tokens 100000:150000 --concurrency 16 --out stress.json
"""
from __future__ import annotations

import argparse
import copy
import http.client
import json
import random
import statistics
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor


def read_key(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read().strip()


def post_chat(url, body, key=None, timeout=7200):
    """POST /v1/chat/completions, streaming or not; returns a dict with the text channels, usage,
    status, time to first token and total time."""
    parsed = urllib.parse.urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    started = time.monotonic()
    out = {"status": None, "content": "", "reasoning": "", "tool_calls": [], "usage": None,
           "finish_reason": None, "ttft": None, "seconds": None, "error": None, "logprobs": []}
    try:
        connection.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=headers)
        response = connection.getresponse()
        out["status"] = response.status
        if response.status != 200:
            out["error"] = response.read().decode("utf-8", "replace")[:2000]
            return out
        if not body.get("stream"):
            payload = json.loads(response.read())
            choice = payload["choices"][0]
            message = choice.get("message", {})
            out["content"] = message.get("content") or ""
            out["reasoning"] = message.get("reasoning_content") or message.get("reasoning") or ""
            out["tool_calls"] = message.get("tool_calls") or []
            out["logprobs"] = ((choice.get("logprobs") or {}).get("content")) or []
            out["finish_reason"] = choice.get("finish_reason")
            out["usage"] = payload.get("usage")
            out["ttft"] = time.monotonic() - started
            return out
        calls = {}
        buffer = b""
        while True:
            chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
            if not chunk:
                break
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    message = json.loads(data)
                    if "error" in message:
                        out["error"] = json.dumps(message["error"])[:2000]
                        continue
                    if message.get("usage"):
                        out["usage"] = message["usage"]
                    for choice in message.get("choices", []):
                        delta = choice.get("delta", {})
                        text = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if (text or reasoning or delta.get("tool_calls")) and out["ttft"] is None:
                            out["ttft"] = time.monotonic() - started
                        out["content"] += text
                        out["reasoning"] += reasoning
                        out["logprobs"].extend(((choice.get("logprobs") or {}).get("content")) or [])
                        for call in delta.get("tool_calls") or []:
                            slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                            function = call.get("function") or {}
                            slot["name"] += function.get("name") or ""
                            slot["arguments"] += function.get("arguments") or ""
                        if choice.get("finish_reason"):
                            out["finish_reason"] = choice["finish_reason"]
        out["tool_calls"] = [calls[i] for i in sorted(calls)]
    except Exception as error:  # noqa: BLE001 - a transport failure is a result here
        out["error"] = f"{type(error).__name__}: {error}"
    finally:
        out["seconds"] = time.monotonic() - started
        connection.close()
    return out


def metrics(url, key=None):
    parsed = urllib.parse.urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    headers = {"Authorization": "Bearer " + key} if key else {}
    try:
        connection.request("GET", "/metrics", headers=headers)
        text = connection.getresponse().read().decode()
    except Exception:  # noqa: BLE001
        return {}
    finally:
        connection.close()
    wanted = ("surogate_kv_preemptions_total", "surogate_kv_recomputed_tokens_total",
              "surogate_kv_growth_refusals_total", "surogate_prefill_tokens_total",
              "surogate_decode_tokens_total", "surogate_kv_on_demand")
    out = {}
    for line in text.splitlines():
        for name in wanted:
            if line.startswith(name + "{") or line.startswith(name + " "):
                out[name] = out.get(name, 0) + float(line.rsplit(" ", 1)[1])
    return out


def load_requests(path, categories=None):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if categories and row["category"] not in categories:
                continue
            rows.append(row)
    return rows


def flatten(result):
    calls = [(c.get("name") or c.get("function", {}).get("name", ""),
              c.get("arguments") or c.get("function", {}).get("arguments", "")) for c in result["tool_calls"]]
    return result["reasoning"] + "\x1e" + result["content"] + "\x1e" + json.dumps(calls)


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def shape(body, args):
    body = copy.deepcopy(body)
    body["model"] = args.model or body.get("model")
    for field in ("max_tokens", "max_completion_tokens"):
        body.pop(field, None)
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
        if args.temperature == 0:
            for field in ("top_p", "top_k", "min_p"):
                body.pop(field, None)
    if args.seed is not None:
        body["seed"] = args.seed
    if getattr(args, "top_logprobs", 0):
        body["logprobs"] = True
        body["top_logprobs"] = args.top_logprobs
    if args.no_stream:
        body["stream"] = False
        body.pop("stream_options", None)
    else:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def selected_rows(args):
    if getattr(args, "images", 0):
        return [{"id": f"image-{i}", "category": "image", "body": image_body(i, args)} for i in range(args.images)]
    rows = load_requests(args.requests, set(args.categories.split(",")) if args.categories else None)
    random.Random(args.shuffle_seed).shuffle(rows)
    return rows[: args.n]


def cmd_generate(args):
    """The requests one at a time against one engine, each answer recorded (identity needs only one
    engine up at a time: generate on each, then compare)."""
    key = read_key(args.api_key_file)
    if args.images and not args.model:
        args.model = served_model(args.url, key)
    rows = selected_rows(args)
    results = [None] * len(rows)
    if args.concurrency > 1:
        # All at once (the batch composition a loaded server gives them), recorded in order.
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, result in enumerate(pool.map(lambda row: post_chat(args.url, shape(row["body"], args), key), rows)):
                results[i] = result
    with open(args.out, "w", encoding="utf-8") as handle:
        for i, row in enumerate(rows):
            result = results[i] or post_chat(args.url, shape(row["body"], args), key)
            scores = [[e.get("token"), e.get("logprob"),
                       [t.get("logprob") for t in (e.get("top_logprobs") or [])]] for e in result["logprobs"]]
            handle.write(json.dumps({"id": row["id"], "category": row["category"], "flat": flatten(result),
                                     "scores": scores,
                                     "usage": result["usage"], "finish_reason": result["finish_reason"],
                                     "error": result["error"], "seconds": result["seconds"]}) + "\n")
            handle.flush()
            print(f"[{i + 1}/{len(rows)}] {row['id']}: {(result['usage'] or {}).get('completion_tokens')} tokens "
                  f"{result['seconds']:.1f}s err={result['error']}", flush=True)
    print(json.dumps({"metrics": metrics(args.url, key)}))
    return 0


def cmd_compare(args):
    def read(path):
        with open(path, encoding="utf-8") as handle:
            return {row["id"]: row for row in map(json.loads, handle)}
    base, test = read(args.base), read(args.test)
    report = {"identical": 0, "different": 0, "errors": 0, "missing": 0, "differences": []}
    for rid, a in base.items():
        b = test.get(rid)
        if b is None:
            report["missing"] += 1
            continue
        if a["error"] or b["error"]:
            report["errors"] += 1
            continue
        if a["flat"] == b["flat"]:
            report["identical"] += 1
            continue
        report["different"] += 1
        at = first_difference(a["flat"], b["flat"])
        report["differences"].append({"id": rid, "first_difference_char": at, "chars": len(a["flat"]),
                                      "base_tokens": (a["usage"] or {}).get("completion_tokens"),
                                      "test_tokens": (b["usage"] or {}).get("completion_tokens"),
                                      "base_excerpt": a["flat"][max(0, at - 60): at + 60],
                                      "test_excerpt": b["flat"][max(0, at - 60): at + 60]})
    print(json.dumps(report, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    return 0 if report["different"] == 0 and report["errors"] == 0 else 1


def cmd_gaps(args):
    """Where two recordings with logprobs first differ by token, and how close the base run's top two
    candidates were there: a recompute that only perturbs the arithmetic flips near-ties, a wrong
    state flips confident tokens."""
    def read(path):
        with open(path, encoding="utf-8") as handle:
            return {row["id"]: row for row in map(json.loads, handle)}
    base, test = read(args.base), read(args.test)
    rows = []
    for rid, a in base.items():
        b = test.get(rid)
        if not b or a["error"] or b["error"] or not a.get("scores") or not b.get("scores"):
            continue
        ta = [s[0] for s in a["scores"]]
        tb = [s[0] for s in b["scores"]]
        at = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
        if at is None:
            rows.append({"id": rid, "identical": ta == tb, "tokens": len(ta), "scored_test": len(b["scores"]),
                         "tokens_test": (b["usage"] or {}).get("completion_tokens")})
            continue
        def gap(entry):
            tops = sorted([x for x in entry[2] if x is not None], reverse=True)
            return tops[0] - tops[1] if len(tops) >= 2 else None

        def drift(lo, hi):
            # Mean |logprob difference| of the shared tokens in [lo, hi): how far the two runs'
            # arithmetic has moved apart before they pick different tokens.
            values = [abs(a["scores"][i][1] - b["scores"][i][1]) for i in range(lo, min(hi, at))
                      if a["scores"][i][1] is not None and b["scores"][i][1] is not None]
            return sum(values) / len(values) if values else None
        rows.append({"id": rid, "identical": False, "first_token": at, "tokens": len(ta),
                     "base_gap": gap(a["scores"][at]), "test_gap": gap(b["scores"][at]),
                     "base_token": ta[at], "test_token": tb[at],
                     "base_logprob": a["scores"][at][1], "test_logprob": b["scores"][at][1],
                     "drift_before": drift(0, args.resume_at), "drift_after": drift(args.resume_at, at)})
    differing = [r for r in rows if not r["identical"]]
    gaps = sorted(r["base_gap"] for r in differing if r.get("base_gap") is not None)
    summary = {"compared": len(rows), "identical": len(rows) - len(differing), "different": len(differing),
               "first_token_min": min((r["first_token"] for r in differing), default=None),
               "gap_p50": gaps[len(gaps) // 2] if gaps else None, "gap_max": gaps[-1] if gaps else None,
               "gaps_over_0.5": sum(1 for g in gaps if g > 0.5), "rows": rows}
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    for r in differing:
        print(json.dumps(r))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)
    return 0


def cmd_identity(args):
    key = read_key(args.api_key_file)
    rows = load_requests(args.requests, set(args.categories.split(",")) if args.categories else None)
    random.Random(args.shuffle_seed).shuffle(rows)
    rows = rows[: args.n]
    report = {"requests": [], "identical": 0, "different": 0, "errors": 0}
    for i, row in enumerate(rows):
        body = shape(row["body"], args)
        a = post_chat(args.base, body, key)
        b = post_chat(args.test, body, key)
        entry = {"id": row["id"], "category": row["category"],
                 "base_tokens": (a["usage"] or {}).get("completion_tokens"),
                 "test_tokens": (b["usage"] or {}).get("completion_tokens"),
                 "base_error": a["error"], "test_error": b["error"]}
        if a["error"] or b["error"]:
            report["errors"] += 1
            entry["verdict"] = "error"
        else:
            fa, fb = flatten(a), flatten(b)
            if fa == fb:
                report["identical"] += 1
                entry["verdict"] = "identical"
            else:
                report["different"] += 1
                at = first_difference(fa, fb)
                entry.update(verdict="different", first_difference_char=at, total_chars=len(fa),
                             base_excerpt=fa[max(0, at - 80): at + 80], test_excerpt=fb[max(0, at - 80): at + 80])
        report["requests"].append(entry)
        print(f"[{i + 1}/{len(rows)}] {row['id']}: {entry['verdict']} "
              f"({entry['base_tokens']} / {entry['test_tokens']} tokens)", flush=True)
    report["metrics_base"] = metrics(args.base, key)
    report["metrics_test"] = metrics(args.test, key)
    print(json.dumps({k: report[k] for k in ("identical", "different", "errors", "metrics_base", "metrics_test")}))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    return 0 if report["errors"] == 0 else 1


WORDS = ("system log record value process network request response cache memory thread worker "
         "queue error warning module function service token index buffer file stream batch").split()


def served_model(url, key=None):
    parsed = urllib.parse.urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    headers = {"Authorization": "Bearer " + key} if key else {}
    try:
        connection.request("GET", "/v1/models", headers=headers)
        return json.loads(connection.getresponse().read())["data"][0]["id"]
    finally:
        connection.close()


def png_blocks(seed, width=448, height=336):
    """A PNG of random coloured rectangles (stdlib only), for image requests."""
    import struct
    import zlib
    rng = random.Random(seed)
    pixels = [[(255, 255, 255)] * width for _ in range(height)]
    for _ in range(6):
        x0, y0 = rng.randrange(width - 40), rng.randrange(height - 40)
        w, h = rng.randrange(30, width - x0), rng.randrange(30, height - y0)
        colour = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for y in range(y0, y0 + h):
            row = pixels[y]
            for x in range(x0, x0 + w):
                row[x] = colour
    raw = b"".join(b"\x00" + bytes(c for px in row for c in px) for row in pixels)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def image_body(i, args):
    import base64
    url = "data:image/png;base64," + base64.b64encode(png_blocks(500 + i)).decode()
    return {"model": args.model or "model",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "Describe every coloured rectangle in this image: its colour, position and "
                                         "size, then write a short story inspired by it."}]}],
            "chat_template_kwargs": {"enable_thinking": False}}


def synthetic_body(i, tokens, args):
    rng = random.Random(1000 + i)
    # About 2.17 tokens a word for this vocabulary: the numbers (which make it non-repetitive)
    # tokenize digit by digit. Measured on 24 prompts (2026-10-19): 2.15-2.18. Until then this
    # divided by 1.35, so earlier "--prompt-tokens 100000:160000" runs got 164-254k tokens.
    words = int(tokens / 2.17)
    text = " ".join(f"{rng.choice(WORDS)}{rng.randint(0, 9999)}" if rng.random() < 0.3 else rng.choice(WORDS)
                    for _ in range(words))
    ask = ("Write a long, detailed report on what this log shows, section by section." if args.long_answer
           else "In one short sentence: what is the most frequent word in it?")
    return {"model": args.model or "model",
            "messages": [{"role": "user", "content": "Here is a log dump:\n" + text + "\n\n" + ask}],
            "chat_template_kwargs": {"enable_thinking": False}}


def cmd_load(args):
    key = read_key(args.api_key_file)
    bodies = []
    if args.images:
        args.model = args.model or served_model(args.url, key)
        bodies = [(f"image-{i}", "image", image_body(i, args)) for i in range(args.images)]
    elif args.synthetic and not args.model:
        args.model = served_model(args.url, key)
    if args.synthetic and not args.images:
        lo, hi = (int(x) for x in args.prompt_tokens.split(":"))
        rng = random.Random(args.shuffle_seed)
        for i in range(args.synthetic):
            bodies.append((f"synthetic-{i}", "synthetic", synthetic_body(i, rng.randint(lo, hi), args)))
    elif not args.images:
        rows = load_requests(args.requests, set(args.categories.split(",")) if args.categories else None)
        random.Random(args.shuffle_seed).shuffle(rows)
        while len(rows) < args.n:
            rows = rows + rows
        bodies = [(r["id"], r["category"], r["body"]) for r in rows[: args.n]]
    bodies = [(i, c, shape(b, args)) for i, c, b in bodies]
    before = metrics(args.url, key)
    started = time.monotonic()
    results = [None] * len(bodies)
    lock = threading.Lock()
    done = [0]

    def run(index):
        rid, category, body = bodies[index]
        result = post_chat(args.url, body, key)
        result.update(id=rid, category=category)
        results[index] = result
        with lock:
            done[0] += 1
            if done[0] % max(1, len(bodies) // 20) == 0 or result["error"]:
                print(f"[{done[0]}/{len(bodies)}] {rid} status={result['status']} "
                      f"{result['seconds']:.1f}s err={result['error']}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(run, range(len(bodies))))
    wall = time.monotonic() - started
    after = metrics(args.url, key)
    ok = [r for r in results if not r["error"]]
    completion = sum((r["usage"] or {}).get("completion_tokens", 0) for r in ok)
    prompt = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in ok)
    ttft = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    latency = sorted(r["seconds"] for r in ok)

    def pct(values, p):
        return values[min(len(values) - 1, int(p * len(values)))] if values else None

    summary = {
        "requests": len(results), "ok": len(ok), "errors": len(results) - len(ok),
        "error_samples": [r["error"] for r in results if r["error"]][:5],
        "wall_seconds": wall, "requests_per_second": len(ok) / wall if wall else None,
        "completion_tokens": completion, "prompt_tokens": prompt,
        "output_tokens_per_second": completion / wall if wall else None,
        "ttft_p50": pct(ttft, 0.5), "ttft_p95": pct(ttft, 0.95),
        "latency_p50": pct(latency, 0.5), "latency_p95": pct(latency, 0.95),
        "latency_mean": statistics.mean(latency) if latency else None,
        "finish_reasons": {k: sum(1 for r in ok if r["finish_reason"] == k) for k in {r["finish_reason"] for r in ok}},
        "metrics_delta": {k: after.get(k, 0) - before.get(k, 0) for k in after},
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "results": [{k: r[k] for k in ("id", "category", "status", "error",
                                                                           "usage", "finish_reason", "ttft", "seconds")}
                                                       for r in results]}, handle, indent=1)
    return 0 if summary["errors"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("identity", "load", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--requests")
        p.add_argument("--categories")
        p.add_argument("--n", type=int, default=64)
        p.add_argument("--max-tokens", type=int, default=0, help="0 = none (the engine's default)")
        p.add_argument("--temperature", type=float)
        p.add_argument("--seed", type=int)
        p.add_argument("--model")
        p.add_argument("--no-stream", action="store_true")
        p.add_argument("--top-logprobs", type=int, default=0)
        p.add_argument("--shuffle-seed", type=int, default=7)
        p.add_argument("--api-key-file")
        p.add_argument("--out")
    sub.choices["identity"].add_argument("--base", required=True)
    sub.choices["identity"].add_argument("--test", required=True)
    sub.choices["generate"].add_argument("--url", required=True)
    sub.choices["generate"].add_argument("--concurrency", type=int, default=1)
    gaps = sub.add_parser("gaps")
    gaps.add_argument("--resume-at", type=int, default=64, help="first forced preemption (tokens)")
    gaps.add_argument("--base", required=True)
    gaps.add_argument("--test", required=True)
    gaps.add_argument("--out")
    compare = sub.add_parser("compare")
    compare.add_argument("--base", required=True)
    compare.add_argument("--test", required=True)
    compare.add_argument("--out")
    sub.choices["load"].add_argument("--url", required=True)
    sub.choices["load"].add_argument("--concurrency", type=int, default=16)
    sub.choices["load"].add_argument("--synthetic", type=int, default=0)
    sub.choices["load"].add_argument("--long-answer", action="store_true")
    for name in ("load", "generate"):
        sub.choices[name].add_argument("--images", type=int, default=0)
    sub.choices["load"].add_argument("--prompt-tokens", default="100000:150000")
    args = parser.parse_args()
    return {"identity": cmd_identity, "load": cmd_load, "generate": cmd_generate,
            "compare": cmd_compare, "gaps": cmd_gaps}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
