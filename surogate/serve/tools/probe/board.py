"""Board-shape load: the definitions behind surogate/serve/BENCHMARKS.md.

`users` closed-loop clients stream salted prompts (`prompt_tokens` filler words behind a
unique tag, so nothing shares a prefix) with a fixed output length for `seconds`; every
request asks for `stream_options.include_usage` so the token accounting is the engine's own.

    python board.py PORT MODEL USERS SECONDS PROMPT_TOKENS MAX_TOKENS [WARMUP_SECONDS]

Reports, over the measured window only (a warm-up of the same shape runs first when asked):
  decode tok/s   = completion tokens / wall
  prefill tok/s  = prompt tokens / wall (a throughput share, decode phases included)
  TTFT p50 / p90 = time to the first streamed content chunk
  requests completed, errors, mean request latency.
"""

import json
import statistics
import sys
import threading
import time
import urllib.request


def run(port, model, users, seconds, prompt_tokens, max_tokens, label):
    stop = time.time() + seconds
    lock = threading.Lock()
    serial = [0]
    ttfts, prompt_sum, completion_sum, latencies, ok, err = [], [0], [0], [], [0], [0]
    errors = []

    def worker():
        while time.time() < stop:
            with lock:
                serial[0] += 1
                tag = serial[0]
            prompt = f"request {label}-{tag}: " + "word " * prompt_tokens
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", body,
                {"Content-Type": "application/json"})
            t0, first, chunks, usage = time.time(), None, 0, None
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    for line in r:
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            break
                        try:
                            d = json.loads(payload)
                        except ValueError:
                            continue
                        if d.get("error"):
                            # An in-stream error (e.g. "expired while waiting for admission")
                            # arrives inside a 200 response; it is a failed request.
                            raise RuntimeError(str(d["error"])[:80])
                        if d.get("usage"):
                            usage = d["usage"]
                        choices = d.get("choices") or [{}]
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            if first is None:
                                first = time.time() - t0
                            chunks += 1
                latency = time.time() - t0
                with lock:
                    if first is not None:
                        ttfts.append(first * 1e3)
                    completion_sum[0] += (usage or {}).get("completion_tokens", chunks)
                    prompt_sum[0] += (usage or {}).get("prompt_tokens", prompt_tokens)
                    latencies.append(latency)
                    ok[0] += 1
            except Exception as exc:  # noqa: BLE001
                with lock:
                    err[0] += 1
                    errors.append(str(exc)[:80])

    begin = time.time()
    threads = [threading.Thread(target=worker) for _ in range(users)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - begin
    q = (lambda p: statistics.quantiles(ttfts, n=100)[p - 1] if len(ttfts) >= 2
         else (ttfts[0] if ttfts else 0.0))
    return {
        "users": users, "wall": wall, "decode": completion_sum[0] / wall,
        "prefill": prompt_sum[0] / wall, "ttft_p50": q(50), "ttft_p90": q(90),
        "ok": ok[0], "err": err[0],
        "latency_mean": statistics.mean(latencies) if latencies else 0.0,
        "errors": errors,
    }


def main():
    port, model = sys.argv[1], sys.argv[2]
    users, seconds = int(sys.argv[3]), float(sys.argv[4])
    prompt_tokens, max_tokens = int(sys.argv[5]), int(sys.argv[6])
    warmup = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    if warmup > 0:
        w = run(port, model, users, warmup, prompt_tokens, max_tokens, "warm")
        print(f"warm-up users={users} {warmup:.0f}s: decode={w['decode']:.1f} tok/s "
              f"ok={w['ok']} err={w['err']}", flush=True)
    r = run(port, model, users, seconds, prompt_tokens, max_tokens, "board")
    print(f"board users={users} {r['wall']:.0f}s {prompt_tokens}/{max_tokens}: "
          f"decode={r['decode']:.1f} tok/s prefill={r['prefill']:.1f} tok/s "
          f"ttft_p50={r['ttft_p50'] / 1e3:.2f}s ttft_p90={r['ttft_p90'] / 1e3:.2f}s "
          f"latency_mean={r['latency_mean']:.1f}s ok={r['ok']} err={r['err']}", flush=True)
    if r["errors"]:
        from collections import Counter
        for text, n in Counter(r["errors"]).most_common(3):
            print(f"  error x{n}: {text}", flush=True)


if __name__ == "__main__":
    main()
