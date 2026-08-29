# Time-to-first-token under load: U concurrent users stream unique-tagged prompts for D
# seconds; reports TTFT p50/p90 (first SSE content chunk) and decode tok/s.
#
#   python ttft.py <port> <model> <users> <seconds> <prompt_tokens> <max_tokens>
import json, statistics, sys, threading, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
users, dur, ptok, mtok = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
stop, ttfts, out_tokens, ok, err = time.time() + dur, [], [], [0], [0]
lock = threading.Lock()
serial = [0]

def worker():
    while time.time() < stop:
        with lock:
            serial[0] += 1
            tag = serial[0]
        prompt = f"request {tag}: " + "word " * ptok
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": mtok, "temperature": 0, "ignore_eos": True,
                           "stream": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        t0, first, n = time.time(), None, 0
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
                    delta = d.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        if first is None:
                            first = time.time() - t0
                        n += 1
            with lock:
                if first is not None:
                    ttfts.append(first * 1e3)
                out_tokens.append(n)
                ok[0] += 1
        except Exception:
            with lock:
                err[0] += 1

begin = time.time()
threads = [threading.Thread(target=worker) for _ in range(users)]
[t.start() for t in threads]
[t.join() for t in threads]
elapsed = time.time() - begin
q = lambda p: statistics.quantiles(ttfts, n=100)[p - 1] if len(ttfts) >= 2 else (ttfts[0] if ttfts else 0)
print(f"users={users}: ttft_p50={q(50):.0f}ms ttft_p90={q(90):.0f}ms "
      f"decode={sum(out_tokens)/elapsed:.1f} chunks/s ok={ok[0]} err={err[0]}")
