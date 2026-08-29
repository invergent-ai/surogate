# Decode throughput against an OpenAI-compatible endpoint: U concurrent users, fixed prompt and
# output lengths, for D seconds.
#
#   python tput.py <port> <model> <users> <seconds> <prompt_tokens> <max_tokens>
#
# Every request carries a unique tag so prompts are all distinct: with prefix reuse enabled,
# identical prompts collapse to one sampling round each after the first (the retained prompt
# prefix plus its tail hidden re-serve instantly) and the numbers become meaningless
# (2026-08-29 lesson: 995 "tok/s" at 10 ms latency).
import json, statistics, sys, threading, time, urllib.request

port, model = sys.argv[1], sys.argv[2]
users, dur, ptok, mtok = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
stop, out_tokens, lats, ok, err = time.time() + dur, [], [], [0], [0]
lock = threading.Lock()
serial = [0]

def worker():
    while time.time() < stop:
        with lock:
            serial[0] += 1
            tag = serial[0]
        prompt = f"request {tag}: " + "word " * ptok
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": mtok, "temperature": 0, "ignore_eos": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.load(r)
            n = d.get("usage", {}).get("completion_tokens", 0)
            with lock:
                out_tokens.append(n)
                lats.append((time.time() - t0) * 1e3)
                ok[0] += 1
        except Exception:
            with lock:
                err[0] += 1

begin = time.time()
threads = [threading.Thread(target=worker) for _ in range(users)]
[t.start() for t in threads]
[t.join() for t in threads]
elapsed = time.time() - begin
print(f"users={users}: decode={sum(out_tokens)/elapsed:.1f} tok/s "
      f"latency_p50={statistics.median(lats) if lats else 0:.0f}ms ok={ok[0]} err={err[0]}")
