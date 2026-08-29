# Answer-coherence rate under load: N loader users saturate the endpoint while four short
# factual questions are asked repeatedly; the score is the fraction of answers matching each
# question's expected pattern. A rate over >=100 probes with a control arm is the unit of
# evidence here — single answers prove nothing (the engine is not run-to-run deterministic and
# the model itself drifts on terse prompts).
#
#   python coherence.py <port> <model> <loader_users> <seconds> [rounds=25]
import json, re, sys, threading, time, urllib.request

port, model, users, dur = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 25
QUESTIONS = [("What is the capital of France? Answer in one word.", r"paris"),
             ("Name three prime numbers.", r"\b2\b|\btwo\b"),
             ("What is 17 plus 25? Answer with the number.", r"\b42\b"),
             ("What colour is the sky on a clear day? One word.", r"blue")]
stop = time.time() + dur
lock = threading.Lock()
serial = [0]

def loader():
    while time.time() < stop:
        with lock:
            serial[0] += 1
            tag = serial[0]
        body = json.dumps({"model": model,
                           "messages": [{"role": "user", "content": f"load {tag}: " + "word " * 512}],
                           "max_tokens": 128, "temperature": 0}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", body,
                {"Content-Type": "application/json"}), timeout=1800).read()
        except Exception:
            pass

threads = [threading.Thread(target=loader, daemon=True) for _ in range(users)]
[t.start() for t in threads]
time.sleep(20)
ok = bad = 0
for _ in range(rounds):
    for q, pattern in QUESTIONS:
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": q}],
                           "max_tokens": 40, "temperature": 0}).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions", body,
                    {"Content-Type": "application/json"}), timeout=1800) as r:
                answer = json.load(r)["choices"][0]["message"]["content"]
        except Exception as exc:
            answer = f"<error {exc}>"
        hit = re.search(pattern, answer, re.I) is not None
        ok, bad = ok + hit, bad + (not hit)
        if not hit:
            print(f"  MISMATCH q={q[:34]!r} -> {answer[:110]!r}", flush=True)
stop = 0
print(f"coherence under {users} users: {ok} ok, {bad} suspicious")
