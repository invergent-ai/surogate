# Answer-coherence rate under load: N loader users saturate the endpoint while four short
# factual questions are asked repeatedly; the score is the fraction of answers matching each
# question's expected pattern. A rate over >=100 probes with a control arm is the unit of
# evidence here — single answers prove nothing (the engine is not run-to-run deterministic and
# the model itself drifts on terse prompts).
#
# The models this engine serves reason before they answer, so the budget has to cover the
# reasoning block *and* the answer, and only the answer is scored (see probe/chat.py for why
# scoring the reasoning would be a false pass). A request whose budget expired mid-reasoning is
# counted as `truncated`, not as a wrong answer: it says raise max_tokens, not fix the engine.
#
#   python coherence.py <port> <model> <loader_users> <seconds> [rounds=25] [key=value ...]
#     rounds       probe rounds of four questions each
#     max_tokens=  answer budget, default 512 (measured: these four questions cost the
#                  35B under 512 completion tokens including the reasoning block)
#     thinking=    on | off; default is whatever the server was started with
import json, re, sys, threading, time, urllib.request

import chat

port, model, users, dur = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
rounds = int(sys.argv[5]) if len(sys.argv) > 5 and "=" not in sys.argv[5] else 25
tail = [a for a in sys.argv[5:] if "=" in a]
opts = chat.options(tail)
max_tokens = int(opts.get("max_tokens", 512))
thinking = chat.thinking_option(opts)

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
ok = bad = truncated = 0
for _ in range(rounds):
    for q, pattern in QUESTIONS:
        try:
            reply = chat.ask(int(port), model, q, max_tokens, timeout=1800, thinking=thinking)
        except Exception as exc:
            reply = chat.Reply(f"<error {exc}>", "", "error", 0, 0)
        if reply.truncated_in_reasoning:
            truncated += 1
            print(f"  TRUNCATED q={q[:34]!r} after {reply.completion_tokens} tokens, all reasoning",
                  flush=True)
            continue
        hit = re.search(pattern, reply.answer, re.I) is not None
        ok, bad = ok + hit, bad + (not hit)
        if not hit:
            print(f"  MISMATCH q={q[:34]!r} -> {reply.answer[:110]!r}", flush=True)
stop = 0
print(f"coherence under {users} users (thinking={chat.thinking_label(thinking)}, "
      f"max_tokens={max_tokens}): {ok} ok, {bad} suspicious, {truncated} truncated")
