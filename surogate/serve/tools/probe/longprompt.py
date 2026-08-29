"""Long-prompt recall probe: does a prompt that spans several prefill chunks keep its
own beginning?

Each request plants a unique code in the first sentence of a filler passage long
enough to exceed the engine's prefill window, then asks for the code back. A
correct answer needs attention over the earlier chunks' KV and the recurrent
state carried across chunk boundaries; a prompt that lost either answers with
something else. Runs `users` concurrent loaders for `rounds` rounds so the
prompts prefill in mixed rounds under decode load (or alone with users=1).

    python longprompt.py PORT MODEL USERS ROUNDS WORDS

Prints one summary line: ok / wrong / error counts and the first few failures.
"""

import json
import random
import sys
import threading
import time
import urllib.request

WORDS = (
    "river stone harbour lantern meadow copper signal window garden orchard "
    "silver marble thunder valley compass feather velvet timber saddle pillar "
    "canyon ember anchor summit blossom mirror ribbon shadow candle glacier "
    "beacon cellar pebble forest ladder needle parcel quiver rocket saffron"
).split()


def passage(rng: random.Random, words: int) -> str:
    out = []
    while len(out) < words:
        sentence = [rng.choice(WORDS) for _ in range(rng.randint(7, 14))]
        sentence[0] = sentence[0].capitalize()
        out.extend(sentence)
        out[-1] += "."
    return " ".join(out[:words])


def make_prompt(seed: int, words: int):
    rng = random.Random(seed)
    code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(3)) + \
        "-" + "".join(rng.choice("23456789") for _ in range(4))
    text = (
        f"Remember this code: {code}. Keep it in mind while reading the passage below.\n\n"
        + passage(rng, words)
        + "\n\nQuestion: what was the code I asked you to remember at the start? "
        "Reply with the code only."
    )
    return code, text


def ask(port: int, model: str, prompt: str):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 24,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


def main() -> None:
    port, model, users, rounds, words = (
        int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    lock = threading.Lock()
    tally = {"ok": 0, "wrong": 0, "error": 0}
    failures = []

    def loader(user: int) -> None:
        for round_index in range(rounds):
            seed = 1000 * user + round_index + int(time.time()) % 1000
            code, prompt = make_prompt(seed, words)
            try:
                answer = ask(port, model, prompt)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    tally["error"] += 1
                    failures.append(f"user {user} round {round_index}: error {exc}")
                continue
            with lock:
                if code in answer:
                    tally["ok"] += 1
                else:
                    tally["wrong"] += 1
                    failures.append(
                        f"user {user} round {round_index}: expected {code}, got {answer!r}")

    started = time.time()
    threads = [threading.Thread(target=loader, args=(u,)) for u in range(users)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"longprompt: users={users} rounds={rounds} words={words}: ok={tally['ok']} "
          f"wrong={tally['wrong']} error={tally['error']} wall={time.time() - started:.0f}s",
          flush=True)
    for line in failures[:6]:
        print("  " + line[:200], flush=True)


if __name__ == "__main__":
    main()
