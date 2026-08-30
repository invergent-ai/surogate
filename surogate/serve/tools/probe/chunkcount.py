"""Chunked-prefill integrity probe with a near-zero model floor.

Each request is a filler passage long enough to span several prefill chunks
followed by an instruction to count upwards from a random start, comma
separated. At temperature 0 the model reproduces the sequence reliably, so any
deviation in the first `max_tokens` tokens is evidence the prompt's last chunk
(the instruction) or the decode lane itself was corrupted — the two failure
sites of a mixed round. Runs `users` concurrent loaders for `rounds` rounds.

    python chunkcount.py PORT MODEL USERS ROUNDS WORDS [MAX_TOKENS]

Prints one summary line (ok / short / garbage / wrong / error) and the first failures;
`garbage` — characters a count cannot contain — is the corruption class.
"""

import json
import random
import re
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
    start = rng.randint(3, 400)
    text = (
        "Read the passage below, then follow the instruction after it.\n\n"
        + passage(rng, words)
        + f"\n\nInstruction: count upwards from {start}, one number after another, "
        "separated by commas, with nothing else in the reply. Keep counting until you "
        "are stopped."
    )
    return start, text


def ask(port: int, model: str, prompt: str, max_tokens: int):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


def classify(start: int, answer: str) -> str:
    """ok: the counting sequence from `start` (or `start` + 1, a reading the model
    takes about a third of the time); short: a valid sequence that stopped early;
    garbage: characters outside the digits/commas of a count — the corruption class;
    wrong: a well-formed but incorrect sequence."""
    numbers = re.findall(r"\d+", answer)
    stripped = re.sub(r"[\d,\s.]", "", answer)
    if stripped:
        return "garbage"
    if not numbers:
        return "wrong"
    first = int(numbers[0])
    if first not in (start, start + 1):
        return "wrong"
    complete = numbers[:-1] if len(numbers) > 1 else numbers
    if any(int(value) != first + i for i, value in enumerate(complete)):
        return "wrong"
    return "ok" if len(numbers) >= 8 else "short"


def main() -> None:
    port, model, users, rounds, words = (
        int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    max_tokens = int(sys.argv[6]) if len(sys.argv) > 6 else 128
    lock = threading.Lock()
    tally = {"ok": 0, "short": 0, "garbage": 0, "wrong": 0, "error": 0}
    failures = []

    def loader(user: int) -> None:
        for round_index in range(rounds):
            seed = 1000 * user + round_index + int(time.time()) % 1000
            start, prompt = make_prompt(seed, words)
            try:
                answer = ask(port, model, prompt, max_tokens)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    tally["error"] += 1
                    failures.append(f"user {user} round {round_index}: error {exc}")
                continue
            verdict = classify(start, answer)
            with lock:
                tally[verdict] += 1
                if verdict != "ok":
                    failures.append(
                        f"user {user} round {round_index} {verdict}: from {start}, got {answer!r}")

    started = time.time()
    threads = [threading.Thread(target=loader, args=(u,)) for u in range(users)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"chunkcount: users={users} rounds={rounds} words={words} max_tokens={max_tokens}: "
          f"ok={tally['ok']} short={tally['short']} garbage={tally['garbage']} "
          f"wrong={tally['wrong']} error={tally['error']} "
          f"wall={time.time() - started:.0f}s", flush=True)
    for line in failures[:12]:
        print("  " + line[:160], flush=True)


if __name__ == "__main__":
    main()
