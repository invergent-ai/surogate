# Cross-request KV isolation check: two short prompts with disjoint answers, sent together,
# repeatedly. A request that answers the other prompt's question has read the other request's
# KV (or its own page's previous occupant).
#
#   python pairleak.py <port> [rounds] [--model NAME]
#
# The clean case is silent by construction when the same prompts recycle the same pages: a
# request that reads a page's previous occupant reads its own previous answer's KV. Run the
# engine with SUROGATE_SERVE_KV_ZERO_TAKE=126 to fill every page with 0x7E as it is taken:
# any read of a cell the request never wrote then turns its output into garbage instead of a
# plausible answer, which this script reports as LEAK too (non-ASCII garbage in a reasoning
# preamble). The 2026-09-02 lone-prefill table-row leak reproduced 7/8 rounds this way.
import json
import sys
import threading
import time
import urllib.request

P1 = "Name three primary colors."
P2 = "What is the capital of France?"
OTHER = {P1: ("France", "Paris"), P2: ("color", "red", "blue")}


def ask(port, model, prompt, max_tokens=48):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        message = json.load(response)["choices"][0]["message"]
    text = (message.get("reasoning_content") or "") + " || " + (message.get("content") or "")
    return text.replace("\n", " ")


def classify(prompt, text):
    if any(key in text for key in OTHER[prompt]):
        return "LEAK"
    head = text[:60]
    if sum(1 for ch in head if ord(ch) > 127) > 5:
        return "LEAK"  # poisoned-cell garbage, see the header
    return "own"


def main():
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    model = "big"
    if "--model" in sys.argv:
        model = sys.argv[sys.argv.index("--model") + 1]
        args = [arg for arg in args if arg != model]
    port = args[0]
    rounds = int(args[1]) if len(args) > 1 else 8
    print("solo P1:", ask(port, model, P1)[:120])
    print("solo P2:", ask(port, model, P2)[:120], flush=True)
    leaks = 0
    for r in range(rounds):
        results = [None, None]
        threads = [
            threading.Thread(target=lambda i=i, p=p: results.__setitem__(i, ask(port, model, p)))
            for i, p in enumerate((P1, P2))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for name, prompt, text in (("P1", P1, results[0]), ("P2", P2, results[1])):
            tag = classify(prompt, text)
            leaks += tag == "LEAK"
            print(f"round {r} {name}[{tag}]: {text[:110]}", flush=True)
        time.sleep(0.5)
    print(f"{'FAIL' if leaks else 'PASS'}: {leaks} leaks in {rounds} rounds")
    sys.exit(1 if leaks else 0)


if __name__ == "__main__":
    main()
