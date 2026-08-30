"""Agent-shaped traffic: a conversation that grows by append, re-sent every turn.

`board.py` salts every prompt so nothing shares a prefix — it measures raw prompt processing
with the prefix cache disarmed, which is the worst case. An agent harness is the opposite
extreme: the system prompt and tool definitions are byte-identical on every turn and the history
only grows at the end, so almost all of a 20k-token prompt has been prefilled before. Which of
those two an engine is good at is a different question, and this probe asks the second one.

Each user drives its own conversation: a fixed preamble (standing in for a system prompt plus
tool definitions), then `turns` rounds of user message → assistant reply, with the whole history
re-sent each round. The prompt therefore grows by a known amount per turn while its prefix stays
constant.

**The measurement is behavioural, not telemetric.** Chat Completions does not report cached
tokens on every engine — this server reports them on `/v1/responses` and in its request log, vLLM
reports them in `usage.prompt_tokens_details` — so the probe reads that field when it is offered
but does not depend on it. What it always has is TTFT against prompt length, and those two
settle the question on their own:

* a working prefix cache makes TTFT track the tokens **new** since the last turn, so it stays
  roughly flat while the prompt grows;
* no cache makes TTFT track the **whole** prompt, so it climbs turn after turn.

The summary reports both readings of the same seconds — tokens ÷ TTFT over the new tokens and
over the whole prompt — and their ratio is the speed-up the cache is actually delivering.

    python agentloop.py PORT MODEL TURNS [key=value ...]
      users=1        independent conversations in parallel
      preamble=4000  filler tokens standing in for the system prompt and tool definitions
      turn=120       user-message tokens added per turn
      reply=200      max_tokens per turn
      thinking=on|off   default: whatever the server was started with
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
import urllib.request

import chat

WORDS = ("river stone harbour lantern meadow copper signal window garden orchard silver marble "
         "thunder valley compass feather velvet timber saddle pillar canyon ember anchor summit "
         "blossom mirror ribbon shadow candle glacier beacon cellar pebble forest ladder needle").split()


def filler(tokens: int, seed: int) -> str:
    """Roughly `tokens` words, deterministic for a given seed so a turn is reproducible."""

    return " ".join(WORDS[(seed * 7 + i * 13) % len(WORDS)] for i in range(tokens))


def stream_turn(port: int, model: str, messages: list[dict], max_tokens: int,
                thinking: bool | None, timeout: float = 1800.0):
    """One streamed turn. Returns (ttft_seconds, answer, usage)."""

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking is not None:
        body["enable_thinking"] = thinking
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    first: float | None = None
    answer: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                event = json.loads(payload)
            except ValueError:
                continue
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if chat.is_first_token(delta) and first is None:
                first = time.time() - started
            if delta.get("content"):
                answer.append(delta["content"])
    return (first if first is not None else time.time() - started), "".join(answer), usage


def cached_tokens(usage: dict) -> int | None:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and "cached_tokens" in details:
        return int(details["cached_tokens"])
    return None


def conversation(port: int, model: str, user: int, turns: int, preamble: int, turn_tokens: int,
                 reply: int, thinking: bool | None, out: list) -> None:
    messages = [
        {"role": "system", "content": "You are a coding agent. Tools and context follow.\n"
                                      + filler(preamble, seed=1)},
    ]
    previous_prompt = 0
    for index in range(turns):
        messages.append({"role": "user",
                         "content": f"step {index}: " + filler(turn_tokens, seed=user * 100 + index)})
        try:
            ttft, answer, usage = stream_turn(port, model, messages, reply, thinking)
        except Exception as exc:  # noqa: BLE001
            out.append({"user": user, "turn": index, "error": str(exc)[:80]})
            return
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        out.append({
            "user": user,
            "turn": index,
            "prompt": prompt_tokens,
            "new": prompt_tokens - previous_prompt,
            "ttft": ttft,
            "cached": cached_tokens(usage),
            "completion": int(usage.get("completion_tokens") or 0),
        })
        previous_prompt = prompt_tokens
        messages.append({"role": "assistant", "content": answer})


def main() -> None:
    port, model, turns = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    opts = chat.options([a for a in sys.argv[4:] if "=" in a])
    users = int(opts.get("users", 1))
    preamble = int(opts.get("preamble", 4000))
    turn_tokens = int(opts.get("turn", 120))
    reply = int(opts.get("reply", 200))
    thinking = chat.thinking_option(opts)

    records: list = []
    lock = threading.Lock()
    sink: list = []

    def run(user: int) -> None:
        local: list = []
        conversation(port, model, user, turns, preamble, turn_tokens, reply, thinking, local)
        with lock:
            sink.extend(local)

    started = time.time()
    threads = [threading.Thread(target=run, args=(u,)) for u in range(users)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = sorted(sink, key=lambda r: (r.get("turn", 0), r.get("user", 0)))

    errors = [r for r in records if "error" in r]
    good = [r for r in records if "error" not in r]
    if not good:
        print(f"agentloop: no successful turns ({errors[:1]})")
        return

    print(f"agentloop users={users} turns={turns} preamble={preamble} turn={turn_tokens} "
          f"reply={reply} thinking={chat.thinking_label(thinking)}")
    print("  turn   prompt      new   ttft_ms   new_tok/s  whole_tok/s   cached")
    for index in range(turns):
        rows = [r for r in good if r["turn"] == index]
        if not rows:
            continue
        prompt = statistics.median(r["prompt"] for r in rows)
        new = statistics.median(r["new"] for r in rows)
        ttft = statistics.median(r["ttft"] for r in rows)
        cached = [r["cached"] for r in rows if r["cached"] is not None]
        print(f"  {index:4d} {prompt:8.0f} {new:8.0f} {ttft * 1e3:9.0f} "
              f"{new / max(ttft, 1e-9):11.0f} {prompt / max(ttft, 1e-9):12.0f} "
              f"{(statistics.median(cached) if cached else float('nan')):8.0f}")

    later = [r for r in good if r["turn"] > 0]
    first = [r for r in good if r["turn"] == 0]
    if later and first:
        ttft_first = statistics.median(r["ttft"] for r in first)
        ttft_later = statistics.median(r["ttft"] for r in later)
        prompt_last = statistics.median(r["prompt"] for r in good if r["turn"] == turns - 1)
        new_rate = statistics.median(r["new"] / max(r["ttft"], 1e-9) for r in later)
        whole_rate = statistics.median(r["prompt"] / max(r["ttft"], 1e-9) for r in later)
        print(f"\n  first turn (cold prefix) TTFT {ttft_first * 1e3:.0f} ms; later turns median "
              f"{ttft_later * 1e3:.0f} ms at a {prompt_last:.0f}-token prompt")
        print(f"  effective rate on the tokens new since the last turn: {new_rate:,.0f} tok/s")
        print(f"  the same seconds read against the whole prompt:       {whole_rate:,.0f} tok/s")
        print(f"  prefix speed-up (whole ÷ new): {whole_rate / max(new_rate, 1e-9):.1f}×  "
              f"— 1.0× means nothing was reused")
    if errors:
        print(f"  errors: {len(errors)} (first: {errors[0]})")
    print(f"  wall {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
