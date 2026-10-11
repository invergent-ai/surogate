#!/usr/bin/env python3
"""Long-context agentic benchmark: how prompt caching holds up across a multi-turn tool loop.

Each agent runs one conversation over /v1/chat/completions. The conversation opens with a system
prompt shared by every agent (SYSTEM tokens), the agent's own task, and a history of earlier tool
calls and results that brings it to about START tokens. Then every turn the model writes a tool
call of OUT tokens, a tool result of TOOL_MIN-TOOL_MAX tokens is appended, and the whole
conversation goes back as the next request, until the prompt reaches MAX tokens. Turn 0 is the
cold start (only the shared system prompt can be cached); every later turn should reuse
everything but its new tail.

Per turn it records the prompt length, the cached tokens the server reports
(usage.prompt_tokens_details.cached_tokens; vLLM needs --enable-prompt-tokens-details), the time
to first token, and the decode speed. One JSON line per turn goes to --jsonl, and one SUMMARY
line per agent count to stdout.

    python agentic_long_bench.py http://127.0.0.1:18080 <model dir> surogate --corpus wiki.test.raw \
        --agents 1,4,8 --jsonl out.jsonl
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("tokenizer")
parser.add_argument("label")
parser.add_argument("--corpus", required=True)
parser.add_argument("--agents", default="1,4,8")
parser.add_argument("--start", type=int, default=100_000)
parser.add_argument("--max", type=int, default=200_000)
parser.add_argument("--system", type=int, default=6_000)
parser.add_argument("--out", type=int, default=200)
parser.add_argument("--tool-min", type=int, default=1_000)
parser.add_argument("--tool-max", type=int, default=4_000)
parser.add_argument("--stagger", type=float, default=5.0)
parser.add_argument("--jsonl", required=True)
parser.add_argument(
    "--tools",
    action="store_true",
    help="Real tool calls: the history is read_file calls and their results, the model gets "
    "read_file/search/run_shell, writes its own calls and stops on its own (no ignore_eos), and "
    "every call it makes is answered with a corpus piece. Requests are not streamed, so a turn "
    "records its latency, not its first token: take TTFT, decode speed and draft acceptance from "
    "the server's request log.",
)
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.tokenizer)
ids = tok.encode(open(args.corpus, encoding="utf-8").read(), add_special_tokens=False)
if len(ids) < args.system + 2 * args.tool_max:
    raise SystemExit(f"corpus too small: {len(ids)} tokens")
SYSTEM = (
    "You are a software engineering agent. Use the tools to inspect and edit the repository, "
    "then answer.\n\n" + tok.decode(ids[: args.system])
)


def piece(rng, tokens):
    start = rng.randrange(args.system, len(ids) - tokens - 1)
    return tok.decode(ids[start : start + tokens])


def tool_result(rng):
    return "<tool_result>\n" + piece(rng, rng.randint(args.tool_min, args.tool_max)) + "\n</tool_result>"


TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search", "description": "Search the repository for a string; returns matching lines with context.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


def opening_tools(rng, agent):
    """--tools: the task, then read_file calls and their results up to about --start tokens."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Task {agent}-{rng.randrange(10**9)}: find where the behaviour described "
         "below is implemented and explain it, reading and searching files as needed.\n\n" + piece(rng, 300)},
    ]
    approx = args.system + 300
    step = 0
    while approx < args.start:
        call_id = f"call_{agent}_{step}"
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": f"src/module_{agent}_{step}.py"})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": piece(rng, rng.randint(args.tool_min, args.tool_max))},
        ]
        approx += 40 + (args.tool_min + args.tool_max) // 2 + 12
        step += 1
    return messages


async def request_tools(session, messages):
    body = {
        "model": "bench",
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": args.out,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    async with session.post(args.url + "/v1/chat/completions", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
        reply = await response.json()
    ended = time.perf_counter()
    choice = reply["choices"][0]
    usage = reply["usage"]
    details = usage.get("prompt_tokens_details") or {}
    return {
        "prompt": usage["prompt_tokens"],
        "cached": details.get("cached_tokens"),
        "completion": usage["completion_tokens"],
        "latency_s": round(ended - started, 4),
        "finish": choice.get("finish_reason"),
        "tool_calls": len(choice["message"].get("tool_calls") or []),
        "message": choice["message"],
    }


def opening(rng, agent):
    """System prompt, task, and earlier tool calls and results up to about --start tokens."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Task {agent}-{rng.randrange(10**9)}: " + piece(rng, 300)},
    ]
    approx = args.system + 300
    step = 0
    while approx < args.start:
        call = f"read_file(path='src/module_{agent}_{step}.py')  # " + piece(rng, args.out - 20)
        result = tool_result(rng)
        messages += [{"role": "assistant", "content": call}, {"role": "user", "content": result}]
        approx += args.out + (args.tool_min + args.tool_max) // 2 + 12
        step += 1
    return messages


async def request(session, messages):
    body = {
        "model": "bench",
        "messages": messages,
        "max_tokens": args.out,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started, first, usage, done, pieces, thought = time.perf_counter(), None, None, False, [], []
    async with session.post(args.url + "/v1/chat/completions", json=body) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:300]}")
        async for line in response.content:
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(str(chunk["error"])[:300])
            for choice in chunk.get("choices", []):
                # The first token is the first of either channel: a server that splits a thought
                # off (thinking asked off, but the model opened one) streams it as reasoning, where
                # a server without a reasoning parser streams the same tokens as content.
                delta = choice.get("delta") or {}
                text = delta.get("content")
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if (text or reasoning) and first is None:
                    first = time.perf_counter()
                if reasoning:
                    thought.append(reasoning)
                if text:
                    pieces.append(text)
            usage = chunk.get("usage") or usage
    if not done or usage is None or first is None:
        raise RuntimeError("incomplete stream")
    ended = time.perf_counter()
    details = usage.get("prompt_tokens_details") or {}
    completion = usage["completion_tokens"]
    return {
        "prompt": usage["prompt_tokens"],
        "cached": details.get("cached_tokens"),
        "completion": completion,
        "ttft_s": round(first - started, 4),
        "latency_s": round(ended - started, 4),
        "decode_tok_s": round((completion - 1) / max(ended - first, 1e-9), 2),
        "reasoning_chars": sum(len(piece) for piece in thought),
        "text": "".join(pieces),
    }


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def summarize(agents, records, errors, wall):
    if args.tools:
        return summarize_tools(agents, records, errors, wall)
    cold = [r for r in records if r["turn"] == 0]
    warm = [r for r in records if r["turn"] > 0]
    known = [r for r in warm if r["cached"] is not None]
    new = [r["prompt"] - r["cached"] for r in known]
    out = {
        "label": args.label,
        "agents": agents,
        "wall_s": round(wall, 1),
        "turns": len(records),
        "errors": len(errors),
        "first_error": errors[:1],
        "turns_per_min": round(len(records) * 60 / wall, 1),
        "gen_tok_s": round(sum(r["completion"] for r in records) / wall, 1),
        "cold_ttft_p50_s": pct([r["ttft_s"] for r in cold], 0.5),
        "cold_ttft_max_s": max((r["ttft_s"] for r in cold), default=None),
        "cold_prompt_mean": round(statistics.mean(r["prompt"] for r in cold)) if cold else None,
        "cold_cached": [r["cached"] for r in cold],
        "warm_turns": len(warm),
        "warm_ttft_p50_s": pct([r["ttft_s"] for r in warm], 0.5),
        "warm_ttft_p90_s": pct([r["ttft_s"] for r in warm], 0.9),
        "warm_ttft_max_s": max((r["ttft_s"] for r in warm), default=None),
        "warm_cached_reported": len(known),
        "warm_new_tokens_p50": pct(new, 0.5),
        "warm_new_tokens_max": max(new, default=None),
        # A turn whose server recomputed more than the last turn's reply plus one tool result
        # lost part of its conversation's cache.
        "warm_cache_misses": sum(1 for n in new if n > args.out + args.tool_max + 512),
        "warm_cached_fraction_min": (round(min(r["cached"] / r["prompt"] for r in known), 4) if known else None),
        "decode_tok_s_p50": pct([r["decode_tok_s"] for r in warm], 0.5),
        "turns_with_reasoning": sum(1 for r in records if r.get("reasoning_chars")),
    }
    buckets = {}
    for r in warm:
        low = (r["prompt"] // 25_000) * 25
        buckets.setdefault(f"{low}-{low + 25}k", []).append(r)
    out["by_context"] = {
        k: {
            "turns": len(v),
            "ttft_p50_s": pct([r["ttft_s"] for r in v], 0.5),
            "decode_tok_s_p50": pct([r["decode_tok_s"] for r in v], 0.5),
        }
        for k, v in sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))
    }
    return out


def summarize_tools(agents, records, errors, wall):
    warm = [r for r in records if r["turn"] > 0]
    known = [r for r in warm if r["cached"] is not None]
    return {
        "label": args.label,
        "mode": "tools",
        "agents": agents,
        "wall_s": round(wall, 1),
        "turns": len(records),
        "errors": len(errors),
        "first_error": errors[:1],
        "turns_per_min": round(len(records) * 60 / wall, 1),
        "gen_tok_s": round(sum(r["completion"] for r in records) / wall, 1),
        "completion_mean": round(statistics.mean(r["completion"] for r in records), 1) if records else None,
        "turns_with_tool_calls": sum(1 for r in records if r["tool_calls"]),
        "finish": {k: sum(1 for r in records if r["finish"] == k) for k in sorted({r["finish"] for r in records})},
        "cold_latency_max_s": max((r["latency_s"] for r in records if r["turn"] == 0), default=None),
        "warm_latency_p50_s": pct([r["latency_s"] for r in warm], 0.5),
        "warm_latency_p90_s": pct([r["latency_s"] for r in warm], 0.9),
        "warm_new_tokens_p50": pct([r["prompt"] - r["cached"] for r in known], 0.5),
        "warm_cache_misses": sum(1 for r in known if r["prompt"] - r["cached"] > 2 * args.out + 3 * args.tool_max + 512),
    }


async def run(agents, sink):
    records, errors = [], []

    async def agent(session, i):
        rng = random.Random(1000 * agents + i)
        await asyncio.sleep(i * args.stagger)
        messages = (opening_tools if args.tools else opening)(rng, i)
        turn = 0
        while True:
            try:
                r = await (request_tools if args.tools else request)(session, messages)
            except Exception as error:  # noqa: BLE001
                errors.append(f"agent {i} turn {turn}: {error}")
                return
            message = r.pop("message", None)
            text = r.pop("text", None)
            r.update(label=args.label, agents=agents, agent=i, turn=turn, at_s=round(time.perf_counter() - begin, 2))
            records.append(r)
            sink.write(json.dumps(r) + "\n")
            sink.flush()
            if r["prompt"] + args.out + args.tool_max * max(1, r.get("tool_calls", 1)) >= args.max:
                return
            if args.tools and message.get("tool_calls"):
                messages = messages + [
                    {"role": "assistant", "content": message.get("content") or "", "tool_calls": message["tool_calls"]}
                ] + [
                    {"role": "tool", "tool_call_id": call["id"],
                     "content": piece(rng, rng.randint(args.tool_min, args.tool_max))}
                    for call in message["tool_calls"]
                ]
            elif args.tools:
                messages = messages + [
                    {"role": "assistant", "content": message.get("content") or ""},
                    {"role": "user", "content": "Keep going: check the next relevant file before you conclude."},
                ]
            else:
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": tool_result(rng)},
                ]
            turn += 1

    timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=64)) as session:
        begin = time.perf_counter()
        await asyncio.gather(*(agent(session, i) for i in range(agents)))
        wall = time.perf_counter() - begin
    return summarize(agents, records, errors, wall)


with open(args.jsonl, "a", encoding="utf-8") as sink:
    for count in (int(n) for n in args.agents.split(",")):
        print("SUMMARY " + json.dumps(asyncio.run(run(count, sink))), flush=True)
