"""Measure a running server with staggered, persistent HTTP clients.

Run with ``python -m surogate.serve.tools.bench.serve_http_bench --help``.
Start generation servers with --no-prefix-reuse. Warmup and model loading are
excluded. Throughput counts whole requests completed inside the fixed measurement
window; in-flight requests drain afterward and are excluded from that window.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import itertools
import json
import math
import statistics
import time

import aiohttp

PROSE = """A database stores information in pages and keeps frequently used pages in memory.
When a client sends a query, the server chooses an execution plan and returns the matching rows.
An index can reduce the amount of data read, but maintaining it also adds work to each update.
To compare two designs, measure both throughput and response latency under the same workload.
Weather observations provide another example: a sensor records temperature and pressure each hour.
Researchers check for missing readings before computing daily averages and looking for seasonal trends.
They retain the original measurements so that another team can reproduce the analysis.
In a garden, different plants need different amounts of water, sunlight, and space to grow.
A careful planting plan leaves enough room for roots and makes harvesting easier later in the year.
Software teams face similar tradeoffs when they choose how much work to run concurrently.
"""


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


async def benchmark(args: argparse.Namespace) -> dict:
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(32, args.concurrency))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        url = args.url.rstrip("/")
        pool = []
        for index in range(max(32, args.concurrency)):
            if args.embedding:
                pool.append([2000 + (index * 7919 + p * 31) % 200000 for p in range(args.prompt_tokens)])
            else:
                # Tokenize prose once, then submit exact-length token sequences.
                # Distinct prefixes plus disabled reuse prevent cache hits.
                text = f"Document {index:06d}. " + PROSE * (args.prompt_tokens // 100 + 2)
                async with session.post(
                    url + "/tokenize",
                    json={
                        "model": args.model,
                        "prompt": text,
                    },
                ) as response:
                    response.raise_for_status()
                    tokens = (await response.json())["tokens"]
                if len(tokens) < args.prompt_tokens:
                    raise RuntimeError("tokenized corpus is shorter than the requested prompt")
                pool.append(tokens[: args.prompt_tokens])

        counter = itertools.count()

        async def request() -> dict:
            index = next(counter)
            started = time.perf_counter()
            first = None
            if args.embedding:
                inputs = [pool[(index * args.batch + i) % len(pool)] for i in range(args.batch)]
                async with session.post(
                    url + "/v1/embeddings",
                    json={
                        "model": args.model,
                        "input": inputs,
                    },
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
                vectors = payload["data"]
                if len(vectors) != args.batch:
                    raise RuntimeError("embedding count differs from input count")
                dims = len(vectors[0]["embedding"])
                if not dims or any(
                    len(v["embedding"]) != dims or not all(math.isfinite(x) for x in v["embedding"]) for v in vectors
                ):
                    raise RuntimeError("invalid embedding vectors")
                prompt_tokens, completion_tokens = args.batch * args.prompt_tokens, 0
            else:
                body = {
                    "model": args.model,
                    "tokens": pool[index % len(pool)],
                    "messages": [{"role": "user", "content": "Continue the document."}],
                    "max_tokens": args.output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "return_token_ids": True,
                }
                usage, seen_tokens, finished = None, 0, False
                async with session.post(url + "/v1/chat/completions", json=body) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if not line.startswith(b"data: "):
                            continue
                        if line.strip() == b"data: [DONE]":
                            finished = True
                            break
                        chunk = json.loads(line[6:])
                        if "error" in chunk:
                            raise RuntimeError(str(chunk["error"]))
                        for choice in chunk.get("choices", []):
                            ids = choice.get("token_ids", [])
                            delta = choice.get("delta") or {}
                            if first is None and (
                                delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                            ):
                                first = time.perf_counter()
                            seen_tokens += len(ids)
                        usage = chunk.get("usage") or usage
                if not finished or usage is None or first is None:
                    raise RuntimeError("incomplete stream or missing token/usage events")
                prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
                if prompt_tokens != args.prompt_tokens or completion_tokens != args.output_tokens:
                    raise RuntimeError(f"unexpected token accounting: {usage}")
                if seen_tokens != completion_tokens:
                    raise RuntimeError("stream token count differs from final usage")
                dims = 0
            ended = time.perf_counter()
            return dict(
                started=started,
                ended=ended,
                latency=ended - started,
                ttft=None if first is None else first - started,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                dimensions=dims,
            )

        # Each client completes a warmup before the timed settling period begins.
        async def warm_client(index: int) -> None:
            await asyncio.sleep(index / args.concurrency)
            await request()

        await asyncio.gather(*(warm_client(i) for i in range(args.concurrency)))
        begin = time.perf_counter() + args.warmup
        end = begin + args.seconds
        records, errors = [], []
        drained = 0

        async def worker(index: int) -> None:
            nonlocal drained
            await asyncio.sleep(index / args.concurrency * min(args.stagger, args.warmup))
            while time.perf_counter() < end:
                try:
                    result = await request()
                    if begin <= result["ended"] < end:
                        records.append(result)
                    elif result["ended"] >= end:
                        drained += 1
                except Exception as error:
                    errors.append(str(error))
                    return

        measured_at = datetime.datetime.now(datetime.UTC).isoformat()
        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
        if not records:
            raise RuntimeError(f"no requests completed in the measurement window: {errors}")
        prompts = sum(r["prompt_tokens"] for r in records)
        completions = sum(r["completion_tokens"] for r in records)
        latencies = [r["latency"] for r in records]
        result = {
            "label": args.label,
            "measured_at": measured_at,
            "endpoint": "embeddings" if args.embedding else "chat/completions",
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": 0 if args.embedding else args.output_tokens,
            "batch": args.batch,
            "concurrency": args.concurrency,
            "window_s": args.seconds,
            "warmup_s": args.warmup,
            "stagger_s": min(args.stagger, args.warmup),
            "requests_ok": len(records),
            "requests_failed": len(errors),
            "errors": errors[:10],
            "drained_requests": drained,
            "prompt_tok_s": prompts / args.seconds,
            "decode_tok_s": completions / args.seconds,
            "total_tok_s": (prompts + completions) / args.seconds,
            "latency_p50_ms": statistics.median(latencies) * 1000,
            "latency_p99_ms": percentile(latencies, 0.99) * 1000,
        }
        if args.embedding:
            result.update(vectors_s=len(records) * args.batch / args.seconds, dimensions=records[0]["dimensions"])
        else:
            result.update(
                ttft_p50_ms=statistics.median(r["ttft"] for r in records) * 1000,
                ttft_p99_ms=percentile([r["ttft"] for r in records], 0.99) * 1000,
            )
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="bench")
    parser.add_argument("--label", default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--warmup", type=float, default=15)
    parser.add_argument("--stagger", type=float, default=3, help="spread client starts over this many seconds")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--batch", type=int, default=1, help="embedding sequences per request")
    args = parser.parse_args()
    if (
        min(
            args.concurrency,
            args.prompt_tokens,
            args.output_tokens,
            args.batch,
            args.seconds,
            args.warmup,
            args.timeout,
            args.stagger,
        )
        <= 0
    ):
        parser.error("counts and durations must be positive")
    result = asyncio.run(benchmark(args))
    print(json.dumps(result, indent=2))
    return 1 if result["requests_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
