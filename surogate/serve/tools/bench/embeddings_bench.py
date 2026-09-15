"""Measure an OpenAI-compatible ``/v1/embeddings`` endpoint.

Written to compare engines, so it says only what the wire says: it sends batches
of sequences of a known token length and reports how fast the vectors come back.
Any server implementing the endpoint can be measured, which is the point -- the
llama.cpp baseline and our own encoder path run through exactly this harness.

Requests carry token ids rather than text, which the OpenAI schema permits. A
request labelled 512 tokens is then exactly 512 tokens on both sides, and two
engines are not compared partly on their tokenizers.

    python -m surogate.serve.tools.bench.embeddings_bench \\
        --url http://127.0.0.1:8411 --tokens 512 --batch 8 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def build_inputs(tokens: int, count: int, corpus: Sequence[int] | None = None) -> list[list[int]]:
    """`count` distinct sequences of exactly `tokens` ids.

    Both engines receive identical token IDs, excluding tokenizer differences
    from the fixed-length measurement.

    Ids are drawn from a band well clear of the byte-fallback and special ranges,
    so nothing here depends on a particular vocabulary beyond its size.
    """
    if corpus is not None:
        if not corpus or any(type(token) is not int or token < 0 for token in corpus):
            raise ValueError("token corpus must be a nonempty array of nonnegative integer IDs")
        return [[corpus[(index * 7919 + position) % len(corpus)] for position in range(tokens)]
                for index in range(count)]
    pool_size = 200_000
    return [
        [2000 + ((index * 7919 + position * 31) % pool_size) for position in range(tokens)]
        for index in range(count)
    ]


@dataclass
class Result:
    latencies: list[float] = field(default_factory=list)
    vectors: int = 0
    failures: int = 0
    dims: int = 0
    elapsed: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, seconds: float, vectors: int, dims: int) -> None:
        with self.lock:
            self.latencies.append(seconds)
            self.vectors += vectors
            self.dims = dims

    def fail(self) -> None:
        with self.lock:
            self.failures += 1


def post(url: str, model: str | None, inputs: Sequence[Sequence[int]], timeout: float) -> tuple[int, int]:
    fields = {"input": [list(i) for i in inputs], "encoding_format": "float"}
    if model is not None:
        fields["model"] = model
    body = json.dumps(fields).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    data = payload["data"]
    if len(data) != len(inputs) or not data:
        raise ValueError("server returned the wrong number of embeddings")
    dims = len(data[0]["embedding"])
    for index, item in enumerate(data):
        vector = item["embedding"]
        if item["index"] != index or not dims or len(vector) != dims or not all(map(math.isfinite, vector)):
            raise ValueError("server returned an invalid embedding")
    if payload.get("usage", {}).get("prompt_tokens") != sum(map(len, inputs)):
        raise ValueError("server processed a different token count")
    return len(data), dims


def run(url: str, model: str | None, texts: list[list[int]], batch: int, concurrency: int,
        requests: int, timeout: float) -> Result:
    result = Result()
    counter = iter(range(requests))
    counter_lock = threading.Lock()

    def worker() -> None:
        while True:
            with counter_lock:
                index = next(counter, None)
            if index is None:
                return
            chunk = [texts[(index * batch + i) % len(texts)] for i in range(batch)]
            start = time.perf_counter()
            try:
                vectors, dims = post(url, model, chunk, timeout)
            except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError):
                result.fail()
                continue
            result.record(time.perf_counter() - start, vectors, dims)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    began = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    result.elapsed = time.perf_counter() - began
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", help="served model ID; omit to use the running model")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--token-corpus", type=Path, help="JSON array of token IDs; use natural text windows instead of synthetic IDs")
    parser.add_argument("--bos-token-id", type=int, help="prepend this token within --tokens")
    parser.add_argument("--eos-token-id", type=int, help="append this token within --tokens")
    parser.add_argument("--batch", type=int, default=8, help="sequences per request")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    if min(args.tokens, args.batch, args.concurrency, args.requests, args.timeout) <= 0 or args.warmup < 0:
        parser.error("tokens, batch, concurrency, requests and timeout must be positive; warmup must be nonnegative")

    pool = max(args.batch * 4, 32)
    corpus = json.loads(args.token_corpus.read_text()) if args.token_corpus else None
    if corpus is not None and not isinstance(corpus, list):
        parser.error("token corpus must be a JSON array")
    prefix = [] if args.bos_token_id is None else [args.bos_token_id]
    suffix = [] if args.eos_token_id is None else [args.eos_token_id]
    inner_tokens = args.tokens - len(prefix) - len(suffix)
    if inner_tokens < 0 or any(token < 0 for token in prefix + suffix):
        parser.error("special token IDs must be nonnegative and fit within --tokens")
    texts = [prefix + ids + suffix for ids in build_inputs(inner_tokens, pool, corpus)]

    if args.warmup:
        warmup = run(args.url, args.model, texts, args.batch, args.concurrency, args.warmup, args.timeout)
        if warmup.failures:
            print(f"warmup failed: {warmup.failures} requests")
            return 1

    result = run(args.url, args.model, texts, args.batch, args.concurrency,
                 args.requests, args.timeout)
    if not result.latencies:
        print(f"all {result.failures} requests failed")
        return 1

    ordered = sorted(result.latencies)
    elapsed = result.elapsed
    print(json.dumps({
        "label": args.label,
        "tokens": args.tokens,
        "token_corpus": str(args.token_corpus) if args.token_corpus else None,
        "bos_token_id": args.bos_token_id,
        "eos_token_id": args.eos_token_id,
        "batch": args.batch,
        "concurrency": args.concurrency,
        "requests_ok": len(result.latencies),
        "requests_failed": result.failures,
        "dims": result.dims,
        "elapsed_s": round(elapsed, 3),
        "embeddings_per_s": round(result.vectors / elapsed, 1),
        "tokens_per_s": round(result.vectors * args.tokens / elapsed, 1),
        "latency_ms": {
            "mean": round(1000 * statistics.fmean(ordered), 2),
            "p50": round(1000 * ordered[len(ordered) // 2], 2),
            "p99": round(1000 * ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 2),
        },
    }))
    return int(result.failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
