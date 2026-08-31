"""Measure an OpenAI-compatible ``/v1/embeddings`` endpoint.

Written to compare engines, so it says only what the wire says: it sends batches
of texts of a known token length and reports how fast the vectors come back. Any
server implementing the endpoint can be measured, which is the point -- the
llama.cpp baseline and our own encoder path run through exactly this harness.

Token lengths are exact, not approximate: texts are built by decoding real token
ids from the model's own tokenizer, so a request labelled 512 tokens is 512
tokens on both sides of the comparison.

    python -m surogate.serve.tools.bench.embeddings_bench \\
        --url http://127.0.0.1:8411 --tokens 512 --batch 8 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

DEFAULT_TOKENIZER = (
    "~/.cache/huggingface/hub/models--google--embeddinggemma-300m/"
    "snapshots/57c266a740f537b4dc058e1b0cda161fd15afa75"
)


def build_texts(tokenizer_dir: str, tokens: int, count: int) -> list[str]:
    """`count` distinct texts, each exactly `tokens` tokens once re-encoded.

    Built from the tokenizer's own vocabulary rather than from prose, so the
    length is exact and the same on any engine sharing the tokenizer.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(Path(tokenizer_dir).expanduser()))
    # A deterministic slice of ordinary word-like tokens; ids well past the byte
    # and special ranges decode to printable pieces.
    pool = list(range(2000, 2000 + max(tokens * 4, 4096)))
    texts = []
    for index in range(count):
        offset = (index * 97) % (len(pool) - tokens)
        ids = pool[offset : offset + tokens]
        text = tok.decode(ids)
        # Re-encoding rarely lands exactly; trim or pad in token space until it does.
        for _ in range(8):
            actual = len(tok(text)["input_ids"])
            if actual == tokens:
                break
            ids = ids[: max(1, len(ids) - (actual - tokens))] if actual > tokens else ids + pool[:1] * (tokens - actual)
            text = tok.decode(ids)
        texts.append(text)
    return texts


@dataclass
class Result:
    latencies: list[float] = field(default_factory=list)
    vectors: int = 0
    failures: int = 0
    dims: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, seconds: float, vectors: int, dims: int) -> None:
        with self.lock:
            self.latencies.append(seconds)
            self.vectors += vectors
            self.dims = dims

    def fail(self) -> None:
        with self.lock:
            self.failures += 1


def post(url: str, model: str, texts: Sequence[str], timeout: float) -> tuple[int, int]:
    body = json.dumps({"model": model, "input": list(texts)}).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    data = payload["data"]
    return len(data), len(data[0]["embedding"])


def run(url: str, model: str, texts: list[str], batch: int, concurrency: int,
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
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
                result.fail()
                continue
            result.record(time.perf_counter() - start, vectors, dims)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    began = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    result.elapsed = time.perf_counter() - began  # type: ignore[attr-defined]
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="embeddinggemma")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8, help="texts per request")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    pool = max(args.batch * 4, 32)
    texts = build_texts(args.tokenizer, args.tokens, pool)

    if args.warmup:
        run(args.url, args.model, texts, args.batch, args.concurrency, args.warmup, args.timeout)

    result = run(args.url, args.model, texts, args.batch, args.concurrency,
                 args.requests, args.timeout)
    if not result.latencies:
        print(f"all {result.failures} requests failed")
        return 1

    ordered = sorted(result.latencies)
    elapsed = result.elapsed  # type: ignore[attr-defined]
    print(json.dumps({
        "label": args.label,
        "tokens": args.tokens,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
