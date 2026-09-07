"""Perplexity of the served model on a text corpus, chunked exactly as llama-perplexity chunks it.

llama-perplexity tokenises the whole corpus, cuts the token stream into windows of ``ctx``
tokens, and scores the second half of every window against the first half as context. The same
windows are built here from llama.cpp's own tokeniser (a ``llama-server`` on the same GGUF
answers ``/tokenize`` and ``/detokenize``), so the numbers are comparable window for window.

The engine side is its NLL probe: with ``SUROGATE_SERVE_NLL_DUMP=<file>`` set, an eager prefill
appends one line per prompt position holding the token, the token that follows, the negative
log-likelihood of that following token, and the token ranked first. Each window is sent as one
user message, and ``SUROGATE_SERVE_RAW_PROMPT=1`` makes the engine take that message as the bare
prompt: inside a chat turn a thinking model scores prose entirely differently (it expects the
user turn to end, and lands two orders of magnitude off llama-perplexity's number), so the raw
text is the only condition comparable with the reference. The probe's
rows are matched against the window's token ids, and only positions whose token and successor
both match are scored, so a tokenisation difference at a window edge is excluded rather than
mis-scored.

    SUROGATE_SERVE_NLL_DUMP=/tmp/nll.txt SUROGATE_SERVE_RAW_PROMPT=1 SUROGATE_SERVE_PREFILL_GRAPH=0 \\
        SUROGATE_SERVE_NO_MIXED_GRAPH=1 surogate-engine <artifact> --port 8999 ...
    python -m surogate.serve.tools.eval.perplexity --corpus wiki.test.raw --dump /tmp/nll.txt \\
        --prepare windows.json --llama-server http://127.0.0.1:8998
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
from typing import Any


def _post(url: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def prepare_windows(corpus: str, llama_server: str, ctx: int) -> list[dict[str, Any]]:
    """Token windows and their text, from llama.cpp's tokeniser."""
    text = open(corpus, encoding="utf-8").read()
    tokens = _post(f"{llama_server}/tokenize", {"content": text, "add_special": False})["tokens"]
    windows = []
    for start in range(0, len(tokens) - ctx + 1, ctx):
        ids = tokens[start:start + ctx]
        content = _post(f"{llama_server}/detokenize", {"tokens": ids})["content"]
        windows.append({"ids": ids, "text": content})
    return windows


def read_dump_rows(path: str, offset: int) -> tuple[list[tuple[int, int, float]], int]:
    rows: list[tuple[int, int, float]] = []
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        rows.append((int(fields[0]), int(fields[1]), float(fields[2])))
    return rows, offset + len(data)


def locate(rows: list[tuple[int, int, float]], ids: list[int]) -> int | None:
    """Index of the row where the window's ids begin.

    Found from sixteen tokens well inside the window rather than its first ones: a window cut
    mid-word re-merges differently when its text is tokenised again, so the first few ids are the
    ones most likely to differ, and the second half is what gets scored anyway.
    """
    anchor = 64
    probe = ids[anchor:anchor + 16]
    for start in range(anchor, len(rows) - len(probe) + 1):
        if all(rows[start + i][0] == probe[i] for i in range(len(probe))):
            return start - anchor
    return None


def score_window(rows: list[tuple[int, int, float]], ids: list[int], ctx: int) -> tuple[float, int, int]:
    """(sum of nll, scored positions, mismatched positions) over the window's second half."""
    start = locate(rows, ids)
    if start is None:
        return 0.0, 0, ctx - 1 - ctx // 2
    total = 0.0
    scored = 0
    mismatched = 0
    for position in range(ctx // 2, ctx - 1):
        row = start + position
        if row < len(rows) and rows[row][0] == ids[position] and rows[row][1] == ids[position + 1]:
            total += rows[row][2]
            scored += 1
        else:
            mismatched += 1
    return total, scored, mismatched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", help="text file (e.g. wikitext-2 test)")
    parser.add_argument("--llama-server", default="http://127.0.0.1:8998",
                        help="llama-server on the same GGUF, for /tokenize and /detokenize")
    parser.add_argument("--prepare", required=True, help="windows file: written from --corpus, else read")
    parser.add_argument("--engine", default="http://127.0.0.1:8999")
    parser.add_argument("--dump", required=True, help="the file SUROGATE_SERVE_NLL_DUMP names")
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--chunks", type=int, default=0, help="windows to score (0: all)")
    parser.add_argument("--model", default=None, help="served model id (default: the first of /v1/models)")
    args = parser.parse_args()

    if args.corpus:
        windows = prepare_windows(args.corpus, args.llama_server, args.ctx)
        json.dump(windows, open(args.prepare, "w"))
        print(f"prepared {len(windows)} windows of {args.ctx} tokens -> {args.prepare}")
    else:
        windows = json.load(open(args.prepare))
    if args.chunks:
        windows = windows[:args.chunks]

    if args.model is None:
        with urllib.request.urlopen(f"{args.engine}/v1/models", timeout=30) as response:
            args.model = json.loads(response.read().decode("utf-8"))["data"][0]["id"]
    open(args.dump, "a").close()
    offset = os.path.getsize(args.dump)
    total = 0.0
    total_sq = 0.0
    scored = 0
    mismatched = 0
    started = time.time()
    for index, window in enumerate(windows, 1):
        _post(f"{args.engine}/v1/chat/completions", {
            "model": args.model, "max_tokens": 1, "temperature": 0.0,
            "messages": [{"role": "user", "content": window["text"]}]})
        rows, offset = read_dump_rows(args.dump, offset)
        window_total, window_scored, window_mismatched = score_window(rows, window["ids"], args.ctx)
        if window_scored:
            start = locate(rows, window["ids"])
            for position in range(args.ctx // 2, args.ctx - 1):
                row = start + position
                if row < len(rows) and rows[row][0] == window["ids"][position] and rows[row][1] == window["ids"][position + 1]:
                    total_sq += rows[row][2] ** 2
        total += window_total
        scored += window_scored
        mismatched += window_mismatched
        mean = total / max(scored, 1)
        print(f"[{index}] {math.exp(mean):.4f}", end=" ", flush=True)
        if index % 8 == 0:
            print(flush=True)
    print()
    if not scored:
        print("no positions scored: is SUROGATE_SERVE_NLL_DUMP set and prefill eager?", file=sys.stderr)
        return 1
    mean = total / scored
    variance = max(total_sq / scored - mean * mean, 0.0)
    ppl = math.exp(mean)
    print(f"Final estimate: PPL = {ppl:.4f} +/- {ppl * math.sqrt(variance / scored):.4f} "
          f"over {scored} positions in {len(windows)} windows "
          f"({mismatched} excluded at window edges) in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
