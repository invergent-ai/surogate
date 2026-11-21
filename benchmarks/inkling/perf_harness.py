#!/usr/bin/env python3
"""ENGINE-INKLING lane h (M4): the performance harness for Inkling-2-Mini serving, for OUR engine and the vLLM image.

Same requests for both engines over the OpenAI chat API, DISTINCT prompts per stream (each stream starts at its own
offset of a cyclic text corpus, so no two streams share a prefix or a token sequence). Measures:
  decode   tok/s at S streams x context C (prompt of ~C tokens, G generated tokens, ignore_eos): per-stream and
           aggregate decode rate (aggregate over the window where every stream is decoding), TTFT; the prompts are
           pre-filled once (max_tokens 1) so the measured requests hit the prefix cache and decode together
  prefill  cold prompts of N tokens, 1 output token: prefill seconds and tok/s
  cached   TTFT of (prefix P + suffix of S tokens) after P was served once, vs a cold S-token prompt
  swap     POST /v1/load_lora_adapter latency while one stream decodes: the stream's largest inter-token gap
  mtp      our engine's --request-log-jsonl records: rounds, drafted, accepted per position, tokens per round
Every row is printed as one JSON line; with --bounds (perf/results/bounds-*.json from perf/bounds.py) decode rows
also carry the re-derived bound, target and the measured fraction of the bound.

RULE (DESIGN 4 / LANE-RULES 9): no speed number of OUR engine counts before logit parity holds on the measured path.
For --engine ours the harness refuses to run without --parity-record naming the PASS record of that path's gate.

Examples:
  perf_harness.py --engine vllm --base http://127.0.0.1:18471 --model inkling --corpus a.txt b.txt \
      --decode 1x8k,8x8k --prefill 97k --out vllm.jsonl
  perf_harness.py --engine ours --parity-record rl/engine-inkling-2026-11-21/gate/<path>-PASS.md --base ... \
      --decode 1x8k,1x64k,1x262k,4x8k,8x8k,8x64k --prefill 8k,32k,97k,262k --cached 64k+256 \
      --request-log /path/requests.jsonl --bounds perf/results/bounds-measured-1121.json --out ours.jsonl
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

K = 1024


def parse_tokens(s):
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * K) if s[:-1] not in ("97",) else 97000
    return int(s)


class Corpus:
    """Cyclic text; slice(offset_frac, n_chars) never repeats a prefix across distinct offsets."""

    def __init__(self, paths):
        self.text = "".join(open(p, encoding="utf-8", errors="replace").read() for p in paths)
        if len(self.text) < 10000:
            raise SystemExit("corpus too small")
        self.chars_per_token = 3.6  # calibrated against the server on first use

    def slice(self, start, n_chars):
        t, L = self.text, len(self.text)
        start %= L
        out = []
        while n_chars > 0:
            piece = t[start:start + n_chars]
            out.append(piece)
            n_chars -= len(piece)
            start = 0
        return "".join(out)

    def prompt(self, stream, n_streams, tokens, salt=0):
        n_chars = int(tokens * self.chars_per_token)
        off = (stream * len(self.text)) // max(1, n_streams) + salt * 7919
        return "Stream %d.\n" % stream + self.slice(off, n_chars) + "\n\nContinue this text in detail."


class Client:
    def __init__(self, base, model):
        self.base, self.model = base.rstrip("/"), model

    def post(self, path, body, timeout=600):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()

    def stream(self, prompt, max_tokens, out, temperature=1.0):
        """Streams one chat completion; records send / first-token / per-chunk / end times and usage."""
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "ignore_eos": True, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        out["t0"] = time.time()
        stamps, usage = [], None
        with urllib.request.urlopen(req, timeout=7200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                ch = obj.get("choices") or []
                if ch:
                    d = ch[0].get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        stamps.append(time.time())
        out["t_end"] = time.time()
        out["stamps"] = stamps
        out["t_first"] = stamps[0] if stamps else out["t_end"]
        out["usage"] = usage or {"prompt_tokens": None, "completion_tokens": len(stamps)}
        return out


def parallel(client, prompts, max_tokens):
    outs = [dict() for _ in prompts]
    ths = [threading.Thread(target=client.stream, args=(p, max_tokens, o)) for p, o in zip(prompts, outs)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return outs


def emit(fh, **row):
    row["time"] = time.strftime("%H:%M:%S")
    line = json.dumps(row)
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def decode_summary(outs):
    t_all_first = max(o["t_first"] for o in outs)
    t_end = max(o["t_end"] for o in outs)
    per = [(o["usage"]["completion_tokens"] - 1) / max(1e-9, o["t_end"] - o["t_first"]) for o in outs]
    if len(outs) == 1:
        agg = per[0]
    else:  # tokens emitted after every stream has started decoding, over that window
        agg = sum(sum(1 for s in o["stamps"] if s > t_all_first) for o in outs) / max(1e-9, t_end - t_all_first)
    return dict(aggregate_tok_s=round(agg, 1), per_stream_tok_s=[round(x, 1) for x in per],
                ttft_s=[round(o["t_first"] - o["t0"], 3) for o in outs],
                prompt_tokens=[o["usage"]["prompt_tokens"] for o in outs],
                completion_tokens=[o["usage"]["completion_tokens"] for o in outs])


def bound_lookup(bounds, streams, ctx):
    if not bounds:
        return {}
    for r in bounds.get("decode", []):
        if r["batch"] == streams and r["ctx"] == ctx:
            return dict(bound_tok_s=round(r["bound"], 1), target_tok_s=round(r["target"], 1))
    return {}


def read_request_log(path, since):
    """Our engine's --request-log-jsonl: speculative stats of requests finished after `since` (best effort)."""
    rows = []
    if not path or not os.path.exists(path):
        return rows
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        spec = r.get("speculative") or (r.get("metrics") or {}).get("speculative")
        if spec and spec.get("rounds"):
            rows.append(spec)
    return rows[-since:] if since else rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", choices=["ours", "vllm"], required=True)
    ap.add_argument("--parity-record", help="PASS record of the logit-parity gate for the measured path (ours)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--decode", default="", help="comma list of SxC, e.g. 1x8k,8x64k")
    ap.add_argument("--gen", type=int, default=512, help="generated tokens per decode stream")
    ap.add_argument("--no-prewarm", action="store_true",
                    help="do not pre-fill each decode row's prompts (default: one max_tokens=1 pass first)")
    ap.add_argument("--prefill", default="", help="comma list of cold prompt sizes, e.g. 8k,97k")
    ap.add_argument("--cached", default="", help="comma list of P+S, e.g. 64k+256")
    ap.add_argument("--swap", nargs=2, metavar=("NAME", "PATH"), help="adapter to load while a stream decodes")
    ap.add_argument("--request-log", help="our engine's --request-log-jsonl (MTP stats)")
    ap.add_argument("--bounds", help="perf/bounds.py --json output")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    if a.engine == "ours":
        if not a.parity_record or not os.path.exists(a.parity_record) or "PASS" not in open(a.parity_record).read():
            raise SystemExit("REFUSED: our engine's speed numbers need a parity PASS record for this path "
                             "(--parity-record <file containing PASS>); rule: no speed number before parity")
    fh = open(a.out, "a") if a.out else None
    bounds = json.load(open(a.bounds)) if a.bounds else None
    c = Client(a.base, a.model)
    corpus = Corpus(a.corpus)
    meta = dict(engine=a.engine, base=a.base, model=a.model, label=a.label, parity_record=a.parity_record)

    # warm-up + chars-per-token calibration (one 4k-token prompt, 1 output token)
    o = c.stream(corpus.prompt(0, 1, 4096, salt=999), 1, {})
    pt = o["usage"]["prompt_tokens"]
    if pt:
        corpus.chars_per_token *= 4096 / pt
    emit(fh, kind="calibration", prompt_tokens=pt, chars_per_token=round(corpus.chars_per_token, 3), **meta)
    parallel(c, [corpus.prompt(i, 8, 64, salt=555) for i in range(8)], 16)

    salt = 1
    for item in filter(None, a.decode.split(",")):
        s, ctx = item.lower().split("x")
        s, ctx = int(s), parse_tokens(ctx)
        prompts = [corpus.prompt(i, s, ctx, salt) for i in range(s)]
        if not a.no_prewarm:  # fill the prefix cache first, so the decode window holds no prefill of other streams
            parallel(c, prompts, 1)
        log_before = len(read_request_log(a.request_log, 0))
        outs = parallel(c, prompts, a.gen)
        salt += 1
        row = dict(kind="decode", streams=s, ctx=ctx, gen=a.gen, **decode_summary(outs), **meta)
        b = bound_lookup(bounds, s, ctx)
        if b:
            row.update(b, fraction_of_bound=round(row["aggregate_tok_s"] / b["bound_tok_s"], 3),
                       meets_target=row["aggregate_tok_s"] >= b["target_tok_s"])
        spec = read_request_log(a.request_log, 0)[log_before:]
        if spec:
            rounds = sum(x["rounds"] for x in spec)
            acc = sum(x["accepted_tokens"] for x in spec)
            row["mtp"] = dict(rounds=rounds, drafted=sum(x["drafted_tokens"] for x in spec), accepted=acc,
                              tokens_per_round=round(1 + acc / max(1, rounds), 3),
                              accepted_per_position=[sum(v) for v in zip(*[x["accepted_per_position"] for x in spec
                                                                           if x.get("accepted_per_position")])])
        emit(fh, **row)

    for item in filter(None, a.prefill.split(",")):
        n = parse_tokens(item)
        o = c.stream(corpus.prompt(0, 1, n, salt), 1, {})
        salt += 1
        pt = o["usage"]["prompt_tokens"]
        dt = o["t_first"] - o["t0"]
        emit(fh, kind="prefill", prompt_tokens=pt, seconds=round(dt, 3), tok_s=round((pt or n) / dt, 1), **meta)

    for item in filter(None, a.cached.split(",")):
        p, sfx = item.split("+")
        p, sfx = parse_tokens(p), parse_tokens(sfx)
        prefix = corpus.prompt(0, 1, p, salt)
        suffix = corpus.slice(123457 + salt, int(sfx * corpus.chars_per_token))
        salt += 1
        first = c.stream(prefix, 1, {})
        hit = c.stream(prefix + suffix, 1, {})
        cold = c.stream(corpus.prompt(0, 1, sfx, salt + 1000), 1, {})
        emit(fh, kind="cached_ttft", prefix_tokens=first["usage"]["prompt_tokens"],
             total_tokens=hit["usage"]["prompt_tokens"], ttft_first_s=round(first["t_first"] - first["t0"], 3),
             ttft_hit_s=round(hit["t_first"] - hit["t0"], 3), ttft_cold_suffix_only_s=round(cold["t_first"] - cold["t0"], 3),
             **meta)

    if a.swap:
        out = {}
        th = threading.Thread(target=c.stream, args=(corpus.prompt(3, 4, 8192, salt), 1024, out))
        th.start()
        time.sleep(3.0)
        t0 = time.time()
        st, body = c.post("/v1/load_lora_adapter", {"lora_name": a.swap[0], "lora_path": a.swap[1]})
        t_swap = time.time() - t0
        th.join()
        gaps = [b - a_ for a_, b in zip(out["stamps"], out["stamps"][1:])]
        gaps_sorted = sorted(gaps)
        emit(fh, kind="swap", status=st, swap_s=round(t_swap, 4), stream_max_gap_ms=round(1e3 * max(gaps), 2),
             stream_median_gap_ms=round(1e3 * gaps_sorted[len(gaps) // 2], 2), **meta)
    emit(fh, kind="done", **meta)


if __name__ == "__main__":
    main()
