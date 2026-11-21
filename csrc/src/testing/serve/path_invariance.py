#!/usr/bin/env python3
"""Path-invariance gate (lane f1, rl/engine-inkling-2026-11-21 FEASIBILITY 10:05 item 3).

The log-probs of the same tokens must not depend on the path the engine took to the prefix. This harness reaches one
token sequence S = P + X by several paths and compares the log-probs of the K tokens Z generated greedily after it,
BITWISE (the server prints float log-probs at full precision). Target-agnostic: it drives `surogate-engine` over HTTP
and starts a fresh engine per path, so every cache starts cold. Stdlib only; runs on the GPU box.

Paths
  prefill   prompt S on a fresh engine (one prefill when |S| fits the engine's prefill chunk): the REFERENCE
  decode    prompt P, max_tokens |X| + K: X and Z come from decode rounds (X is P's greedy continuation, so S = P + X)
  verify    as decode, on an engine started with --verify-args (MTP / DFlash): X and Z from verify rounds
  cut@b     prompt S[:b] (max_tokens 1), then S: the suffix prefills from b (a GPU prefix hit = a chunk cut at b)
  host@b    as cut@b on an engine with a host tier and a KV capacity of --host-capacity tokens, with one long
            unrelated prompt in between that needs the retained lane's pages: the prefix spills to the host tier and
            S restores it from there (checked: GET /kv_stats host "restores" grows across S's request)
Exit status 0 iff every path equals the reference bitwise; the JSON report has the per-path detail.

Example (gpubox2, Gemma 3 270M fixture):
  path_invariance.py --engine ~/en-f1/bin3/surogate-engine --artifact ~/en-f1/fixtures/gemma-3-270m-it.sinfer \
      --card 7 --port 18761 --material material.txt --engine-args "--kv-cache-dtype fp8" --out report.json
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TOP = 20


def post(port, path, body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Engine:
    def __init__(self, args, tag, extra_args="", lanes=None, host_gb=None, max_model_len=None):
        self.port = args.port
        self.log_path = os.path.join(args.workdir, f"engine-{tag}.log")
        cmd = [args.engine, args.artifact, "--port", str(args.port), "--served-model-name", "m",
               "--max-model-len", str(max_model_len or args.max_model_len), "--max-num-seqs", str(lanes or args.lanes)]
        if host_gb is not None:
            cmd += ["--prefix-cache-host-gb", str(host_gb)]
        cmd += shlex.split(args.engine_args) + shlex.split(extra_args)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.card),
                   LD_LIBRARY_PATH=os.path.dirname(os.path.abspath(args.engine)))
        for item in args.env:
            key, _, value = item.partition("=")
            env[key] = value
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(300):
            if self.proc.poll() is not None:
                raise RuntimeError(f"engine {tag} exited at startup; see engine-{tag}.log")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2).read()
                return
            except OSError:
                time.sleep(1)
        raise RuntimeError(f"engine {tag} did not become ready")

    def complete(self, prompt, max_tokens):
        resp = post(self.port, "/v1/completions", {"model": "m", "prompt": prompt, "max_tokens": max_tokens,
                                                   "temperature": 0, "logprobs": TOP, "return_token_ids": True})
        choice = resp["choices"][0]
        lp = choice.get("logprobs") or {}
        return {"prompt_ids": resp.get("prompt_token_ids"), "ids": choice.get("token_ids"),
                "texts": lp.get("tokens"), "logprobs": lp.get("token_logprobs"), "top": lp.get("top_logprobs"),
                "cached": ((resp.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")}

    def host_restores(self):
        """Host-tier restores so far, from GET /kv_stats (any nested "restores" counter; the request log line does
        not name a host restore: it prints the reuse PATH, e.g. append_frontier, for GPU and host hits alike)."""
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/kv_stats", timeout=30) as r:
            stats = json.load(r)
        total, stack = 0, [stats]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "restores" and isinstance(value, (int, float)):
                        total += value
                    else:
                        stack.append(value)
            elif isinstance(node, list):
                stack.extend(node)
        return total

    def tokenize(self, text):
        resp = post(self.port, "/tokenize", {"model": "m", "prompt": text, "with_token_strings": True})
        return resp["tokens"], resp.get("token_strs")

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def tail(result, k):
    """The last k generated tokens: ids, chosen log-probs, top-k alternatives."""
    return {"ids": result["ids"][-k:], "logprobs": result["logprobs"][-k:], "top": result["top"][-k:],
            "cached": result["cached"]}


def prefix_text(engine, s_text, s_ids, b):
    """A character prefix of S whose tokenization is exactly s_ids[:b'] with b' as close to b as possible.
    Cuts are tried before spaces (a BPE token usually starts with one), nearest to the estimated position first."""
    estimate = int(b * len(s_text) / max(1, len(s_ids)))
    candidates = sorted((c for c in range(max(1, estimate - 400), min(len(s_text), estimate + 400))
                         if s_text[c] == " "), key=lambda c: abs(c - estimate))
    best = None
    for c in candidates[:120]:
        ids, _ = engine.tokenize(s_text[:c])
        if ids == s_ids[:len(ids)] and (best is None or abs(len(ids) - b) < abs(best[0] - b)):
            best = (len(ids), s_text[:c])
            if best[0] == b:
                break
    if best is None:
        raise RuntimeError(f"no text prefix tokenizes to a prefix of S near {b}")
    return best


def compare(ref, got):
    out = {"tokens_equal": ref["ids"] == got["ids"], "logprobs_bitwise": ref["logprobs"] == got["logprobs"],
           "top_bitwise": ref["top"] == got["top"], "cached": got.get("cached")}
    diffs = [i for i, (a, b) in enumerate(zip(ref["logprobs"], got["logprobs"])) if a != b]
    out["first_logprob_mismatch"] = diffs[0] if diffs else None
    out["max_abs_logprob_diff"] = max((abs(a - b) for a, b in zip(ref["logprobs"], got["logprobs"])), default=0.0)
    out["equal"] = out["tokens_equal"] and out["logprobs_bitwise"] and out["top_bitwise"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--material", required=True, help="text the prompt P is cut from")
    ap.add_argument("--engine-args", default="", help="flags every engine gets (e.g. --kv-cache-dtype fp8)")
    ap.add_argument("--verify-args", default="", help="flags of the speculative engine for the verify path")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for every engine")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--lanes", type=int, default=8, help="--max-num-seqs of every engine (Gemma 3 270M needs >= 8 "
                    "for its CUDA graph allowance on the base engine too)")
    ap.add_argument("--prompt-chars", type=int, default=6000, help="length of P in characters")
    ap.add_argument("--x-tokens", type=int, default=48, help="|X|: tokens decoded after P that become part of S")
    ap.add_argument("--k-tokens", type=int, default=16, help="|Z|: tokens whose log-probs are compared")
    ap.add_argument("--cuts", default="512,1061", help="prefix-hit cut points b (tokens)")
    ap.add_argument("--host-capacity", type=int, default=4096,
                    help="max-model-len = kv-capacity of the host-path engine (small, so the evicting prompt spills)")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    material = " ".join(open(args.material, encoding="utf-8").read().split())
    p_text = (material * (1 + args.prompt_chars // max(1, len(material))))[:args.prompt_chars]
    k, nx = args.k_tokens, args.x_tokens
    report = {"args": vars(args), "paths": {}}

    # decode: P, then X and Z from decode rounds; S = P + X must re-tokenize to P_ids + X_ids.
    eng = Engine(args, "decode")
    try:
        dec = eng.complete(p_text, nx + k)
        x_texts = dec["texts"][:nx]
        s_text = p_text + "".join(x_texts)
        s_ids, _ = eng.tokenize(s_text)
        expected = dec["prompt_ids"] + dec["ids"][:nx]
        if s_ids != expected:
            report["error"] = "S = P + X does not re-tokenize to P_ids + X_ids (boundary merge); change --x-tokens"
            json.dump(report, open(args.out, "w"), indent=1)
            print(report["error"])
            return 2
        cuts = []
        for b in (int(c) for c in args.cuts.split(",") if c):
            cuts.append(prefix_text(eng, s_text, s_ids, min(b, len(s_ids) - 1)))
    finally:
        eng.close()
    report["s_tokens"] = len(s_ids)
    decode = tail(dec, k)

    eng = Engine(args, "prefill")
    try:
        ref = tail(eng.complete(s_text, k), k)
    finally:
        eng.close()
    report["reference"] = ref
    report["paths"]["decode"] = compare(ref, decode)

    if args.verify_args:
        eng = Engine(args, "verify", extra_args=args.verify_args)
        try:
            report["paths"]["verify"] = compare(ref, tail(eng.complete(p_text, nx + k), k))
        finally:
            eng.close()

    for cut, text in cuts:
        eng = Engine(args, f"cut{cut}")
        try:
            eng.complete(text, 1)
            report["paths"][f"cut@{cut}"] = compare(ref, tail(eng.complete(s_text, k), k))
        finally:
            eng.close()
        cap = args.host_capacity
        eng = Engine(args, f"host{cut}", host_gb=1, max_model_len=cap, extra_args=f"--kv-capacity {cap}")
        try:
            eng.complete(text, 1)
            # An unrelated prompt that needs nearly the whole cache: the retained lane must give its pages back.
            # Sized by the tokenizer to cap - 300 tokens (the request must fit the context; prompt + 1 token).
            chars_per_token = len(s_text) / max(1, len(s_ids))
            long_text = (material * (2 + int(4 * cap * chars_per_token) // max(1, len(material))))[777:]
            length = int((cap - 300) * chars_per_token)
            def fits(n):
                try:
                    return len(eng.tokenize(long_text[:n])[0]) <= cap - 300
                except urllib.error.HTTPError:  # /tokenize refuses a prompt past the context too
                    return False
            while length > 0 and not fits(length):
                length = int(length * 0.9)
            filler = long_text[:length]
            eng.complete(filler, 1)
            before = eng.host_restores()
            result = compare(ref, tail(eng.complete(s_text, k), k))
            result["restored_from_host"] = eng.host_restores() > before
            report["paths"][f"host@{cut}"] = result
        finally:
            eng.close()

    ok = all(path["equal"] for path in report["paths"].values())
    report["pass"] = ok
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"S = {len(s_ids)} tokens, K = {k}")
    for name, path in report["paths"].items():
        spilled = "" if "restored_from_host" not in path else f" host_restore={path['restored_from_host']}"
        print(f"  {name:12s} {'EQUAL' if path['equal'] else 'DIFF '} cached={path['cached']}{spilled} "
              f"first_mismatch={path['first_logprob_mismatch']} max|d|={path['max_abs_logprob_diff']:.3g}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
