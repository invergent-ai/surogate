"""Micro-step reads: one serving engine's prompt log-probs of its share of the plan's rows, adapter after adapter, written per adapter as
<out>/<adapter>.part<k>.npz with lp_<plan row> = float32 [T], the value at each loss token and 0 elsewhere (t8_micro.lp_at_mask).
"base" = requests to the served model id with no adapter; any other name is loaded with /v1/load_lora_adapter, read, and unloaded.
  t8_micro_reads.py --url U --model inkling --plan DIR --part k --parts n --adapters base,Z:/path,... [--only-rows 0,5,9 for NAME] --out DIR
Rows are split over the parts by length (longest first to the part with the fewest tokens), the same split on every part."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def split(Ts: dict, parts: int) -> list:
    load, out = [0] * parts, [[] for _ in range(parts)]
    for n in sorted(Ts, key=lambda n: (-Ts[n], n)):
        k = min(range(parts), key=lambda j: (load[j], j))
        out[k].append(n)
        load[k] += Ts[n]
    return out


def score(url, model, ids):
    d = post(url, "/v1/completions", {"model": model, "prompt": "", "tokens": [int(x) for x in ids], "max_tokens": 1, "temperature": 0.0,
                                      "prompt_logprobs": 1, "logprobs": 1, "return_token_ids": True})
    plp = d["prompt_logprobs"]
    echo = d.get("prompt_token_ids")
    if echo is None:
        echo = (d.get("choices") or [{}])[0].get("prompt_token_ids")
    if echo is None or [int(x) for x in echo] != [int(x) for x in ids]:
        raise SystemExit(f"the engine's prompt ids differ from the row's ({len(echo) if echo is not None else 'none'} vs {len(ids)})")
    if len(plp) != len(ids):
        raise SystemExit(f"{len(plp)} prompt log-probs for {len(ids)} ids")
    return np.array([0.0] + [float(plp[t][str(int(ids[t]))]["logprob"]) for t in range(1, len(ids))], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--parts", type=int, required=True)
    ap.add_argument("--adapters", required=True, help="comma list: base or name:path")
    ap.add_argument("--subset", default="", help="name=comma list of plan rows read for that adapter only (e.g. Z=0,1,2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(os.path.join(a.plan, "rows.npz"), allow_pickle=False)
    R = sum(1 for k in z.files if k.startswith("ids_") and k[4:].isdigit())
    Ts = {n: int(z["T"][n]) for n in range(R)}
    mine = split(Ts, a.parts)[a.part]
    subset = {}
    if a.subset:
        name, _, lst = a.subset.partition("=")
        subset[name] = {int(x) for x in lst.split(",") if x}
    os.makedirs(a.out, exist_ok=True)
    for spec in a.adapters.split(","):
        name, _, path = spec.partition(":")
        rows = [n for n in mine if name not in subset or n in subset[name]]
        model = a.model if name == "base" else name
        t0 = time.time()
        if name != "base":
            post(a.url, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, timeout=900)
        res = {}
        try:
            for n in rows:
                ids, mask = z[f"ids_{n}"], z[f"mask_{n}"].astype(bool)
                lp = score(a.url, model, ids)
                out = np.zeros(len(ids), dtype=np.float32)
                out[mask] = lp[mask]
                res[f"lp_{n}"] = out
        finally:
            if name != "base":
                post(a.url, "/v1/unload_lora_adapter", {"lora_name": name}, timeout=900)
        tok = sum(Ts[n] for n in rows)
        np.savez(os.path.join(a.out, f"{name}.part{a.part}.npz"), **res)
        print(f"part {a.part} {name}: {len(rows)} rows, {tok} tokens, {time.time() - t0:.0f} s, {tok / max(time.time() - t0, 1e-9):.0f} tok/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
