"""Lane T M1 / M2: the trainer's forward through the shim == the engine's own forward, BIT FOR BIT.

Inputs (made by lanes/T/card/m2_remote.sh on the same card, same libsinfer.so, just before):
  --dump DIR     SUROGATE_SERVE_INKLING_DUMP of surogate-engine's first prompt forward (o00): per visit attn_in, qkvr,
                 gated, attn_out, mlp_in, moe_out, partial (forward.cpp `dump`)
  --resp DIR     /v1/completions responses with prompt_logprobs (<name>.json), token ids in --prompts DIR/<name>.json
Checks:
  M1  visit 0: every dumped op output of the engine == the shim's taps (attention, MoE, convs, mixes), bitwise
  M2  all 48 visits' dumped tensors == the shim's, and the prompt log-probs == the engine's returned values, bitwise,
      on every prompt given (o25 = 4,963 tokens crosses the engine's 4,096 prefill chunk; ours is one segment)
Usage: python m2_engine_parity.py --bin B --artifact A --dump D --resp R --prompts P --names o00,o25 --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ift_engine import InklingEngine  # noqa: E402
import ift_forward as F  # noqa: E402

D = 2304


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


def read_dump(path):
    with open(path, "rb") as f:
        magic, dtype, rows, cols, planes = struct.unpack("<5i", f.read(20))
        assert magic == 0x314B4E49, path
        raw = f.read()
    t = torch.frombuffer(bytearray(raw), dtype=torch.int16 if dtype == 0 else torch.int32)
    return t.reshape(planes, cols, rows)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--resp", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--names", default="o00")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = {"checks": {}, "prompts": {}}
    t0 = time.time()
    eng = InklingEngine(a.bin, a.artifact, 0)
    log(f"loaded in {time.time() - t0:.1f}s; constants {eng.c}; mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
        f"(torch) / free {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")
    names = a.names.split(",")
    for k, name in enumerate(names):
        rec = json.load(open(os.path.join(a.prompts, name + ".json")))
        ids = torch.tensor(rec["ids"], dtype=torch.int32, device=eng.dev)
        T = ids.numel()
        visit_rows = []
        dump_dir = a.dump if (k == 0 and a.dump) else ""

        def on_visit(v, x, bufs):
            if not dump_dir:
                return
            row = {"v": v}
            for n in ("attn_in", "qkvr", "gated", "attn_out", "mlp_in", "moe_out", "partial"):
                p = os.path.join(dump_dir, f"v{v}_{n}.bin")
                if not os.path.exists(p):
                    row[n] = "missing"
                    continue
                ref = read_dump(p)
                if n == "partial":
                    mine = bits(F.slot(x, eng.visits[v].next_partial)).reshape(1, T, D)
                elif n == "attn_in":
                    mine = bits(bufs[n])
                else:
                    mine = bits(bufs[n]).reshape(1, T, -1)
                if ref.shape != mine.shape:
                    row[n] = f"shape {tuple(ref.shape)} vs {tuple(mine.shape)}"
                    continue
                diff = (ref != mine).sum().item()
                row[n] = "EQUAL" if diff == 0 else f"{diff} / {ref.numel()} differ"
            visit_rows.append(row)
            bad = [f"{n}:{s}" for n, s in row.items() if n != "v" and s != "EQUAL"]
            if bad or v in (0, 5, 6, 18, 47):
                log(f"{name} visit {v}: {'ALL EQUAL' if not bad else ' '.join(bad)}")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        out = F.forward(eng, ids, save=True,
                        tap_names=("attn_in", "qkvr", "gated", "attn_out", "mlp_in", "moe_out") if dump_dir else None,
                        on_visit=on_visit if dump_dir else None)
        lp = F.logprobs(eng, out["hidden"], ids)
        torch.cuda.synchronize()
        dt = time.time() - t1
        # the engine's returned prompt log-probs (index 0 is None)
        resp = json.load(open(os.path.join(a.resp, name + ".json")))
        r = resp.get("response", resp)
        plp = r["prompt_logprobs"]
        mine = lp.cpu()
        eq = ne = miss = 0
        worst = 0.0
        first_bad = None
        for i in range(1, T):
            ent = plp[i]
            if ent is None or str(rec["ids"][i]) not in ent:
                miss += 1
                continue
            ref = torch.tensor(ent[str(rec["ids"][i])]["logprob"], dtype=torch.float32)
            if ref.view(torch.int32).item() == mine[i - 1].view(torch.int32).item():
                eq += 1
            else:
                ne += 1
                worst = max(worst, abs(ref.item() - mine[i - 1].item()))
                first_bad = first_bad if first_bad is not None else i
        res = {"T": T, "logprob_equal": eq, "logprob_differ": ne, "logprob_missing": miss, "worst_abs": worst,
               "first_differ_pos": first_bad, "mean_nll_mine": float(-mine.mean()), "forward_s": dt,
               "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30}
        if visit_rows:
            all_eq = all(all(s == "EQUAL" for n, s in row.items() if n != "v") for row in visit_rows)
            res["visits_checked"] = len(visit_rows)
            res["visits_all_equal"] = all_eq
            res["visit0"] = visit_rows[0]
            res["visit_rows"] = visit_rows
            report["checks"]["M1_visit0_bitwise"] = all(s == "EQUAL" for n, s in visit_rows[0].items() if n != "v")
            report["checks"]["M2_visits_bitwise"] = all_eq and len(visit_rows) == 48
        report["prompts"][name] = res
        log(f"{name}: T {T} logprobs equal {eq} / {eq + ne} (missing {miss}, worst |d| {worst:.3g}); forward + head "
            f"{dt:.2f}s; peak {res['peak_alloc_gib']:.2f} GiB")
        del out
    report["checks"]["M2_logprobs_bitwise"] = all(p["logprob_differ"] == 0 and p["logprob_equal"] > 0
                                                  for p in report["prompts"].values())
    json.dump(report, open(a.out, "w"), indent=1)
    log("CHECKS " + json.dumps(report["checks"]))
    eng.close()
    ok = all(report["checks"].values())
    print("M2 PASS" if ok else "M2 FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
