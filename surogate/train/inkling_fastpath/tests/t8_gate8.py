"""The 8-card gate of U1. PASS LINES, written before any run. One test step; nothing is trained and the test adapter is deleted after the check.
Real Inkling-2-Mini rows picked by length only: batch = the 8 longest <= 58,010 + 24 spread; K = 18, centred head, seeds "weights", B keys, 8 workers on ONE box.
  G8a every hard gate holds on every row (replay, support, zero adapter on every worker, finite), no row refused.
  G8b host peak per worker (VmHWM) <= 20 GiB.
  G8c lowest box MemAvailable during the step >= 60 GB (the K = 18 batch; the lowest reading over the whole phase is reported beside it).
  G8d A/A: the same 8 rows twice on the 8 workers, gradient and segment sums bitwise equal.
  G9  adapter round trip: after a test update of B (L2 0.03 along the step's gradient) the workers' log-probs == a FRESH engine (release M13b, the rollout engine) loaded with the exported BF16
      adapter, bitwise at every position of 3 rows of about 14k tokens; and the update changed at least one log-prob (otherwise the check is vacuous).
Reported, no line: tok/s per card at K = 18 on the batch, and at K = 6 / 18 / 30 / 48 on 8 rows of about 30k.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(path):
    from surogate.inkling_rl.records import Row

    z = np.load(path)
    n = sum(1 for k in z.files if k.startswith("ids_"))
    rows = [Row(ids=z[f"ids_{i}"].astype(np.int64), loss_mask=z[f"mask_{i}"].astype(bool), engine_lp=z[f"lp_{i}"].astype(np.float64), advantage=0.0, versions=None,
                teacher_lp=None, routed=None, notes={"weight": 1.0}) for i in range(n)]
    return rows, {k[6:]: [int(x) for x in z[k]] for k in z.files if k.startswith("group_")}


class MemWatch:
    """The lowest MemAvailable (GB) while the step runs."""

    def __init__(self):
        self.low, self.stop_ev = 1e9, threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_ev.wait(2.0):
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable"):
                    self.low = min(self.low, int(line.split()[1]) / 1048576)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        self.t.join(5)


def phase_step(a):
    with MemWatch() as whole:
        rc, rep = _phase_step(a)
        rep["min_mem_available_whole_phase_gb"] = whole.low
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    log(f"lowest MemAvailable over the whole phase: {whole.low:.0f} GB")
    print(f"T8 GATE8 step: {'PASS' if rc == 0 else 'FAIL'}", flush=True)
    return rc


def _phase_step(a):
    from surogate.inkling_rl.step import StepConfig
    import t8_step as TS

    rows, groups = load(a.rows)
    cards = tuple(int(x) for x in a.cards.split(","))
    scfg = StepConfig(max_policy_lag=None)
    base = dict(bin_dir=a.bin, artifact=a.artifact, cards=cards, head="centred", max_row_tokens=58010, segment_keys=("B13", "B2"), train_keys=("B13", "B2"), run_dir=a.run_dir)
    pick = lambda name: [rows[i] for i in groups[name]]
    rep = {"cards": list(cards), "speed": {}}
    for K in (6, 30, 48):                                     # before the K = 18 trainer: the same rows at every K
        tr = TS.T8Trainer(TS.T8Config(K=K, **{**base, "segment_keys": ()}))  # no segment sums here: at K = 48 they would be the host-memory peak (review M1)
        try:
            o = tr.compute(pick("speed"), 0, scfg, seeds="weights")
        finally:
            tr.close()
        rep["speed"][str(K)] = {"tok_per_s_per_card": o.metrics["tok_per_s_per_card"], "wall_s": o.metrics["wall_s"], "row_tokens": o.metrics["row_tokens"],
                                "host_peak_gib": max(w["VmHWM_gib"] for w in o.metrics["workers"]), "oom_retries": sum(r["oom_retries"] for r in o.rows)}
        log(f"K={K}: {rep['speed'][str(K)]}")
        del o
    tr = TS.T8Trainer(TS.T8Config(K=18, **base))
    try:
        rep["zero_adapter"] = len(tr.check_zero_adapter(pick("aa")[:1]))
        a1 = tr.compute(pick("aa"), 0, scfg, seeds="weights")
        a2 = tr.compute(pick("aa"), 0, scfg, seeds="weights")
        rep["G8d"] = bool(all(torch.equal(x, y) for x, y in zip(a1.grads, a2.grads)) and set(a1.segments) == set(a2.segments)
                          and all(torch.equal(a1.segments[k], a2.segments[k]) for k in a1.segments) and any(bool(x.ne(0).any()) for x in a1.grads))
        log(f"G8d A/A on 8 workers: {rep['G8d']}")
        del a1, a2
        o = tr.compute(pick("speed"), 0, scfg, seeds="weights")
        rep["speed"]["18"] = {"tok_per_s_per_card": o.metrics["tok_per_s_per_card"], "wall_s": o.metrics["wall_s"], "row_tokens": o.metrics["row_tokens"],
                              "host_peak_gib": max(w["VmHWM_gib"] for w in o.metrics["workers"]), "oom_retries": sum(r["oom_retries"] for r in o.rows)}
        del o
        batch = pick("batch")
        with MemWatch() as mw:
            o = tr.compute(batch, 0, scfg, seeds="weights")
        rep["batch"] = {k: o.metrics[k] for k in ("row_tokens", "wall_s", "tok_per_s_per_card", "engine_mismatch_kl", "total_tokens", "replay_checked_rows", "replay_skipped_rows",
                                                  "refused_long_rows", "workers")}
        rep["batch"]["oom_retries"] = sum(r["oom_retries"] for r in o.rows)
        rep["batch"]["rows"] = len(o.rows)
        rep["min_mem_available_gb"] = mw.low
        peak = max(w["VmHWM_gib"] for w in o.metrics["workers"])
        rep["host_peak_per_worker_gib"] = peak
        rep["G8a"] = bool(rep["zero_adapter"] == len(cards) and len(o.rows) == len(batch) and o.metrics["replay_skipped_rows"] == 0 and o.metrics["replay_checked_rows"] == len(batch))
        rep["G8b"] = bool(peak <= 20.0)
        rep["G8c"] = bool(mw.low >= 60.0)
        json.dump(rep, open(a.out, "w"), indent=1, default=float)  # a later crash keeps G8a-d
        log(f"batch: {rep['batch']['rows']} rows, {rep['batch']['tok_per_s_per_card']:.0f} tok/s per card, host peak {peak:.2f} GiB, lowest MemAvailable {mw.low:.0f} GB; G8a {rep['G8a']} G8b {rep['G8b']} G8c {rep['G8c']}")
        # the test update of G9
        rt = pick("rt")
        lp0 = tr.logprobs(rt)
        norm = float(sum(g.double().pow(2).sum() for g in o.grads)) ** 0.5
        if not norm > 0:
            raise SystemExit("zero gradient: G9 cannot be read")
        M = tr.masters()
        for L, d in enumerate(M):
            for j, k in enumerate(TS.LORA_KEYS):
                if k in ("B13", "B2"):
                    d[k] = d[k] + (o.grads[L * 4 + j].double() * (0.03 / norm)).float()
        del o
        tr.set_masters(M)
        lp1 = tr.logprobs(rt)
        rep["update_changed_logprobs"] = int(sum(int((x != y).sum()) for x, y in zip(lp0, lp1)))
        os.makedirs(a.adapter, exist_ok=True)
        rep["adapter_sha256"] = tr.export_adapter(a.adapter, "BF16", meta={"step": "t8-gate-test"})
        np.savez(os.path.join(a.run_dir, "trainer_v1.npz"), **{f"lp_{i}": x for i, x in enumerate(lp1)})
        json.dump([{"ids": r.ids.tolist()} for r in rt], open(os.path.join(a.run_dir, "rows_ids.json"), "w"))
    finally:
        tr.close()
    ok = rep["G8a"] and rep["G8b"] and rep["G8c"] and rep["G8d"] and rep["update_changed_logprobs"] > 0
    return (0 if ok else 4), rep


def phase_compare(a):
    rep = json.load(open(a.out))
    z = np.load(os.path.join(a.run_dir, "trainer_v1.npz"))
    eng = json.load(open(os.path.join(a.run_dir, "engine_v1.json")))
    eq = ne = 0
    for i, e in enumerate(eng):
        x = z[f"lp_{i}"].astype(np.float32)
        y = np.asarray(e["lp"][1:], dtype=np.float32)
        if x.shape != y.shape:
            raise SystemExit(f"row {i}: {x.shape} log-probs from the workers, {y.shape} from the engine")
        same = x.view(np.int32) == y.view(np.int32)
        eq += int(same.sum())
        ne += int((~same).sum())
    rep["G9"] = {"equal": eq, "differ": ne, "pass": bool(ne == 0 and eq > 0 and rep["update_changed_logprobs"] > 0)}
    rep["pass"] = bool(rep["G8a"] and rep["G8b"] and rep["G8c"] and rep["G8d"] and rep["G9"]["pass"])
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(f"G9 {rep['G9']}\nT8 GATE8: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("step", "compare"))
    ap.add_argument("--bin", default="")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--cards", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return phase_step(a) if a.phase == "step" else phase_compare(a)


if __name__ == "__main__":
    sys.exit(main())
