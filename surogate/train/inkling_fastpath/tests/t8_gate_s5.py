"""Lane T8, S5: the adapter round trip of SkyRL's LoRA shape (attention r64 / alpha 32 on the seven slices + experts r16 / alpha 32, A and B) through the
SERVING engine that loads the trainer's exported file. PASS LINES, written before any run (also in the lane notes):
  G18a the trainer (T8 shim, release m8) at the ZERO adapter == the engine's BASE model (requests to "inkling", no adapter slot) on 3 real rows of about
       14k tokens: bitwise at every position.
  G18b a test adapter (A = the trainer's kaiming init, B = N(0, 0.02), seed 21, every key of every layer) exported by T8Trainer.export_adapter (the
       PEFT-native form) and loaded fresh by the engine (--enable-lora --max-lora-rank 64 --max-expert-lora-rank 16): the trainer's log-probs with
       that adapter == the engine's for that module, bitwise at every position of the same rows; and the adapter changes at least one log-prob against
       the base (otherwise the check is vacuous).
  (A tensor-by-tensor dump of the engine's banks is not available from outside the engine; the bitwise forward at every position, plus the CPU
   round trip of the file in tests/t8_attn_cpu.py, stand in for "every tensor bitwise after load".)
  phase trainer: writes rows_ids.json, trainer_zero.npz, trainer_adapter.npz and the adapter directory. phase compare: reads the engine's two files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def phase_trainer(a):
    import t8_gate8 as G8
    import t8_step as TS

    rows_all, groups = G8.load(a.rows)
    rows = [rows_all[i] for i in groups["rt"]]
    tr = TS.T8Trainer(TS.T8Config(bin_dir=a.bin, artifact=a.artifact, cards=(int(a.card),), K=48, head="plain", attn_rank=64, attn_alpha=32.0,
                                  train_keys=TS.param_keys(64), segment_keys=(), run_dir=a.run_dir))
    try:
        lz = tr.logprobs(rows)
        M = tr.masters()
        g = torch.Generator().manual_seed(21)
        for d in M:
            for k in d:
                if k.endswith("B") or k in ("B13", "B2"):
                    d[k] = torch.randn(d[k].shape, generator=g) * 0.02
        tr.set_masters(M)
        la = tr.logprobs(rows)
        sha = tr.export_adapter(a.adapter, "BF16", meta={"step": "t8-s5-test"})
    finally:
        tr.close()
    json.dump([{"ids": r.ids.tolist()} for r in rows], open(os.path.join(a.run_dir, "rows_ids.json"), "w"))
    np.savez(os.path.join(a.run_dir, "trainer_zero.npz"), **{f"lp_{i}": x for i, x in enumerate(lz)})
    np.savez(os.path.join(a.run_dir, "trainer_adapter.npz"), **{f"lp_{i}": x for i, x in enumerate(la)})
    json.dump({"adapter_sha256": sha, "rows": [r.T for r in rows]}, open(a.out, "w"))
    print(f"trainer phase done: rows {[r.T for r in rows]}, adapter sha {sha[:16]}", flush=True)
    return 0


def cmp(trainer_npz, engine_json):
    z = np.load(trainer_npz)
    eng = json.load(open(engine_json))
    eq = ne = 0
    for i, e in enumerate(eng):
        x = z[f"lp_{i}"].astype(np.float32)
        y = np.asarray(e["lp"][1:], dtype=np.float32)
        if x.shape != y.shape:
            raise SystemExit(f"row {i}: {x.shape} trainer log-probs, {y.shape} engine")
        same = x.view(np.int32) == y.view(np.int32)
        eq += int(same.sum())
        ne += int((~same).sum())
    return eq, ne


def phase_compare(a):
    rep = json.load(open(a.out))
    d = a.run_dir
    e0, n0 = cmp(os.path.join(d, "trainer_zero.npz"), os.path.join(d, "engine_base.json"))
    e1, n1 = cmp(os.path.join(d, "trainer_adapter.npz"), os.path.join(d, "engine_adapter.json"))
    za, zb = np.load(os.path.join(d, "trainer_zero.npz")), np.load(os.path.join(d, "trainer_adapter.npz"))
    moved = int(sum(int((za[k] != zb[k]).sum()) for k in za.files))
    rep["G18a"] = {"equal": e0, "differ": n0, "pass": bool(n0 == 0 and e0 > 0)}
    rep["G18b"] = {"equal": e1, "differ": n1, "adapter_changed_positions": moved, "pass": bool(n1 == 0 and e1 > 0 and moved > 0)}
    rep["pass"] = bool(rep["G18a"]["pass"] and rep["G18b"]["pass"])
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"G18a {rep['G18a']}\nG18b {rep['G18b']}\nT8 GATE S5: {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
    return 0 if rep["pass"] else 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("trainer", "compare"))
    ap.add_argument("--bin", default="")
    ap.add_argument("--artifact", default="")
    ap.add_argument("--rows", default="")
    ap.add_argument("--card", default="0")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return phase_trainer(a) if a.phase == "trainer" else phase_compare(a)


if __name__ == "__main__":
    sys.exit(main())
