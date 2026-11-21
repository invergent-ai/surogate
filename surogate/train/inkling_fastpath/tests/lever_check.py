"""Lane T LEVEL-1 acceptance check, ONE process on ONE card (lanes/T/NOTES.md, LEVEL-1 plan). For the cumulative arms
(each adds one lever to the previous arm):

  gradient  every LoRA tensor, per row length: arm vs the arm before it (the lever alone), arm vs A0 (the whole stack),
            and at the short length arm vs ift_backward (the path validated against lane L's rig). The line per lever:
            cos >= 0.9999 and rel err <= 1 %, finite.
  floor     the same arm run twice (A/A): noise-1 measured 0.46 % max at 8k with identical flags, so the 1 % line is read
            against the floor measured IN THIS SESSION.
  speed     warm, unprofiled, on the same flags as the gradient. noise-1 showed the first of two identical runs 3.9 s slower
            (22.95 vs 19.03 s at 8k), and a cold forward costs 2.0 s vs 0.7 s warm: every length gets an untimed warm-up,
            the arms alternate direction between repetitions, the median counts.
  long      the last arm vs the arm named by --long-base at a long row (gradient + speed + peak memory).
  profile   one extra profiled pass of the last arm (section ticks synchronise: never a speed number).

The verdict names, in arm order, the arms that pass; an arm after a failing one is not accepted (the stack is cumulative).
Usage: python lever_check.py --bin B --artifact A --prompts DIR --out report.json [--lengths 1024,8192] [--reps 2]
       [--long 32768] [--long-base A4] [--arms "A0:FLAG=v,...;A1:..."]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ift_engine import InklingEngine  # noqa: E402
import ift_backward as B  # noqa: E402
import ift_long as LG  # noqa: E402

COS_LINE, REL_LINE = 0.9999, 0.01
# every arm names every flag. MOE_DET=1 from A1 on: lev-all-1 showed the MoEGM arms do not repeat (A/A rel err 0.36-0.92 % at 1k,
# A0 repeats bit for bit), so a lever's difference could not be read below that floor.
OFF = "MOE_GM=0,MOE_DET=0,FAST_GEMM=0,ATTN_ADAPT=0,CHUNK=1024,MOE_FAST=0,ATTN_TC=0"
DEFAULT_ARMS = ";".join((
    "A0:" + OFF,
    "A1:MOE_GM=1,MOE_DET=1,FAST_GEMM=0,ATTN_ADAPT=0,CHUNK=1024,MOE_FAST=0,ATTN_TC=0",
    "A2:MOE_GM=1,MOE_DET=1,FAST_GEMM=1,ATTN_ADAPT=0,CHUNK=1024,MOE_FAST=0,ATTN_TC=0",
    "A3:MOE_GM=1,MOE_DET=1,FAST_GEMM=1,ATTN_ADAPT=1,CHUNK=1024,MOE_FAST=0,ATTN_TC=0",
    "A4:MOE_GM=1,MOE_DET=1,FAST_GEMM=1,ATTN_ADAPT=1,CHUNK=4096,MOE_FAST=0,ATTN_TC=0",
    "A5:MOE_GM=1,MOE_DET=1,FAST_GEMM=1,ATTN_ADAPT=1,CHUNK=4096,MOE_FAST=1,ATTN_TC=0",
    "A6:MOE_GM=1,MOE_DET=1,FAST_GEMM=1,ATTN_ADAPT=1,CHUNK=4096,MOE_FAST=1,ATTN_TC=1",
))


def log(msg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    print(f"[T {datetime.now(ZoneInfo('Europe/Bucharest')).strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_arms(spec):
    arms = []
    for part in [x for x in spec.split(";") if x]:
        name, flags = part.split(":")
        arms.append((name, {k: int(v) for k, v in (kv.split("=") for kv in flags.split(",") if kv)}))
    return arms


def apply_flags(flags):
    for k, v in flags.items():
        old = getattr(LG, k)
        setattr(LG, k, type(old)(v))


@torch.no_grad()
def compare(g_new, g_ref, dev):
    """Per-tensor cosine / relative error (fp64 on the card, one tensor at a time)."""
    cos, rel = [], []
    for x, y in zip(g_new, g_ref):
        xd = x.to(dev, non_blocking=True).flatten().double()
        yd = y.to(dev, non_blocking=True).flatten().double()
        cos.append(float(xd @ yd / (xd.norm() * yd.norm()).clamp_min(1e-300)))
        rel.append(float((xd - yd).norm() / yd.norm().clamp_min(1e-300)))
    o = sorted(rel)
    return {"cos_min": min(cos), "rel_err_max": o[-1], "rel_err_median": o[len(o) // 2],
            "worst_tensor": int(max(range(len(rel)), key=rel.__getitem__))}


def line_ok(c):
    return c["cos_min"] >= COS_LINE and c["rel_err_max"] <= REL_LINE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--lengths", default="1024,8192")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--long", type=int, default=32768)
    ap.add_argument("--long-base", default="A4")
    a = ap.parse_args()
    arms = parse_arms(a.arms)
    names = [n for n, _ in arms]
    eng = InklingEngine(a.bin, a.artifact, 0)
    W = B.EngineWeights(eng)
    lora = B.make_lora(eng.dev, seed=7, b_std=1e-3)
    banks = B.EngineBanks(lora, eng.dev)
    params = [lo[k] for lo in lora for k in B.LORA_KEYS]
    pool = []
    for n in ("o25", "o00"):
        pool += json.load(open(os.path.join(a.prompts, n + ".json")))["ids"]
    while len(pool) < max(a.long, 140000):
        pool = pool + pool
    rep = {"arms": {n: f for n, f in arms}, "lines": {"cos": COS_LINE, "rel_err": REL_LINE}, "by_length": {}}

    def save():
        json.dump(rep, open(a.out + ".tmp", "w"), indent=1)
        os.replace(a.out + ".tmp", a.out)

    def run(build, T, keep=True):
        for p in params:
            p.grad = None
        ids = torch.tensor(pool[:T], dtype=torch.int32, device=eng.dev)
        torch.cuda.synchronize()
        t0 = time.time()
        lp, _ = build(ids, lora, banks)
        loss = -lp.mean()
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        finite = all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
        g = [p.grad.detach().to("cpu") for p in params] if keep else None
        val = float(loss.detach())
        del lp, loss
        for p in params:
            p.grad = None
        return g, val, dt, finite

    B.PROFILE["on"] = False
    lengths = [int(x) for x in a.lengths.split(",")]
    for T in lengths:
        row = {"arms": {}}
        apply_flags(arms[-1][1])
        run(LG.TrainForwardLong(eng, W).build, T, keep=False)  # untimed warm-up (allocator, kernels, cuBLAS handles)
        ref_old = None
        if T == lengths[0]:  # the rig-validated path, on the short row only (its memory grows quickly with T)
            ref_old, loss_old, _, _ = run(B.TrainForward(eng, W).build, T)
            row["loss_ift_backward"] = loss_old
        first, times, losses, finite = {}, {n: [] for n in names}, {}, {}
        floors = {}
        for r in range(a.reps):
            order = arms if r % 2 == 0 else list(reversed(arms))
            for name, flags in order:
                apply_flags(flags)
                g, l, dt, fin = run(LG.TrainForwardLong(eng, W).build, T)
                times[name].append(dt)
                losses[name], finite[name] = l, finite.get(name, True) and fin
                if name not in first:
                    first[name] = g
                elif name not in floors:
                    floors[name] = compare(g, first[name], eng.dev)  # A/A: this arm, second run vs first
                del g
                log(f"T={T} rep {r} {name}: {dt:.2f} s loss {l:.6f} finite {fin}")
        for k, (name, _) in enumerate(arms):
            med = sorted(times[name])[len(times[name]) // 2]
            ar = {"seconds": times[name], "seconds_median": med, "tok_per_s": T / med, "loss": losses[name],
                  "finite": finite[name], "aa_floor": floors.get(name)}
            if k > 0:
                ar["vs_previous"] = compare(first[name], first[names[k - 1]], eng.dev)
                ar["vs_A0"] = compare(first[name], first[names[0]], eng.dev)
            if ref_old is not None:
                ar["vs_ift_backward"] = compare(first[name], ref_old, eng.dev)
            row["arms"][name] = ar
            log(f"T={T} {name}: median {med:.2f} s = {T / med:.0f} tok/s " + json.dumps(
                {x: {"cos": round(ar[x]["cos_min"], 6), "rel": round(ar[x]["rel_err_max"], 5)}
                 for x in ("aa_floor", "vs_previous", "vs_A0", "vs_ift_backward") if ar.get(x)}))
        rep["by_length"][str(T)] = row
        del first, ref_old
        save()

    # verdict: cumulative, stops at the first arm that misses a line
    accepted = []
    for k, name in enumerate(names):
        ok = True
        for T in lengths:
            ar = rep["by_length"][str(T)]["arms"][name]
            ok = ok and ar["finite"]
            for key in ("vs_previous", "vs_A0", "vs_ift_backward"):
                if ar.get(key):
                    ok = ok and line_ok(ar[key])
        if not ok:
            break
        accepted.append(name)
    rep["accepted_arms"] = accepted
    rep["last_accepted"] = accepted[-1] if accepted else None
    save()
    log("ACCEPTED " + json.dumps(accepted))

    # long row: the last arm vs the base arm
    if a.long and rep["last_accepted"] and a.long_base in names:
        fl = dict(arms)
        lg = {}
        pair = (("new", rep["last_accepted"]),) if rep["last_accepted"] == a.long_base else (
            ("base", a.long_base), ("new", rep["last_accepted"]))  # no second run when the stack IS the base arm
        for label, name in pair:
            apply_flags(fl[name])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                g, l, dt, fin = run(LG.TrainForwardLong(eng, W).build, a.long)
            except torch.OutOfMemoryError as e:
                lg[label] = {"arm": name, "oom": str(e).split(". ")[0][:200]}
                log(f"long {label} {name}: OOM")
                break
            lg[label] = {"arm": name, "seconds": dt, "tok_per_s": a.long / dt, "loss": l, "finite": fin,
                         "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            lg["g_" + label] = g
            log(f"long {label} {name}: {dt:.1f} s = {a.long / dt:.0f} tok/s peak {lg[label]['peak_gib']:.1f} GiB")
        if "g_base" in lg and "g_new" in lg:
            lg["compare"] = compare(lg["g_new"], lg["g_base"], eng.dev)
            log("long compare " + json.dumps(lg["compare"]))
        lg.pop("g_base", None)
        lg.pop("g_new", None)
        rep["long"] = {"T": a.long, **lg}
        save()

    # one profiled pass of the last accepted arm at the longest checked length (where the time goes next)
    if rep["last_accepted"]:
        apply_flags(dict(arms)[rep["last_accepted"]])
        B.PROFILE["on"] = True
        B.PROFILE["t"].clear()
        T = lengths[-1]
        _, _, dt, _ = run(LG.TrainForwardLong(eng, W).build, T, keep=False)
        rep["profile"] = {"T": T, "arm": rep["last_accepted"], "seconds_synchronising": dt,
                          "sections": {k: round(v, 3) for k, v in sorted(B.PROFILE["t"].items())}}
        B.PROFILE["on"] = False
        save()
    eng.close()
    last = rep["last_accepted"]
    if last:
        best = rep["by_length"][str(lengths[-1])]["arms"][last]["tok_per_s"]
        log(f"RESULT accepted through {last}: {best:.0f} tok/s at {lengths[-1]} (target 894); see {a.out}")
    else:
        log("RESULT no arm accepted")
    return 0 if len(accepted) == len(names) else 3


if __name__ == "__main__":
    sys.exit(main())
