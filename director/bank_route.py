"""Stage-2 (feasible): optimize PER-TASK routing against the precomputed agentic bank — INSTANT,
GPU-only, ZERO live rollouts.

Why this and not live per-turn ES: Trinity's live ES was only feasible because it SELF-HOSTED its
workers (60 vLLM instances of 32B-class models across 4 GPUs -> 60-way rollout concurrency, no API
limits, max_turns=5). Our pool is frontier-scale OPEN models (deepseek-v4-pro, glm-5.2 ~ Opus) that
are API-only and our tau tasks are ~30 turns -> live per-candidate ES would take years. The PRODUCTION
Trinity framework already decouples this: orchestrator traces -> exported fitness dataset -> external
ES. Our agentic_bank.jsonl IS that exported fitness evidence.

Per-task routing reward = bank[item, router(instruction)] is EXACT (a solo rollout = routing the whole
task to one worker). We warm-start with an SFT on the bank's per-task soft targets, then run sep-CMA-ES
on the routing reward. The HELD-OUT capture is the thesis test: can the router pick the right worker
for an UNSEEN tau task from its instruction features?

Env: MANIFEST_DIR, TAU, CMAES_GENERATIONS, CMAES_SIGMA0, CMAES_POPSIZE, VAL_FRAC, SFT_EPOCHS, SFT_LR.
"""
from __future__ import annotations

import importlib
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

MANIFEST_DIR = os.getenv("MANIFEST_DIR", "manifests/fugu_clean_v1")
BANK = os.path.join(MANIFEST_DIR, "agentic_bank.jsonl")
SFT_CKPT = os.path.join(MANIFEST_DIR, "sft_router.pt")  # Stage-1 (single-step) router, baseline only
TAU = float(os.getenv("TAU", "0.1"))
GENERATIONS = int(os.getenv("CMAES_GENERATIONS", "60"))
SIGMA0 = float(os.getenv("CMAES_SIGMA0", "0.03"))
POPSIZE = int(os.getenv("CMAES_POPSIZE", "0")) or None
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.25"))
SFT_EPOCHS = int(os.getenv("SFT_EPOCHS", "300"))
SFT_LR = float(os.getenv("SFT_LR", "1e-4"))
SEED = int(os.getenv("SEED", "0"))


def _soft(r_bar: np.ndarray, tau: float) -> list[float]:
    z = r_bar / max(tau, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).tolist()


def tau_instructions() -> dict[str, str]:
    out: dict[str, str] = {}
    for env in ["retail", "airline"]:
        mod = importlib.import_module(f"tau_bench.envs.{env}.tasks_test")
        tasks = getattr(mod, "TASKS_TEST", None) or getattr(mod, "TASKS", [])
        for i, t in enumerate(tasks):
            instr = getattr(t, "instruction", None) or (t.get("instruction") if isinstance(t, dict) else None)
            out[f"tau-{env}-{i}"] = instr or ""
    return out


def main():
    from director.config import DirectorConfig, FeaturizerConfig, default_frontier_pool
    from director.fugu.cmaes import evolve
    from director.fugu.inference import select_worker
    from director.fugu.labels import SoftLabel
    from director.fugu.model import save_router
    from director.fugu.run import build_router
    from director.fugu.sft import train_sft
    from director.shared.transcript import raw_query

    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
    wids = cfg.worker_ids
    idx = {w: j for j, w in enumerate(wids)}

    # bank -> R[item, worker]; keep only items with ALL workers graded AND a real instruction
    cells: dict[str, dict[str, float]] = defaultdict(dict)
    for l in open(BANK):
        if l.strip():
            c = json.loads(l)
            cells[c["item_id"]][c["worker"]] = c["reward"]
    instr = tau_instructions()
    items = sorted(i for i in cells if all(w in cells[i] for w in wids) and instr.get(i))
    if len(items) < 12:
        print(f"only {len(items)} complete bank items — need more tau cells", flush=True)
        return
    R = np.array([[cells[i][w] for w in wids] for i in items], dtype=float)
    texts = [raw_query(instr[i]) for i in items]

    # train / held-out val split
    order = list(range(len(items)))
    random.Random(SEED).shuffle(order)
    nval = max(4, int(len(items) * VAL_FRAC))
    va_i = sorted(order[:nval])
    tr_i = sorted(order[nval:])

    def stats(ix):
        Rs = R[ix]
        return float(Rs.mean(0).max()), float(Rs.max(1).mean())

    btr, otr = stats(tr_i)
    bva, ova = stats(va_i)
    print(f"[bank] items={len(items)} (train={len(tr_i)} val={len(va_i)}) workers={wids}", flush=True)
    print(f"[bank] TRAIN best-single={btr:.3f} oracle={otr:.3f} (headroom {otr-btr:+.3f})", flush=True)
    print(f"[bank] VAL   best-single={bva:.3f} oracle={ova:.3f} (headroom {ova-bva:+.3f})", flush=True)

    def routed(router, ix):
        picks = [idx[select_worker(router, texts[i])] for i in ix]
        return float(R[np.array(ix), picks].mean()), picks

    def cap(orch, b, o):
        return (orch - b) / (o - b) if o > b else 0.0

    def report(tag, router):
        rtr, _ = routed(router, tr_i)
        rva, pva = routed(router, va_i)
        dist = Counter(wids[p] for p in pva)
        print(f"[{tag}] TRAIN routed={rtr:.3f} (cap {cap(rtr,btr,otr):.0%}) | "
              f"VAL routed={rva:.3f} (cap {cap(rva,bva,ova):.0%}) | val-dist={dict(dist)}", flush=True)
        return rtr, rva

    # baseline: the Stage-1 (single-step) router, untuned for agentic
    if os.path.exists(SFT_CKPT):
        report("single-step-SFT", build_router(cfg, ckpt=SFT_CKPT))

    # Fugu two-stage, all on the bank (instant): agentic SFT warm-start -> sep-CMA-ES on routing reward
    router = build_router(cfg)
    labels = [SoftLabel(task_id=items[i], prompt=instr[items[i]], worker_ids=wids,
                        r_bar=R[i].tolist(), p=_soft(R[i], TAU), grader="agentic") for i in tr_i]
    train_sft(router, labels, epochs=SFT_EPOCHS, lr=SFT_LR, batch_size=64, micro_batch=16, log_every=50)
    report("agentic-SFT", router)

    def fitness():  # maximize TRAIN routed reward (held-out val is never optimized against)
        r, _ = routed(router, tr_i)
        return r

    print(f"[cmaes] {GENERATIONS} gens (sigma0={SIGMA0}, popsize={POPSIZE or 'auto'}) on bank routing reward", flush=True)
    res = evolve(router, fitness, generations=GENERATIONS, sigma0=SIGMA0, popsize=POPSIZE,
                 checkpoint_dir=os.path.join(MANIFEST_DIR, "bank_cmaes_ckpt"), resume=True, verbose=True)
    rtr, rva = report("agentic-CMA-ES", router)
    save_router(router, os.path.join(MANIFEST_DIR, "agentic_router.pt"), worker_ids=wids)

    gates = {
        "TRAIN captures > 25%": cap(rtr, btr, otr) > 0.25,
        "VAL captures > 0 (generalizes)": cap(rva, bva, ova) > 0.0,
        "VAL routed > best-single": rva > bva,
    }
    lines = [
        "", "## Stage-2 (bank-based per-task routing — INSTANT)",
        f"- bank items={len(items)} train={len(tr_i)} val={len(va_i)} | gens={res.generations_run}",
        f"- TRAIN best={btr:.3f} oracle={otr:.3f} -> CMA-ES {rtr:.3f} (capture {cap(rtr,btr,otr):.0%})",
        f"- VAL   best={bva:.3f} oracle={ova:.3f} -> CMA-ES {rva:.3f} (capture {cap(rva,bva,ova):.0%})  [held-out: the thesis]",
        "", "### gates",
        *[f"- [{'PASS' if ok else 'FAIL'}] {n}" for n, ok in gates.items()],
    ]
    with open(os.path.join(MANIFEST_DIR, "data_report.md"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nSTAGE-2 GATES: {'ALL PASS' if all(gates.values()) else 'SOME FAILED'}", flush=True)


if __name__ == "__main__":
    main()
