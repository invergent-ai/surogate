"""FULL-STRENGTH VERDICT eval — Fugu-Ultra (step-145 conductor, live vLLM :8007) vs
full-strength workers on the held-out trend60 set. The endgame comparison MISSION
reserved ("full-strength comparison is reserved for the endgame verdict"), pulled
forward per the 2026-07-08 pivot, subject corrected 2026-07-08: the thing under test
is FUGU-ULTRA, with a Zenith-inspired feedback-loop variant.

Arms:
  solo__{opus,gpt,gemini,glm}  one full-strength call per task (the bar)
  solo2__W                     solo + oracle-retry: if wrong, one "incorrect, try again"
                               turn (derived from solo — retry runs only on failures)
  fu1                          Fugu-Ultra one-shot: conductor plan -> execute_workflow
                               with FULL-strength workers -> grade
  fu2                          Fugu-Ultra + feedback loop (Zenith gap-finding borrowed):
                               fu1 outcome handed back via REVISE_INSTRUCTION -> new plan
                               -> execute -> grade. Turn-1 == fu1 (same sample), so
                               fu2-vs-fu1 is exactly paired; fairness partner is solo2
                               (both loops get the same binary incorrect signal).

Full strength = max_tokens 16384 + reasoning_effort "high" (the Fugu report §4.1.1
setting), temp 0.2 / top_p 1.0 unchanged from the historical series so the ONLY moved
variables are effort + cap. Conductor generation itself is unchanged (temp 1.0,
1024 tok, no-think — the trained distribution).

Run:
  ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra .venv/bin/python \
    scratchpad/eval_fullstrength_verdict.py --label fs_verdict --conc 8 --n 60
Smoke (2 tasks per capability, all arms):
  ... --label fs_smoke --conc 4 --n 2
Resume: rows are keyed (task_id, arm) in --out; existing rows are skipped.
"""
import argparse, asyncio, importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "ultra")
os.environ.setdefault("ULTRA_GPT_FIRST_TOKEN_S", "300")   # full-effort GPT reasons long before first token
os.environ.setdefault("ULTRA_GPT_TOTAL_S", "600")

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="fs_verdict")
ap.add_argument("--conc", type=int, default=8)
ap.add_argument("--n", type=int, default=60)          # tasks per capability (caps at available)
ap.add_argument("--vllm", default="http://localhost:8007/v1")
ap.add_argument("--adapter", default="default")       # step-145 policy (trainer stopped at 144/145)
ap.add_argument("--out", default="scratchpad/fs_verdict_rows.jsonl")
ap.add_argument("--budget-usd", type=float, default=120.0)  # hard stop on provider-REPORTED spend
ap.add_argument("--timeout-s", type=float, default=900.0)   # per worker call (hardest tasks need more)
ap.add_argument("--pool-upgrade", action="store_true")      # retired no-op (kept so old commands don't error)
ap.add_argument("--pool-swaps", default="")                  # slot swaps, e.g. "st_glm=grok"
ap.add_argument("--handicap", action="store_true")           # TREND protocol: workers 4096/minimal (historical series)
ap.add_argument("--extra-solos", default="")                 # extra baseline workers, e.g. "legacy_opus=opus,legacy_gpt=gpt" (solo arms only — NEVER in the conductor's 4-slot pool)
ap.add_argument("--conductor-model", default=None)           # vLLM model id override for the conductor (base-model ablation)
ap.add_argument("--retry-workers", default="st_opus,st_gpt")  # solo2 arms (bar-plausible workers only)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

# HARD RULE (2026-07-09, the 3h57m batch-157 incident): while a training orchestrator is
# live, evals must not exceed conc 3 — training runs 12-wide against Yunwu's ~16 ceiling.
import subprocess as _sp
if _sp.run(["pgrep", "-f", "surogate grpo-orch"], capture_output=True).stdout.strip():
    if args.conc > 3:
        print(f"TRAINING LIVE: clamping eval concurrency {args.conc} -> 3 (see MISSION invariants)", flush=True)
        args.conc = 3

spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from openai import AsyncOpenAI
from ultra.workers import Sampling
from ultra.workflow import parse_workflow
from ultra.executor import execute_workflow
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
# (--pool-upgrade retired 2026-07-09: user directive — no gemini-3.1-pro anywhere.)
for swap in filter(None, args.pool_swaps.split(",")):
    slot, model = swap.split("=")
    cfg["worker_pool"][slot.strip()]["model"] = model.strip()
    print(f"POOL SWAP: {slot.strip()} -> {model.strip()}", flush=True)
EXTRA_SOLOS = {}
for pair in filter(None, args.extra_solos.split(",")):
    wname, model = pair.split("=")
    EXTRA_SOLOS[wname.strip()] = model.strip()
for wname, model in EXTRA_SOLOS.items():
    tmpl = dict(cfg["worker_pool"][cfg["worker_pool_names"][0]])
    tmpl.update({"name": wname, "model": model, "worker_id": len(cfg["worker_pool_names"])})
    cfg["worker_pool"][wname] = tmpl
    cfg["worker_pool_names"] = cfg["worker_pool_names"] + [wname]
    print(f"EXTRA SOLO BASELINE: {wname} -> {model}", flush=True)
MANIFEST = os.environ.get("EVAL_MANIFEST", "heldout_trend60_taskspecs.jsonl")
tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/{MANIFEST}")]
import random
rng = random.Random(7); by = defaultdict(list)
for t in tasks:
    by[t.capability].append(t)
sample = []
for cap, ts in by.items():
    rng.shuffle(ts); sample += ts[: args.n]
ALL_WORKERS = cfg["worker_pool_names"]
WORKERS = [w for w in ALL_WORKERS if w not in EXTRA_SOLOS]  # the conductor's 4 ordinal slots
# solo2 (retry) arms only for workers that can plausibly BE the bar — retry rows for weak
# workers are spend with no decision value (user directive 2026-07-09: eval only what we need).
RETRY_WORKERS = [w for w in ALL_WORKERS if w in {x.strip() for x in args.retry_workers.split(",")}]

if args.handicap:
    # the historical trend protocol (identical to training + every trend row ever)
    FS_SAMP = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
    WORKER_MAX_TOKENS = {}
else:
    FS_SAMP = Sampling(temperature=0.2, top_p=1.0, max_tokens=16384, reasoning_effort="high")
    # Per-worker full-strength output caps. 16384 truncated gemini/glm on hard tasks (smoke:
    # finish=length) — raise them to their OpenRouter top-provider ceilings' safe zone.
    # Opus never truncates at 16384; GPT ignores caps (wall-clock streaming budget instead).
    WORKER_MAX_TOKENS = {"st_gemini": 32768, "st_glm": 32768}

REVISE_INSTRUCTION = (
    "The workflow above was executed and its final answer was evaluated as INCORRECT.\n"
    "Outcome of the previous attempt:\n{outcome}\n\n"
    "Using what the previous attempt revealed, design a NEW workflow that diagnoses the error and "
    "produces a correct solution to the original question. Output the three lists as before."
)
SOLO_RETRY_INSTRUCTION = (
    "Your previous answer was evaluated as INCORRECT.\n\n"
    "Reconsider the problem from scratch, identify where the previous attempt went wrong, "
    "and give a corrected final answer in the same required format."
)

print(f"FULL-STRENGTH VERDICT: {len(sample)} tasks "
      f"{dict((c, len([t for t in sample if t.capability == c])) for c in by)} | adapter='{args.adapter}' "
      f"conc={args.conc} samp=16384/high budget=${args.budget_usd}", flush=True)

vllm = AsyncOpenAI(base_url=args.vllm, api_key="x", timeout=180.0)
vsem = asyncio.Semaphore(4)

# ---------- accounting: wrap pool.call so EVERY worker call (solo + inside workflows) is metered
TOK = defaultdict(lambda: [0, 0, 0.0, 0])  # worker -> [prompt_toks, completion_toks, cost_usd, calls]
TRUNC = defaultdict(int)                    # worker -> finish_reason=="length" count


def _instrument(pool):
    orig = pool.call
    import dataclasses

    async def call(worker, messages, sampling, **kw):
        cap = WORKER_MAX_TOKENS.get(worker)
        if cap and sampling.max_tokens < cap:
            sampling = dataclasses.replace(sampling, max_tokens=cap)
        c = await orig(worker, messages, sampling, **kw)
        t = TOK[worker]
        t[0] += c.prompt_tokens or 0; t[1] += c.completion_tokens or 0
        t[2] += c.cost_usd or 0.0; t[3] += 1
        if c.finish_reason == "length":
            TRUNC[worker] += 1
        if pool.budget and pool.budget.spent_usd > args.budget_usd:
            raise RuntimeError(f"BUDGET STOP: reported spend ${pool.budget.spent_usd:.2f} > cap")
        return c

    pool.call = call
    return pool


# ---------- rows / resume
DONE: dict[tuple, dict] = {}
if Path(args.out).exists():
    for line in open(args.out):
        r = json.loads(line)
        DONE[(r["task_id"], r["arm"])] = r
outf = open(args.out, "a")
outlock = asyncio.Lock()


async def emit(row):
    async with outlock:
        outf.write(json.dumps(row) + "\n"); outf.flush()
    DONE[(row["task_id"], row["arm"])] = row


# ---------- conductor helpers
async def gen_plan(msgs):
    async with vsem:
        for _ in range(3):
            try:
                r = await vllm.chat.completions.create(
                    model=args.conductor_model or args.adapter, messages=[dict(m) for m in msgs],
                    temperature=1.0, top_p=1.0, max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                txt = r.choices[0].message.content or ""
                if txt.strip():
                    return txt
            except Exception:
                await asyncio.sleep(3)
        return ""


async def run_workflow(task, raw, pool, rid):
    """Parse + execute + grade one conductor plan. Returns (score, steps, status, outcome_text)."""
    try:
        wf = parse_workflow(env._extract_workflow_payload(raw))
    except Exception:
        return 0.0, 0, ("parse_fail" if raw.strip() else "empty_gen"), "(unparseable workflow)"
    try:
        rec = await execute_workflow(task, wf, pool, FS_SAMP, rollout_id=rid,
                                     worker_ids=WORKERS, worker_harnesses={}, max_steps=5)
        ok = 1.0 if (rec.grade and rec.grade.success) else 0.0
        outcome = rec.execution.steps[-1].text[:1500] if rec.execution.steps else ""
        return ok, len(wf.steps), "ok", outcome
    except Exception as e:
        return 0.0, len(wf.steps), "exec_error", f"(execution error: {type(e).__name__})"


async def grade(task, text):
    s = await asyncio.wait_for(
        asyncio.to_thread(get_grader(task.grader.type), text or "", task.grader.expected_answer), timeout=90.0)
    return float(s >= task.grader.success_threshold)


# ---------- arms
async def arm_solo(task, w, pool):
    key = (task.task_id, f"solo__{w}")
    if key in DONE:
        return
    try:
        c = await pool.call(w, [dict(m) for m in task.input.messages], FS_SAMP)
        score = await grade(task, c.text)
        await emit({"task_id": task.task_id, "arm": f"solo__{w}", "cap": task.capability,
                    "score": score, "status": "ok", "answer": (c.text or "")[:1500],
                    "finish": c.finish_reason})
    except Exception as e:
        await emit({"task_id": task.task_id, "arm": f"solo__{w}", "cap": task.capability,
                    "score": 0.0, "status": f"error:{type(e).__name__}", "answer": ""})


async def arm_solo2(task, w, pool):
    key = (task.task_id, f"solo2__{w}")
    if key in DONE:
        return
    base = DONE.get((task.task_id, f"solo__{w}"))
    if base is None:
        return  # solo phase failed to produce a row; skip
    if base["score"] >= 1.0:
        await emit({"task_id": task.task_id, "arm": f"solo2__{w}", "cap": task.capability,
                    "score": 1.0, "status": "carried", "retried": False})
        return
    msgs = [dict(m) for m in task.input.messages]
    msgs.append({"role": "assistant", "content": (base.get("answer") or "")[:4000]})
    msgs.append({"role": "user", "content": SOLO_RETRY_INSTRUCTION})
    try:
        c = await pool.call(w, msgs, FS_SAMP)
        score = await grade(task, c.text)
        await emit({"task_id": task.task_id, "arm": f"solo2__{w}", "cap": task.capability,
                    "score": score, "status": "ok", "retried": True})
    except Exception as e:
        await emit({"task_id": task.task_id, "arm": f"solo2__{w}", "cap": task.capability,
                    "score": 0.0, "status": f"error:{type(e).__name__}", "retried": True})


async def arm_fu1(task, pool, i):
    key = (task.task_id, "fu1")
    if key in DONE:
        return
    msgs = env._prompt_for_task(task, cfg, "single_turn", max_task_chars=12000)
    raw = await gen_plan(msgs)
    score, steps, status, outcome = await run_workflow(task, raw, pool, f"fs1_{args.label}_{i}")
    await emit({"task_id": task.task_id, "arm": "fu1", "cap": task.capability, "score": score,
                "status": status, "steps": steps, "plan": raw[:2000], "outcome": outcome})


async def arm_fu2(task, pool, i):
    key = (task.task_id, "fu2")
    if key in DONE:
        return
    base = DONE.get((task.task_id, "fu1"))
    if base is None:
        return
    if base["score"] >= 1.0:
        await emit({"task_id": task.task_id, "arm": "fu2", "cap": task.capability,
                    "score": 1.0, "status": "carried", "retried": False})
        return
    msgs = list(env._prompt_for_task(task, cfg, "single_turn", max_task_chars=12000))
    msgs = [dict(m) for m in msgs]
    msgs.append({"role": "assistant", "content": base.get("plan") or ""})
    msgs.append({"role": "user", "content": REVISE_INSTRUCTION.format(outcome=base.get("outcome") or "")})
    raw = await gen_plan(msgs)
    score, steps, status, outcome = await run_workflow(task, raw, pool, f"fs2_{args.label}_{i}")
    await emit({"task_id": task.task_id, "arm": "fu2", "cap": task.capability, "score": score,
                "status": status, "steps": steps, "retried": True, "plan": raw[:2000], "outcome": outcome})


# ---------- driver
async def main():
    cache = ".ultra_cache/eval" if args.handicap else ".ultra_cache/eval_fullstrength"
    pool = _instrument(env._build_pool(
        pilot_config=cfg, provider_mode="live", cache_dir=cache,
        max_concurrency=args.conc, requests_per_minute=None, timeout_s=args.timeout_s, max_retries=3))
    t0 = time.time()
    # phase 1: solos (the bar) + fu1, concurrently — independent work
    await asyncio.gather(*(
        [arm_solo(t, w, pool) for t in sample for w in ALL_WORKERS]
        + [arm_fu1(t, pool, i) for i, t in enumerate(sample)]))
    # phase 2: retry arms (need phase-1 rows)
    await asyncio.gather(*(
        [arm_solo2(t, w, pool) for t in sample for w in RETRY_WORKERS]
        + [arm_fu2(t, pool, i) for i, t in enumerate(sample)]))
    wall = time.time() - t0

    # ---------- verdict
    N = len(sample)
    arms = sorted({a for (_, a) in DONE})
    rate = {}
    percap = {}
    for a in arms:
        rows = [DONE[(t.task_id, a)] for t in sample if (t.task_id, a) in DONE]
        if not rows:
            continue
        rate[a] = sum(r["score"] for r in rows) / len(rows)
        pc = defaultdict(list)
        for r in rows:
            pc[r["cap"]].append(r["score"])
        percap[a] = {c: round(sum(v) / len(v), 3) for c, v in pc.items()}
    solos = {a: r for a, r in rate.items() if a.startswith("solo__")}
    bw = max(solos, key=solos.get) if solos else None
    oracle = 0.0
    if solos:
        oracle = sum(1 for t in sample if any(
            DONE.get((t.task_id, f"solo__{w}"), {}).get("score", 0) > 0 for w in WORKERS)) / N

    print("\n=============== FULL-STRENGTH VERDICT ===============", flush=True)
    for a in sorted(rate, key=rate.get, reverse=True):
        print(f"{a:16s} {rate[a]:.3f}  {percap[a]}", flush=True)
    if bw:
        print(f"\nbest solo worker: {bw} = {solos[bw]:.3f} | oracle {oracle:.3f}", flush=True)
        for a in ("fu1", "fu2"):
            if a in rate:
                print(f"{a} vs best solo: {rate[a] - solos[bw]:+.3f}", flush=True)
        s2 = {a: r for a, r in rate.items() if a.startswith("solo2__")}
        if s2 and "fu2" in rate:
            b2 = max(s2, key=s2.get)
            print(f"fu2 vs best solo2 ({b2} {s2[b2]:.3f}): {rate['fu2'] - s2[b2]:+.3f}  "
                  f"(the fair loop-vs-loop comparison)", flush=True)
        # paired discordants fu2 vs best solo2
        if "fu2" in rate and s2:
            b2 = max(s2, key=s2.get)
            up = dn = 0
            for t in sample:
                x = DONE.get((t.task_id, "fu2"), {}).get("score")
                y = DONE.get((t.task_id, b2), {}).get("score")
                if x is None or y is None:
                    continue
                up += int(x > y); dn += int(x < y)
            print(f"paired fu2 vs {b2}: +{up}/-{dn} discordant of {N}", flush=True)
    tok_out = {w: t[1] for w, t in TOK.items()}
    spent = pool.budget.spent_usd if pool.budget else 0.0
    print(f"\naccounting: reported ${spent:.2f} | completion tokens {tok_out} | "
          f"truncated(length) {dict(TRUNC) or 'none'} | wall {wall / 60:.0f} min", flush=True)
    with open(os.environ.get("EVAL_TREND_LOG", "output/fugu_ultra_paper/heldout_trend.log"), "a") as f:
        f.write(json.dumps({"label": args.label, "mode": "fullstrength", "n": N,
                            "rates": {a: round(r, 3) for a, r in rate.items()},
                            "percap": percap, "best_solo": bw, "oracle": round(oracle, 3),
                            "reported_usd": round(spent, 2),
                            "completion_tokens": tok_out, "truncated": dict(TRUNC)}) + "\n")
    print("DONE", flush=True)


asyncio.run(main())
