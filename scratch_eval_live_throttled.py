"""LIVE throttled held-out eval of the CURRENT GRPO policy via the running training vLLM -- NO stop,
NO restart penalty. Queries the live 'default' LoRA adapter (the trained policy broadcast every step)
at localhost:8007, then executes workflows + best-worker solos at LOW worker concurrency so it stays
under Yunwu's ~16-concurrent limit while training already uses 12. Reuses the cached solos
(.ultra_cache/eval) so the best-worker baseline is ~free. Reports conductor vs best-worker vs oracle
+ per-capability, and appends to the trend log.

Evaluates a *moving* adapter (current policy ~ current step) -- fine for the trend, not a frozen ckpt.
Safe to run concurrently with training.

  # smoke test (validate live-adapter query only -- ZERO worker calls, safe anytime):
  PYTHONPATH=ultra .venv/bin/python scratch_eval_live_throttled.py --smoke --n 2
  # full live eval at step 20 (concurrent with training):
  ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra .venv/bin/python scratch_eval_live_throttled.py --label step20 --conc 3
"""
import argparse, asyncio, importlib.util, json, os, sys
from collections import defaultdict

sys.path.insert(0, "ultra")

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="live")            # trend-log label (e.g. "step20")
ap.add_argument("--conc", type=int, default=3)         # worker concurrency (train 12 + this < 16 Yunwu)
ap.add_argument("--n", type=int, default=40)            # tasks per capability (caps at available)
ap.add_argument("--vllm", default="http://localhost:8007/v1")
ap.add_argument("--adapter", default="default")        # LoRA adapter name the training broadcasts
ap.add_argument("--smoke", action="store_true")        # only gen+parse a couple workflows, no worker calls
ap.add_argument("--think", action="store_true")        # enable_thinking=True + larger gen cap (base-ceiling probe)
args = ap.parse_args()

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
_manifest = os.environ.get("EVAL_MANIFEST", "heldout_trend_taskspecs.jsonl")
tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/{_manifest}")]
import random
rng = random.Random(7); by = defaultdict(list)
for t in tasks:
    by[t.capability].append(t)
sample = []
for cap, ts in by.items():
    rng.shuffle(ts); sample += ts[: args.n]
print(f"LIVE eval: {len(sample)} tasks {dict((c, min(args.n, len(ts))) for c, ts in by.items())} "
      f"| adapter='{args.adapter}' conc={args.conc} smoke={args.smoke}", flush=True)

vllm = AsyncOpenAI(base_url=args.vllm, api_key="x", timeout=180.0)
vsem = asyncio.Semaphore(4)  # bound concurrent vLLM conductor gens (light; queued behind training)
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")  # workers (== heldout eval)
WORKERS = cfg["worker_pool_names"]

async def gen_workflow(task):
    """Generate the conductor workflow from the LIVE trained adapter (temp 1.0 == heldout eval)."""
    msgs = env._prompt_for_task(task, cfg, "single_turn", max_task_chars=12000)
    async with vsem:
        for _ in range(3):  # retry if vLLM returns empty (busy behind training rollouts)
            try:
                r = await vllm.chat.completions.create(
                    model=args.adapter, messages=[dict(m) for m in msgs],
                    temperature=1.0, top_p=1.0, max_tokens=(6144 if args.think else 1024),
                    extra_body={"chat_template_kwargs": {"enable_thinking": args.think}})
                txt = r.choices[0].message.content or ""
                if txt.strip():
                    return txt
            except Exception:
                await asyncio.sleep(3)
        return ""

async def exec_wf(task, raw, pool, i):
    try:
        wf = parse_workflow(env._extract_workflow_payload(raw))
        rec = await execute_workflow(task, wf, pool, samp, rollout_id=f"live_{args.label}_{i}",
                                     worker_harnesses={}, max_steps=5)
        return (1.0 if (rec.grade and rec.grade.success) else 0.0, len(wf.steps))
    except Exception:
        return (0.0, 0)

async def worker_solo(task, w, pool):
    try:
        c = await pool.call(w, [dict(m) for m in task.input.messages], samp)
        score = await asyncio.wait_for(
            asyncio.to_thread(get_grader(task.grader.type), c.text or "", task.grader.expected_answer), timeout=90.0)
        return float(score >= task.grader.success_threshold)
    except Exception:
        return 0.0

async def smoke():
    """Validate the live-adapter query end to end WITHOUT touching the worker pool."""
    raws = await asyncio.gather(*[gen_workflow(t) for t in sample[:2]])
    for i, raw in enumerate(raws):
        print(f"\n=== task {i} ({sample[i].capability}) live '{args.adapter}' output ===", flush=True)
        try:
            wf = parse_workflow(env._extract_workflow_payload(raw))
            print(f"PARSED OK: {len(wf.steps)} steps", flush=True)
            print(json.dumps([{"worker_id": s.worker_id, "subtask": s.subtask[:80]} for s in wf.steps], indent=1)[:600], flush=True)
        except Exception as e:
            print(f"PARSE FAILED: {e}\nraw[:400]: {raw[:400]}", flush=True)

async def full():
    pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/eval",
                           max_concurrency=args.conc, requests_per_minute=None, timeout_s=600.0, max_retries=3)
    raws = await asyncio.gather(*[gen_workflow(t) for t in sample])          # live vLLM (bounded by vsem)
    cond_res = await asyncio.gather(*[exec_wf(sample[i], raws[i], pool, i) for i in range(len(sample))])
    solo = await asyncio.gather(*[worker_solo(t, w, pool) for t in sample for w in WORKERS])  # cached => fast
    return cond_res, solo

if args.smoke:
    asyncio.run(smoke()); print("\nSMOKE DONE", flush=True); sys.exit(0)

cond_res, solo = asyncio.run(full())
N = len(sample)
cond_scores = [c for c, _ in cond_res]; steps = [s for _, s in cond_res]
cond_rate = sum(cond_scores) / N
sp = defaultdict(dict); k = 0
for t in sample:
    for w in WORKERS:
        sp[t.task_id][w] = solo[k]; k += 1
single = {w: sum(sp[tid][w] for tid in sp) / N for w in WORKERS}
bw = max(single, key=single.get)
oracle = sum(1 for tid in sp if max(sp[tid].values()) > 0) / N
cond_by_cap = defaultdict(list)
for i, t in enumerate(sample):
    cond_by_cap[t.capability].append(cond_scores[i])
cond_percap = {c: round(sum(v) / len(v), 3) for c, v in cond_by_cap.items()}
ms = [s for s in steps if s]
print("\n==================== LIVE HELD-OUT VERDICT ====================", flush=True)
print(f"CONDUCTOR (live '{args.adapter}'): {cond_rate:.3f}  per-cap {cond_percap}", flush=True)
print(f"best single worker ({bw}):        {single[bw]:.3f}   all: {dict((w, round(single[w], 3)) for w in WORKERS)}", flush=True)
print(f"oracle (best-per-task):           {oracle:.3f}", flush=True)
print(f">>> conductor {'BEATS' if cond_rate > single[bw] else 'does NOT beat'} best worker: {cond_rate - single[bw]:+.3f}", flush=True)
print(f"workflow steps: mean={sum(ms) / len(ms):.2f} max={max(ms) if ms else 0}  (n={N})", flush=True)
with open(os.environ.get("EVAL_TREND_LOG", "output/fugu_ultra_lcb/heldout_trend.log"), "a") as f:
    f.write(json.dumps({"label": args.label, "mode": "live", "conductor": round(cond_rate, 3),
        "conductor_percap": cond_percap, "best_worker": bw, "best_worker_rate": round(single[bw], 3),
        "workers": {w: round(single[w], 3) for w in WORKERS}, "oracle": round(oracle, 3),
        "gap_vs_best": round(cond_rate - single[bw], 3)}) + "\n")
print("DONE", flush=True)
