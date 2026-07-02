"""Headroom + complementarity probe on the PAPER data.
Runs each frontier worker SOLO (single call, paper settings) on a sample of tasks,
grades, and reports: per-worker pass rate, oracle (best-per-task), best-single-worker,
HEADROOM (oracle - best-single), and cross-worker complementarity.
This is the objective's prerequisite: can the conductor beat the best single worker?
"""
import asyncio, json, importlib.util, random, sys
from collections import defaultdict
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/probe",
                       max_concurrency=16, requests_per_minute=None, timeout_s=600.0, max_retries=1)
tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/heldout_eval_taskspecs.jsonl")]
# add the knowledge domain (MMLU-Pro) for a proper CROSS-domain headroom read (math+code+knowledge)
tasks += [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/paper_train_taskspecs.jsonl")
          if json.loads(l)["capability"] == "factual_qa"]
rng = random.Random(11)
by_cap = defaultdict(list)
for t in tasks:
    by_cap[t.capability].append(t)
sample = []
PER = 8
for cap, ts in by_cap.items():
    rng.shuffle(ts); sample += ts[:PER]
print(f"probe: {len(sample)} tasks ({ {c: min(PER, len(ts)) for c, ts in by_cap.items()} }) x 4 workers", flush=True)
workers = ["st_opus", "st_gemini", "st_gpt", "st_glm"]
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

async def one(w, t):
    try:
        c = await pool.call(w, [dict(m) for m in t.input.messages], samp)
        try:
            score = await asyncio.wait_for(
                asyncio.to_thread(get_grader(t.grader.type), c.text or "", t.grader.expected_answer),
                timeout=20.0)
        except asyncio.TimeoutError:
            score = 0.0  # grader hang (e.g. code blocks on stdin) -> treat as fail
        return (t.task_id, t.capability, w, float(score >= t.grader.success_threshold))
    except Exception:
        return (t.task_id, t.capability, w, 0.0)

async def go():
    return await asyncio.gather(*[one(w, t) for t in sample for w in workers])

res = asyncio.run(go())
pw = defaultdict(list); task_pass = defaultdict(dict); cap_of = {}
for tid, cap, w, p in res:
    pw[w].append(p); task_pass[tid][w] = p; cap_of[tid] = cap
N = len(task_pass)
print("\n=== per-worker SOLO pass rate ===", flush=True)
for w in workers:
    print(f"  {w}: {sum(pw[w])/len(pw[w]):.3f}")
oracle = sum(1 for tid in task_pass if max(task_pass[tid].values()) > 0) / N
single = {w: sum(task_pass[tid][w] for tid in task_pass) / N for w in workers}
bw = max(single, key=single.get)
print(f"\noracle (best-per-task): {oracle:.3f} | best single worker ({bw}): {single[bw]:.3f} | HEADROOM: {oracle - single[bw]:+.3f}", flush=True)
comp = sum(1 for tid in task_pass if task_pass[tid][bw] == 0 and max(task_pass[tid].values()) > 0)
print(f"tasks {bw} FAILS but another worker SOLVES (complementarity): {comp}/{N}", flush=True)
capstat = defaultdict(lambda: [0, 0, 0])
for tid in task_pass:
    capstat[cap_of[tid]][0] += (max(task_pass[tid].values()) > 0)
    capstat[cap_of[tid]][1] += task_pass[tid][bw]
    capstat[cap_of[tid]][2] += 1
print("\n=== by capability: oracle / best-single / n ===", flush=True)
for c, (orc, bs, n) in capstat.items():
    print(f"  {c}: oracle {orc}/{n} ({orc/n:.2f}) | {bw} {bs}/{n} ({bs/n:.2f})")
print("DONE", flush=True)
