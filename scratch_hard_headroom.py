"""HARD-task headroom probe. The easy-set probe showed gemini-flash dominating (no headroom).
User's hypothesis: on HARD tasks the cheap fast worker FAILS, exposing complementarity.
Runs each worker SOLO at minimal reasoning (paper's worker handicap) on the 120 hard tasks
(HLE-MC + GPQA-Diamond + AIME), grades exactly, reports oracle vs best-single + complementarity,
broken down BY SOURCE (they differ sharply in difficulty)."""
import asyncio, json, importlib.util, sys
from collections import defaultdict
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/hardprobe",
                       max_concurrency=16, requests_per_minute=None, timeout_s=600.0, max_retries=1)
tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/hard_eval_taskspecs.jsonl")]
def src_of(t): return t.task_id.split("__", 1)[0]
bysrc = defaultdict(int)
for t in tasks: bysrc[src_of(t)] += 1
print(f"hard probe: {len(tasks)} tasks {dict(bysrc)} x 4 workers @ minimal reasoning", flush=True)
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
            score = 0.0
        return (t.task_id, src_of(t), w, float(score >= t.grader.success_threshold))
    except Exception as e:
        return (t.task_id, src_of(t), w, 0.0)

async def go():
    return await asyncio.gather(*[one(w, t) for t in tasks for w in workers])

res = asyncio.run(go())
pw = defaultdict(list); task_pass = defaultdict(dict); src_of_tid = {}
for tid, src, w, p in res:
    pw[w].append(p); task_pass[tid][w] = p; src_of_tid[tid] = src
N = len(task_pass)
print("\n=== per-worker SOLO pass rate (all hard tasks) ===", flush=True)
for w in workers:
    print(f"  {w}: {sum(pw[w])/len(pw[w]):.3f}")
oracle = sum(1 for tid in task_pass if max(task_pass[tid].values()) > 0) / N
single = {w: sum(task_pass[tid][w] for tid in task_pass) / N for w in workers}
bw = max(single, key=single.get)
print(f"\noracle (best-per-task): {oracle:.3f} | best single ({bw}): {single[bw]:.3f} | HEADROOM: {oracle - single[bw]:+.3f}", flush=True)
comp = sum(1 for tid in task_pass if task_pass[tid][bw] == 0 and max(task_pass[tid].values()) > 0)
print(f"tasks {bw} FAILS but another worker SOLVES (complementarity): {comp}/{N}", flush=True)
# per-source: who wins where (the cross-domain routing signal)
print("\n=== by source: per-worker pass | oracle | best-single ===", flush=True)
for src in sorted(bysrc):
    tids = [tid for tid in task_pass if src_of_tid[tid] == src]
    n = len(tids)
    wr = {w: sum(task_pass[tid][w] for tid in tids)/n for w in workers}
    orc = sum(1 for tid in tids if max(task_pass[tid].values()) > 0)/n
    bs = max(wr.values())
    cells = " ".join(f"{w.replace('st_','')}={wr[w]:.2f}" for w in workers)
    print(f"  {src} (n={n}): {cells} | oracle={orc:.2f} best={bs:.2f} head={orc-bs:+.2f}")
print("DONE", flush=True)
