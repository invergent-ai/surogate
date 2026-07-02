"""Stochasticity/guessing CONTROL for the hard-set headroom read.
The hard set is MC-heavy -> cross-worker oracle can be inflated by guessing (4 guessers on a
4-choice MC = 0.68 oracle by luck). This isolates genuine complementarity from luck:
  cross-oracle  = best-of-4 DIFFERENT workers (1 sample each)   <- what the easy/hard probe reports
  self-oracle   = best-of-4 SAME worker (4 samples, temp>0)     <- pure stochasticity/guessing baseline
If cross-oracle <= self-oracle, the apparent 'headroom' is NOT routable complementarity.
Run only AFTER the raw probe, and only if it shows apparent headroom (set WHICH to the sources to test)."""
import asyncio, json, importlib.util, sys
from collections import defaultdict
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

WHICH = set(sys.argv[1:]) or {"hle_mc", "gpqa_diamond", "aime"}  # sources to control
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/hardctrl",
                       max_concurrency=16, requests_per_minute=None, timeout_s=600.0, max_retries=1)
def src_of(t): return t.task_id.split("__", 1)[0]
tasks = [t for t in (TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/hard_eval_taskspecs.jsonl"))
         if src_of(t) in WHICH]
workers = ["st_opus", "st_gemini", "st_gpt", "st_glm"]
SELF_WORKER = "st_gemini"  # the easy-set winner: its 4-sample self-oracle is the luck baseline
K = 4
print(f"control: {len(tasks)} tasks {sorted(WHICH)} | cross={workers} vs self-oracle({SELF_WORKER} x{K})", flush=True)

async def grade(c, t):
    try:
        return await asyncio.wait_for(asyncio.to_thread(
            get_grader(t.grader.type), c.text or "", t.grader.expected_answer), timeout=20.0)
    except Exception:
        return 0.0

async def call(w, t, temp, tag):
    try:
        c = await pool.call(w, [dict(m) for m in t.input.messages],
                            Sampling(temperature=temp, top_p=1.0, max_tokens=4096, reasoning_effort="minimal"), cache_salt=tag)
        s = await grade(c, t)
        return float(s >= t.grader.success_threshold)
    except Exception:
        return 0.0

async def per_task(t):
    cross = await asyncio.gather(*[call(w, t, 0.2, "x") for w in workers])
    selfs = await asyncio.gather(*[call(SELF_WORKER, t, 0.7, f"s{k}") for k in range(K)])
    return (src_of(t), max(cross) > 0, max(selfs) > 0, dict(zip(workers, cross)))

async def go(): return await asyncio.gather(*[per_task(t) for t in tasks])
res = asyncio.run(go())
bysrc = defaultdict(lambda: [0, 0, 0])
for src, cross_ok, self_ok, _ in res:
    bysrc[src][0] += cross_ok; bysrc[src][1] += self_ok; bysrc[src][2] += 1
print("\n=== cross-worker oracle vs same-worker self-oracle (luck baseline) ===", flush=True)
tot = [0, 0, 0]
for src in sorted(bysrc):
    co, so, n = bysrc[src]; tot[0]+=co; tot[1]+=so; tot[2]+=n
    print(f"  {src} (n={n}): cross-oracle={co/n:.2f} self-oracle={so/n:.2f} | genuine-lift={co/n - so/n:+.2f}")
print(f"  ALL (n={tot[2]}): cross-oracle={tot[0]/tot[2]:.2f} self-oracle={tot[1]/tot[2]:.2f} | genuine-lift={tot[0]/tot[2]-tot[1]/tot[2]:+.2f}", flush=True)
print("(genuine-lift > 0 => real routable complementarity beyond stochasticity/guessing)", flush=True)
print("DONE", flush=True)
