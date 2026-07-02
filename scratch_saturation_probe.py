"""Measure the TRUE learnable band per source AT OUR ACTUAL WORKER SETTINGS (4096 tokens, temp 0.2,
minimal reasoning) -- NOT the full-strength benchmark. The handicap is the paper's mechanism: it pulls
strong models down into the learnable band. Per source: per-worker pass, oracle, best-single, and band
classification (saturated=drop / learnable=keep / all-fail=drop). This answers 'does it add value'."""
import asyncio, json, importlib.util, random, sys
from collections import defaultdict
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
SOURCES = {
    "TACO(code)": "code_hard_taco_taskspecs.jsonl",
    "LCB-hard(code)": "lcb_hard_train_taskspecs.jsonl",
    "Omni-MATH": "reasoning_hard_omnimath_taskspecs.jsonl",
    "RLPR": "reasoning_rlpr_taskspecs.jsonl",
    "Reasoning-Gym": "reasoning_hard_rgym_taskspecs.jsonl",
    "MMLU-Pro": "reasoning_mmlu_pro_taskspecs.jsonl",
    "ARC-AGI-2": "reasoning_arc_agi2_taskspecs.jsonl",
    "BBEH": "reasoning_bbeh_taskspecs.jsonl",
}
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/saturation",
                       max_concurrency=8, requests_per_minute=None, timeout_s=300.0, max_retries=3)
WORKERS = ["st_opus", "st_gemini", "st_gpt", "st_glm"]
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")  # EXACT training settings
gsem = asyncio.Semaphore(4)

async def one(w, t):
    try:
        c = await pool.call(w, [dict(m) for m in t.input.messages], samp)
        async with gsem:
            s = await asyncio.wait_for(asyncio.to_thread(get_grader(t.grader.type), c.text or "", t.grader.expected_answer), timeout=90.0)
        return float(s >= t.grader.success_threshold)
    except Exception:
        return 0.0

async def probe_source(name, path):
    try:
        rows = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/{path}")]
        random.Random(9).shuffle(rows); rows = rows[:PER]
        res = await asyncio.gather(*[one(w, t) for t in rows for w in WORKERS])
        bt = defaultdict(dict); i = 0
        for t in rows:
            for w in WORKERS:
                bt[t.task_id][w] = res[i]; i += 1
        n = len(rows)
        perw = {w: sum(bt[tid][w] for tid in bt)/n for w in WORKERS}
        oracle = sum(1 for tid in bt if max(bt[tid].values()) > 0)/n
        best = max(perw.values())
        learnable = sum(1 for tid in bt if 0 < sum(bt[tid].values()) < len(WORKERS))
        band = "SATURATED(drop)" if best >= 0.9 else ("ALL-FAIL(drop)" if oracle <= 0.1 else "LEARNABLE(keep)")
        cells = " ".join(f"{w.replace('st_','')}={perw[w]:.2f}" for w in WORKERS)
        print(f"  {name:16s} n={n} | {cells} | oracle={oracle:.2f} best={best:.2f} learnable-band={learnable}/{n} | {band}", flush=True)
    except Exception as e:
        print(f"  {name:16s} PROBE-ERROR: {type(e).__name__}: {str(e)[:120]}", flush=True)

print(f"SATURATION @ 4096-tok/minimal (PER={PER}) -- workers {WORKERS}", flush=True)
async def go():
    await asyncio.gather(*[probe_source(n, p) for n, p in SOURCES.items()], return_exceptions=True)
asyncio.run(go())
print("DONE", flush=True)
