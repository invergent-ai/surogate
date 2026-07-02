"""LCB FOOTHOLD GATE — the responsible check before spending on GRPO training.
On a sample of LCB-V1 train tasks at the paper's worker setting (minimal reasoning, 4096 tok),
measure the three things that decide whether GRPO can learn:
  (1) HEADROOM   — oracle (best solo/task) vs best single worker
  (2) LEARNABLE  — fraction of tasks with MIXED pass/fail across arms (within-group reward variance)
  (3) DECOMP LIFT— GLM-draft -> Opus-review (proven coordination) vs best single worker; failures recovered
Positive (1)+(3) => launch training. This is decomposition headroom on VERIFIABLE code, not routing oracle."""
import asyncio, json, importlib.util, random, sys, copy
from collections import defaultdict
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/lcb_foothold",
                       max_concurrency=16, requests_per_minute=None, timeout_s=300.0, max_retries=1)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MAXTESTS = 8  # trim for probe-grading speed; affects all arms equally
rng = random.Random(7)
allt = [json.loads(l) for l in open(f"{D}/lcb_train_taskspecs.jsonl")]
rng.shuffle(allt)
tasks = []
for r in allt[:N]:
    ea = r["grader"]["expected_answer"]
    ea["tests"] = ea["tests"][:MAXTESTS]
    tasks.append(TaskSpec.model_validate(r))
print(f"LCB foothold: {len(tasks)} tasks x [4 solo + glm->opus + gemini->opus] @ minimal reasoning, {MAXTESTS} tests/task", flush=True)
SOLO = ["st_opus", "st_gemini", "st_gpt", "st_glm"]
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
grade_sem = asyncio.Semaphore(4)
grader = get_grader("code_exec_stdio")

async def grade(text, t):
    async with grade_sem:
        try:
            s = await asyncio.wait_for(asyncio.to_thread(grader, text or "", t.grader.expected_answer), timeout=60.0)
            return float(s >= t.grader.success_threshold)
        except Exception:
            return 0.0

async def solo(w, t):
    try:
        c = await pool.call(w, [dict(m) for m in t.input.messages], samp)
        return await grade(c.text, t)
    except Exception:
        return 0.0

async def decomp(drafter, t):
    """drafter writes a solution, opus reviews+fixes -> grade the corrected solution."""
    try:
        c1 = await pool.call(drafter, [dict(m) for m in t.input.messages], samp)
        draft = c1.text or ""
    except Exception:
        return 0.0
    prob = t.input.messages[-1]["content"]
    review_msgs = [
        {"role": "system", "content": "You are an expert reviewer. Find and fix any bugs or missed edge cases in the candidate solution. Output the COMPLETE corrected Python program in one code block."},
        {"role": "user", "content": f"Problem:\n{prob}\n\nCandidate solution:\n{draft}\n\nReturn the complete corrected program."},
    ]
    try:
        c2 = await pool.call("st_opus", review_msgs, samp)
        return await grade(c2.text, t)
    except Exception:
        return 0.0

async def one(t):
    res = await asyncio.gather(
        *[solo(w, t) for w in SOLO],
        decomp("st_glm", t),
        decomp("st_gemini", t),
    )
    solos = dict(zip(SOLO, res[:4]))
    return {"tid": t.task_id, "solo": solos, "glm_opus": res[4], "gemini_opus": res[5]}

async def go(): return await asyncio.gather(*[one(t) for t in tasks])
rows = asyncio.run(go())

M = len(rows)
print("\n=== per-worker SOLO pass rate ===", flush=True)
for w in SOLO:
    print(f"  {w}: {sum(r['solo'][w] for r in rows)/M:.3f}")
single = {w: sum(r["solo"][w] for r in rows)/M for w in SOLO}
bw = max(single, key=single.get)
oracle = sum(1 for r in rows if max(r["solo"].values()) > 0)/M
print(f"\noracle (best solo/task): {oracle:.3f} | best single ({bw}): {single[bw]:.3f} | ROUTING-HEADROOM: {oracle-single[bw]:+.3f}", flush=True)
# learnable band: tasks with mixed pass/fail across the 6 arms (solo x4 + 2 decomp)
def arms(r): return list(r["solo"].values()) + [r["glm_opus"], r["gemini_opus"]]
mixed = sum(1 for r in rows if 0 < sum(arms(r)) < len(arms(r)))
allfail = sum(1 for r in rows if sum(arms(r)) == 0)
allpass = sum(1 for r in rows if sum(arms(r)) == len(arms(r)))
print(f"LEARNABLE band (mixed pass/fail across arms): {mixed}/{M} ({mixed/M:.0%}) | all-fail {allfail} | all-pass {allpass}", flush=True)
# decomposition lift vs best single worker
go_pass = sum(r["glm_opus"] for r in rows)/M
ge_pass = sum(r["gemini_opus"] for r in rows)/M
print(f"\n=== DECOMPOSITION (draft->opus-review) vs best single ({bw}={single[bw]:.2f}) ===", flush=True)
print(f"  glm->opus:    {go_pass:.3f}  (lift {go_pass-single[bw]:+.3f})", flush=True)
print(f"  gemini->opus: {ge_pass:.3f}  (lift {ge_pass-single[bw]:+.3f})", flush=True)
rec = sum(1 for r in rows if r["solo"][bw] == 0 and (r["glm_opus"] > 0 or r["gemini_opus"] > 0))
lost = sum(1 for r in rows if r["solo"][bw] > 0 and r["glm_opus"] == 0 and r["gemini_opus"] == 0)
print(f"  decomp RECOVERS {bw} failures: {rec} | decomp LOSES {bw} wins: {lost}", flush=True)
best_decomp_oracle = sum(1 for r in rows if max(r["glm_opus"], r["gemini_opus"], r["solo"][bw]) > 0)/M
print(f"  [{bw} + decomp] oracle: {best_decomp_oracle:.3f} (vs {bw}-solo {single[bw]:.3f})", flush=True)
print("DONE", flush=True)
