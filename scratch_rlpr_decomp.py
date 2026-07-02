"""DECOMPOSITION probe on RLPR (the all-fail source). Single workers score ~0.1 at 4096/minimal,
but the conductor's value is recovering best-worker FAILURES via decomposition. Test the paper's
emergent patterns (plan->execute; draft->verify->refine) vs best-single -- does multi-step recover
tasks single workers fail? This is the correct 'does it add value' test for all-fail sources."""
import asyncio, json, importlib.util, random, sys
from collections import defaultdict
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
MANIFEST = sys.argv[2] if len(sys.argv) > 2 else "reasoning_rlpr_taskspecs.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/rlpr_decomp",
                       max_concurrency=8, requests_per_minute=None, timeout_s=300.0, max_retries=3)
rng = random.Random(13)
allt = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/{MANIFEST}")]
rng.shuffle(allt); tasks = allt[:N]
print(f"DECOMP probe: {N} tasks from {MANIFEST} @ 4096/minimal", flush=True)
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
WORKERS = ["st_opus", "st_gemini", "st_gpt", "st_glm"]
gsem = asyncio.Semaphore(4)

def prob(t): return t.input.messages[-1]["content"]
async def grade(text, t):
    async with gsem:
        try:
            return float(await asyncio.wait_for(asyncio.to_thread(get_grader(t.grader.type), text or "", t.grader.expected_answer), timeout=60.0))
        except Exception:
            return 0.0
async def gen(w, msgs):
    try:
        c = await pool.call(w, msgs, samp); return c.text or ""
    except Exception:
        return ""
def M(sys_, user): return [{"role": "system", "content": sys_}, {"role": "user", "content": user}]

async def solo(w, t): return await grade(await gen(w, [dict(m) for m in t.input.messages]), t)

async def plan_exec(t):  # step1 plan (no answer) -> step2 execute
    plan = await gen("st_opus", M("You are an expert. Analyze the problem: identify the relevant principle/method and outline the solution steps precisely. Do NOT compute the final numeric answer.", prob(t)))
    if len(plan) < 5: return 0.0
    ans = await gen("st_opus", M("You are an expert. Follow the given plan to solve the problem and compute the final answer. Put the final answer in \\boxed{}.", f"Problem:\n{prob(t)}\n\nPlan:\n{plan}\n\nExecute it and give the final answer."))
    return await grade(ans, t)

async def draft_verify_refine(t):  # draft -> critique -> revise
    draft = await gen("st_opus", [dict(m) for m in t.input.messages])
    if len(draft) < 5: return 0.0
    crit = await gen("st_opus", M("You are a meticulous checker. Find every error in reasoning or computation in the candidate solution. Be specific.", f"Problem:\n{prob(t)}\n\nCandidate:\n{draft}\n\nList the errors."))
    ans = await gen("st_opus", M("You are an expert. Given the problem, a candidate solution, and a critique, produce the corrected final answer. Put it in \\boxed{}.", f"Problem:\n{prob(t)}\n\nCandidate:\n{draft}\n\nCritique:\n{crit}\n\nCorrected final answer:"))
    return await grade(ans, t)

async def one(t):
    solos = await asyncio.gather(*[solo(w, t) for w in WORKERS])
    pe, dvr = await asyncio.gather(plan_exec(t), draft_verify_refine(t))
    return dict(zip(WORKERS, solos)), pe, dvr

async def _go(): return await asyncio.gather(*[one(t) for t in tasks])
res = asyncio.run(_go())
n = len(res)
best_single = {w: sum(r[0][w] for r in res)/n for w in WORKERS}
bw = max(best_single, key=best_single.get)
oracle4 = sum(1 for r in res if max(r[0].values()) > 0)/n
pe_rate = sum(r[1] for r in res)/n; dvr_rate = sum(r[2] for r in res)/n
best_wf = sum(1 for r in res if max(r[1], r[2]) > 0)/n
recov = sum(1 for r in res if max(r[0].values()) == 0 and max(r[1], r[2]) > 0)  # workflows solve what NO worker did
print(f"\nbest single worker ({bw}): {best_single[bw]:.2f} | oracle(best-of-4 solo): {oracle4:.2f}", flush=True)
print(f"plan->execute: {pe_rate:.2f} | draft->verify->refine: {dvr_rate:.2f}", flush=True)
print(f"[best decomposition workflow]: {best_wf:.2f}  (vs best-single {best_single[bw]:.2f}, oracle {oracle4:.2f})", flush=True)
print(f">>> DECOMPOSITION recovers tasks NO single worker solved: {recov}/{n}", flush=True)
print("VERDICT:", "DECOMP ADDS VALUE (keep)" if best_wf > oracle4 + 0.01 else "decomp does NOT beat routing (drop justified)", flush=True)
print("DONE", flush=True)
