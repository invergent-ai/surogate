"""Test option (b): is there a TIGHTER handicap where CODE (decomposable+verifiable) leaves
decomposition headroom? At 4096/minimal Opus aces code (no room). Tighten the token budget so
Opus fails ~30%, then check if draft->review-fix decomposition BEATS Opus-solo (the paper's mechanism
on a decomposable domain). Budgets 2048 + 1024. No big run -- a small diagnostic."""
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
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/code_handicap",
                       max_concurrency=8, requests_per_minute=None, timeout_s=300.0, max_retries=3)
N = 12
rng = random.Random(21)
tasks = []
for f in ["lcb_hard_train_taskspecs.jsonl", "code_hard_taco_taskspecs.jsonl"]:
    rows = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/{f}")]
    rng.shuffle(rows); tasks += rows[:N // 2]
gsem = asyncio.Semaphore(4)
def prob(t): return t.input.messages[-1]["content"]
async def grade(text, t):
    async with gsem:
        try: return float(await asyncio.wait_for(asyncio.to_thread(get_grader(t.grader.type), text or "", t.grader.expected_answer), timeout=60.0))
        except Exception: return 0.0
async def gen(w, msgs, budget):
    try:
        c = await pool.call(w, msgs, Sampling(temperature=0.2, top_p=1.0, max_tokens=budget, reasoning_effort="minimal"))
        return c.text or ""
    except Exception: return ""
def M(s, u): return [{"role": "system", "content": s}, {"role": "user", "content": u}]

async def solo(w, t, b): return await grade(await gen(w, [dict(m) for m in t.input.messages], b), t)
async def draft_review(t, b):  # gpt drafts -> opus reviews+fixes (code review is reliable, unlike math verify)
    d = await gen("st_gpt", [dict(m) for m in t.input.messages], b)
    if len(d) < 5: return 0.0
    fix = await gen("st_opus", M("You are an expert code reviewer. The candidate solution may have bugs or miss edge cases. Output the COMPLETE corrected program.", f"Problem:\n{prob(t)}\n\nCandidate:\n{d}\n\nCorrected program:"), b)
    return await grade(fix, t)

async def one(t, b):
    o, g, dr = await asyncio.gather(solo("st_opus", t, b), solo("st_gpt", t, b), draft_review(t, b))
    return o, g, dr

for budget in [2048, 1024]:
    async def go(): return await asyncio.gather(*[one(t, budget) for t in tasks])
    res = asyncio.run(go())
    n = len(res)
    opus = sum(r[0] for r in res)/n; gpt = sum(r[1] for r in res)/n; dr = sum(r[2] for r in res)/n
    oracle = sum(1 for r in res if max(r[0], r[1]) > 0)/n
    best_solo = max(opus, gpt)
    recov = sum(1 for r in res if r[0] == 0 and r[2] > 0)
    print(f"@budget={budget}: opus-solo={opus:.2f} gpt-solo={gpt:.2f} | draft->review-fix={dr:.2f} | oracle(2)={oracle:.2f} | decomp-vs-best-solo={dr-best_solo:+.2f} | recovers-opus-fails={recov}/{n}", flush=True)
print("DONE", flush=True)
