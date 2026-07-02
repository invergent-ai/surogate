"""LCB foothold, INSTRUMENTED. The first probe silently turned call-errors into 0.0, so Yunwu
rate-limiting the Opus/GPT burst (concurrency 16) looked like 0/30 capability. Here: moderate
concurrency, retries, and per-worker status {ok-pass, ok-wrong, CALL_ERROR, GRADE_ERROR} so we
separate infra failures from real wrong answers. Also measures decomposition lift on VERIFIABLE code."""
import asyncio, json, importlib.util, random, sys
from collections import defaultdict, Counter
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/lcb_foothold2",
                       max_concurrency=CONC, requests_per_minute=None, timeout_s=300.0, max_retries=3)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
rng = random.Random(7)
allt = [json.loads(l) for l in open(f"{D}/lcb_train_taskspecs.jsonl")]
rng.shuffle(allt)
tasks = [TaskSpec.model_validate(r) for r in allt[:N]]
print(f"LCB foothold2: {N} tasks x [4 solo + gemini->opus + glm->opus] @ minimal, concurrency={CONC}, retries=3", flush=True)
SOLO = ["st_opus", "st_gemini", "st_gpt", "st_glm"]
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
grader = get_grader("code_exec_stdio")
gsem = asyncio.Semaphore(4)
callfail = Counter()

async def grade(text, t):
    async with gsem:
        try:
            return float(await asyncio.wait_for(asyncio.to_thread(grader, text or "", t.grader.expected_answer), timeout=90.0))
        except Exception:
            return -1.0  # GRADE_ERROR

async def gen(w, msgs):
    try:
        c = await pool.call(w, msgs, samp)
        return c.text or ""
    except Exception as e:
        callfail[w] += 1
        return None  # CALL_ERROR

async def solo(w, t):
    txt = await gen(w, [dict(m) for m in t.input.messages])
    if txt is None:
        return "CALL_ERROR"
    s = await grade(txt, t)
    return "GRADE_ERROR" if s < 0 else ("pass" if s >= 1.0 else "wrong")

async def decomp(drafter, t):
    draft = await gen(drafter, [dict(m) for m in t.input.messages])
    if draft is None or len(draft) < 5:
        return "CALL_ERROR"
    prob = t.input.messages[-1]["content"]
    rev = [
        {"role": "system", "content": "You are an expert reviewer. Find and fix any bugs or missed edge cases in the candidate solution. Output the COMPLETE corrected Python program in one code block."},
        {"role": "user", "content": f"Problem:\n{prob}\n\nCandidate solution:\n{draft}\n\nReturn the complete corrected program."},
    ]
    txt = await gen("st_opus", rev)
    if txt is None:
        return "CALL_ERROR"
    s = await grade(txt, t)
    return "GRADE_ERROR" if s < 0 else ("pass" if s >= 1.0 else "wrong")

async def one(t):
    r = await asyncio.gather(*[solo(w, t) for w in SOLO], decomp("st_gemini", t), decomp("st_glm", t))
    return {"solo": dict(zip(SOLO, r[:4])), "gem_opus": r[4], "glm_opus": r[5]}

async def go(): return await asyncio.gather(*[one(t) for t in tasks])
rows = asyncio.run(go())
M = len(rows)
print(f"\n=== per-worker status (n={M}) ===", flush=True)
for w in SOLO:
    c = Counter(r["solo"][w] for r in rows)
    pr = c["pass"]/M
    print(f"  {w}: pass={c['pass']} wrong={c['wrong']} CALL_ERR={c['CALL_ERROR']} GRADE_ERR={c['GRADE_ERROR']} | pass_rate(of all)={pr:.2f}", flush=True)
print(f"call failures: {dict(callfail)}", flush=True)
# headroom + decomposition on tasks where ALL solo calls succeeded (clean)
def ok(v): return v in ("pass", "wrong")
single = {w: sum(r["solo"][w] == "pass" for r in rows)/M for w in SOLO}
bw = max(single, key=single.get)
oracle = sum(1 for r in rows if any(r["solo"][w] == "pass" for w in SOLO))/M
print(f"\noracle(best solo/task)={oracle:.2f} | best single({bw})={single[bw]:.2f} | routing-headroom={oracle-single[bw]:+.2f}", flush=True)
gem = sum(r["gem_opus"] == "pass" for r in rows)/M
glm = sum(r["glm_opus"] == "pass" for r in rows)/M
print(f"\n=== DECOMPOSITION (draft->opus-review) vs best single ({bw}={single[bw]:.2f}) ===", flush=True)
print(f"  gemini->opus: {gem:.2f} (lift {gem-single[bw]:+.2f}) | glm->opus: {glm:.2f} (lift {glm-single[bw]:+.2f})", flush=True)
rec = sum(1 for r in rows if r["solo"][bw] != "pass" and (r["gem_opus"]=="pass" or r["glm_opus"]=="pass"))
print(f"  decomp RECOVERS {bw} non-passes: {rec}/{M}", flush=True)
allarms = lambda r: [r["solo"][w] for w in SOLO] + [r["gem_opus"], r["glm_opus"]]
mixed = sum(1 for r in rows if 0 < sum(a=="pass" for a in allarms(r)) < sum(ok(a) or a=="CALL_ERROR" for a in allarms(r)))
print(f"  LEARNABLE band (mixed pass across arms): ~{mixed}/{M}", flush=True)
print("DONE", flush=True)
