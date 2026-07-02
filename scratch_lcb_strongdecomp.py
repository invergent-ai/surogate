"""DECISIVE cheap test: does STRONG decomposition beat GPT-solo on LCB-V1? foothold2 only tried
weak drafters (gemini/glm->opus) and they lost to gpt-solo. Here test the coordination the conductor
would actually learn with STRONG workers: gpt->opus-review, opus->gpt-review, best-of-3->opus-synth.
If the best arm > gpt-solo AND recovers gpt failures => real headroom on LCB-V1 => train here.
If not => LCB-V1 is too easy (one worker dominates) => need harder code."""
import asyncio, json, importlib.util, random, sys
from collections import Counter
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/lcb_strong",
                       max_concurrency=8, requests_per_minute=None, timeout_s=300.0, max_retries=3)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
manifest = sys.argv[2] if len(sys.argv) > 2 else "lcb_train_taskspecs.jsonl"
rng = random.Random(13)
allt = [json.loads(l) for l in open(f"{D}/{manifest}")]
rng.shuffle(allt)
tasks = [TaskSpec.model_validate(r) for r in allt[:N]]
print(f"LCB strong-decomp: {N} tasks from {manifest} @ minimal, concurrency 8", flush=True)
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
grader = get_grader("code_exec_stdio")
gsem = asyncio.Semaphore(4)

async def grade(text, t):
    async with gsem:
        try:
            return float(await asyncio.wait_for(asyncio.to_thread(grader, text or "", t.grader.expected_answer), timeout=90.0))
        except Exception:
            return 0.0

async def gen(w, msgs):
    try:
        c = await pool.call(w, msgs, samp); return c.text or ""
    except Exception:
        return ""

def umsg(t): return [dict(m) for m in t.input.messages]
def prob(t): return t.input.messages[-1]["content"]
_REV = "You are an expert competitive-programming reviewer. The candidate solution may have bugs or miss edge cases. Output the COMPLETE corrected Python program in one code block."
_SYN = "You are an expert. Below are candidate solutions to the problem. Synthesize the single best, fully-correct Python program (fix bugs, merge strengths). Output one complete program in a code block."

async def solo(w, t): return await grade(await gen(w, umsg(t)), t)

async def review(drafter, reviewer, t):
    d = await gen(drafter, umsg(t))
    if len(d) < 5: return 0.0
    msgs = [{"role":"system","content":_REV}, {"role":"user","content":f"Problem:\n{prob(t)}\n\nCandidate:\n{d}\n\nReturn the corrected program."}]
    return await grade(await gen(reviewer, msgs), t)

async def synth(t):
    cands = await asyncio.gather(gen("st_gpt", umsg(t)), gen("st_opus", umsg(t)), gen("st_gemini", umsg(t)))
    body = "\n\n".join(f"Candidate {i+1}:\n{c}" for i, c in enumerate(cands) if c)
    msgs = [{"role":"system","content":_SYN}, {"role":"user","content":f"Problem:\n{prob(t)}\n\n{body}\n\nSynthesize the best program."}]
    return await grade(await gen("st_opus", msgs), t)

async def one(t):
    gpt, opus, gO, oG, syn = await asyncio.gather(
        solo("st_gpt", t), solo("st_opus", t),
        review("st_gpt","st_opus",t), review("st_opus","st_gpt",t), synth(t))
    return dict(gpt=gpt, opus=opus, gpt_opus=gO, opus_gpt=oG, synth=syn)

async def go(): return await asyncio.gather(*[one(t) for t in tasks])
rows = asyncio.run(go())
M = len(rows)
arms = ["gpt","opus","gpt_opus","opus_gpt","synth"]
rate = {a: sum(r[a] for r in rows)/M for a in arms}
print("\n=== arm pass rates ===", flush=True)
for a in arms: print(f"  {a}: {rate[a]:.3f}")
gpt = rate["gpt"]
print(f"\nGPT-solo baseline: {gpt:.3f}", flush=True)
for a in ["gpt_opus","opus_gpt","synth"]:
    rec = sum(1 for r in rows if r["gpt"] < 1.0 and r[a] >= 1.0)
    lost = sum(1 for r in rows if r["gpt"] >= 1.0 and r[a] < 1.0)
    print(f"  {a}: {rate[a]:.3f} (lift {rate[a]-gpt:+.3f}) | recovers gpt-fails {rec} | loses gpt-wins {lost}", flush=True)
# best coordination oracle (if conductor could pick the best arm per task)
best_arm_oracle = sum(1 for r in rows if max(r["gpt_opus"], r["opus_gpt"], r["synth"], r["gpt"]) >= 1.0)/M
print(f"\n[gpt-solo + 3 coord arms] oracle: {best_arm_oracle:.3f} (vs gpt-solo {gpt:.3f}) = decomposition-headroom {best_arm_oracle-gpt:+.3f}", flush=True)
print("DONE", flush=True)
