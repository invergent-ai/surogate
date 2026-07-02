"""Decomposition headroom: does a fixed multi-step workflow beat gemini-SOLO (the best worker)?
gemini-solo  vs  [gemini drafts -> opus verifies+corrects -> final]  on held-out + knowledge.
If the workflow > gemini-solo, decomposition (not routing) is the conductor's path."""
import asyncio, json, importlib.util, sys
sys.path.insert(0,"ultra")
spec=importlib.util.spec_from_file_location('fpe','environments/fugu-ultra-pilot/fugu_ultra_pilot.py')
env=importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec
import random
D="director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg=json.load(open(f"{D}/pilot_config_singleturn.json"))
pool=env._build_pool(pilot_config=cfg, provider_mode='live', cache_dir='.ultra_cache/decomp', max_concurrency=12, requests_per_minute=None, timeout_s=600.0, max_retries=1)
tasks=[TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/heldout_eval_taskspecs.jsonl")]
tasks+=[TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/paper_train_taskspecs.jsonl") if json.loads(l)["capability"]=="factual_qa"]
rng=random.Random(11); from collections import defaultdict
by=defaultdict(list)
for t in tasks: by[t.capability].append(t)
sample=[]
for c,ts in by.items(): rng.shuffle(ts); sample+=ts[:8]
samp=Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
def grade(t,txt): return float(get_grader(t.grader.type)(txt or "", t.grader.expected_answer)>=t.grader.success_threshold)
async def solo(t):
    c=await pool.call("st_gemini",[dict(m) for m in t.input.messages],samp); return grade(t,c.text)
async def workflow(t):
    # step1: gemini drafts
    d=await pool.call("st_gemini",[dict(m) for m in t.input.messages],samp)
    # step2: opus verifies+corrects with the draft in context
    msgs=[dict(m) for m in t.input.messages]+[
        {"role":"assistant","content":d.text or ""},
        {"role":"user","content":"Carefully verify the above solution. If it is wrong, correct it. Give the final answer in the required format."}]
    c=await pool.call("st_opus",msgs,samp); return grade(t,c.text)
async def go():
    s=await asyncio.gather(*[solo(t) for t in sample])
    w=await asyncio.gather(*[workflow(t) for t in sample])
    return s,w
s,w=asyncio.run(go())
N=len(sample)
print(f"n={N}")
print(f"gemini-SOLO:                  {sum(s)/N:.3f}")
print(f"workflow(gemini->opus-verify): {sum(w)/N:.3f}")
print(f">>> decomposition {'BEATS' if sum(w)>sum(s) else 'does NOT beat'} best solo worker: {(sum(w)-sum(s))/N:+.3f}")
rec=sum(1 for i in range(N) if s[i]==0 and w[i]==1); lost=sum(1 for i in range(N) if s[i]==1 and w[i]==0)
print(f"recovered (solo fail->wf pass): {rec} | lost (solo pass->wf fail): {lost}")
print("DONE")
