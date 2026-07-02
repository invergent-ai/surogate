"""THE OBJECTIVE VERDICT: trained conductor vs best single worker on HELD-OUT tasks.

For each held-out task:
  - conductor: generate workflow (base + GRPO LoRA checkpoint) -> parse -> execute_workflow -> grade
  - each worker SOLO: single call -> grade
Reports conductor pass-rate vs best-single-worker pass-rate (+ oracle, complementarity).
Run AFTER training frees the GPUs:
  ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra .venv/bin/python scratch_eval_heldout.py \
     --adapter output/fugu_ultra_singleturn/checkpoints/step_00000040 --n 40
"""
import argparse, asyncio, importlib.util, json, sys
from collections import defaultdict
sys.path.insert(0, "ultra")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="output/fugu_ultra_workflow_sft_qwen3_8b")
ap.add_argument("--adapter", default="")  # GRPO LoRA checkpoint dir; "" = base (untrained) baseline
ap.add_argument("--n", type=int, default=40)  # tasks per capability
ap.add_argument("--conc", type=int, default=12)  # worker pool concurrency; Yunwu-safe max ~12 (16 rate-limits -> 0.000)
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.workflow import parse_workflow
from ultra.executor import execute_workflow
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/eval",
                       max_concurrency=args.conc, requests_per_minute=None, timeout_s=600.0, max_retries=3)
import os as _os
_manifest = _os.environ.get("EVAL_MANIFEST", "heldout_trend_taskspecs.jsonl")  # trend set = LCB-V6 + AIME
tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/{_manifest}")]
import random
rng = random.Random(7); by = defaultdict(list)
for t in tasks: by[t.capability].append(t)
sample = []
for cap, ts in by.items():
    rng.shuffle(ts); sample += ts[:args.n]
print(f"held-out eval: {len(sample)} tasks {dict((c, min(args.n, len(ts))) for c, ts in by.items())}", flush=True)

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda").eval()
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    print(f"loaded conductor adapter: {args.adapter}", flush=True)
else:
    print("NO adapter (untrained base baseline)", flush=True)

def gen_workflow(task):
    msgs = env._prompt_for_task(task, cfg, "single_turn", max_task_chars=12000)
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(txt, return_tensors="pt", truncation=True, max_length=7000).to("cuda")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=1024, do_sample=True, temperature=1.0, top_p=1.0,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
WORKERS = cfg["worker_pool_names"]

async def exec_wf(task, wf, i):
    if wf is None:
        return 0.0
    try:
        rec = await execute_workflow(task, wf, pool, samp, rollout_id=f"eval_{i}", worker_harnesses={}, max_steps=5)
        return 1.0 if (rec.grade and rec.grade.success) else 0.0
    except Exception:
        return 0.0

async def worker_solo(task, w):
    try:
        c = await pool.call(w, [dict(m) for m in task.input.messages], samp)
        score = await asyncio.wait_for(
            asyncio.to_thread(get_grader(task.grader.type), c.text or "", task.grader.expected_answer), timeout=90.0)
        return float(score >= task.grader.success_threshold)
    except Exception:
        return 0.0

async def main():
    # 1) batch-generate all conductor workflows (GPU, sequential but fast)
    wfs = []
    for i, t in enumerate(sample):
        raw = gen_workflow(t)
        try:
            wf = parse_workflow(env._extract_workflow_payload(raw)); wfs.append((wf, len(wf.steps)))
        except Exception:
            wfs.append((None, 0))
        if (i + 1) % 10 == 0: print(f"  generated {i+1}/{len(sample)} workflows", flush=True)
    steps = [ns for _, ns in wfs]
    # 2) execute all workflows + all worker solos CONCURRENTLY
    print("  executing all workflows + solos concurrently...", flush=True)
    cond_scores = await asyncio.gather(*[exec_wf(sample[i], wfs[i][0], i) for i in range(len(sample))])
    solo = await asyncio.gather(*[worker_solo(t, w) for t in sample for w in WORKERS])
    cond = [(sample[i].task_id, sample[i].capability, cond_scores[i]) for i in range(len(sample))]
    return cond, steps, solo

cond, steps, solo = asyncio.run(main())
# aggregate
sp = defaultdict(dict); k = 0
for t in sample:
    for w in WORKERS:
        sp[t.task_id][w] = solo[k]; k += 1
N = len(sample)
cond_rate = sum(c for _, _, c in cond) / N
single = {w: sum(sp[tid][w] for tid in sp) / N for w in WORKERS}
bw = max(single, key=single.get)
oracle = sum(1 for tid in sp if max(sp[tid].values()) > 0) / N
ms = [s for s in steps if s]
print("\n==================== HELD-OUT VERDICT ====================", flush=True)
print(f"CONDUCTOR pass rate:        {cond_rate:.3f}", flush=True)
print(f"best single worker ({bw}):  {single[bw]:.3f}", flush=True)
print(f"  all workers: {dict((w, round(single[w],3)) for w in WORKERS)}", flush=True)
print(f"oracle (best-per-task):     {oracle:.3f}", flush=True)
print(f">>> conductor {'BEATS' if cond_rate > single[bw] else 'does NOT beat'} best single worker: {cond_rate - single[bw]:+.3f}", flush=True)
print(f"conductor workflow steps: mean={sum(ms)/len(ms):.2f} max={max(ms)} (decomposition: >1 = multi-step)", flush=True)
# per-capability breakdown (code vs math) — conductor and each worker
cond_by_cap = defaultdict(list)
for tid, cap, c in cond:
    cond_by_cap[cap].append(c)
cond_percap = {cap: round(sum(v) / len(v), 3) for cap, v in cond_by_cap.items()}
cap_of = {t.task_id: t.capability for t in sample}
wk_percap = {w: {} for w in WORKERS}
for w in WORKERS:
    bycap = defaultdict(list)
    for tid in sp:
        bycap[cap_of[tid]].append(sp[tid][w])
    wk_percap[w] = {cap: round(sum(v) / len(v), 3) for cap, v in bycap.items()}
print(f"conductor per-cap: {cond_percap}", flush=True)
print(f"workers per-cap:   {wk_percap}", flush=True)
# append trend record
with open("output/fugu_ultra_lcb/heldout_trend.log", "a") as _tf:
    _tf.write(json.dumps({
        "adapter": args.adapter or "BASE", "conductor": round(cond_rate, 3),
        "conductor_percap": cond_percap, "best_worker": bw,
        "best_worker_rate": round(single[bw], 3),
        "workers": {w: round(single[w], 3) for w in WORKERS},
        "workers_percap": wk_percap, "oracle": round(oracle, 3),
        "gap_vs_best": round(cond_rate - single[bw], 3),
        "wf_steps_mean": round(sum(ms) / len(ms), 2),
    }) + "\n")
print("DONE", flush=True)
