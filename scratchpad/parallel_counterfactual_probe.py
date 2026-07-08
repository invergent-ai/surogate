"""PARALLEL-TOPOLOGY COUNTERFACTUAL PROBE (offline; zero interaction with the live run).

Question: would parallel+aggregator plans - which the policy NEVER samples - beat chains on our
setup? Sharpest testbed: the EVICTED FORTRESSES (group avg <= 0.5 at eviction; chains ~never won).
Any parallel win here is strict-improvement evidence; none = the collapse-to-chains is fine.

Design: up to 25 fortresses from the latest orch ckpt hard pool, x2 hand-built workflows:
  P1: leaves [gpt, gemini] independent (access [],[]) -> aggregator opus (access [0,1])
  P2: leaves [opus, glm]  independent                 -> aggregator gpt
Same handicapped workers as training (temp 0.2 / 4096 / minimal / short budget), same
execute_workflow + graders. Workflows built through parse_workflow so semantics are identical.
"""
import asyncio, glob, importlib.util, json, os, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate")
sys.path.insert(0, "/home/densemax/work/flavius/surogate/ultra")
spec = importlib.util.spec_from_file_location(
    "fpe", "/home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.schemas import TaskSpec

D = "/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train"
OUT = os.path.dirname(os.path.abspath(__file__)) + "/parallel_probe_results.jsonl"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
WORKERS = cfg["worker_pool_names"]                       # ordinals: 0 opus, 1 gemini, 2 gpt, 3 glm
N_TASKS = int(os.environ.get("PROBE_TASKS", "25"))

# ---- fortress selection from the latest orch checkpoint ----
ckpts = sorted(glob.glob("output/fugu_ultra_paper/run_default/checkpoints/step_*/orchestrator/buffer/hard_examples.jsonl"),
               key=lambda p: int(p.split("step_")[1].split("/")[0]))
hard = [json.loads(l) for l in open(ckpts[-1])]
print(f"hard pool: {len(hard)} entries from {ckpts[-1].split('run_default/')[1]}", flush=True)

specs = {}
for fn in ("hard_mix_math_taskspecs.jsonl", "hard_mix_code_taskspecs.jsonl",
           "hard_mix_rlpr_taskspecs.jsonl", "hard_mix_repair_taskspecs.jsonl"):
    for l in open(f"{D}/{fn}"):
        t = TaskSpec.model_validate(json.loads(l))
        specs[t.task_id] = t

# round-robin across envs for diversity; cap N_TASKS
by_env = {}
for h in hard:
    info = h.get("info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except Exception:
            info = {}
    tid = (info or {}).get("task_id") or h.get("answer")
    if tid in specs:
        by_env.setdefault(h.get("task", "?"), []).append(tid)
sel, i = [], 0
while len(sel) < N_TASKS and any(by_env.values()):
    for k in sorted(by_env):
        if by_env[k] and len(sel) < N_TASKS:
            sel.append((k, by_env[k].pop(0)))
print(f"selected {len(sel)} fortresses: " +
      ", ".join(f"{k.split('_')[-1]}:{sum(1 for e,_ in sel if e==k)}" for k in sorted(set(e for e,_ in sel))), flush=True)

pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/parallel_probe",
                       max_concurrency=6, requests_per_minute=None, timeout_s=300.0, max_retries=3)
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")
harn = env._worker_harness_overrides(cfg)
MAXS = int(cfg["workflow_policy"]["max_workflow_steps"])

LEAF = "Independently solve the original task end-to-end. Give your complete final answer in the required format."
AGG = ("You are given two candidate solutions produced independently. Compare them, identify any errors "
       "or disagreements, and produce the single correct final answer to the original task in the required format.")

def wf_text(models):
    subs = json.dumps([LEAF, LEAF, AGG])
    return f"model_id = {models}\nsubtasks = {subs}\naccess_list = [[], [], [0, 1]]"

VARIANTS = {"P1_gpt+gem->opus": [2, 1, 0], "P2_opus+glm->gpt": [0, 3, 2]}

async def run_one(env_name, tid, vname, models):
    task = specs[tid]
    raw = wf_text(models)
    wf = env.parse_workflow(env._extract_workflow_payload(raw))
    wf = wf.model_copy(update={"steps": [s.model_copy(update={"budget": "short"}) for s in wf.steps]})
    try:
        rec = await env.execute_workflow(task, wf, pool, samp, f"pprobe-{tid[:18]}-{vname[:2]}",
                                         worker_ids=WORKERS, worker_harnesses=harn,
                                         raw_output=raw, max_steps=MAXS)
        r = float(rec.reward or 0.0)
    except Exception as ex:
        print(f"  {tid[:40]} {vname}: ERROR {str(ex)[:120]}", flush=True)
        return None
    row = {"env": env_name, "task_id": tid, "variant": vname, "reward": r}
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    if r >= 1.0:
        print(f"  CRACKED: {tid[:46]} by {vname}", flush=True)
    return row

async def main():
    open(OUT, "w").close()
    jobs = [run_one(e, t, v, m) for e, t in sel for v, m in VARIANTS.items()]
    rows = [r for r in await asyncio.gather(*jobs) if r]
    print(f"\n=== PARALLEL COUNTERFACTUAL RESULTS ({len(rows)} runs on {len(sel)} fortresses) ===", flush=True)
    for v in VARIANTS:
        vs = [r for r in rows if r["variant"] == v]
        w = sum(1 for r in vs if r["reward"] >= 1.0)
        print(f"{v:20} cracked {w}/{len(vs)}  (win {w/max(1,len(vs)):.1%})", flush=True)
    per = {}
    for r in rows:
        per.setdefault(r["task_id"], []).append(r["reward"])
    union = sum(1 for rs in per.values() if max(rs) >= 1.0)
    print(f"{'UNION (either)':20} cracked {union}/{len(per)}  (chains baseline on these: ~0 by construction)", flush=True)
    by_env_w = {}
    for r in rows:
        by_env_w.setdefault(r["env"], [0, 0])
        by_env_w[r["env"]][1] += 1
        by_env_w[r["env"]][0] += int(r["reward"] >= 1.0)
    print("per env: " + "  ".join(f"{k.split('_')[-1]} {v[0]}/{v[1]}" for k, v in sorted(by_env_w.items())), flush=True)
    print("PROBE DONE", flush=True)

asyncio.run(main())
