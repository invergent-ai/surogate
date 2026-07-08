"""Build the REPAIR lane (recursion-lite, SingleTurnEnv-compatible): for each code task,
precompute one failed SOLO builder attempt (GPT-5.5, the report's builder), embed the broken
artifact into the task prompt, and emit a task whose grader is UNCHANGED (original tests). The
conductor's single trained generation becomes a repair workflow -> trains build-then-debug.

Data-only: no executor/grader/MultiTurnEnv changes. Keeps ONLY tasks the builder FAILED (there
is a real bug to fix) and that aren't hopeless -- the repair band. Round 1 is precomputed
offline (~1 builder call/task); the lane then costs the same as any single-turn lane.
"""
import asyncio, copy, importlib.util, json, os, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate/ultra")
sys.path.insert(0, "/home/densemax/work/flavius/surogate")
spec = importlib.util.spec_from_file_location(
    "fpe", "/home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

D = "/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
BUILDER = "st_gpt"   # report's builder role for code
N_SCAN = int(os.environ.get("REPAIR_SCAN", "70"))
N_KEEP = int(os.environ.get("REPAIR_KEEP", "40"))

code_tasks = [TaskSpec.model_validate(json.loads(l))
              for l in open(f"{D}/hard_mix_code_taskspecs.jsonl")][:N_SCAN]
print(f"scanning {len(code_tasks)} code tasks with solo builder {BUILDER}", flush=True)

pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/repair_build",
                       max_concurrency=8, requests_per_minute=None, timeout_s=300.0, max_retries=2)
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

async def round1(task):
    try:
        c = await pool.call(BUILDER, [dict(m) for m in task.input.messages], samp)
        text = c.text or ""
        grader = get_grader(task.grader.type)
        score = await asyncio.wait_for(
            asyncio.to_thread(grader, text, task.grader.expected_answer), timeout=90.0)
        return text, float(score >= task.grader.success_threshold)
    except Exception as e:
        return f"<builder error: {e}>", -1.0

FEEDBACK_TMPL = (
    "{orig}\n\n"
    "--- A PREVIOUS SOLUTION ATTEMPT (evaluated as INCORRECT) ---\n"
    "{attempt}\n"
    "--- END PREVIOUS ATTEMPT ---\n\n"
    "The attempt above did not pass the tests. Design a workflow that diagnoses what is wrong "
    "and produces a corrected, fully-passing solution to the original problem."
)

async def main():
    results = await asyncio.gather(*[round1(t) for t in code_tasks])
    kept, dead, aced, err = [], 0, 0, 0
    for task, (attempt, score) in zip(code_tasks, results):
        if score < 0:
            err += 1; continue
        if score >= 1.0:
            aced += 1; continue          # builder solved it solo -> nothing to repair
        # failed (score 0): a real bug to fix. Cap attempt length so the prompt stays in budget.
        orig = " ".join(m.get("content", "") for m in task.input.messages if m.get("role") == "user")
        d = copy.deepcopy(json.loads(task.model_dump_json()))
        d["task_id"] = "repair_" + d["task_id"]
        d["capability"] = "unit_code"
        d["input"]["messages"] = [{"role": "user",
            "content": FEEDBACK_TMPL.format(orig=orig, attempt=attempt[:2500])}]
        # grader UNCHANGED: same type + expected_answer -> grades the repair's final output
        d["source"] = {"name": "repair_code", "version": "v1", "policy": "train_allowed",
                       "url_or_ref": "derived:hard_mix_code+gpt_solo_round1", "license": None,
                       "source_commit": None}
        kept.append(d)
        if len(kept) >= N_KEEP:
            break
    print(f"builder outcomes: {aced} aced (dropped), {len(code_tasks)-aced-err-len([1 for _,(_,s) in zip(code_tasks,results) if s>=1 or s<0])} , {err} errored", flush=True)
    print(f"REPAIR BAND kept: {len(kept)} tasks (builder failed -> real bug to fix)", flush=True)

    with open(f"{D}/hard_mix_repair_taskspecs.jsonl", "w") as f:
        for d in kept:
            f.write(json.dumps(d) + "\n")
    # validate + lane-map append
    for d in kept:
        t = TaskSpec.model_validate(d); get_grader(t.grader.type)
    import shutil
    shutil.copy(f"{D}/pilot_config_singleturn.json", f"{D}/pilot_config_singleturn.json.bak_repair")
    c2 = json.load(open(f"{D}/pilot_config_singleturn.json"))
    ids = [d["task_id"] for d in kept]
    before = len(c2["task_ids_by_lane"]["single_turn"])
    c2["task_ids_by_lane"]["single_turn"] = list(dict.fromkeys(c2["task_ids_by_lane"]["single_turn"] + ids))
    json.dump(c2, open(f"{D}/pilot_config_singleturn.json", "w"), indent=1)
    print(f"lane map {before} -> {len(c2['task_ids_by_lane']['single_turn'])}; all TaskSpecs validate", flush=True)
    if kept:
        print("\n--- sample repair prompt (first kept) ---")
        print(kept[0]["input"]["messages"][0]["content"][:700])

asyncio.run(main())
