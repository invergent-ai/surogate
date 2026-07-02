"""Arm C/D probe: does PROMPT-ENFORCED CoT (paper-style prose plan, thinking OFF) beat
native <think> deliberation -- and does deliberation help at all?

  C_cot:    enable_thinking=False + system instruction: brief prose plan, then the JSON.
  D_direct: enable_thinking=False, unmodified prompt (no-deliberation control).

Same 45 tasks / seed 7 / temp 1.0 / n=2 as the thinking probes. Two modes:
  (default)  generate on GPU, save stats + first-parsed workflows (no worker calls)
  --grade    execute + grade the saved workflows (run only when the worker pool is free)
"""
import asyncio, importlib.util, json, os, random, sys
from collections import Counter, defaultdict

RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
N_PER_CAP = 15
N_SAMPLES = 2
BUDGET = 8192
COT_LINE = ("\n\nBefore the JSON, write a brief plan in plain prose -- a few sentences analyzing "
            "how hard the task is, which workers fit which subtasks, and whether a verification "
            "step is worth it. Then output the workflow as a single JSON object.")
OUT = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_cot"


def load_env_and_tasks():
    sys.path.insert(0, "ultra")
    spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    from ultra.schemas import TaskSpec
    cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
    tasks_all = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/hard_mix_all_taskspecs.jsonl")]
    by = defaultdict(list)
    for t in tasks_all:
        by[t.capability].append(t)
    rng = random.Random(7)
    sample = []
    for cap, ts in sorted(by.items()):
        rng.shuffle(ts); sample += ts[:N_PER_CAP]
    return env, cfg, sample


def generate():
    env, cfg, sample = load_env_and_tasks()
    from ultra.workflow import parse_workflow
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(RAW)

    arms = {}
    for arm, cot in (("C_cot", True), ("D_direct", False)):
        ps = []
        for t in sample:
            msgs = env._prompt_for_task(t, cfg, "single_turn", max_task_chars=12000)
            if cot:
                msgs = [dict(m) for m in msgs]
                msgs[0]["content"] = msgs[0]["content"] + COT_LINE
            ps.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        arms[arm] = ps

    llm = LLM(model=RAW, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=BUDGET, n=N_SAMPLES, seed=7)

    os.makedirs(OUT, exist_ok=True)
    for arm, prompts in arms.items():
        outs = llm.generate(prompts, sp)
        rows, first = [], {}
        for i, o in enumerate(outs):
            for c in o.outputs:
                text = c.text
                stripped = text.rsplit("</think>", 1)[1] if "</think>" in text else text
                brace = stripped.find("{")
                plan = stripped[:brace] if brace > 0 else ""
                try:
                    wf = parse_workflow(env._extract_workflow_payload(text))
                except Exception:
                    wf = None
                rows.append({"task_id": sample[i].task_id, "cap": sample[i].capability,
                             "parsed": wf is not None, "steps": len(wf.steps) if wf else 0,
                             "plan_tokens": len(tok(plan).input_ids) if plan.strip() else 0,
                             "total_tokens": len(tok(text).input_ids),
                             "truncated": c.finish_reason == "length", "text": text})
                if wf is not None and sample[i].task_id not in first:
                    first[sample[i].task_id] = json.loads(env._extract_workflow_payload(text))
        with open(f"{OUT}/completions_{arm}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(f"{OUT}/workflows_{arm}.json", "w") as f:
            json.dump(first, f)
        n = len(rows)
        plans = sorted(r["plan_tokens"] for r in rows if r["plan_tokens"])
        tot = sorted(r["total_tokens"] for r in rows)
        s = {"parse_rate": round(sum(r["parsed"] for r in rows) / n, 3),
             "trunc_rate": round(sum(r["truncated"] for r in rows) / n, 3),
             "plan_p50": plans[len(plans) // 2] if plans else 0,
             "total_p50": tot[len(tot) // 2],
             "tasks_with_wf": len(first),
             "steps_dist": dict(sorted(Counter(r["steps"] for r in rows if r["parsed"]).items()))}
        print(f"\n===== Qwen3-8B {arm} (no-think, budget {BUDGET}, {n} samples) =====", flush=True)
        for k, v in s.items():
            print(f"  {k}: {v}", flush=True)


def grade():
    env, cfg, sample = load_env_and_tasks()
    from ultra.workflow import parse_workflow
    from ultra.workers import Sampling
    from ultra.executor import execute_workflow
    task_of = {t.task_id: t for t in sample}
    pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/probe_think",
                           max_concurrency=12, requests_per_minute=None, timeout_s=600.0, max_retries=3)
    samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

    async def run_one(t, wf, tag, i):
        try:
            rec = await execute_workflow(t, wf, pool, samp, rollout_id=f"{tag}_{i}", worker_harnesses={}, max_steps=5)
            return 1.0 if (rec.grade and rec.grade.success) else 0.0
        except Exception:
            return 0.0

    async def run_all():
        results = {}
        for arm in ("C_cot", "D_direct"):
            first = json.load(open(f"{OUT}/workflows_{arm}.json"))
            items = [(task_of[tid], parse_workflow(json.dumps(w))) for tid, w in first.items() if tid in task_of]
            scores = await asyncio.gather(*[run_one(t, wf, arm, i) for i, (t, wf) in enumerate(items)])
            percap = defaultdict(list)
            for (t, _), s in zip(items, scores):
                percap[t.capability].append(s)
            results[arm] = {"n": len(items), "grade": round(sum(scores) / max(1, len(scores)), 3),
                            "percap": {c: round(sum(v) / len(v), 2) for c, v in sorted(percap.items())}}
            print(f"  {arm}: {results[arm]}", flush=True)
        return results

    print("===== EXECUTION (one wf/task, handicapped workers) =====", flush=True)
    results = asyncio.run(run_all())
    with open(f"{OUT}/grades.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    grade() if "--grade" in sys.argv else generate()
