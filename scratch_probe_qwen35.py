"""Head-to-head conductor-base probe: Qwen3.5-9B vs Qwen3-8B (raw, thinking ON, budget 8192).

Phase 1 (GPU):  Qwen3.5-9B generates arms A_plain / B_brief -- identical tasks, prompts,
                seed, and sampling as the Qwen3-8B length probe (scratch_probe_think_len).
Phase 2 (API):  execute + grade one arm-A workflow per task for Qwen3.5-9B.
Phase 3 (API):  same execution for Qwen3-8B's stored arm-A completions -> same-budget grade.

Output: side-by-side parse / truncation / think-length / steps / grade table.
"""
import asyncio, importlib.util, json, os, random, sys
from collections import Counter, defaultdict

Q35 = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
Q3_ARM_A = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_think_len/completions_A_plain.jsonl"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
N_PER_CAP = 15
N_SAMPLES = 2
BUDGET = 8192
BRIEF_LINE = ("\n\nKeep your private reasoning brief -- a few hundred tokens of planning at most -- "
              "then emit the workflow JSON.")
OUT = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_q35"


def main():
    sys.path.insert(0, "ultra")
    spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    from ultra.schemas import TaskSpec
    from ultra.workflow import parse_workflow
    from ultra.workers import Sampling
    from ultra.executor import execute_workflow

    cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
    tasks_all = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/hard_mix_all_taskspecs.jsonl")]
    by = defaultdict(list)
    for t in tasks_all:
        by[t.capability].append(t)
    rng = random.Random(7)
    sample = []
    for cap, ts in sorted(by.items()):
        rng.shuffle(ts); sample += ts[:N_PER_CAP]
    task_of = {t.task_id: t for t in sample}
    os.makedirs(OUT, exist_ok=True)

    # ---------- Phase 1: Qwen3.5-9B generation, both arms ----------
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(Q35)

    arms = {}
    for arm, brief in (("A_plain", False), ("B_brief", True)):
        ps = []
        for t in sample:
            msgs = env._prompt_for_task(t, cfg, "single_turn", max_task_chars=12000)
            if brief:
                msgs = [dict(m) for m in msgs]
                msgs[0]["content"] = msgs[0]["content"] + BRIEF_LINE
            ps.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
        arms[arm] = ps

    llm = LLM(model=Q35, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=BUDGET, n=N_SAMPLES, seed=7)

    stats, q35_first = {}, {}
    for arm, prompts in arms.items():
        outs = llm.generate(prompts, sp)
        rows = []
        for i, o in enumerate(outs):
            for c in o.outputs:
                text = c.text
                think = text.split("</think>")[0].replace("<think>", "") if "</think>" in text else ""
                try:
                    wf = parse_workflow(env._extract_workflow_payload(text))
                except Exception:
                    wf = None
                rows.append({"task_id": sample[i].task_id, "cap": sample[i].capability,
                             "parsed": wf is not None, "steps": len(wf.steps) if wf else 0,
                             "think_tokens": len(tok(think).input_ids) if think else 0,
                             "truncated": c.finish_reason == "length", "text": text})
                if arm == "A_plain" and wf is not None and sample[i].task_id not in q35_first:
                    q35_first[sample[i].task_id] = wf
        with open(f"{OUT}/completions_q35_{arm}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        thinks = sorted(r["think_tokens"] for r in rows if r["think_tokens"])
        stats[arm] = {"parse_rate": round(sum(r["parsed"] for r in rows) / n, 3),
                      "trunc_rate": round(sum(r["truncated"] for r in rows) / n, 3),
                      "think_p50": thinks[len(thinks) // 2] if thinks else 0,
                      "think_p90": thinks[int(len(thinks) * 0.9)] if thinks else 0,
                      "steps_dist": dict(sorted(Counter(r["steps"] for r in rows if r["parsed"]).items()))}
        print(f"\n===== Qwen3.5-9B {arm} (budget {BUDGET}, {n} samples) =====", flush=True)
        for k, v in stats[arm].items():
            print(f"  {k}: {v}", flush=True)
    del llm

    # ---------- Phase 2+3: execute one arm-A workflow per task, BOTH models ----------
    q3_first = {}
    for line in open(Q3_ARM_A):
        r = json.loads(line)
        if r["task_id"] in q3_first or r["task_id"] not in task_of:
            continue
        try:
            wf = parse_workflow(env._extract_workflow_payload(r["text"]))
            q3_first[r["task_id"]] = wf
        except Exception:
            pass

    pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/probe_think",
                           max_concurrency=12, requests_per_minute=None, timeout_s=600.0, max_retries=3)
    samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

    async def run_one(t, wf, tag, i):
        try:
            rec = await execute_workflow(t, wf, pool, samp, rollout_id=f"{tag}_{i}", worker_harnesses={}, max_steps=5)
            return 1.0 if (rec.grade and rec.grade.success) else 0.0
        except Exception:
            return 0.0

    async def grade_model(first_map, tag):
        items = [(task_of[tid], wf) for tid, wf in first_map.items()]
        scores = await asyncio.gather(*[run_one(t, wf, tag, i) for i, (t, wf) in enumerate(items)])
        percap = defaultdict(list)
        for (t, _), s in zip(items, scores):
            percap[t.capability].append(s)
        return {"n": len(items), "grade": round(sum(scores) / max(1, len(scores)), 3),
                "percap": {c: round(sum(v) / len(v), 2) for c, v in sorted(percap.items())}}

    async def run_both():
        g35 = await grade_model(q35_first, "q35")
        g3 = await grade_model(q3_first, "q3")
        return g35, g3

    g35, g3 = asyncio.run(run_both())
    print(f"\n===== EXECUTION (arm A, one wf/task, handicapped workers) =====", flush=True)
    print(f"  Qwen3.5-9B: {g35}", flush=True)
    print(f"  Qwen3-8B:   {g3}", flush=True)

    summary = {"qwen3.5-9b": {**stats, "exec": g35}, "qwen3-8b_armA_exec": g3}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
