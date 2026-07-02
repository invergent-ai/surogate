"""Head-to-head conductor-candidate probe: Qwen3.5-9B vs Qwen3-8B (raw, thinking ON).

Protocol (identical for both): few-shot Conductor prompt, temp 1.0, budget 8192, seed 7,
45 train tasks x 2 samples. Qwen3.5-9B generates fresh (arms A_plain/B_brief); Qwen3-8B's
arm-A completions are reloaded from the earlier length probe. Then BOTH models' arm-A
parsed-first-per-task workflows execute against the handicapped worker pool (shared,
conc 12) and get graded -- same-protocol grade comparison.

Run: CUDA_VISIBLE_DEVICES=0 ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra .venv/bin/python scratch_probe_compare.py
"""
import asyncio, importlib.util, json, os, random, sys
from collections import Counter, defaultdict

Q35 = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
SAVED_Q3 = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_think_len/completions_A_plain.jsonl"
OUT = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_compare"
N_PER_CAP, N_SAMPLES, BUDGET = 15, 2, 8192
BRIEF_LINE = ("\n\nKeep your private reasoning brief -- a few hundred tokens of planning at most -- "
              "then emit the workflow JSON.")


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
    task_by_id = {t.task_id: t for t in sample}

    # ---------- 1. Qwen3.5-9B generation (arms A/B) ----------
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

    llm = LLM(model=Q35, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.90, trust_remote_code=True)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=BUDGET, n=N_SAMPLES, seed=7)

    os.makedirs(OUT, exist_ok=True)
    summary = {"qwen35_9b": {}, "qwen3_8b": {}}
    q35_first = {}
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
        with open(f"{OUT}/q35_completions_{arm}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        thinks = sorted(r["think_tokens"] for r in rows if r["think_tokens"])
        s = {"parse_rate": round(sum(r["parsed"] for r in rows) / n, 3),
             "trunc_rate": round(sum(r["truncated"] for r in rows) / n, 3),
             "think_p50": thinks[len(thinks) // 2] if thinks else 0,
             "think_p90": thinks[int(len(thinks) * 0.9)] if thinks else 0,
             "steps_dist": dict(sorted(Counter(r["steps"] for r in rows if r["parsed"]).items()))}
        summary["qwen35_9b"][arm] = s
        print(f"\n===== Qwen3.5-9B {arm} (budget {BUDGET}, {n} samples) =====", flush=True)
        for k, v in s.items():
            print(f"  {k}: {v}", flush=True)
    del llm

    # ---------- 2. rebuild Qwen3-8B arm-A parsed set from the saved probe ----------
    q3_first = {}
    for line in open(SAVED_Q3):
        r = json.loads(line)
        if r["task_id"] in q3_first or r["task_id"] not in task_by_id:
            continue
        try:
            wf = parse_workflow(env._extract_workflow_payload(r["text"]))
        except Exception:
            continue
        q3_first[r["task_id"]] = wf
    print(f"\nexec sets: qwen35={len(q35_first)} tasks, qwen3_8b={len(q3_first)} tasks", flush=True)

    # ---------- 3. execute + grade both sets (shared pool) ----------
    pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/probe_think",
                           max_concurrency=12, requests_per_minute=None, timeout_s=600.0, max_retries=3)
    samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

    async def run_one(tag, t, wf, i):
        try:
            rec = await execute_workflow(t, wf, pool, samp, rollout_id=f"{tag}_{i}", worker_harnesses={}, max_steps=5)
            return tag, t, 1.0 if (rec.grade and rec.grade.success) else 0.0
        except Exception:
            return tag, t, 0.0

    async def run_all():
        jobs = [run_one("q35", task_by_id[tid], wf, i) for i, (tid, wf) in enumerate(sorted(q35_first.items()))]
        jobs += [run_one("q3", task_by_id[tid], wf, i) for i, (tid, wf) in enumerate(sorted(q3_first.items()))]
        return await asyncio.gather(*jobs)

    results = asyncio.run(run_all())
    for tag, name in (("q35", "qwen35_9b"), ("q3", "qwen3_8b")):
        scores = [(t, s) for g, t, s in results if g == tag]
        bycap = defaultdict(list)
        for t, s in scores:
            bycap[t.capability].append(s)
        g = round(sum(s for _, s in scores) / max(1, len(scores)), 3)
        summary[name]["exec"] = {"n": len(scores), "grade": g,
                                 "percap": {c: round(sum(v) / len(v), 2) for c, v in sorted(bycap.items())}}
        print(f"\n===== {name} EXECUTION: n={len(scores)} grade={g} percap={summary[name]['exec']['percap']} =====", flush=True)

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
