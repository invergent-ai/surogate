"""FAST re-probe under the PAPER prompt (Fig-13 verbatim, three-list output, anon models).

Arms (raw Qwen3-8B, temp 1.0, 24 tasks x 2 samples = 48 obs/arm):
  P_nothink: enable_thinking=False, budget 2048  -- the paper-faithful mode (prose plan -> lists)
  P_think:   enable_thinking=True,  budget 8192  -- generation-only (parse + think-length)

Execution/grade: P_nothink only, one workflow per task (<=24), conc 12, timeout 240s.
Also reports the worker-selection distribution (models are anonymized ordinals now).
"""
import asyncio, importlib.util, json, os, random, sys
from collections import Counter, defaultdict

RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
N_PER_CAP = 8
N_SAMPLES = 2
OUT = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_paper"


def main():
    sys.path.insert(0, "ultra")
    spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    from ultra.schemas import TaskSpec
    from ultra.workflow import parse_workflow, validate_workflow
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
    print(f"tasks: {len(sample)} ({N_PER_CAP}/cap)", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(RAW)

    def render(think: bool):
        ps = []
        for t in sample:
            msgs = env._prompt_for_task(t, cfg, "single_turn", max_task_chars=12000)
            ps.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think))
        return ps

    llm = LLM(model=RAW, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.90)
    os.makedirs(OUT, exist_ok=True)
    stats, nothink_first = {}, {}
    for arm, think, budget in (("P_nothink", False, 2048), ("P_think", True, 8192)):
        sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=budget, n=N_SAMPLES, seed=7)
        outs = llm.generate(render(think), sp)
        rows = []
        for i, o in enumerate(outs):
            for c in o.outputs:
                text = c.text
                pre = text.split("</think>")[0].replace("<think>", "") if "</think>" in text else ""
                body = text.rsplit("</think>", 1)[1] if "</think>" in text else text
                m = body.find("model")
                plan = body[:m] if m > 0 else ""
                wf = None
                try:
                    wf = parse_workflow(env._extract_workflow_payload(text))
                    validate_workflow(wf, worker_count=4)
                except Exception:
                    wf = None
                rows.append({"task_id": sample[i].task_id, "cap": sample[i].capability,
                             "parsed": wf is not None, "steps": len(wf.steps) if wf else 0,
                             "workers": [s.worker_id for s in wf.steps] if wf else [],
                             "plan_tokens": len(tok(plan).input_ids) if plan.strip() else 0,
                             "think_tokens": len(tok(pre).input_ids) if pre else 0,
                             "truncated": c.finish_reason == "length", "text": text})
                if arm == "P_nothink" and wf is not None and sample[i].task_id not in nothink_first:
                    nothink_first[sample[i].task_id] = wf
        with open(f"{OUT}/completions_{arm}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        plans = sorted(r["plan_tokens"] for r in rows if r["plan_tokens"])
        thinks = sorted(r["think_tokens"] for r in rows if r["think_tokens"])
        wdist = Counter(w for r in rows for w in r["workers"])
        s = {"parse_rate": round(sum(r["parsed"] for r in rows) / n, 3),
             "trunc_rate": round(sum(r["truncated"] for r in rows) / n, 3),
             "plan_p50": plans[len(plans) // 2] if plans else 0,
             "think_p50": thinks[len(thinks) // 2] if thinks else 0,
             "steps_dist": dict(sorted(Counter(r["steps"] for r in rows if r["parsed"]).items())),
             "worker_dist": dict(sorted(wdist.items()))}
        stats[arm] = s
        print(f"\n===== {arm} (budget {budget}, {n} samples) =====", flush=True)
        for k, v in s.items():
            print(f"  {k}: {v}", flush=True)
    del llm

    pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/probe_think",
                           max_concurrency=12, requests_per_minute=None, timeout_s=240.0, max_retries=3)
    samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

    async def run_one(t, wf, i):
        try:
            rec = await execute_workflow(t, wf, pool, samp, rollout_id=f"paper_{i}", worker_harnesses={}, max_steps=5)
            return 1.0 if (rec.grade and rec.grade.success) else 0.0
        except Exception:
            return 0.0

    async def run_all():
        items = [(t, nothink_first[t.task_id]) for t in sample if t.task_id in nothink_first]
        scores = await asyncio.gather(*[run_one(t, wf, i) for i, (t, wf) in enumerate(items)])
        return items, scores

    items, scores = asyncio.run(run_all())
    percap = defaultdict(list)
    for (t, _), s in zip(items, scores):
        percap[t.capability].append(s)
    grade = {"n": len(items), "grade": round(sum(scores) / max(1, len(scores)), 3),
             "percap": {c: round(sum(v) / len(v), 2) for c, v in sorted(percap.items())}}
    print(f"\n===== EXECUTION P_nothink ({len(items)} workflows, handicapped workers) =====", flush=True)
    print(f"  {grade}", flush=True)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({**stats, "exec_nothink": grade}, f, indent=2)
    print(f"\nsummary -> {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
