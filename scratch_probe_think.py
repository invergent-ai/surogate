"""STEP-1 PROBE for the thinking-conductor track: can RAW Qwen3-8B (native thinking, no SFT)
produce parseable, scoring workflows under the few-shot Conductor prompt?

Measures, at the TRAINING operating point (temp 1.0, handicapped workers):
  - parse rate (the paper's format condition) over n_samples per task
  - grade rate of one executed workflow per task (vs the SFT conductor's ~0.65 train grade)
  - think length (tokens), truncation rate at the 4096 cap, workflow step distribution

Verdict rule: parse rate >= ~0.7 -> skip SFT (paper-style raw+few-shot GRPO);
much lower -> self-distilled cold-start SFT first.

Run: CUDA_VISIBLE_DEVICES=0 ULTRA_ALLOW_YUNWU=1 PYTHONPATH=ultra .venv/bin/python scratch_probe_think.py
(All work lives under main() -- vLLM v1 spawns its engine core, which re-imports this module.)
"""
import asyncio, importlib.util, json, os, random, sys
from collections import Counter, defaultdict

RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
N_PER_CAP = 15
N_SAMPLES = 2          # completions per task -> 60 parse observations
MAX_THINK_GEN = 4096   # the training budget we'd use
OUT = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_think"


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
    print(f"probe tasks: {len(sample)} {dict((c, min(N_PER_CAP, len(ts))) for c, ts in by.items())}", flush=True)

    # ---------- 1. generate with RAW Qwen3-8B, thinking ON ----------
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(RAW)
    prompts = []
    for t in sample:
        msgs = env._prompt_for_task(t, cfg, "single_turn", max_task_chars=12000)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
    print(f"prompt tokens: max={max(len(tok(p).input_ids) for p in prompts)}", flush=True)

    llm = LLM(model=RAW, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_THINK_GEN, n=N_SAMPLES, seed=7)
    outs = llm.generate(prompts, sp)

    os.makedirs(OUT, exist_ok=True)
    rows, parsed_first = [], {}
    for i, o in enumerate(outs):
        for j, c in enumerate(o.outputs):
            text = c.text
            think = text.split("</think>")[0].replace("<think>", "") if "</think>" in text else ""
            wf, err = None, ""
            try:
                wf = parse_workflow(env._extract_workflow_payload(text))
            except Exception as e:
                err = str(e)[:120]
            rows.append({
                "task_id": sample[i].task_id, "cap": sample[i].capability, "sample": j,
                "parsed": wf is not None, "steps": len(wf.steps) if wf else 0,
                "think_tokens": len(tok(think).input_ids) if think else 0,
                "truncated": c.finish_reason == "length", "err": err, "text": text,
            })
            if wf is not None and sample[i].task_id not in parsed_first:
                parsed_first[sample[i].task_id] = wf
    with open(f"{OUT}/completions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n = len(rows)
    parse_rate = sum(r["parsed"] for r in rows) / n
    trunc_rate = sum(r["truncated"] for r in rows) / n
    thinks = sorted(r["think_tokens"] for r in rows if r["think_tokens"])
    steps_dist = Counter(r["steps"] for r in rows if r["parsed"])
    print(f"\n===== GENERATION ({n} samples) =====", flush=True)
    print(f"parse rate:      {parse_rate:.2f}   (truncated: {trunc_rate:.2f})", flush=True)
    if thinks:
        print(f"think tokens:    mean={sum(thinks)/len(thinks):.0f}  p50={thinks[len(thinks)//2]}  p90={thinks[int(len(thinks)*0.9)]}  max={thinks[-1]}", flush=True)
    print(f"steps dist:      {dict(sorted(steps_dist.items()))}", flush=True)
    print(f"tasks with >=1 parsed workflow: {len(parsed_first)}/{len(sample)}", flush=True)
    del llm  # free GPU before the API phase

    # ---------- 2. execute + grade one workflow per task (handicapped workers) ----------
    pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/probe_think",
                           max_concurrency=12, requests_per_minute=None, timeout_s=600.0, max_retries=3)
    samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

    async def run_one(t, wf, i):
        try:
            rec = await execute_workflow(t, wf, pool, samp, rollout_id=f"probe_{i}", worker_harnesses={}, max_steps=5)
            return 1.0 if (rec.grade and rec.grade.success) else 0.0
        except Exception:
            return 0.0

    async def run_all():
        items = [(t, parsed_first[t.task_id]) for t in sample if t.task_id in parsed_first]
        scores = await asyncio.gather(*[run_one(t, wf, i) for i, (t, wf) in enumerate(items)])
        return items, scores

    items, scores = asyncio.run(run_all())
    by_cap = defaultdict(list)
    for (t, _), s in zip(items, scores):
        by_cap[t.capability].append(s)
    grade = sum(scores) / max(1, len(scores))
    print(f"\n===== EXECUTION ({len(items)} workflows, handicapped workers) =====", flush=True)
    print(f"grade rate:      {grade:.2f}   per-cap: {dict((c, round(sum(v)/len(v), 2)) for c, v in by_cap.items())}", flush=True)

    summary = {"parse_rate": round(parse_rate, 3), "trunc_rate": round(trunc_rate, 3),
               "tasks_parsed": f"{len(parsed_first)}/{len(sample)}", "grade": round(grade, 3),
               "grade_percap": {c: round(sum(v)/len(v), 3) for c, v in by_cap.items()},
               "think_p50": thinks[len(thinks)//2] if thinks else 0,
               "think_p90": thinks[int(len(thinks)*0.9)] if thinks else 0,
               "steps_dist": dict(sorted(steps_dist.items()))}
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    verdict = "SKIP SFT (paper-style raw+few-shot GRPO viable)" if parse_rate >= 0.7 else "COLD-START SFT NEEDED (self-distill)"
    print(f"\n===== VERDICT: {verdict} =====\nsummary -> {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
