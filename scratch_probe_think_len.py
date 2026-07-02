"""Follow-up probe (generation-only, zero API cost): think-length vs budget.

Q: at budget 8192, does raw Qwen3-8B actually finish (parse) -- or expand to fill?
Two arms x 45 tasks x 2 samples at temp 1.0:
  A. plain few-shot conductor prompt (same as probe 1)
  B. plain + one system line asking for brief private reasoning
Reports per-arm: parse rate, truncation, think-token distribution, steps distribution.
"""
import importlib.util, json, os, random, sys
from collections import Counter, defaultdict

RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
N_PER_CAP = 15
N_SAMPLES = 2
BUDGET = 8192
BRIEF_LINE = ("\n\nKeep your private reasoning brief -- a few hundred tokens of planning at most -- "
              "then emit the workflow JSON.")
OUT = "/tmp/claude-1000/-home-densemax-work-flavius-surogate/7a24ada0-f10a-4e2c-a6e4-7d77090f15ca/scratchpad/probe_think_len"


def main():
    sys.path.insert(0, "ultra")
    spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    from ultra.schemas import TaskSpec
    from ultra.workflow import parse_workflow

    cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
    tasks_all = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/hard_mix_all_taskspecs.jsonl")]
    by = defaultdict(list)
    for t in tasks_all:
        by[t.capability].append(t)
    rng = random.Random(7)
    sample = []
    for cap, ts in sorted(by.items()):
        rng.shuffle(ts); sample += ts[:N_PER_CAP]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(RAW)

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

    llm = LLM(model=RAW, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=BUDGET, n=N_SAMPLES, seed=7)

    os.makedirs(OUT, exist_ok=True)
    summary = {}
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
        with open(f"{OUT}/completions_{arm}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        thinks = sorted(r["think_tokens"] for r in rows if r["think_tokens"])
        s = {"parse_rate": round(sum(r["parsed"] for r in rows) / n, 3),
             "trunc_rate": round(sum(r["truncated"] for r in rows) / n, 3),
             "think_p50": thinks[len(thinks) // 2] if thinks else 0,
             "think_p90": thinks[int(len(thinks) * 0.9)] if thinks else 0,
             "think_max": thinks[-1] if thinks else 0,
             "steps_dist": dict(sorted(Counter(r["steps"] for r in rows if r["parsed"]).items()))}
        summary[arm] = s
        print(f"\n===== {arm} (budget {BUDGET}, {n} samples) =====", flush=True)
        for k, v in s.items():
            print(f"  {k}: {v}", flush=True)

    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {OUT}/summary.json", flush=True)


if __name__ == "__main__":
    main()
