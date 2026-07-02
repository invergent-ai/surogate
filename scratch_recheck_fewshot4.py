"""Gen-only recheck: does the 4-example few-shot set (1/3/4/5-step span) break the
3-step cloning seen with 2 examples, without denting the 1.00 parse rate?"""
import importlib.util, json, random, sys
from collections import Counter, defaultdict

RAW = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"


def main():
    sys.path.insert(0, "ultra")
    spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    from ultra.schemas import TaskSpec
    from ultra.workflow import parse_workflow, validate_workflow
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
    tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/hard_mix_all_taskspecs.jsonl")]
    by = defaultdict(list)
    for t in tasks:
        by[t.capability].append(t)
    rng = random.Random(7); sample = []
    for cap, ts in sorted(by.items()):
        rng.shuffle(ts); sample += ts[:8]
    tok = AutoTokenizer.from_pretrained(RAW)
    prompts = [tok.apply_chat_template(env._prompt_for_task(t, cfg, "single_turn", max_task_chars=12000),
               tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in sample]
    llm = LLM(model=RAW, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.90)
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048, n=2, seed=7))
    n = parsed = 0
    steps = Counter(); workers = Counter(); bycap = defaultdict(Counter)
    for i, o in enumerate(outs):
        for c in o.outputs:
            n += 1
            try:
                wf = parse_workflow(env._extract_workflow_payload(c.text))
                validate_workflow(wf, worker_count=4)
                parsed += 1; steps[len(wf.steps)] += 1
                bycap[sample[i].capability][len(wf.steps)] += 1
                for s in wf.steps:
                    workers[s.worker_id] += 1
            except Exception:
                pass
    print(f"4-fewshot no-think: parse {parsed}/{n} = {parsed/n:.2f}")
    print(f"steps_dist: {dict(sorted(steps.items()))}")
    print(f"steps by capability: {({c: dict(sorted(d.items())) for c, d in sorted(bycap.items())})}")
    print(f"worker_dist: {dict(sorted(workers.items()))}")


if __name__ == "__main__":
    main()
