"""Can the SFT+step40 conductor follow the arm-C prose-plan instruction?

Loads the workflow-SFT base + step_00000040 LoRA in vLLM, sends the arm-C prompt
(brief prose plan, then JSON), measures: plan-emission rate (plan_tokens>0), parse rate,
plan length. Prediction: SFT conditioning wins and it emits bare JSON regardless.
"""
import importlib.util, json, random, sys
from collections import defaultdict

SFT = "/home/densemax/work/flavius/surogate/output/fugu_ultra_workflow_sft_qwen3_8b"
LORA = "/home/densemax/work/flavius/surogate/output/fugu_ultra_lcb/run_default/broadcasts/step_40"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
COT_LINE = ("\n\nBefore the JSON, write a brief plan in plain prose -- a few sentences analyzing "
            "how hard the task is, which workers fit which subtasks, and whether a verification "
            "step is worth it. Then output the workflow as a single JSON object.")


def main():
    sys.path.insert(0, "ultra")
    spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
    env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
    from ultra.schemas import TaskSpec
    from ultra.workflow import parse_workflow
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
    tasks_all = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/hard_mix_all_taskspecs.jsonl")]
    by = defaultdict(list)
    for t in tasks_all:
        by[t.capability].append(t)
    rng = random.Random(7)
    sample = []
    for cap, ts in sorted(by.items()):
        rng.shuffle(ts); sample += ts[:15]

    tok = AutoTokenizer.from_pretrained(SFT)
    prompts = []
    for t in sample:
        msgs = env._prompt_for_task(t, cfg, "single_turn", max_task_chars=12000)
        msgs = [dict(m) for m in msgs]
        msgs[0]["content"] = msgs[0]["content"] + COT_LINE
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    llm = LLM(model=SFT, dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.90,
              enable_lora=True, max_lora_rank=16)
    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048, n=2, seed=7)
    outs = llm.generate(prompts, sp, lora_request=LoRARequest("step40", 1, LORA))

    n = parsed = with_plan = 0
    plan_lens = []
    for i, o in enumerate(outs):
        for c in o.outputs:
            n += 1
            text = c.text
            stripped = text.rsplit("</think>", 1)[1] if "</think>" in text else text
            brace = stripped.find("{")
            plan = stripped[:brace].strip() if brace > 0 else ""
            plan_tok = len(tok(plan).input_ids) if plan else 0
            if plan_tok >= 20:  # >= a real sentence, not stray chars
                with_plan += 1
                plan_lens.append(plan_tok)
            try:
                parse_workflow(env._extract_workflow_payload(text))
                parsed += 1
            except Exception:
                pass
    plan_lens.sort()
    print(f"SFT+step40 under the arm-C CoT instruction ({n} samples):")
    print(f"  emits a real plan (>=20 tok): {with_plan}/{n} = {with_plan/n:.2f}")
    print(f"  plan p50: {plan_lens[len(plan_lens)//2] if plan_lens else 0}")
    print(f"  parse rate: {parsed/n:.2f}")


if __name__ == "__main__":
    main()
