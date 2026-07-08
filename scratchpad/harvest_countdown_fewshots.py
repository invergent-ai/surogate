"""Harvest paper-faithful Countdown few-shot examples: generate conductor plans for two
Countdown problems from the LIVE vLLM (base model = coldstart flavor, plus the adapter for
comparison), parse them, and print diverse candidates. Generation only — zero worker calls.
The two finalists get executed (few worker calls) at the post-90 GO before entering the prompt."""
import asyncio, copy, json, sys

sys.path.insert(0, "/home/densemax/work/flavius/surogate/ultra")
sys.path.insert(0, "/home/densemax/work/flavius/surogate")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "fpe", "/home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from openai import AsyncOpenAI
from ultra.schemas import TaskSpec

D = "/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
template = json.loads(open(f"{D}/hard_mix_math_taskspecs.jsonl").readline())

COUNTDOWN = [
    ("Using the numbers [3, 7, 25, 50] and the operations +, -, *, / (each number used at most "
     "once), write an arithmetic expression that evaluates exactly to 481. Provide the final "
     "expression in <answer> </answer> tags.", "481"),
    ("Using the numbers [2, 5, 8, 9, 75] and the operations +, -, *, / (each number used at most "
     "once), write an arithmetic expression that evaluates exactly to 632. Provide the final "
     "expression in <answer> </answer> tags.", "632"),
]

def make_task(q, ans, i):
    d = copy.deepcopy(template)
    d["task_id"] = f"countdown_fewshot_{i}"
    d["input"]["messages"] = [{"role": "user", "content": q}]
    d["grader"]["expected_answer"] = ans
    return TaskSpec.model_validate(d)

BASE = "/var/lib/mesh/flavius/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
vllm = AsyncOpenAI(base_url="http://localhost:8007/v1", api_key="x", timeout=180.0)
sem = asyncio.Semaphore(3)

async def gen(task, model, n=6):
    msgs = env._prompt_for_task(task, cfg, "single_turn", max_task_chars=12000)
    outs = []
    async with sem:
        for _ in range(n):
            try:
                r = await vllm.chat.completions.create(
                    model=model, messages=[dict(m) for m in msgs], temperature=1.0, top_p=1.0,
                    max_tokens=1024, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                outs.append(r.choices[0].message.content or "")
            except Exception as e:
                outs.append(f"<ERR {e}>")
    return outs

async def main():
    tasks = [make_task(q, a, i) for i, (q, a) in enumerate(COUNTDOWN)]
    for model_tag, model in (("BASE", BASE), ("ADAPTER", "default")):
        for t in tasks:
            outs = await gen(t, model)
            print(f"\n########## {model_tag} | {t.task_id} ##########")
            for j, o in enumerate(outs):
                try:
                    wf = json.loads(env._extract_workflow_payload(o))
                    shape = [s["worker_id"] for s in wf["steps"]]
                    print(f"--- cand {j}: steps={len(shape)} models={shape} ---")
                    print(o.strip()[:600], "\n")
                except Exception:
                    print(f"--- cand {j}: PARSE FAIL ---")

asyncio.run(main())
