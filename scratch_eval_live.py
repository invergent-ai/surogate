"""LIVE held-out eval of the CURRENT conductor policy via the running vLLM endpoint (no pause, no GPU
conflict -- GRPO broadcasts LoRA to vLLM each step, so this evaluates the step-N policy). Generates
the conductor workflow through vLLM, executes it via the worker pool, grades on the held-out trend set
(LCB-V6 code + AIME math). Reports conductor accuracy (overall + per-capability) for the ascending trend.
Usage: .venv/bin/python scratch_eval_live.py <step-label>"""
import asyncio, json, importlib.util, sys
from collections import defaultdict
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from openai import AsyncOpenAI
from ultra.workers import Sampling
from ultra.workflow import parse_workflow
from ultra.executor import execute_workflow
from ultra.grading import get_grader
from ultra.schemas import TaskSpec

LABEL = sys.argv[1] if len(sys.argv) > 1 else "?"
D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/eval_live",
                       max_concurrency=6, requests_per_minute=None, timeout_s=300.0, max_retries=3)
import os as _os
tasks = [TaskSpec.model_validate(json.loads(l)) for l in open(f"{D}/heldout_trend_taskspecs.jsonl")]
tasks = tasks[:int(_os.environ.get("EVAL_N", "999"))]
VLLM_MODEL = "/home/densemax/work/flavius/surogate/output/fugu_ultra_workflow_sft_qwen3_8b"
vllm = AsyncOpenAI(base_url="http://localhost:8007/v1", api_key="x", timeout=120.0)
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")  # worker settings
gsem = asyncio.Semaphore(4); vsem = asyncio.Semaphore(6)

async def gen_workflow(task):
    msgs = env._prompt_for_task(task, cfg, "single_turn", max_task_chars=12000)
    async with vsem:
        try:
            r = await vllm.chat.completions.create(model=VLLM_MODEL, messages=[dict(m) for m in msgs],
                    temperature=0.2, top_p=1.0, max_tokens=1024,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content or ""
        except Exception:
            return ""

async def one(i, task):
    raw = await gen_workflow(task)
    try:
        wf = parse_workflow(env._extract_workflow_payload(raw))
        rec = await execute_workflow(task, wf, pool, samp, rollout_id=f"evl_{LABEL}_{i}", worker_harnesses={}, max_steps=5)
        ok = 1.0 if (rec.grade and rec.grade.success) else 0.0
        steps = len(wf.steps)
    except Exception:
        ok, steps = 0.0, 0
    return task.capability, ok, steps

async def go(): return await asyncio.gather(*[one(i, t) for i, t in enumerate(tasks)])
res = asyncio.run(go())
by = defaultdict(list); allok = []; allsteps = []
for cap, ok, steps in res:
    by[cap].append(ok); allok.append(ok); allsteps.append(steps)
acc = sum(allok)/len(allok)
percap = {c: round(sum(v)/len(v), 3) for c, v in by.items()}
ms = [s for s in allsteps if s]
print(f"STEP {LABEL} | conductor held-out accuracy = {acc:.3f} | per-cap {percap} | mean-workflow-steps {sum(ms)/len(ms) if ms else 0:.2f} | n={len(allok)}", flush=True)
# append to the trend log
with open("output/fugu_ultra_lcb/heldout_trend.log", "a") as f:
    f.write(json.dumps({"step": LABEL, "acc": acc, "percap": percap}) + "\n")
print("DONE", flush=True)
