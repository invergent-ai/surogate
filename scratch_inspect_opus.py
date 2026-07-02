"""DIAGNOSE: Opus + GPT (via yunwu) grade 0.000 on every LCB task while Gemini/GLM (openrouter) work.
Inspect the RAW worker output to find the failure mode: empty? truncated mid-reasoning? no code block?
provider error? Try reasoning_effort + max_tokens variations to find a setting that emits gradeable code."""
import asyncio, json, importlib.util, sys
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.grading.verifiers import extract_code
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/inspect_opus",
                       max_concurrency=8, requests_per_minute=None, timeout_s=300.0, max_retries=1)
t = TaskSpec.model_validate(json.loads(open(f"{D}/lcb_train_taskspecs.jsonl").readline()))
msgs = [dict(m) for m in t.input.messages]
print(f"task: {t.task_id} | prompt {len(msgs[-1]['content'])} chars\n", flush=True)

async def probe(label, worker, **samp_kw):
    s = Sampling(temperature=0.2, top_p=1.0, **samp_kw)
    try:
        c = await pool.call(worker, msgs, s)
        txt = c.text or ""
        code = extract_code(txt)
        meta = {k: getattr(c, k, None) for k in ("finish_reason", "completion_tokens", "prompt_tokens", "cost_usd", "model")}
        print(f"--- {label} [{worker}] {samp_kw} ---", flush=True)
        print(f"  text_len={len(txt)} code_len={len(code)} meta={meta}", flush=True)
        print(f"  HEAD: {txt[:300]!r}", flush=True)
        print(f"  TAIL: {txt[-300:]!r}", flush=True)
        print("", flush=True)
    except Exception as e:
        print(f"--- {label} [{worker}] EXC: {type(e).__name__}: {str(e)[:200]} ---\n", flush=True)

async def go():
    # baseline that works (gemini/glm via openrouter)
    await probe("gemini-minimal-4096", "st_gemini", max_tokens=4096, reasoning_effort="minimal")
    # the broken ones (opus/gpt via yunwu) at the training setting
    await probe("opus-minimal-4096", "st_opus", max_tokens=4096, reasoning_effort="minimal")
    await probe("gpt-minimal-4096", "st_gpt", max_tokens=4096, reasoning_effort="minimal")
    # variations to find a fix
    await probe("opus-NOreason-4096", "st_opus", max_tokens=4096)
    await probe("opus-minimal-16384", "st_opus", max_tokens=16384, reasoning_effort="minimal")
    await probe("gpt-NOreason-4096", "st_gpt", max_tokens=4096)
    await probe("gpt-minimal-16384", "st_gpt", max_tokens=16384, reasoning_effort="minimal")

asyncio.run(go())
print("DONE", flush=True)
