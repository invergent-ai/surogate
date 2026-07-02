"""Definitive: do Opus/GPT actually PASS the LCB tests, or fail (and why)? Grade the cached
outputs on the real task with FULL tests; on failure, run test[0] manually and show rc/stdout/expected."""
import asyncio, json, importlib.util, sys
sys.path.insert(0, "ultra")
spec = importlib.util.spec_from_file_location("fpe", "environments/fugu-ultra-pilot/fugu_ultra_pilot.py")
env = importlib.util.module_from_spec(spec); spec.loader.exec_module(env)
from ultra.workers import Sampling
from ultra.grading import get_grader
from ultra.grading.verifiers import extract_code, _run_capped
from ultra.schemas import TaskSpec

D = "director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/inspect_opus",
                       max_concurrency=8, requests_per_minute=None, timeout_s=300.0, max_retries=1)
t = TaskSpec.model_validate(json.loads(open(f"{D}/lcb_train_taskspecs.jsonl").readline()))
ea = t.grader.expected_answer
print(f"task {t.task_id}: {len(ea['tests'])} full tests, threshold={t.grader.success_threshold}\n", flush=True)
grader = get_grader("code_exec_stdio")
samp = Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal")

async def check(w):
    c = await pool.call(w, [dict(m) for m in t.input.messages], samp)  # cache hit
    txt = c.text or ""
    code = extract_code(txt)
    score = grader(txt, ea)
    print(f"=== {w}: score={score} | code_len={len(code)} ===", flush=True)
    if score == 0.0 and code:
        t0 = ea["tests"][0]
        rc, out = _run_capped([sys.executable, "-c", code], stdin_data=t0.get("input", ""), timeout=ea.get("timeout", 10))
        print(f"  test[0] rc={rc}", flush=True)
        print(f"  STDIN:    {t0.get('input','')[:160]!r}", flush=True)
        print(f"  GOT:      {out.strip()[:200]!r}", flush=True)
        print(f"  EXPECTED: {str(t0.get('output','')).strip()[:200]!r}", flush=True)
        # how many of the full tests pass?
        npass = 0
        for tt in ea["tests"]:
            rc2, o2 = _run_capped([sys.executable, "-c", code], stdin_data=tt.get("input",""), timeout=ea.get("timeout",10))
            if rc2 == 0 and o2.strip() == str(tt.get("output","")).strip():
                npass += 1
        print(f"  passes {npass}/{len(ea['tests'])} full tests", flush=True)

async def go():
    for w in ["st_opus", "st_gpt", "st_gemini", "st_glm"]:
        await check(w)
asyncio.run(go())
print("DONE", flush=True)
