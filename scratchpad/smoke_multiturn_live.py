"""LIVE 2-turn smoke of the MultiTurnEnv — read-only vs the training vLLM (like an eval), low conc.
Drives the real sequence on a few code tasks: turn0 plan -> execute (real workers) -> grade;
on failure, feed the outcome back -> turn1 repair plan -> execute -> grade. Reports whether the
repair plan DIFFERS from turn0 and whether turn1 improved. Does NOT touch training state."""
import asyncio, importlib.util, json, os, sys, types

sys.path.insert(0, "/home/densemax/work/flavius/surogate")
sys.path.insert(0, "/home/densemax/work/flavius/surogate/ultra")

env = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "fpe", "/home/densemax/work/flavius/surogate/environments/fugu-ultra-pilot/fugu_ultra_pilot.py"))
env.__spec__.loader.exec_module(env)
mt = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "mt", "/tmp/claude-1000/-home-densemax-work-flavius-surogate/1636be7a-c882-47c0-8ed5-6ece7392008f/scratchpad/fugu_ultra_multiturn.py"))
mt.__spec__.loader.exec_module(mt)

from openai import AsyncOpenAI
from ultra.workers import Sampling
from ultra.schemas import TaskSpec

D = "/home/densemax/work/flavius/surogate/director/manifests/fugu_clean_v1/grpo_pilot_train"
cfg = json.load(open(f"{D}/pilot_config_singleturn.json"))
WORKERS = cfg["worker_pool_names"]              # ordinals 0..3 = [opus, gemini, gpt, glm]
N_TASKS = int(os.environ.get("SMOKE_TASKS", "2"))

code_tasks = [TaskSpec.model_validate(json.loads(l))
              for l in open(f"{D}/hard_mix_code_taskspecs.jsonl")][:8][:N_TASKS]

pool = env._build_pool(pilot_config=cfg, provider_mode="live", cache_dir=".ultra_cache/eval",
                       max_concurrency=3, requests_per_minute=None, timeout_s=300.0, max_retries=2)

rt = types.SimpleNamespace(
    pool=pool,
    sampling=Sampling(temperature=0.2, top_p=1.0, max_tokens=4096, reasoning_effort="minimal"),
    tasks_by_id={t.task_id: t for t in code_tasks},
    lane_masks={"single_turn": WORKERS},
    worker_harnesses=env._worker_harness_overrides(cfg),
    max_workflow_steps=int(cfg["workflow_policy"]["max_workflow_steps"]),
    force_step_budget="short",
)
e = mt.FuguUltraMultiTurnEnv.__new__(mt.FuguUltraMultiTurnEnv)
e.rt = rt

vllm = AsyncOpenAI(base_url="http://localhost:8007/v1", api_key="x", timeout=180.0)

async def gen(messages):
    for _ in range(3):
        try:
            r = await vllm.chat.completions.create(
                model="default", messages=[dict(m) for m in messages],
                temperature=1.0, top_p=1.0, max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            t = r.choices[0].message.content or ""
            if t.strip():
                return t
        except Exception:
            await asyncio.sleep(2)
    return ""

def shape(raw):
    try:
        wf = mt.parse_workflow(mt._extract_workflow_payload(raw))
        return [s.worker_id for s in wf.steps]
    except Exception:
        return None

async def drive(task):
    print(f"\n===== {task.task_id} =====", flush=True)
    t0_prompt = env._prompt_for_task(task, cfg, "single_turn", max_task_chars=12000)
    raw0 = await gen(t0_prompt)
    state = await e.setup_state({"info": {"task_id": task.task_id, "lane": "single_turn"},
                                 "trajectory": [{"prompt": t0_prompt, "completion": raw0}]})
    resp = await e.env_response(None, state)
    r0 = state["turn_records"][0]
    print(f"turn0: shape={shape(raw0)} grade={r0['reward']} success={r0['success']}", flush=True)

    if state.get("final_env_response"):
        print("turn0 SOLVED -> early terminate (reward 1.0). No repair needed.", flush=True)
        return
    # build turn-1 prompt exactly as vf get_prompt_messages would: prior prompt + completion + env_response
    t1_prompt = list(t0_prompt) + [{"role": "assistant", "content": raw0}] + [dict(m) for m in resp]
    raw1 = await gen(t1_prompt)
    state["trajectory"].append({"prompt": t1_prompt, "completion": raw1})
    reward = await e.reward(state)
    r1 = state["turn_records"][-1]
    s0, s1 = shape(raw0), shape(raw1)
    print(f"turn1: shape={s1} grade={r1['reward']} success={r1['success']}", flush=True)
    print(f"  plan CHANGED: {s0 != s1} | terminal reward: {reward} | repair references failure: "
          f"{'diagnos' in raw1.lower() or 'fix' in raw1.lower() or 'error' in raw1.lower() or 'correct' in raw1.lower()}", flush=True)

async def main():
    print(f"LIVE 2-turn smoke: {len(code_tasks)} code tasks, conc=3 (read-only vs :8007)", flush=True)
    for t in code_tasks:
        try:
            await drive(t)
        except Exception as ex:
            print(f"  {t.task_id} ERROR: {ex}", flush=True)
    print("\nSMOKE DONE", flush=True)

asyncio.run(main())
