"""Per-step escalation probe (the correct unit for the product). On a few tau tasks, compare:
  - constant policies: always-GLM / always-Gemini / always-Opus (baselines)
  - escalation policies: GLM drives each turn; on the FIRST stall (env error / repeated action / no
    tool call) hand off to the strong model (Opus or Gemini) for the rest of the episode.
Measures per-task cost (summed per-turn cost_usd, rollout-local) + success + #escalated turns.
Question: does GLM-default-with-escalation match always-Opus success at much lower cost?
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict

import numpy as np

TASKS = int(os.getenv("TASKS", "3"))  # per env
USER_MODEL = "openrouter/openai/gpt-5-mini"
MAXT = int(os.getenv("MAXT", "30"))


async def rollout(env_factory, pool, sampling, mode, default="glm", strong="opus"):
    from director.agentic.taubench_env import TauBenchEnv  # noqa
    from director.agentic.toolcall import TOOL_SYSTEM
    from director.agentic.toolenv import RESPOND, ToolAction

    env = env_factory()
    user_msg, tools = await asyncio.to_thread(env.reset)
    messages = [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": user_msg}]
    cost = 0.0; turns = 0; n_strong = 0; escalated = False; last_action = None
    try:
        for _ in range(MAXT):
            wid = (strong if escalated else default) if mode == "escalate" else default
            if wid != "glm":
                n_strong += 1
            resp = await pool.call_tools(wid, messages, tools, sampling)
            cost += resp.cost_usd; turns += 1
            if resp.tool_calls:
                tc = resp.tool_calls[0]
                messages.append({"role": "assistant", "content": resp.content or "",
                                 "tool_calls": [{"id": tc.id, "type": "function",
                                                 "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                step = await asyncio.to_thread(env.step, ToolAction(name=tc.name, arguments=tc.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": step.observation})
                action = (tc.name, json.dumps(tc.arguments, sort_keys=True))
            else:
                messages.append({"role": "assistant", "content": resp.content or ""})
                step = await asyncio.to_thread(env.step, ToolAction(name=RESPOND, arguments={"content": resp.content or ""}))
                messages.append({"role": "user", "content": step.observation})
                action = ("respond", (resp.content or "")[:40])
            obs = (step.observation or "").lower()
            if not escalated and ((" error" in obs or "invalid" in obs or "not found" in obs)
                                  or (not resp.tool_calls) or (action == last_action)):
                escalated = True  # latch: hand off to strong for the rest
            last_action = action
            if step.done:
                break
        reward = await asyncio.to_thread(env.reward)
    finally:
        await asyncio.to_thread(env.close)
    return float(reward), cost, turns, n_strong


def main():
    from director.config import DirectorConfig, FeaturizerConfig, PoolConfig, default_frontier_pool
    from director.agentic.runners import load_tau_tasks
    from director.agentic.taubench_env import TauBenchEnv
    from director.fugu.run import _sampling

    from director.shared.providers import build_pool
    cfg = DirectorConfig(workers=default_frontier_pool(), featurizer=FeaturizerConfig(context_window=4096))
    pool = build_pool(PoolConfig(budget_usd=None, max_concurrency=12, timeout_s=300, max_retries=2), cfg.workers)
    samp = _sampling(cfg)

    tasks = [("retail", it) for it in load_tau_tasks("retail", TASKS)] \
        + [("airline", it) for it in load_tau_tasks("airline", TASKS)]
    POLICIES = [("always-glm", "const", "glm", None), ("always-gemini", "const", "gemini", None),
                ("always-opus", "const", "opus", None),
                ("esc-glm->opus", "escalate", "glm", "opus"),
                ("esc-glm->gemini", "escalate", "glm", "gemini")]
    print(f"per-step probe: {len(tasks)} tau tasks x {len(POLICIES)} policies", flush=True)

    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    sem = asyncio.Semaphore(6)
    res = defaultdict(list)

    async def one(env_name, item, pname, mode, dflt, strong):
        async with sem:
            ef = lambda: TauBenchEnv(env_name, item["payload"][1], user_model=USER_MODEL, user_provider="openrouter")
            try:
                r, c, t, ns = await rollout(ef, pool, samp, mode, default=dflt, strong=strong or "opus")
            except Exception as e:
                print(f"  ! {pname} {item['item_id']}: {type(e).__name__}: {str(e)[:60]}", flush=True)
                return
            res[pname].append((r, c, t, ns))
            print(f"  {pname:16} {item['item_id']:16} reward={r:.0f} ${c:.4f} turns={t} strong={ns}", flush=True)

    async def run():
        await asyncio.gather(*[one(s, it, *p) for s, it in tasks for p in POLICIES])
    loop.run_until_complete(run())

    print("\n=== PER-POLICY ===", flush=True)
    print(f"  {'policy':16} {'success':>8} {'avg $/task':>11} {'avg turns':>10} {'avg escal':>10}  n")
    for pname, *_ in POLICIES:
        a = np.array(res[pname]) if res[pname] else np.zeros((0, 4))
        if len(a):
            print(f"  {pname:16} {a[:,0].mean():>8.2f} {a[:,1].mean():>11.4f} {a[:,2].mean():>10.1f} {a[:,3].mean():>10.1f}  {len(a)}", flush=True)
    print("\nverdict: does esc-glm->X match always-opus success at lower $/task? (tau is GLM-unfavorable; "
          "coding would be the better domain but needs a per-turn coding harness.)", flush=True)


if __name__ == "__main__":
    main()
