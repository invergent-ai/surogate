"""Unified per-(task, worker) agentic runners for the incremental training loop.

Each ``run_*`` runs ONE task with a forced solo worker (``allowed={worker}``) through the
real harness and returns ``(reward in {0,1}, cost_usd)`` plus a routing ``prompt``. They are
async and run on the loop's single event loop; blocking env/grading/harness ops are pushed to
threads (``asyncio.to_thread``) so the shared async WorkerPool's semaphore stays bound to one
loop (avoids the "bound to a different event loop" crash).

Sources:
  swebench      -> mini-swe-agent on SWE-Bench Verified (sync harness, own HTTP) via to_thread
  swebench_pro  -> routed bash rollout over SWEBenchProEnv + ScaleAI grading (--use_local_docker)
  tau           -> routed tool-call rollout over TauBenchEnv (judge-free, programmatic reward)
  terminal      -> terminal-bench Harness, one task, forced solo worker (sync) via to_thread
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid

from ..fugu.inference import select_worker
from ..shared.transcript import Transcript
from ..shared.types import Sampling
from .actions import parse_action
from .prompts import AGENT_SYSTEM, wrap_observation
from .toolcall import TOOL_SYSTEM
from .toolenv import RESPOND, ToolAction

HARNESS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "swe_bench_pro_os")
)

# terminal-bench's Dataset construction is NOT thread-safe: under concurrent per-cell Harnesses it
# races the shared cache and produces malformed ".." task paths -> docker-compose fails (flood of
# tracebacks). Serialize JUST the Dataset __init__ (it's fast, mostly cache reads) with a global
# lock; the actual rollouts still run concurrently. This makes terminal usable at our parallelism.
import threading as _threading  # noqa: E402
try:
    import terminal_bench.dataset.dataset as _tb_ds  # noqa: E402
    _DATASET_INIT_LOCK = _threading.RLock()  # reentrant: Dataset.__init__ may construct a nested
    # Dataset on the same thread (cache-validation path); a plain Lock would self-deadlock there.
    _orig_ds_init = _tb_ds.Dataset.__init__

    def _locked_ds_init(self, *a, **k):
        with _DATASET_INIT_LOCK:
            _orig_ds_init(self, *a, **k)

    _tb_ds.Dataset.__init__ = _locked_ds_init
except Exception:  # terminal-bench not installed / API changed -> skip the patch
    pass


# --- generic async rollouts (env ops pushed off the event loop) -------------
async def _routed_bash_rollout(router, pool, env, *, allowed, sampling, max_turns):
    task = await asyncio.to_thread(env.reset)
    tx = Transcript()
    tx.add("system", AGENT_SYSTEM)
    tx.add("user", task)
    cost = 0.0
    for _ in range(max_turns):
        wid = select_worker(router, tx.render(), allowed=allowed)
        comp = await pool.call(wid, tx.as_messages(), sampling)
        cost += comp.cost_usd
        tx.add("assistant", comp.text)
        action = parse_action(comp.text)
        if action.submit:
            break
        if action.command is None:
            tx.add("user", wrap_observation("No bash block found. Emit one ```bash command."))
            continue
        res = await asyncio.to_thread(env.step, action.command)
        tx.add("user", wrap_observation(res.observation))
        if res.done:
            break
    reward = await asyncio.to_thread(env.evaluate)
    return float(reward), cost


async def _routed_tool_rollout(router, pool, env, *, allowed, sampling, max_turns):
    user_msg, tools = await asyncio.to_thread(env.reset)
    tx = Transcript()
    tx.add("system", TOOL_SYSTEM)
    tx.add("user", user_msg)
    messages = [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": user_msg}]
    cost = 0.0
    for _ in range(max_turns):
        wid = select_worker(router, tx.render(), allowed=allowed)
        resp = await pool.call_tools(wid, messages, tools, sampling)
        cost += resp.cost_usd
        tx.add("assistant", resp.as_text())
        if resp.tool_calls:
            tc = resp.tool_calls[0]
            messages.append({
                "role": "assistant", "content": resp.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}],
            })
            step = await asyncio.to_thread(env.step, ToolAction(name=tc.name, arguments=tc.arguments))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": step.observation})
        else:
            messages.append({"role": "assistant", "content": resp.content or ""})
            step = await asyncio.to_thread(env.step, ToolAction(name=RESPOND, arguments={"content": resp.content or ""}))
            messages.append({"role": "user", "content": step.observation})
        tx.add("user", step.observation)
        if step.done:
            break
    reward = await asyncio.to_thread(env.reward)
    return float(reward), cost


# --- per-source runners ------------------------------------------------------
async def run_swe(ctx, instance, allowed, *, max_turns=80):
    """SWE-Bench Verified via mini-swe-agent (sync, own HTTP) — run in a thread; it does its
    own per-step routing + grading and returns the graded reward."""
    from .swebench_mini import run_instance

    def _go():
        r = run_instance(ctx.router, instance, ctx.worker_slugs, allowed=allowed,
                         cost_limit=0.4, step_limit=max_turns, do_grade=True)
        return float(r["reward"]), float(r["cost"])

    return await asyncio.to_thread(_go)


async def run_swesmith(ctx, instance, allowed, *, max_turns=80):
    """SWE-smith (the agentic-coding TRAINING corpus) via mini-swe-agent + SWE-smith's own grading
    harness. The per-instance image + tests are bundled in the row, so it needs no swebench spec."""
    from .swebench_mini import run_swesmith_instance

    # cost_limit governs how long the agent can iterate; 0.4 starves high-reasoning rollouts (they hit
    # the cap mid-task and submit incomplete patches -> r=0). Bump it (env-tunable) so they can actually
    # reach a passing patch.
    cl = float(os.getenv("SWESMITH_COST_LIMIT", "1.5"))

    def _go():
        r = run_swesmith_instance(ctx.router, instance, ctx.worker_slugs, allowed=allowed,
                                  cost_limit=cl, step_limit=max_turns, do_grade=True)
        return float(r["reward"]), float(r["cost"])

    return await asyncio.to_thread(_go)


async def run_swe_pro(ctx, instance, allowed, *, step_limit=150, cost_limit=3.0):
    """SWE-Bench Pro via mini-swe-agent (the COMPETENT SWE agent that solves Verified) pointed at
    the Pro jefzda container, graded by ScaleAI's harness. mini-swe-agent's swebench config assumes
    /testbed + conda 'testbed'; Pro's repo is at /app, so we remap /testbed->/app and run Pro's
    before_repo_set_cmd at container start. Sync mini-swe-agent runs in a thread (its own HTTP)."""
    import json as _json

    def _go():
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.run.benchmarks.swebench import get_sb_environment

        from .swebench_mini import FuguModel
        from .swebench_pro_env import SWEBenchProEnv, instance_image

        inst = dict(instance)
        inst["docker_image"] = instance_image(inst)  # point mini-swe-agent at the Pro image
        config = get_config_from_spec(str(builtin_config_dir / "benchmarks" / "swebench.yaml"))
        config = _json.loads(_json.dumps(config).replace("/testbed", "/app"))  # Pro repo lives at /app
        # Pro images have ENTRYPOINT=/bin/bash, so mini-swe-agent's `... image sleep 2h` would run
        # `bash sleep 2h` and exit. Clear the entrypoint so the keep-alive `sleep` runs directly.
        config.setdefault("environment", {})["run_args"] = ["--rm", "--entrypoint", ""]
        env = get_sb_environment(config, inst)
        if inst.get("before_repo_set_cmd"):  # set up the repo (reset/checkout base_commit) ourselves
            # NB: mini-swe-agent's own env_startup_command path passes a str to env.execute(), which
            # this version expects as a dict -> we run it via the correct dict form instead.
            env.execute({"command": inst["before_repo_set_cmd"]})
        model = FuguModel(ctx.router, ctx.worker_slugs, allowed=allowed)
        acfg = dict(config.get("agent", {}))
        acfg.pop("agent_class", None)
        acfg.update(cost_limit=cost_limit, step_limit=step_limit)
        agent = DefaultAgent(model, env, **acfg)
        info = agent.run(inst["problem_statement"])
        patch = info.get("submission", "") or ""
        reward = SWEBenchProEnv(inst, harness_dir=HARNESS_DIR).grade_patch(patch)
        return float(reward), float(model.total_cost)

    return await asyncio.to_thread(_go)


async def run_tau(ctx, task, allowed, *, max_turns=30):
    """tau-bench tool-use: routed tool-call rollout, judge-free programmatic reward.
    ``task`` = (env_name, task_index). User simulator routes through OpenRouter via litellm."""
    from .taubench_env import TauBenchEnv

    env_name, idx = task
    env = TauBenchEnv(env_name, idx, user_model=ctx.tau_user_model, user_provider="openrouter")
    try:
        return await _routed_tool_rollout(ctx.router, ctx.pool, env,
                                          allowed=allowed, sampling=ctx.sampling, max_turns=max_turns)
    finally:
        await asyncio.to_thread(env.close)


async def run_terminal(ctx, task_id, allowed, *, max_turns=30):
    """Terminal-Bench: run the real Harness for ONE task with a forced solo worker. Heavy
    (the harness owns the container + native grading); driven in a thread. Returns (reward, 0.0)
    — terminal-bench grades natively; per-$ cost isn't surfaced by the harness here."""
    from pathlib import Path

    from terminal_bench.harness.harness import Harness

    from .terminalbench_agent import _COST_BY_RUN

    worker = next(iter(allowed)) if allowed else None
    cost_key = uuid.uuid4().hex

    # Prefer the pre-resolved LOCAL dataset path (race-free under parallelism); fall back to registry.
    dpath = getattr(ctx, "terminal_dataset_path", None)
    ds_kwargs = ({"dataset_path": Path(dpath)} if dpath
                 else {"dataset_name": ctx.terminal_dataset,
                       "dataset_version": getattr(ctx, "terminal_version", "0.1.1")})

    def _go():
        out = os.path.join(ctx.work_dir, f"tb_{uuid.uuid4().hex[:8]}")
        h = Harness(
            output_path=Path(out),
            run_id=f"director_{uuid.uuid4().hex[:8]}",
            agent_import_path="director.agentic.terminalbench_agent:DirectorAgent",
            agent_kwargs={"director_config": ctx.config_path, "ckpt": ctx.ckpt_path,
                          "allowed": [worker] if worker else None, "cost_key": cost_key,
                          "max_turns": max_turns},
            task_ids=[task_id],
            n_concurrent_trials=1,
            cleanup=True,
            global_test_timeout_sec=180,  # heavy tasks (initramfs/kernel builds) exceed the 60s default
            global_agent_timeout_sec=600,  # generous agent budget for slow builds (default ~360s)
            **ds_kwargs,
        )
        res = h.run()
        trials = res.results
        cost = float(_COST_BY_RUN.pop(cost_key, 0.0))  # real $ spend (always reclaim the registry)
        if not trials:
            raise RuntimeError("terminal: harness returned no trial result")
        tr = trials[0]
        fm = getattr(tr.failure_mode, "value", str(tr.failure_mode))
        if fm in ("unset", "none"):                   # trial completed normally: terminal-bench
            return (1.0 if tr.is_resolved else 0.0), cost  # leaves failure_mode UNSET on success
            #                                              (it's only set on FAILURE); the real
            #                                              verdict is is_resolved -> resolved=1 else 0
        if fm in ("test_timeout", "agent_timeout"):   # test/agent didn't finish in the (generous)
            return 0.0, cost                          # budget -> worker failed; record 0
        # infra/parse error (unknown_agent_error, parse_error, OOM, ...) is NOT a valid grade ->
        # raise so the caller skips it (cell stays missing, retried later).
        raise RuntimeError(f"terminal trial not validly graded (failure_mode={fm})")

    return await asyncio.to_thread(_go)


# --- loaders: return [{"item_id", "prompt", "payload"}] ----------------------
def load_swe_tasks(dataset: str, limit: int):
    from .swebench_env import load_swebench

    rows = load_swebench(dataset=dataset, limit=limit, shuffle=True, seed=0)
    return [{"item_id": r["instance_id"], "prompt": r["problem_statement"], "payload": r} for r in rows]


def load_swesmith_tasks(limit: int, dataset: str = "SWE-bench/SWE-smith"):
    """SWE-smith training instances. Streamed (52k rows); each row bundles image_name + tests +
    problem_statement, so the payload is the whole instance (used for both the rollout and grading)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train", streaming=True).shuffle(seed=0, buffer_size=10000)
    rows = []
    for r in ds:
        if limit and len(rows) >= limit:
            break
        rows.append(dict(r))
    return [{"item_id": r["instance_id"], "prompt": r["problem_statement"], "payload": r} for r in rows]


def load_swe_pro_tasks(limit: int):
    from .swebench_env import load_swebench_pro

    rows = load_swebench_pro(limit=limit, shuffle=True, seed=0)
    return [{"item_id": r["instance_id"], "prompt": r["problem_statement"], "payload": r} for r in rows]


def load_tau_tasks(env_name: str, limit: int, split: str = "test"):
    # Enumerate task indices directly from tau-bench's task modules (no env/user-sim creds needed).
    import importlib

    mod = importlib.import_module(f"tau_bench.envs.{env_name}.tasks_{split}")
    tasks = getattr(mod, f"TASKS_{split.upper()}", None) or getattr(mod, "TASKS", [])
    idxs = list(range(len(tasks)))[:limit]
    return [{"item_id": f"tau-{env_name}-{i}", "prompt": f"[tau-bench {env_name}] task {i}",
             "payload": (env_name, i)} for i in idxs]


def load_terminal_tasks(dataset: str, limit: int, version: str = "0.1.1"):
    from terminal_bench.dataset.dataset import Dataset

    ds = Dataset(name=dataset, version=version)
    ids = sorted(t.name for t in ds.tasks)[:limit]  # stable order so restarts resume the same tasks
    return [{"item_id": tid, "prompt": f"[terminal-bench] {tid}", "payload": tid} for tid in ids]
