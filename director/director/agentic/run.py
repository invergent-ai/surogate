"""Agentic CLI commands: agentic-eval and agentic-train (sep-CMA-ES on SWE-Bench)."""

from __future__ import annotations

import argparse
import asyncio

from ..fugu.model import save_router
from ..fugu.run import build_router, load_config
from ..shared.providers import build_pool
from ..shared.types import Sampling
from .fitness import agentic_eval


def _factories(args):
    """Build env factories for the selected harness."""
    harness = getattr(args, "harness", "swebench")
    if harness == "swebench_pro":
        from .swebench_env import load_swebench_pro
        from .swebench_pro_env import build_swebench_pro_factories

        insts = load_swebench_pro(dataset=args.dataset, limit=args.limit, shuffle=args.shuffle, seed=args.seed)
        return build_swebench_pro_factories(
            insts, harness_dir=getattr(args, "harness_dir", None),
            dockerhub_username=getattr(args, "dockerhub_username", "jefzda"),
            step_timeout=args.step_timeout,
        )
    from .swebench_env import build_swebench_factories, load_swebench

    insts = load_swebench(dataset=args.dataset, limit=args.limit, shuffle=args.shuffle, seed=args.seed)
    return build_swebench_factories(insts, dataset=args.dataset, step_timeout=args.step_timeout)


def cmd_agentic_eval(args) -> None:
    cfg = load_config(args.config)
    pool = build_pool(cfg.pool, cfg.workers)
    router = build_router(cfg, ckpt=args.ckpt)
    factories = _factories(args)
    report = asyncio.run(
        agentic_eval(
            router, pool, factories, max_turns=args.max_turns,
            sampling=Sampling(temperature=0.2, max_tokens=args.max_tokens),
            max_parallel=args.max_parallel,
        )
    )
    print(report.render())


def cmd_agentic_train(args) -> None:
    from ..fugu.cmaes import evolve_parallel
    from .fitness import FitnessConfig, make_agentic_fitness_async

    cfg = load_config(args.config)
    pool = build_pool(cfg.pool, cfg.workers)
    # One router replica per concurrent candidate (cross-candidate parallelism).
    routers = [build_router(cfg, ckpt=args.ckpt) for _ in range(args.router_replicas)]
    factories = _factories(args)
    fit_cfg = FitnessConfig(w_div=args.w_div, w_turn=args.w_turn, w_cost=args.w_cost)
    fitness_async = make_agentic_fitness_async(
        pool, factories, max_turns=args.max_turns,
        sampling=Sampling(temperature=0.2, max_tokens=args.max_tokens),
        replicas=args.replicas, max_parallel=args.max_parallel, cfg=fit_cfg,
    )
    res = evolve_parallel(
        routers, fitness_async, generations=args.generations, sigma0=args.sigma0,
        popsize=args.popsize, checkpoint_dir=args.checkpoint_dir, resume=True, verbose=True,
    )
    save_router(routers[0], args.out, worker_ids=cfg.worker_ids)
    print(f"best shaped fitness={res.best_fitness:.3f} over {res.generations_run} gens; saved to {args.out}")


def cmd_terminal_bench(args) -> None:
    """Run terminal-bench with the Director router as the agent (native grading)."""
    import uuid
    from pathlib import Path

    from terminal_bench.harness.harness import Harness

    h = Harness(
        output_path=Path(args.output),
        run_id=args.run_id or f"director_{uuid.uuid4().hex[:8]}",
        agent_import_path="director.agentic.terminalbench_agent:DirectorAgent",
        agent_kwargs={
            "director_config": args.config, "ckpt": args.ckpt,
            "max_turns": args.max_turns, "max_tokens": args.max_tokens,
        },
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        dataset_path=Path(args.dataset_path) if args.dataset_path else None,
        task_ids=[args.task_id] if args.task_id else None,
        n_tasks=args.n_tasks,
        n_concurrent_trials=args.max_parallel,
        cleanup=True,
    )
    res = h.run()
    trials = res.results
    resolved = sum(1 for t in trials if t.is_resolved)
    print(f"terminal-bench: resolved {resolved}/{len(trials)} = {resolved / max(len(trials), 1):.3f}")
    for t in trials:
        print(f"  {t.task_id}: {'RESOLVED' if t.is_resolved else t.failure_mode}")


def _configure_litellm_openrouter(cfg) -> None:
    """Route tau-bench's user simulator (litellm) through OpenRouter via its
    openai-compatible path. Use ``--user-provider openai`` with these globals set."""
    import litellm

    litellm.api_base = cfg.pool.base_url
    litellm.api_key = cfg.pool.api_key()


def _taubench_factories(args):
    from .taubench_env import build_taubench_factories, load_taubench_tasks

    idxs = load_taubench_tasks(args.domain, task_split=args.task_split, limit=args.limit)
    return build_taubench_factories(
        args.domain, idxs, user_model=args.user_model,
        user_provider=args.user_provider, task_split=args.task_split,
    )


def cmd_taubench_eval(args) -> None:
    from .toolcall import toolcall_eval

    cfg = load_config(args.config)
    _configure_litellm_openrouter(cfg)
    pool = build_pool(cfg.pool, cfg.workers)
    router = build_router(cfg, ckpt=args.ckpt)
    report = asyncio.run(
        toolcall_eval(
            router, pool, _taubench_factories(args), max_turns=args.max_turns,
            sampling=Sampling(temperature=0.2, max_tokens=args.max_tokens), max_parallel=args.max_parallel,
        )
    )
    print(report.render())


def cmd_taubench_train(args) -> None:
    from ..fugu.cmaes import evolve_parallel
    from .fitness import FitnessConfig
    from .toolcall import make_toolcall_fitness_async

    cfg = load_config(args.config)
    _configure_litellm_openrouter(cfg)
    pool = build_pool(cfg.pool, cfg.workers)
    routers = [build_router(cfg, ckpt=args.ckpt) for _ in range(args.router_replicas)]
    fit_cfg = FitnessConfig(w_div=args.w_div, w_turn=args.w_turn, w_cost=args.w_cost)
    fitness_async = make_toolcall_fitness_async(
        pool, _taubench_factories(args), max_turns=args.max_turns,
        sampling=Sampling(temperature=0.2, max_tokens=args.max_tokens),
        replicas=args.replicas, max_parallel=args.max_parallel, cfg=fit_cfg,
    )
    res = evolve_parallel(
        routers, fitness_async, generations=args.generations, sigma0=args.sigma0,
        popsize=args.popsize, checkpoint_dir=args.checkpoint_dir, resume=True, verbose=True,
    )
    save_router(routers[0], args.out, worker_ids=cfg.worker_ids)
    print(f"best shaped fitness={res.best_fitness:.3f} over {res.generations_run} gens; saved to {args.out}")


def _add_taubench_common(p):
    p.add_argument("--config"); p.add_argument("--ckpt", default=None)
    p.add_argument("--domain", default="retail", choices=["retail", "airline"])
    p.add_argument("--task-split", default="test"); p.add_argument("--limit", type=int, default=20)
    # The simulator runs via litellm; we point litellm's openai-compatible path at
    # OpenRouter (see _configure_litellm_openrouter), so provider="openai" + bare slug.
    p.add_argument("--user-model", default="google/gemini-3.5-flash")
    p.add_argument("--user-provider", default="openai")
    p.add_argument("--max-turns", type=int, default=30); p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--max-parallel", type=int, default=2)


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("agentic-eval", help="evaluate the router on SWE-Bench (resolve rate + worker turn share)")
    p.add_argument("--config"); p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--limit", type=int, default=20); p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=4096); p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--step-timeout", type=float, default=120.0)
    p.add_argument("--harness", choices=["swebench", "swebench_pro"], default="swebench")
    p.add_argument("--harness-dir", default=None, help="clone of scaleapi/SWE-bench_Pro-os (Pro grading)")
    p.add_argument("--dockerhub-username", default="jefzda")
    p.set_defaults(func=cmd_agentic_eval)

    p = sub.add_parser("agentic-train", help="refine the router on SWE-Bench with sep-CMA-ES")
    p.add_argument("--config"); p.add_argument("--ckpt", default=None)
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--limit", type=int, default=20); p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--max-turns", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=4096); p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--step-timeout", type=float, default=120.0)
    p.add_argument("--generations", type=int, default=60); p.add_argument("--sigma0", type=float, default=0.03)
    p.add_argument("--popsize", type=int, default=None); p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--router-replicas", type=int, default=1, help="router copies = concurrent candidates per generation")
    p.add_argument("--checkpoint-dir", default=None, help="save/resume CMA-ES state each generation")
    p.add_argument("--w-div", type=float, default=0.15, help="diversity (entropy) bonus weight")
    p.add_argument("--w-turn", type=float, default=0.0, help="turn-count penalty weight")
    p.add_argument("--w-cost", type=float, default=0.0, help="USD-cost penalty weight")
    p.add_argument("--harness", choices=["swebench", "swebench_pro"], default="swebench")
    p.add_argument("--harness-dir", default=None, help="clone of scaleapi/SWE-bench_Pro-os (Pro grading)")
    p.add_argument("--dockerhub-username", default="jefzda")
    p.add_argument("--out", required=True); p.set_defaults(func=cmd_agentic_train)

    p = sub.add_parser("terminal-bench-eval", help="run Terminal-Bench with the router as the agent (native grading)")
    p.add_argument("--config"); p.add_argument("--ckpt", default=None)
    p.add_argument("--output", default="./tb_out"); p.add_argument("--run-id", default=None)
    p.add_argument("--dataset-name", default="terminal-bench-core")
    p.add_argument("--dataset-version", default=None)
    p.add_argument("--dataset-path", default=None, help="local task dir(s) instead of a registry dataset")
    p.add_argument("--task-id", default=None); p.add_argument("--n-tasks", type=int, default=None)
    p.add_argument("--max-turns", type=int, default=30); p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-parallel", type=int, default=2)
    p.set_defaults(func=cmd_terminal_bench)

    p = sub.add_parser("taubench-eval", help="evaluate the router on tau-bench tool-use (judge-free)")
    _add_taubench_common(p)
    p.set_defaults(func=cmd_taubench_eval)

    p = sub.add_parser("taubench-train", help="refine the router on tau-bench tool-use with sep-CMA-ES")
    _add_taubench_common(p)
    p.add_argument("--generations", type=int, default=60); p.add_argument("--sigma0", type=float, default=0.03)
    p.add_argument("--popsize", type=int, default=None); p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--router-replicas", type=int, default=1)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--w-div", type=float, default=0.15); p.add_argument("--w-turn", type=float, default=0.0)
    p.add_argument("--w-cost", type=float, default=0.0)
    p.add_argument("--out", required=True); p.set_defaults(func=cmd_taubench_train)
