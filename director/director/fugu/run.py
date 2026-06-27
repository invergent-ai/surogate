"""Fugu CLI commands: label | train-sft | train-cmaes | eval.

All commands build a live OpenRouter-backed worker pool from a DirectorConfig (or the
default 6-worker pool) and operate on a chosen dataset.
"""

from __future__ import annotations

import argparse
import asyncio

from ..config import DirectorConfig, default_frontier_pool
from ..shared.eval import run_eval
from ..shared.providers import build_pool
from ..shared.tasks import LOADERS, Dataset
from ..shared.types import Sampling
from .cmaes import evolve, pool_fitness
from .inference import attach_worker_ids
from .labels import generate_soft_targets, load_labels
from .model import SelectionRouter, load_router, save_router
from .sft import train_sft


def load_config(path: str | None) -> DirectorConfig:
    if not path:
        return DirectorConfig(workers=default_frontier_pool())
    import json

    with open(path, encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    cfg = DirectorConfig.model_validate(data)
    if not cfg.workers:
        cfg.workers = default_frontier_pool()
    return cfg


def _dataset(name: str, limit: int | None) -> Dataset:
    if name not in LOADERS:
        raise SystemExit(f"unknown dataset {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name](limit=limit)


def _sampling(cfg: DirectorConfig) -> Sampling:
    s = cfg.sampling
    return Sampling(temperature=s.temperature, top_p=s.top_p, max_tokens=s.max_tokens,
                    seed=s.seed, reasoning_effort=s.reasoning_effort)


def build_router(cfg: DirectorConfig, ckpt: str | None = None) -> SelectionRouter:
    """Construct the router from config (or a checkpoint), with config-driven
    featurization (context window/strategy, hidden position, SVF targets) and the
    ordered worker ids attached. Single source of truth for every CLI command."""
    if ckpt:
        router = load_router(ckpt)
    else:
        f = cfg.featurizer
        router = SelectionRouter.from_pretrained(
            cfg.backbone, num_workers=len(cfg.worker_ids),
            svf_targets=f.svf_targets, hidden_position=f.hidden_position,
            context_window=f.context_window, context_strategy=f.context_strategy,
            head_tokens=f.head_tokens,
        )
    attach_worker_ids(router, cfg.worker_ids)
    return router


def cmd_curate(args) -> None:
    cfg = load_config(args.config)
    pool = build_pool(cfg.pool, cfg.workers)
    from ..shared import curate as C
    from ..shared.sources import build_candidates, train_sources

    sources = args.sources.split(",") if args.sources else train_sources()
    if not args.build_only:
        cands = build_candidates(sources, per_source_limit=args.per_source_limit, seed=args.seed)
        print(f"candidate pool: {len(cands)} tasks from {sources}")
        asyncio.run(
            C.probe(
                pool, cands, args.manifest_dir,
                sampling=_sampling(cfg), max_in_flight=cfg.pool.max_concurrency,
            )
        )
    C.curate(
        args.manifest_dir,
        per_domain_target=args.per_domain_target,
        worker_ids=cfg.worker_ids,
        sources=sources,
        train_ratio=args.train_ratio,
        seed=args.seed,
        note=args.note or "",
    )
    print(f"spent ${pool.budget.spent_usd:.4f}")


def cmd_label(args) -> None:
    cfg = load_config(args.config)
    pool = build_pool(cfg.pool, cfg.workers)
    if args.manifest_dir:
        from ..data.manifest import read_manifest
        from ..shared.curate import train_dataset

        ds = train_dataset(read_manifest(args.manifest_dir))
        print(f"labeling {len(ds)} train items from manifest {args.manifest_dir}")
    else:
        ds = _dataset(args.dataset, args.limit)
    labels = asyncio.run(
        generate_soft_targets(
            pool, ds, n_samples=args.n_samples, tau=args.tau,
            sampling=_sampling(cfg), out_path=args.out,
        )
    )
    print(f"wrote {len(labels)} labels to {args.out}  (spent ${pool.budget.spent_usd:.4f})")


def cmd_train_sft(args) -> None:
    cfg = load_config(args.config)
    labels = load_labels(args.labels)
    router = build_router(cfg)
    print(router.summary())
    stats = train_sft(router, labels, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch_size, log_every=args.log_every)
    save_router(router, args.out, worker_ids=cfg.worker_ids)
    print(f"final KL={stats.final_loss:.5f}; saved router to {args.out}")


def cmd_train_cmaes(args) -> None:
    cfg = load_config(args.config)
    pool = build_pool(cfg.pool, cfg.workers)
    router = build_router(cfg, ckpt=args.ckpt)
    ds = _dataset(args.dataset, args.limit)
    eval_fn = pool_fitness(router, pool, ds, sampling=_sampling(cfg), replicas=args.replicas)
    res = evolve(router, eval_fn, generations=args.generations, sigma0=args.sigma0,
                 popsize=args.popsize, checkpoint_dir=args.checkpoint_dir, verbose=True)
    save_router(router, args.out, worker_ids=cfg.worker_ids)
    print(f"best fitness={res.best_fitness:.4f}; saved router to {args.out}")


def cmd_eval(args) -> None:
    cfg = load_config(args.config)
    pool = build_pool(cfg.pool, cfg.workers)
    router = build_router(cfg, ckpt=args.ckpt)
    ds = _dataset(args.dataset, args.limit)
    report = asyncio.run(run_eval(router, pool, ds, sampling=Sampling(temperature=0.0)))
    print(report.render())


def add_subparsers(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("curate", help="probe a candidate pool, keep discriminative items, balance + split")
    p.add_argument("--config"); p.add_argument("--manifest-dir", required=True)
    p.add_argument("--sources", default=None, help="comma list (default: all training sources; eval-only benchmarks like gpqa excluded)")
    p.add_argument("--per-source-limit", type=int, default=None)
    p.add_argument("--per-domain-target", type=int, default=750)
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--note", default=None)
    p.add_argument("--build-only", action="store_true", help="skip probe; rebuild manifest from existing probe.jsonl")
    p.set_defaults(func=cmd_curate)

    p = sub.add_parser("label", help="generate soft-target labels via the worker pool")
    p.add_argument("--config"); p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--manifest-dir", default=None, help="label the manifest's train split instead of a named dataset")
    p.add_argument("--n-samples", type=int, default=4); p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--out", required=True); p.set_defaults(func=cmd_label)

    p = sub.add_parser("train-sft", help="train the router by KL to soft targets")
    p.add_argument("--config"); p.add_argument("--labels", required=True)
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=16); p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--out", required=True); p.set_defaults(func=cmd_train_sft)

    p = sub.add_parser("train-cmaes", help="refine the router with sep-CMA-ES on end-to-end reward")
    p.add_argument("--config"); p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="gsm8k"); p.add_argument("--limit", type=int, default=None)
    p.add_argument("--generations", type=int, default=60); p.add_argument("--sigma0", type=float, default=0.03)
    p.add_argument("--popsize", type=int, default=None); p.add_argument("--replicas", type=int, default=1)
    p.add_argument("--checkpoint-dir", default=None, help="save/resume CMA-ES state each generation")
    p.add_argument("--out", required=True); p.set_defaults(func=cmd_train_cmaes)

    p = sub.add_parser("eval", help="evaluate the router vs each single worker")
    p.add_argument("--config"); p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default="gsm8k"); p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_eval)
