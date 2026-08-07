"""Per-worker SOLO baselines over the heldout-probe task set.

Answers the promotion question "does the trained conductor's orchestration
beat each bound worker alone?" on the LOCAL pool at $0: for every probe task
(same lanes, same fixed seed and lane settings as ultra.heldout_probe) each
worker position runs a canonical single-step workflow — "solve completely,
produce the deliverable" — through the SAME runtime.score() path the probe
uses, so grading, budgets and harnesses are identical and the per-task delta
vs a conductor arm is attributable to orchestration alone.

    PYTHONPATH=ultra .venv/bin/python -m ultra.solo_worker_probe \
        --out solo_baselines.json [--lanes repo_open_repo_terminal]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from ultra.heldout_probe import LANES, MAN, REPO, PROBE_SEED, _ensure_heldout_pilot_config, _wait_repo_quiet

ENV_DIR = REPO / "environments/fugu-ultra-pilot"

SOLO_SUBTASK = (
    "Solve this task completely yourself. Do all analysis, implementation and "
    "verification needed, and produce the final deliverable the task requires."
)


def _solo_plan(ref: str) -> str:
    return json.dumps({
        "action": "replan",
        "reason": "Single strong position: one worker owns the task end to end.",
        "steps": [{"profile_ref": ref, "subtask": SOLO_SUBTASK, "access_positions": []}],
    })


def run(out_path: Path, lanes: list[str] | None = None, max_concurrent: int = 4) -> dict:
    sys.path.insert(0, str(ENV_DIR))
    import os

    import fugu_ultra_pilot as env_mod
    from ultra.live_control import WorkerProfile, capability_reference_map

    _ensure_heldout_pilot_config()
    os.environ.setdefault("FUGU_PROBE_API_KEY", "EMPTY")
    results: dict[str, dict] = {}

    for lane, cfg, manifest, safety, budget, wtok, turns, n in LANES:
        if lanes is not None and lane not in lanes:
            continue
        if lane == "repo_open_repo_terminal":
            _wait_repo_quiet()
        kwargs = dict(
            pilot_config_path=str(MAN / cfg),
            task_name=f"fugu_solo_{lane}",
            lane=lane,
            live_safety_path=str(MAN / safety),
            seed=PROBE_SEED,
            shuffle=True,
            max_examples=n,
            force_step_budget=budget,
            worker_temperature=0.2,
            worker_max_tokens=wtok,
            worker_reasoning_effort="high",
            max_turns=turns,
            max_concurrency=max_concurrent,
            timeout_s=1500.0,
            max_retries=0,
            artifact_dir=str(REPO / "output/fugu_ultra_pool_d/solo_probe_artifacts" / lane),
            cache_dir=str(REPO / ".ultra_cache/solo_probe_completions"),
            workflow_record_cache_dir=str(REPO / ".ultra_cache/solo_probe_records"),
        )
        if manifest:
            kwargs["task_manifest_path"] = str(MAN / manifest)
        env = env_mod.load_environment(**kwargs)
        rt = env.ultra_runtime

        priors = rt.lane_role_priors[lane]
        workers = tuple(WorkerProfile(worker_id=i, capability_tags=tuple(t)) for i, t in enumerate(priors))
        ref_by_worker = {w: r for r, w in capability_reference_map(workers).profile_ref_to_worker_id.items()}

        lane_rows: list[dict] = []

        async def _one(row, wid: int) -> dict:
            info = json.loads(row["info"]) if isinstance(row["info"], str) else dict(row["info"])
            state: dict = {}
            try:
                r = await rt.score(_solo_plan(ref_by_worker[wid]), info, state)
            except Exception as exc:  # noqa: BLE001 — record, don't kill the sweep
                return {"task_id": info.get("task_id"), "worker": wid, "reward": None,
                        "outcome": f"runner_error: {exc}"[:160]}
            return {"task_id": info.get("task_id"), "worker": wid, "reward": float(r),
                    "outcome": state.get("_ultra_outcome_class"),
                    "model": rt.pool.model_for(rt.lane_masks[lane][wid])
                    if hasattr(rt.pool, "model_for") else None}

        async def _lane() -> None:
            sem = asyncio.Semaphore(max_concurrent)

            async def guarded(row, wid):
                async with sem:
                    return await _one(row, wid)

            tasks = [guarded(env.dataset[i], wid)
                     for i in range(min(n, len(env.dataset)))
                     for wid in sorted(ref_by_worker)]
            for fut in asyncio.as_completed(tasks):
                lane_rows.append(await fut)

        asyncio.run(_lane())
        by_worker: dict[int, list[float]] = {}
        for r in lane_rows:
            if r["reward"] is not None:
                by_worker.setdefault(r["worker"], []).append(r["reward"])
        results[lane] = {
            "rows": lane_rows,
            "per_worker_mean": {str(w): round(sum(v) / len(v), 4) for w, v in sorted(by_worker.items()) if v},
        }
        print(f"[solo] {lane}: " + " ".join(
            f"w{w}={m}" for w, m in results[lane]["per_worker_mean"].items()), flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lanes", help="comma-separated lane subset")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    lane_list = [l.strip() for l in args.lanes.split(",") if l.strip()] if args.lanes else None
    run(args.out, lanes=lane_list, max_concurrent=args.concurrency)


if __name__ == "__main__":
    main()
