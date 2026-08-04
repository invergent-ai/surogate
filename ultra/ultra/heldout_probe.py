"""Real-task conductor probe across the REAL lanes, one arm per invocation.

"Is the conductor actually improving on real-world tasks?" — measured, not
guessed: tau2 telecom, CRM, tool dialogue, repo/terminal, plus held-out
math/code, on the SAME local pool with the SAME env grading the campaign
trains on. Only the conductor id differs between arms, so the
parent-vs-checkpoint DELTA isolates conductor quality. Lane settings are
copied verbatim from orch_pool_d.yaml so probe rollouts match the training
regime. Absolute numbers are NOT leaderboard-comparable (training pool, not
the production binding).

Holdout status per lane (recorded, not hidden): math/code come from the
curated non-sealed holdout (heldout_eval_taskspecs.jsonl — never trained).
The four agentic lanes sample from the TRAIN registries with a fixed
probe-only seed; the sealed sets (tau2 114-task benchmark already excluded
from its manifest; crmarena 20% stratified holdout in a separate file) stay
sealed for promotion. With ~30 groups/lane served over 70 steps vs 240-832
registered, most sampled tasks are unseen; the paired delta is meaningful
either way, and generalization proper is measured by the sealed sets at
promotion time.

    PYTHONPATH=ultra .venv/bin/python -m ultra.heldout_probe \
        --model fugu-pool-d-step70 --out probe_step70.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAN = REPO / "director/manifests/fugu_clean_v1/grpo_pilot_train"
CONDUCTOR = "http://127.0.0.1:8011/v1"
ENV_DIR = REPO / "environments/fugu-ultra-pilot"
PROBE_SEED = 20260802  # fixed across arms and checkpoints -> paired samples

# (lane, pilot_config, task_manifest|None=env default, live_safety, budget,
#  worker_max_tokens, max_turns, n_tasks) — mirrors orch_pool_d.yaml verbatim.
# max_turns per lane is from orch_pool_d.yaml and is INVERTED vs intuition:
# the six curated single-turn lanes run max_turns 2 (plan turn + one feedback
# retry turn), while the four agentic lanes use the env default of 1 (one
# conductor action whose workflow executes and scores). Probing repo with
# turns=2 made every episode stop at max_turns_reached BEFORE converting ->
# valid_for_training 0 -> uniform 0.0 (measured; that, not the cap, was the
# second repo-zero cause).
LANES = [
    ("single_turn", "pilot_config_heldout_probe.json", "heldout_eval_taskspecs.jsonl",
     "live_safety_pool_d_singleturn.json", "short", 16384, 2, 30),
    ("office_telecom", "pilot_config_pool_d_office.json", "tau2_telecom_taskspecs.jsonl",
     "live_safety_pool_d_office.json", "long", 8192, 1, 10),
    ("office_crm", "pilot_config_pool_d_office.json", "crmarena_train_taskspecs.jsonl",
     "live_safety_pool_d_office.json", "long", 8192, 1, 10),
    ("tool_dialogue", "pilot_config_pool_d_office.json", "tau_retail_train_full_taskspecs.jsonl",
     "live_safety_pool_d_office.json", "long", 8192, 1, 10),
    ("repo_open_repo_terminal", "pilot_config_pool_d_envlanes.json", None,
     "live_safety_pool_d_envlanes.json", "long", 8192, 1, 8),
]


def _wait_repo_quiet(max_wait_s: int = 3600, poll_s: int = 60) -> None:
    """Block until no CAMPAIGN Harbor jobs are running (max 60 min).

    Probe repo episodes and campaign repo groups share the term workers and
    the docker host; run concurrently, agent turns starve, harbor's internal
    phase timeouts cancel trials before the verifier runs, and every probe
    episode quarantines to 0.0 (measured 2026-08-04: step-80 repo probe 0/8
    while campaign records in the same window graded 0.5-1.0). A repo number
    taken during a campaign repo burst is a measurement of contention, not
    of the conductor — so wait for a quiet window; a campaign repo group
    drains in <= ~50 min.
    """
    import subprocess as _sp
    import time as _time

    waited = 0
    while waited < max_wait_s:
        out = _sp.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        active = sum(1 for l in out.splitlines() if "harbor" in l and "jobs" in l and "start" in l)
        if active == 0:
            if waited:
                print(f"[probe] repo lane: quiet window after {waited}s wait", flush=True)
            return
        if waited == 0:
            print(f"[probe] repo lane: {active} campaign Harbor job(s) active — waiting for a quiet window", flush=True)
        _time.sleep(poll_s)
        waited += poll_s
    print(f"[probe] repo lane: NO quiet window in {max_wait_s}s — proceeding; treat repo row as contention-suspect", flush=True)


def _ensure_heldout_pilot_config() -> None:
    """Derived single_turn config: pool-D binding + the HELDOUT ids registered.

    The campaign registry only lists training task ids, so the env would
    select zero holdout tasks (measured). Same binding/masks/policy — only
    task_ids_by_lane is replaced.
    """
    dst = MAN / "pilot_config_heldout_probe.json"
    base = json.loads((MAN / "pilot_config_singleturn_pool_d.json").read_text())
    ids = [json.loads(l)["task_id"] for l in (MAN / "heldout_eval_taskspecs.jsonl").read_text().splitlines() if l.strip()]
    base["task_ids_by_lane"] = {"single_turn": ids}
    base["task_count"] = len(ids)
    base["lane_counts"] = {"single_turn": len(ids)}
    base["purpose_note"] = "DERIVED probe config: pool-D binding over the non-sealed heldout ids (ultra/heldout_probe.py)"
    dst.write_text(json.dumps(base, indent=2))


def run(model_id: str, out_path: Path, max_concurrent: int = 8,
        lanes: list[str] | None = None) -> dict:
    """`lanes` restricts the sweep to the named LANES entries (default: all).

    Used for lane-scoped re-measurement (e.g. repo after the Harbor env fix)
    without re-paying the other lanes; the fixed PROBE_SEED keeps any subset
    paired with the same lane in every other arm.
    """
    sys.path.insert(0, str(ENV_DIR))
    import os

    import fugu_ultra_pilot as env_mod
    from verifiers.clients import ClientConfig

    if lanes is not None:
        unknown = set(lanes) - {l[0] for l in LANES}
        if unknown:
            raise SystemExit(f"unknown lanes: {sorted(unknown)}")
    _ensure_heldout_pilot_config()
    os.environ.setdefault("FUGU_PROBE_API_KEY", "EMPTY")
    client = ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=CONDUCTOR,
        api_key_var="FUGU_PROBE_API_KEY",
        timeout=1800.0,
    )
    lanes_out: dict[str, dict] = {}
    all_rewards: list[float] = []

    for lane, cfg, manifest, safety, budget, wtok, turns, n in LANES:
        if lanes is not None and lane not in lanes:
            continue
        if lane == "repo_open_repo_terminal":
            _wait_repo_quiet()
        kwargs = dict(
            pilot_config_path=str(MAN / cfg),
            task_name=f"fugu_probe_{lane}",
            lane=lane,
            live_safety_path=str(MAN / safety),
            seed=PROBE_SEED,
            shuffle=True,                    # fixed seed -> same tasks every arm
            max_examples=n,
            force_step_budget=budget,
            worker_temperature=0.2,
            worker_max_tokens=wtok,
            worker_reasoning_effort="high",
            max_turns=turns,
            max_concurrency=max_concurrent,
            timeout_s=600.0,
            max_retries=2,
            artifact_dir=str(REPO / "output/fugu_ultra_pool_d/heldout_probe_artifacts" / lane),
            cache_dir=str(REPO / ".ultra_cache/heldout_probe_completions"),
            workflow_record_cache_dir=str(REPO / ".ultra_cache/heldout_probe_records"),
        )
        if manifest:
            kwargs["task_manifest_path"] = str(MAN / manifest)
        env = env_mod.load_environment(**kwargs)
        # Environment.evaluate is async in this verifiers version.
        results = asyncio.run(env.evaluate(
            client, model_id,
            # enable_thinking: false matches the campaign sampling block —
            # without it the chat template turns thinking ON and the conductor
            # burns the whole 1536-token budget deliberating, never emitting
            # JSON (measured: parse 0.0, stop=max_turns_reached).
            sampling_args={"temperature": 0.0, "max_tokens": 1536,
                           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            num_examples=n, rollouts_per_example=1,
            max_concurrent=max_concurrent,
        ))
        outputs = results["outputs"] if isinstance(results, dict) else results.outputs
        rewards = [float(o["reward"]) for o in outputs]
        all_rewards.extend(rewards)
        lanes_out[lane] = {
            "n": len(rewards),
            "mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "solved_frac": round(sum(1 for r in rewards if r >= 0.7) / len(rewards), 4) if rewards else None,
            "rewards": [round(r, 4) for r in rewards],
        }
        print(f"[probe] {lane}: n={len(rewards)} mean={lanes_out[lane]['mean']}", flush=True)

    summary = {
        "model": model_id,
        "seed": PROBE_SEED,
        "lane_filter": sorted(lanes) if lanes is not None else None,
        "n_total": len(all_rewards),
        "mean_reward": round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else None,
        "lanes": lanes_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--lanes", help="comma-separated LANES subset (default: all)")
    args = ap.parse_args()
    lane_list = [l.strip() for l in args.lanes.split(",") if l.strip()] if args.lanes else None
    s = run(args.model, args.out, max_concurrent=args.concurrency, lanes=lane_list)
    print(json.dumps({**s, "lanes": {k: {kk: vv for kk, vv in v.items() if kk != "rewards"} for k, v in s["lanes"].items()}}, indent=2))


if __name__ == "__main__":
    main()
