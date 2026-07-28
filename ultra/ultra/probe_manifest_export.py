"""Export probe-retained questions as fugu-ultra-pilot campaign artifacts.

The interleaved r5 campaign runs its curated verifiable lanes through the
pilot environment, which selects tasks by `task_ids_by_lane` from a pilot
config and reads full task specs from a manifest. Nothing produced those
from a probe journal before. This module emits, from the probe's retained
set:

  * r5_{math,code,rlpr}_taskspecs.jsonl — the retained rows of the
    hard-mix manifests (split by grader lane), and
  * pilot_config_singleturn_r5.json — derived from the stage2-era
    singleturn pilot config BUT with
      - task_ids_by_lane restricted to the retained ids,
      - the worker pool REBUILT from the CURRENT open-weight binding
        (the stage2 config routes st_* workers via yunwu/commercial —
        wrong pool and wrong providers for r5), and
      - group_size_by_lane set from the campaign's G.

Retention is read from the probe journal (question-keyed, outcome-based);
questions map back to task ids via the same loader the probe used, so the
mapping is byte-consistent with what was probed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultra.grpo_campaign import load_mix_candidates

LANE_BY_GRADER = {
    "math_equal": "math",
    "rlpr_lenient": "rlpr",
    "code_exec_stdio": "code",
    # Office lanes are single-turn like the others, so they curate and
    # export through the same path.
    "sql_exec": "bird",
    "finance_numeric": "finance",
    "dabstep_exec": "dabstep",
}
MANIFEST_BY_GRADER = {
    "math_equal": "hard_mix_math_taskspecs.jsonl",
    "rlpr_lenient": "hard_mix_rlpr_taskspecs.jsonl",
    "code_exec_stdio": "hard_mix_code_taskspecs.jsonl",
    "sql_exec": "bird_probe_candidates_taskspecs.jsonl",
    "finance_numeric": "finance_probe_candidates_taskspecs.jsonl",
    "dabstep_exec": "dabstep_probe_candidates_taskspecs.jsonl",
}
# Single-turn lanes in the orch's env order — drives the ratio suggestion.
LANE_ORDER = ["math", "code", "rlpr", "bird", "finance", "dabstep"]


def retained_questions(probe_journal: Path, probe_size: int = 3) -> set[str]:
    """Questions whose COMPLETE probe groups showed reward variance."""
    retained: set[str] = set()
    for line in probe_journal.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        scored = [r for r in row.get("rewards", []) if r is not None]
        if len(scored) < probe_size:
            continue  # incomplete observation — never classify from outage
        if len(set(scored)) > 1:
            retained.add(row["question"])
    return retained


def build_worker_pool(binding: dict[str, Any]) -> dict[str, Any]:
    """Pilot worker-pool entries from the CURRENT open-weight binding.

    Provider routing is OpenRouter DEFAULT (price+uptime balanced).
    Cheapest-first was tried 2026-07-28 and REVERTED the same day: it routed
    to the slowest hosts (~56 tok/s) and stalled the campaign; wall-clock is
    the binding constraint, and the worker-cap reduction delivers the cost
    saving instead.
    """
    pool: dict[str, Any] = {}
    for slot in binding["slots"]:
        name = f"ow_{slot['worker_id']}"
        pool[name] = {
            "backend": "direct_qa",
            "max_turns": None,
            "model": slot["runtime_model"],
            "name": name,
            "role_prior": list(slot["role_prior"]),
        }
    return pool


def env_ratios_from_retention(
    retained_per_lane: dict[str, int],
    lane_order: list[str],
    e2e_share: float,
    *,
    floor: float = 0.10,
) -> list[float]:
    """Split the single-turn share across lanes by how many tasks each KEPT.

    Ratios in orch_r5.yaml were set from the pre-probe plan, but retention
    differs sharply by lane (math ~44%, code ~47%, rlpr ~23% of very
    different pool sizes). Drawing a lane harder than its retained set
    supports just redraws the same questions, so the single-turn share is
    split proportionally to retained counts — equal draw pressure per task.
    ``floor`` is each lane's minimum fraction OF THE SINGLE-TURN SHARE, so
    a small lane thins out but never vanishes.

    ``lane_order`` must list the single-turn lanes in the orch's env order;
    the returned list covers exactly those lanes and sums to
    ``1 - e2e_share``.
    """
    if not 0.0 <= e2e_share < 1.0:
        raise ValueError(f"e2e_share must be in [0, 1): {e2e_share}")
    counts = [max(0, int(retained_per_lane.get(lane, 0))) for lane in lane_order]
    total = sum(counts)
    if total <= 0:
        raise ValueError("no retained tasks; cannot derive ratios")
    if floor * len(counts) > 1.0:
        raise ValueError("floors exceed the available share")

    shares = [c / total for c in counts]
    # Lift every lane to the floor, then take the excess back from the lanes
    # above it, in proportion to how far above they are.
    lifted = [max(s, floor) for s in shares]
    excess = sum(lifted) - 1.0
    headroom = sum(s - floor for s in lifted if s > floor)
    if excess > 0 and headroom > 0:
        lifted = [s - excess * (s - floor) / headroom if s > floor else s
                  for s in lifted]
    single_share = 1.0 - e2e_share
    ratios = [round(s * single_share, 4) for s in lifted]
    # Rounding must not leave the orch's env_ratios short of their target:
    # the launch preflight rejects a set that does not sum to 1.0, so the
    # residual goes to the largest lane (where it is proportionally least).
    residual = round(single_share - sum(ratios), 6)
    if residual:
        biggest = max(range(len(ratios)), key=ratios.__getitem__)
        ratios[biggest] = round(ratios[biggest] + residual, 6)
    return ratios


def export(
    probe_journal: Path,
    manifest_dir: Path,
    binding_path: Path,
    base_pilot_config: Path,
    out_dir: Path,
    group_size: int,
    probe_size: int = 3,
) -> dict[str, Any]:
    """Write the r5 manifests + pilot config; returns summary counts."""
    retained = retained_questions(probe_journal, probe_size)
    if not retained:
        raise RuntimeError("probe journal retains no questions — do not "
                           "generate empty campaign manifests")

    candidates = load_mix_candidates(manifest_dir)
    by_question = {c.question: c for c in candidates}
    retained_ids: dict[str, set[str]] = {lane: set() for lane in LANE_ORDER}
    unmatched = 0
    unknown_graders: dict[str, int] = {}
    for question in retained:
        candidate = by_question.get(question)
        if candidate is None:
            unmatched += 1
            continue
        lane = LANE_BY_GRADER.get(candidate.grader_type)
        if lane is None:
            # A grader in the mix with no lane mapping would otherwise be
            # dropped silently, quietly shrinking the campaign.
            unknown_graders[candidate.grader_type] = (
                unknown_graders.get(candidate.grader_type, 0) + 1)
            continue
        retained_ids[lane].add(candidate.task_id)
    if unknown_graders:
        raise RuntimeError(
            f"retained questions use graders with no lane mapping: "
            f"{unknown_graders} — add them to LANE_BY_GRADER/MANIFEST_BY_GRADER")
    if not any(retained_ids.values()):
        # Every retained question failed to match a candidate: the manifest
        # directory does not correspond to the probe journal. Writing empty
        # lane manifests here would launch a campaign with no data.
        raise RuntimeError(
            f"none of the {len(retained)} retained questions matched a task in "
            f"{manifest_dir} — wrong manifest directory for this probe journal?")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_counts: dict[str, int] = {}
    for grader_type, manifest_name in MANIFEST_BY_GRADER.items():
        lane = LANE_BY_GRADER[grader_type]
        ids = retained_ids[lane]
        rows = []
        source = manifest_dir / manifest_name
        if not source.exists():
            # A lane absent from this campaign's mix is fine, but only if
            # nothing was retained for it — otherwise we would write an
            # empty manifest for questions the probe paid to keep.
            if ids:
                raise RuntimeError(
                    f"{len(ids)} retained {lane} questions but their source "
                    f"manifest is missing: {source}")
            manifest_counts[lane] = 0
            continue
        for line in source.read_text().split("\n"):
            if not line.strip():
                continue
            spec = json.loads(line)
            if spec.get("task_id") in ids:
                rows.append(line)
        out_path = out_dir / f"r5_{lane}_taskspecs.jsonl"
        out_path.write_text("\n".join(rows) + ("\n" if rows else ""))
        manifest_counts[lane] = len(rows)

    binding = json.loads(binding_path.read_text())
    base = json.loads(base_pilot_config.read_text())
    pool = build_worker_pool(binding)
    all_ids = sorted(set().union(*retained_ids.values()))
    config = dict(base)
    config["version"] = "fugu_ultra_singleturn_r5_probe_curated"
    # The 27B speaks the typed control contract, not the paper
    # three-list format — without this flag the env scores every
    # typed plan 0 (launch-gating catch, plan wf_8ef7952d-305).
    config["conductor_contract"] = "typed_control"
    config["worker_pool"] = pool
    config["worker_pool_names"] = sorted(pool)
    config["lane_worker_masks"] = {"single_turn": sorted(pool)}
    config["task_ids_by_lane"] = {"single_turn": all_ids}
    config["lane_counts"] = {"single_turn": len(all_ids)}
    config["task_count"] = len(all_ids)
    config["group_size_by_lane"] = {"single_turn": int(group_size)}
    # r5 pool routes everything through openrouter; the stage2-era
    # yunwu/commercial routing must not survive into this config.
    config["provider_policy"] = {
        "gpt_never_openrouter": False,
        "openrouter_workers": sorted(pool),
        "yunwu_only_workers": [],
    }
    config_path = out_dir / "pilot_config_singleturn_r5.json"
    config_path.write_text(json.dumps(config, indent=1))

    per_lane = {lane: len(ids) for lane, ids in retained_ids.items()}
    # Suggested single-turn ratios in the orch env order, for both configs:
    # orch_r5.yaml gives the e2e lanes 0.17; the office bounce config gives
    # them 0.32 (tool 0.085 + repo 0.085 + office_telecom 0.15).
    return {
        "retained_questions": len(retained),
        "unmatched_questions": unmatched,
        "per_lane": per_lane,
        "manifest_rows": manifest_counts,
        "pilot_config": str(config_path),
        "suggested_env_ratios": dict(zip(
            LANE_ORDER,
            env_ratios_from_retention(per_lane, LANE_ORDER, e2e_share=0.17),
        )),
        "suggested_env_ratios_office_bounce": dict(zip(
            LANE_ORDER,
            env_ratios_from_retention(per_lane, LANE_ORDER, e2e_share=0.32),
        )),
    }
