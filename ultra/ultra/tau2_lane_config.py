"""Bounce-in artifacts for the tau2-telecom office lane.

Builds, from the exported telecom manifest plus the campaign's existing
env-lane pilot config, the two files a checkpoint-boundary bounce needs:

  * pilot_config_r5_office.json — the env-lane config PLUS an
    ``office_telecom`` lane (tool-calling worker mask, deterministic task
    sample), and
  * live_safety_r5_office.json — the live-worker safety manifest for that
    lane, written UNAPPROVED (the approval flip is the user's gate).

The task sample is a deterministic sha1-ordered draw over the manifest's
2171 train tasks: stable across runs, uniform over scenario families, and
free of any content-based selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ultra.live_worker_safety import VERSION as SAFETY_VERSION

LANE = "office_telecom"
DEFAULT_LANE_SIZE = 240
DEFAULT_GROUP_SIZE = 4
# tau2 solo needs a reliable function-caller; reuse the proven tool_dialogue
# worker mask rather than inventing a new pool for one lane.
SOURCE_WORKER_LANE = "tool_dialogue"


def sample_task_ids(manifest: Path, lane_size: int = DEFAULT_LANE_SIZE) -> list[str]:
    """Deterministic sha1-ordered sample of manifest task ids."""
    ids = [
        json.loads(line)["task_id"]
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]
    ordered = sorted(ids, key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    return sorted(ordered[:lane_size])


def build_lane_config(
    base_config: dict[str, Any],
    task_ids: list[str],
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[str, Any]:
    """Env-lane pilot config extended with the office_telecom lane."""
    if not task_ids:
        raise ValueError("office_telecom lane needs at least one task id")
    workers = list(base_config["lane_worker_masks"][SOURCE_WORKER_LANE])
    config = json.loads(json.dumps(base_config))  # deep copy
    config["version"] = "fugu_ultra_envlanes_r5_office_telecom"
    config["lane_worker_masks"][LANE] = workers
    config["task_ids_by_lane"][LANE] = list(task_ids)
    config["lane_counts"][LANE] = len(task_ids)
    config["group_size_by_lane"][LANE] = int(group_size)
    config["task_count"] = sum(
        len(ids) for ids in config["task_ids_by_lane"].values()
    )
    return config


def build_safety_manifest(
    base_safety: dict[str, Any],
    config: dict[str, Any],
    *,
    lane_size: int,
    approved: bool = False,
) -> dict[str, Any]:
    """Safety manifest covering the existing env lanes plus office_telecom.

    Defaults to UNAPPROVED: approval is the user's launch gate, so it must
    be passed in explicitly (``--approved``) rather than assumed. Carrying
    it as a parameter means regenerating the manifest preserves the
    decision instead of silently reverting a hand edit.
    """
    safety = json.loads(json.dumps(base_safety))
    safety["approved"] = bool(approved)
    safety["version"] = SAFETY_VERSION
    safety["purpose"] = (
        "r5 bounce-in: office_telecom lane (tau2-bench telecom SOLO mode, "
        "harness tau2_solo). Open-weight tool-calling workers, openrouter "
        "only, budget long. Training draws come from telecom_full MINUS the "
        "published 114-task telecom benchmark, which stays sealed for eval."
    )
    safety["allowed_lanes"] = sorted(set(safety["allowed_lanes"]) | {LANE})
    safety["allowed_workers_by_lane"][LANE] = list(
        config["lane_worker_masks"][LANE]
    )
    safety["max_examples_by_lane"][LANE] = int(lane_size)
    return safety


def expand_lane(
    config: dict[str, Any],
    safety: dict[str, Any],
    lane: str,
    manifest: Path,
) -> int:
    """Point an existing lane at every task in ``manifest``.

    Used to grow tau retail from the 60-task pilot slice to the full 500-task
    TRAIN split — same harness, same workers, strictly more office-shaped
    tasks. Returns the new task count.
    """
    task_ids = sorted(
        json.loads(line)["task_id"]
        for line in manifest.read_text().splitlines()
        if line.strip()
    )
    config["task_ids_by_lane"][lane] = task_ids
    config["lane_counts"][lane] = len(task_ids)
    config["task_count"] = sum(len(ids) for ids in config["task_ids_by_lane"].values())
    safety["max_examples_by_lane"][lane] = len(task_ids)
    return len(task_ids)


def export(
    manifest: Path,
    base_config_path: Path,
    base_safety_path: Path,
    out_dir: Path,
    *,
    lane_size: int = DEFAULT_LANE_SIZE,
    group_size: int = DEFAULT_GROUP_SIZE,
    retail_manifest: Path | None = None,
    approved: bool = False,
) -> dict[str, Any]:
    task_ids = sample_task_ids(manifest, lane_size)
    config = build_lane_config(
        json.loads(base_config_path.read_text()), task_ids, group_size=group_size
    )
    safety = build_safety_manifest(
        json.loads(base_safety_path.read_text()), config,
        lane_size=len(task_ids), approved=approved,
    )
    retail_tasks = (
        expand_lane(config, safety, SOURCE_WORKER_LANE, retail_manifest)
        if retail_manifest is not None
        else None
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "pilot_config_r5_office.json"
    safety_path = out_dir / "live_safety_r5_office.json"
    config_path.write_text(json.dumps(config, indent=1))
    safety_path.write_text(json.dumps(safety, indent=1))
    return {
        "lane": LANE,
        "lane_tasks": len(task_ids),
        "group_size": group_size,
        "workers": config["lane_worker_masks"][LANE],
        "tool_dialogue_tasks": retail_tasks,
        "pilot_config": str(config_path),
        "live_safety": str(safety_path),
        "approved": safety["approved"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--base-safety", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lane-size", type=int, default=DEFAULT_LANE_SIZE)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--retail-manifest", type=Path, default=None,
                        help="expand the tool_dialogue lane to this manifest's tasks")
    parser.add_argument("--approved", action="store_true",
                        help="mark the safety manifest approved for live spend "
                             "(operator decision; default is unapproved)")
    args = parser.parse_args(argv)
    summary = export(
        args.manifest,
        args.base_config,
        args.base_safety,
        args.out_dir,
        lane_size=args.lane_size,
        group_size=args.group_size,
        retail_manifest=args.retail_manifest,
        approved=args.approved,
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
