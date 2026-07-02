"""Human-reviewable safety gate for live Ultra worker launches.

The gate is intentionally about launch shape, not dollar estimates. A paid GRPO
run must declare which lanes, workers, providers, and step budgets are allowed;
the Verifiers environment rejects live worker execution if the active config
does not match that declaration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .providers import routed_provider_name, routed_slug

VERSION = "fugu_ultra_live_worker_safety_v1"


class LiveWorkerSafetyError(RuntimeError):
    pass


def _as_set(payload: dict[str, Any], key: str) -> set[str]:
    return {str(item) for item in payload.get(key, [])}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LiveWorkerSafetyError(f"invalid live-worker safety JSON at {path}: {exc}") from exc


def _lane_worker_names(pilot_config: dict[str, Any], lane: str) -> list[str]:
    try:
        return [str(worker) for worker in pilot_config["lane_worker_masks"][lane]]
    except KeyError as exc:
        raise LiveWorkerSafetyError(f"lane {lane!r} is not present in pilot_config lane_worker_masks") from exc


def _selected_example_count(pilot_config: dict[str, Any], lane: str, max_examples: int | None) -> int:
    task_count = len(pilot_config.get("task_ids_by_lane", {}).get(lane, []))
    if max_examples is None or max_examples <= 0:
        return task_count
    return min(task_count, int(max_examples))


def validate_live_worker_safety(
    *,
    pilot_config: dict[str, Any],
    lane: str | None,
    provider_mode: str,
    allow_yunwu_live: bool,
    live_safety_path: str | Path | None,
    max_examples: int | None,
    force_step_budget: str | None,
) -> dict[str, Any]:
    """Validate a live worker launch against a reviewed safety manifest.

    Fake/offline provider mode is always allowed. Live mode requires
    ``live_safety_path`` and the active lane/worker/provider/budget shape must be
    within the manifest's allowlists.
    """

    if provider_mode != "live":
        return {"enforced": False, "provider_mode": provider_mode}
    if lane is None:
        raise LiveWorkerSafetyError("live worker runs must pass a concrete lane")
    if live_safety_path is None:
        raise LiveWorkerSafetyError("live worker runs require live_safety_path")

    safety_path = Path(live_safety_path).expanduser().resolve()
    safety = _read_json(safety_path)
    if safety.get("version") != VERSION:
        raise LiveWorkerSafetyError(f"live safety manifest version must be {VERSION!r}")
    if not bool(safety.get("approved")):
        raise LiveWorkerSafetyError("live safety manifest is not approved")

    allowed_lanes = _as_set(safety, "allowed_lanes")
    if lane not in allowed_lanes:
        raise LiveWorkerSafetyError(f"lane {lane!r} is not allowed by {safety_path}")

    max_workflow_steps = int(pilot_config.get("workflow_policy", {}).get("max_workflow_steps", 0))
    safety_max_steps = int(safety.get("max_workflow_steps", 0))
    if max_workflow_steps <= 0 or safety_max_steps <= 0 or max_workflow_steps > safety_max_steps:
        raise LiveWorkerSafetyError(
            f"pilot max_workflow_steps={max_workflow_steps} exceeds safety max {safety_max_steps}"
        )

    required_budget = safety.get("required_force_step_budget")
    if required_budget is not None and force_step_budget != str(required_budget):
        raise LiveWorkerSafetyError(
            f"force_step_budget must be {required_budget!r} for this live run, got {force_step_budget!r}"
        )

    selected_examples = _selected_example_count(pilot_config, lane, max_examples)
    max_examples_by_lane = safety.get("max_examples_by_lane", {})
    if lane in max_examples_by_lane and selected_examples > int(max_examples_by_lane[lane]):
        raise LiveWorkerSafetyError(
            f"selected {selected_examples} examples for lane {lane!r}, "
            f"exceeds safety max {max_examples_by_lane[lane]}"
        )

    lane_workers = _lane_worker_names(pilot_config, lane)
    allowed_workers_by_lane = {
        str(k): {str(item) for item in v}
        for k, v in safety.get("allowed_workers_by_lane", {}).items()
    }
    allowed_workers = allowed_workers_by_lane.get(lane, _as_set(safety, "allowed_workers"))
    if not allowed_workers:
        raise LiveWorkerSafetyError(f"no allowed workers declared for lane {lane!r}")
    unexpected_workers = sorted(set(lane_workers) - allowed_workers)
    if unexpected_workers:
        raise LiveWorkerSafetyError(f"workers not allowed for lane {lane!r}: {unexpected_workers}")

    forbidden_workers = _as_set(safety, "forbidden_workers")
    forbidden_selected = sorted(set(lane_workers) & forbidden_workers)
    if forbidden_selected:
        raise LiveWorkerSafetyError(f"forbidden workers selected for lane {lane!r}: {forbidden_selected}")

    allowed_providers = _as_set(safety, "allowed_providers")
    allowed_yunwu_workers = _as_set(safety, "allowed_yunwu_workers")
    worker_pool = pilot_config.get("worker_pool", {})
    routes: list[dict[str, str]] = []
    yunwu_workers: list[str] = []
    for worker in lane_workers:
        ident = worker_pool.get(worker)
        if ident is None:
            raise LiveWorkerSafetyError(f"worker {worker!r} is missing from pilot_config worker_pool")
        model = str(ident.get("model") or "")
        provider = routed_provider_name(model)
        provider_model = routed_slug(model, provider)
        if provider not in allowed_providers:
            raise LiveWorkerSafetyError(
                f"provider {provider!r} for worker {worker!r} is not allowed by {safety_path}"
            )
        if provider == "yunwu":
            yunwu_workers.append(worker)
        routes.append(
            {
                "worker": worker,
                "model": model,
                "provider": provider,
                "provider_model": provider_model,
            }
        )

    unexpected_yunwu = sorted(set(yunwu_workers) - allowed_yunwu_workers)
    if unexpected_yunwu:
        raise LiveWorkerSafetyError(f"Yunwu workers not explicitly allowed: {unexpected_yunwu}")
    if yunwu_workers and not allow_yunwu_live:
        raise LiveWorkerSafetyError(
            f"lane {lane!r} includes Yunwu workers {sorted(yunwu_workers)}, but allow_yunwu_live is false"
        )

    return {
        "enforced": True,
        "safety_path": str(safety_path),
        "purpose": safety.get("purpose"),
        "lane": lane,
        "selected_examples": selected_examples,
        "max_workflow_steps": max_workflow_steps,
        "force_step_budget": force_step_budget,
        "routes": routes,
    }
