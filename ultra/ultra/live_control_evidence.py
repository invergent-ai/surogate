"""Audit preserved Harbor trajectories for live-control training evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RUN_NAME_RE = re.compile(r"^(?P<workflow>.+)__s(?P<step>\d+)__w(?P<worker>.+)$")
TERMINALBENCH_MARKERS = ("terminalbench", "terminal-bench", "tb21", "fugu_r")
DEFAULT_ORDINAL_ALIASES = {
    "claude-opus-4-8": 0,
    "gpt-5.6-sol": 0,
    "gemini-3.5-flash": 1,
    "gpt-5.5": 2,
    "gpt-5.6-terra": 2,
    "z-ai/glm-5.2": 3,
    "grok-4.5": 3,
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _normalize_model(model: str) -> str:
    return model.removeprefix("openai/")


def _task_id(workflow_key: str) -> str:
    return workflow_key.split("__vf-", 1)[0]


def _task_source(task_id: str) -> str:
    return task_id.split("__", 1)[0]


def _is_terminalbench(task_id: str, trajectory_path: Path) -> bool:
    haystack = f"{task_id} {trajectory_path}".lower()
    return any(marker in haystack for marker in TERMINALBENCH_MARKERS)


def _reward(trial_dir: Path) -> float | None:
    path = trial_dir / "verifier" / "reward.txt"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _trajectory_record(root: Path, path: Path, current_pool: set[str]) -> dict[str, Any]:
    relative = path.relative_to(root)
    run_name = relative.parts[0]
    match = RUN_NAME_RE.fullmatch(run_name)
    if match is None:
        raise ValueError(f"unrecognized Harbor run directory: {run_name}")

    trajectory = _read_json(path)
    trial_dir = path.parent.parent
    config = _read_json(trial_dir / "config.json")
    agent = trajectory.get("agent") or {}
    config_agent = config.get("agent") or {}
    model = _normalize_model(str(agent.get("model_name") or config_agent.get("model_name") or ""))
    api_base = str((config_agent.get("kwargs") or {}).get("api_base") or "")
    steps = trajectory.get("steps") or []
    agent_steps = [step for step in steps if isinstance(step, dict) and step.get("source") == "agent"]
    tool_calls = [
        call
        for step in agent_steps
        for call in (step.get("tool_calls") or [])
        if isinstance(call, dict)
    ]
    workflow_key = match.group("workflow")
    task_id = _task_id(workflow_key)
    return {
        "workflow_key": workflow_key,
        "step_index": int(match.group("step")),
        "worker_name": match.group("worker"),
        "model": model,
        "current_pool_member": model in current_pool,
        "provider_base": api_base,
        "task_id": task_id,
        "task_source": _task_source(task_id),
        "terminalbench": _is_terminalbench(task_id, path),
        "reward": _reward(trial_dir),
        "agent_turns": len(agent_steps),
        "tool_calls": len(tool_calls),
        "completion_calls": sum(
            call.get("function_name") == "mark_task_complete" for call in tool_calls
        ),
        "trajectory_path": str(path),
        "config_path": str(trial_dir / "config.json"),
    }


def audit_live_control_evidence(
    *,
    harbor_root: Path,
    current_pool: tuple[str, ...],
    yunwu_base: str,
    ordinal_aliases: dict[str, int] | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    """Summarize zero-call training evidence from preserved Harbor trajectories."""
    root = harbor_root.resolve()
    pool = set(current_pool)
    aliases = dict(DEFAULT_ORDINAL_ALIASES if ordinal_aliases is None else ordinal_aliases)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for path in sorted(root.glob("*/jobs/*/*/agent/trajectory.json")):
        try:
            record = _trajectory_record(root, path, pool)
        except Exception as exc:  # noqa: BLE001 - retain a complete audit report
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not record["terminalbench"]:
            records.append(record)

    workflows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        workflows[record["workflow_key"]].append(record)
    ordered_workflows = [
        sorted(steps, key=lambda step: step["step_index"])
        for steps in workflows.values()
    ]
    multi = [steps for steps in ordered_workflows if len(steps) >= 2]
    complete_multi = [
        steps
        for steps in multi
        if [step["step_index"] for step in steps] == list(range(len(steps)))
    ]
    yunwu_multi = [
        steps for steps in complete_multi if all(step["provider_base"].rstrip("/") == yunwu_base.rstrip("/") for step in steps)
    ]
    pool_multi = [steps for steps in yunwu_multi if all(step["current_pool_member"] for step in steps)]
    ordinal_multi = [steps for steps in complete_multi if all(step["model"] in aliases for step in steps)]
    transitions = [
        (steps, previous, current)
        for steps in pool_multi
        for previous, current in zip(steps, steps[1:], strict=False)
        if previous["model"] != current["model"]
    ]
    improvements = [
        (steps, previous, current)
        for steps, previous, current in transitions
        if previous["reward"] == 0.0 and current["reward"] == 1.0
    ]
    ordinal_transitions = [
        (steps, previous, current)
        for steps in ordinal_multi
        for previous, current in zip(steps, steps[1:], strict=False)
        if aliases[previous["model"]] != aliases[current["model"]]
    ]
    ordinal_improvements = [
        (steps, previous, current)
        for steps, previous, current in ordinal_transitions
        if previous["reward"] == 0.0 and current["reward"] == 1.0
    ]
    builder_debugger_builder = [
        steps
        for steps in pool_multi
        if len(steps) >= 3
        and steps[0]["model"] == steps[-1]["model"]
        and steps[0]["model"] != steps[1]["model"]
    ]
    ordinal_builder_debugger_builder = [
        steps
        for steps in ordinal_multi
        if len(steps) >= 3
        and aliases[steps[0]["model"]] == aliases[steps[-1]["model"]]
        and aliases[steps[0]["model"]] != aliases[steps[1]["model"]]
    ]

    def workflow_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "workflow_key": steps[0]["workflow_key"],
            "task_id": steps[0]["task_id"],
            "models": [step["model"] for step in steps],
            "rewards": [step["reward"] for step in steps],
            "agent_turns": [step["agent_turns"] for step in steps],
            "tool_calls": [step["tool_calls"] for step in steps],
            "trajectory_paths": [step["trajectory_path"] for step in steps],
        }

    report = {
        "version": "fugu_live_control_evidence_v1",
        "harbor_root": str(root),
        "external_calls_made": 0,
        "terminalbench_records_excluded": sum(
            1
            for path in root.glob("*/jobs/*/*/agent/trajectory.json")
            if any(marker in str(path).lower() for marker in TERMINALBENCH_MARKERS)
        ),
        "current_pool": list(current_pool),
        "required_provider_base": yunwu_base,
        "counts": {
            "trajectories": len(records),
            "unique_tasks": len({record["task_id"] for record in records}),
            "workflow_groups": len(ordered_workflows),
            "multi_position_workflows": len(multi),
            "complete_multi_position_workflows": len(complete_multi),
            "yunwu_multi_position_workflows": len(yunwu_multi),
            "current_pool_multi_position_workflows": len(pool_multi),
            "current_pool_worker_transitions": len(transitions),
            "current_pool_fail_to_pass_transitions": len(improvements),
            "current_pool_builder_debugger_builder_workflows": len(builder_debugger_builder),
            "ordinal_compatible_multi_position_workflows": len(ordinal_multi),
            "ordinal_worker_transitions": len(ordinal_transitions),
            "ordinal_fail_to_pass_transitions": len(ordinal_improvements),
            "ordinal_builder_debugger_builder_workflows": len(ordinal_builder_debugger_builder),
            "parse_errors": len(errors),
        },
        "models": dict(sorted(Counter(record["model"] for record in records).items())),
        "providers": dict(sorted(Counter(record["provider_base"] for record in records).items())),
        "task_sources": dict(sorted(Counter(record["task_source"] for record in records).items())),
        "rewards": dict(
            sorted(Counter("missing" if record["reward"] is None else str(record["reward"]) for record in records).items())
        ),
        "current_pool_workflow_examples": [workflow_summary(steps) for steps in pool_multi[:20]],
        "yunwu_multi_position_examples": [workflow_summary(steps) for steps in yunwu_multi[:20]],
        "ordinal_compatible_examples": [workflow_summary(steps) for steps in ordinal_multi[:20]],
        "fail_to_pass_examples": [workflow_summary(steps) for steps, _, _ in improvements[:20]],
        "ordinal_fail_to_pass_examples": [
            workflow_summary(steps) for steps, _, _ in ordinal_improvements[:20]
        ],
        "builder_debugger_builder_examples": [workflow_summary(steps) for steps in builder_debugger_builder[:20]],
        "ordinal_builder_debugger_builder_examples": [
            workflow_summary(steps) for steps in ordinal_builder_debugger_builder[:20]
        ],
        "errors": errors[:50],
        "training_readiness": {
            "has_current_pool_multi_position_workflows": bool(pool_multi),
            "has_current_pool_fail_to_pass_transitions": bool(improvements),
            "has_builder_debugger_builder_workflows": bool(builder_debugger_builder),
            "has_ordinal_compatible_multi_position_workflows": bool(ordinal_multi),
            "has_ordinal_fail_to_pass_transitions": bool(ordinal_improvements),
            "exact_current_pool_collection_required": not bool(pool_multi and improvements),
            "verdict": (
                "ready_for_offline_control_warmstart"
                if pool_multi and improvements
                else (
                    "ready_for_ordinal_warmstart_but_current_pool_collection_required"
                    if ordinal_multi and ordinal_improvements
                    else "insufficient_transition_evidence"
                )
            ),
        },
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-root", type=Path, default=Path(".ultra_harbor_runs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--current-pool",
        nargs="+",
        default=["gpt-5.6-sol", "gemini-3.5-flash", "gpt-5.6-terra", "grok-4.5"],
    )
    parser.add_argument("--yunwu-base", default="https://yunwu.ai/v1")
    args = parser.parse_args(argv)
    report = audit_live_control_evidence(
        harbor_root=args.harbor_root,
        current_pool=tuple(args.current_pool),
        yunwu_base=args.yunwu_base,
        ordinal_aliases=DEFAULT_ORDINAL_ALIASES,
        out=args.out,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
