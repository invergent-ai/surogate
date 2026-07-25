"""Recover legacy three-list workflows rejected only by the V1 JSON parser."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from director.agentic import fugu_outcome_corpus as v1


RECOVERY_VERSION = "fugu_outcome_trajectory_corpus_v2_legacy_recovery"
V1_LEGACY_REASON = "invalid_initial_workflow:initial raw_plan is invalid JSON"


def _balanced_list(text: str, start: int) -> str | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in "\"'":
            quote = character
        elif character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_legacy_three_list_workflow(raw: str) -> dict[str, Any]:
    """Parse the deployed paper-format plan without executing source text."""

    if not isinstance(raw, str) or not raw.strip():
        raise v1.OutcomeCorpusError("legacy plan is empty")
    text = raw.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    def grab(name: str) -> Any:
        last: str | None = None
        for match in re.finditer(name + r"\s*=\s*(?=\[)", text):
            candidate = _balanced_list(text, match.end())
            if candidate is not None:
                last = candidate
        if last is None:
            raise v1.OutcomeCorpusError(f"legacy plan lacks {name}")
        try:
            return ast.literal_eval(last)
        except (SyntaxError, ValueError) as exc:
            raise v1.OutcomeCorpusError(f"legacy {name} list is invalid") from exc

    model_ids = grab(r"model[_ ]?id")
    subtasks = grab(r"subtasks")
    access_list = grab(r"access[_ ]?list")
    if not all(isinstance(value, list) for value in (model_ids, subtasks, access_list)):
        raise v1.OutcomeCorpusError("legacy plan fields must be lists")
    if len(model_ids) == 0 or len({len(model_ids), len(subtasks), len(access_list)}) != 1:
        raise v1.OutcomeCorpusError("legacy plan list lengths differ")

    steps: list[dict[str, Any]] = []
    for position_id, (worker_id, subtask, raw_access) in enumerate(
        zip(model_ids, subtasks, access_list, strict=True)
    ):
        if (
            isinstance(worker_id, bool)
            or not isinstance(worker_id, int)
            or not isinstance(subtask, str)
            or not subtask.strip()
        ):
            raise v1.OutcomeCorpusError(f"invalid legacy step {position_id}")
        if isinstance(raw_access, str):
            raw_access = [raw_access]
        if not isinstance(raw_access, list):
            raise v1.OutcomeCorpusError(f"invalid legacy access {position_id}")
        if len(raw_access) == 1 and (
            isinstance(raw_access[0], str)
            and raw_access[0].strip().lower() == "all"
        ):
            access = list(range(position_id))
        elif all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_access
        ):
            access = list(raw_access)
        else:
            raise v1.OutcomeCorpusError(f"invalid legacy access {position_id}")
        steps.append(
            {"worker_id": worker_id, "subtask": subtask.strip(), "access": access}
        )
    return {"steps": steps}


def canonicalize_legacy_initial_workflow(
    routes: Sequence[Mapping[str, Any]], binding: v1.PoolCapabilities
) -> dict[str, Any]:
    workflow_ids = sorted(
        {
            route.get("workflow_id")
            for route in routes
            if isinstance(route.get("workflow_id"), int)
            and not isinstance(route.get("workflow_id"), bool)
        }
    )
    if not workflow_ids:
        raise v1.OutcomeCorpusError("legacy routes have no workflow")
    first_id = workflow_ids[0]
    raw = next(
        (
            route.get("raw_plan")
            for route in routes
            if route.get("workflow_id") == first_id
            and isinstance(route.get("raw_plan"), str)
            and route.get("raw_plan", "").strip()
        ),
        None,
    )
    if not isinstance(raw, str):
        raise v1.OutcomeCorpusError("legacy routes have no raw plan")
    parsed = parse_legacy_three_list_workflow(raw)
    synthetic_routes = [{"workflow_id": first_id, "raw_plan": json.dumps(parsed)}]
    return v1.canonicalize_initial_workflow(synthetic_routes, binding)


def recover_legacy_rows(
    *,
    root: Path,
    result_paths: Sequence[Path],
    split_salt: str = v1.DEFAULT_SPLIT_SALT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pool_registry = v1.load_capability_registry(root)
    split_registry = v1.load_split_registry(root)
    bindings = {binding.registry_key: binding for binding in pool_registry.values()}
    forbidden = v1.forbidden_identity_terms(bindings.values())
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []

    for result_path in sorted(result_paths):
        result_path = result_path.resolve()
        route_path = result_path.parent / "agent/fugu_routes.jsonl"
        reasons: list[str] = []
        try:
            result = v1._read_json(result_path)
        except v1.OutcomeCorpusError as exc:
            rejections.append(
                {"result_path": result_path.relative_to(root).as_posix(), "reason": str(exc)}
            )
            continue
        task_name = result.get("task_name")
        task_name = task_name if isinstance(task_name, str) else ""
        reward = v1._binary_reward(result)
        metadata = v1._result_metadata(result)
        pool_id = metadata.get("pool_id")
        binding = pool_registry.get(pool_id) if isinstance(pool_id, str) else None
        routes = metadata.get("fugu_routes")
        task_checksum = result.get("task_checksum")
        if task_name.startswith("terminal-bench/"):
            reasons.append("terminalbench_excluded")
        if reward is None:
            reasons.append("missing_or_nonbinary_reward")
        if result.get("exception_info") is not None:
            reasons.append("result_exception")
        reasons.extend(v1._operational_invalidity(metadata))
        if binding is None:
            reasons.append("unknown_pool_binding")
        elif not v1._pool_matches_result(binding, metadata):
            reasons.append("pool_binding_mismatch")
        if not isinstance(routes, list) or not routes:
            reasons.append("missing_routes")
        if not isinstance(task_checksum, str) or not task_checksum:
            reasons.append("missing_task_checksum")
        try:
            task_text, instruction_path = v1._task_prompt(result, root)
        except v1.OutcomeCorpusError as exc:
            reasons.append(str(exc))
            task_text = ""
            instruction_path = None
        try:
            workflow = (
                canonicalize_legacy_initial_workflow(routes, binding)
                if isinstance(routes, list) and binding is not None
                else None
            )
        except v1.OutcomeCorpusError as exc:
            reasons.append(f"legacy_recovery_failed:{exc}")
            workflow = None

        if reasons:
            rejections.append(
                {
                    "task_name": task_name,
                    "result_path": result_path.relative_to(root).as_posix(),
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        assert reward is not None
        assert binding is not None
        assert workflow is not None
        assert instruction_path is not None
        assert isinstance(task_checksum, str)
        split_assignment = v1.assign_split(task_name, split_registry, salt=split_salt)
        record_seed = (
            result_path.relative_to(root).as_posix()
            + "\0"
            + v1.sha256_file(result_path)
            + "\0"
            + v1.sha256_file(route_path)
        )
        record_id = v1.sha256_bytes(record_seed.encode("utf-8"))
        task_key = v1.sha256_bytes(task_name.encode("utf-8"))
        learned_input = {
            "task": task_text,
            "task_family": v1._task_family(task_name, split_registry),
            "capability_profiles": workflow["capability_profiles"],
            "workflow": {
                "steps": workflow["steps"],
                "topology_signature": workflow["topology_signature"],
                "workflow_signature": workflow["workflow_signature"],
            },
            "runtime_constraints": {
                "worker_call_limit": v1._as_int(
                    metadata.get("fair_position_call_budget"), default=120
                )
                or 120,
                "task_wall_time_seconds": metadata.get("terminal_task_budget_s"),
                "max_agent_turns": metadata.get("max_agent_turns"),
                "live_control_available": isinstance(
                    metadata.get("live_control_decisions"), list
                ),
            },
        }
        leaks = v1.find_identity_leaks(learned_input, forbidden)
        if leaks:
            rejections.append(
                {
                    "task_name": task_name,
                    "result_path": result_path.relative_to(root).as_posix(),
                    "reasons": ["model_identity_leak:" + ",".join(leaks)],
                }
            )
            continue
        rows.append(
            {
                "version": RECOVERY_VERSION,
                "record_id": record_id,
                "task_key": task_key,
                "task_checksum": task_checksum,
                "split": split_assignment.split,
                "input": learned_input,
                "target": {
                    "verified_success": int(reward),
                    "worker_calls": v1._as_int(
                        metadata.get("paid_worker_call_attempts")
                    ),
                },
            }
        )
        provenance.append(
            {
                "record_id": record_id,
                "task_name": task_name,
                "task_checksum": task_checksum,
                "split": split_assignment.split,
                "split_sources": list(split_assignment.sources),
                "split_source_conflict_resolved_to_holdout": split_assignment.conflict,
                "pool_registry_key": binding.registry_key,
                "pool_id": pool_id,
                "pool_fingerprint": metadata.get("pool_fingerprint"),
                "worker_models": metadata.get("worker_models"),
                "runtime_revision": metadata.get("runtime_revision"),
                "result_path": result_path.relative_to(root).as_posix(),
                "result_sha256": v1.sha256_file(result_path),
                "route_path": route_path.relative_to(root).as_posix(),
                "route_sha256": v1.sha256_file(route_path),
                "instruction_path": instruction_path.relative_to(root).as_posix(),
                "instruction_sha256": v1.sha256_file(instruction_path),
                "binding_path": binding.source_path.relative_to(root).as_posix(),
                "binding_sha256": v1.sha256_file(binding.source_path),
                "legacy_parser_recovery": True,
            }
        )
    rows.sort(key=lambda row: row["record_id"])
    provenance.sort(key=lambda row: row["record_id"])
    rejections.sort(key=lambda row: row.get("result_path", ""))
    return rows, provenance, rejections

