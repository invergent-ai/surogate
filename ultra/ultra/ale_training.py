"""Admit scored ALE train episodes with conservative coordination credit."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from .behavior_likelihood import (
    full_vocabulary_behavior_likelihood_contract,
    has_full_vocabulary_behavior_likelihood_contract,
)


class AleTrainingAdmissionError(ValueError):
    """A run is not clean, train-split outcome evidence."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AleTrainingAdmissionError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AleTrainingAdmissionError(f"{path} must contain one JSON object")
    return value


def _inventory_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AleTrainingAdmissionError(f"invalid inventory JSON at line {line_number}: {exc}") from exc
            task_id = row.get("task_id") if isinstance(row, dict) else None
            if not isinstance(task_id, str) or task_id in rows:
                raise AleTrainingAdmissionError("inventory task IDs must be unique strings")
            rows[task_id] = row
    return rows


def _context_path(run_dir: Path) -> Path:
    matches = sorted((run_dir / "origin_log").glob("*/fugu_context.json"))
    if len(matches) != 1:
        raise AleTrainingAdmissionError(f"expected one Fugu context under origin_log, found {len(matches)}")
    return matches[0]


def _training_token_data(trace: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "prompt_token_ids",
        "completion_token_ids",
        "completion_logprobs",
        "temperature",
        "seed",
        "behavior_likelihood_contract",
    )
    present = [field in trace for field in fields]
    if not any(present):
        raise AleTrainingAdmissionError(
            "conductor exact token evidence and behavior-likelihood contract are missing"
        )
    if not all(present):
        raise AleTrainingAdmissionError("conductor token evidence is incomplete")
    prompt_ids = trace["prompt_token_ids"]
    completion_ids = trace["completion_token_ids"]
    logprobs = trace["completion_logprobs"]
    temperature = trace["temperature"]
    seed = trace["seed"]
    likelihood_contract = trace["behavior_likelihood_contract"]
    valid_ids = lambda values: (  # noqa: E731 - compact validation predicate
        isinstance(values, list)
        and all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values)
    )
    if not valid_ids(prompt_ids) or not valid_ids(completion_ids) or not completion_ids:
        raise AleTrainingAdmissionError("conductor token IDs are invalid")
    if (
        not isinstance(logprobs, list)
        or len(logprobs) != len(completion_ids)
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in logprobs
        )
    ):
        raise AleTrainingAdmissionError("conductor token log-probabilities are invalid")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or float(temperature) != 1.0
    ):
        raise AleTrainingAdmissionError(
            "conductor training temperature must be exactly 1.0"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AleTrainingAdmissionError("conductor sampling seed is invalid")
    if not has_full_vocabulary_behavior_likelihood_contract(likelihood_contract):
        raise AleTrainingAdmissionError(
            "conductor behavior-likelihood contract is not full-vocabulary training v2"
        )
    return {
        "prompt_token_ids": list(prompt_ids),
        "completion_token_ids": list(completion_ids),
        "completion_logprobs": [float(value) for value in logprobs],
        "temperature": float(temperature),
        "seed": seed,
        "behavior_likelihood_contract": (
            full_vocabulary_behavior_likelihood_contract()
        ),
    }


_CONTROL_ACTIONS = frozenset({"continue", "handoff", "replan", "complete"})


def is_correction_attempt(decision: dict[str, Any]) -> bool:
    """Return whether exact policy tokens came from a rejected-action correction."""

    messages = decision.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    last_message = messages[-1]
    return (
        len(messages) > 2
        and isinstance(last_message, dict)
        and last_message.get("role") == "user"
        and isinstance(last_message.get("content"), str)
        and last_message["content"].startswith("Correction attempt ")
    )


def _decision_action(decision: dict[str, Any]) -> str:
    value = decision.get("action")
    if isinstance(value, dict):
        value = value.get("action")
    if value is None:
        response = decision.get("response")
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as exc:
                raise AleTrainingAdmissionError("conductor decision response is not valid JSON") from exc
            if isinstance(parsed, dict):
                value = parsed.get("action")
    if value not in _CONTROL_ACTIONS:
        raise AleTrainingAdmissionError("conductor decision has an invalid action")
    return str(value)


def _structural_route_signature(route: dict[str, Any]) -> tuple[Any, ...]:
    """Return only realized, model-agnostic coordination state.

    Worker identity, subtask wording, planned-but-unexecuted steps, and the
    declared workflow length are deliberately absent.  Those fields can vary
    while the conductor still executes the same coordination path.
    """

    route_source = route.get("route_source")
    if route_source is not None and not isinstance(route_source, str):
        raise AleTrainingAdmissionError("worker route source is invalid")
    step_index = route.get("workflow_step_index")
    if step_index is not None and (isinstance(step_index, bool) or not isinstance(step_index, int)):
        raise AleTrainingAdmissionError("worker route step index is invalid")
    access = route.get("workflow_access")
    if access is None:
        access = []
    if not isinstance(access, list) or any(
        isinstance(position, bool) or not isinstance(position, int) for position in access
    ):
        raise AleTrainingAdmissionError("worker route access topology is invalid")
    return route_source, step_index, tuple(access)


def _realized_coordination_path(episode: dict[str, Any]) -> list[tuple[Any, ...]]:
    decisions = episode.get("decisions")
    routes = episode.get("routes")
    if not isinstance(decisions, list) or not decisions:
        raise AleTrainingAdmissionError("group episode contains no conductor decisions")
    if not isinstance(routes, list):
        raise AleTrainingAdmissionError("group episode worker routes are invalid")

    routes_by_runtime_turn: dict[int, tuple[int, dict[str, Any]]] = {}
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise AleTrainingAdmissionError("group episode contains an invalid worker route")
        runtime_turn = route.get("runtime_turn")
        if runtime_turn is None:
            continue
        if (
            isinstance(runtime_turn, bool)
            or not isinstance(runtime_turn, int)
            or runtime_turn in routes_by_runtime_turn
        ):
            raise AleTrainingAdmissionError("worker route runtime turn is invalid")
        routes_by_runtime_turn[runtime_turn] = (index, route)

    path: list[tuple[Any, ...]] = []
    used_route_indices: set[int] = set()
    positional_route_index = 0
    for decision in decisions:
        if not isinstance(decision, dict):
            raise AleTrainingAdmissionError("group episode contains an invalid decision")
        action = _decision_action(decision)
        target = None
        action_value = decision.get("action")
        if action == "handoff" and isinstance(action_value, dict):
            target = action_value.get("target_position_id")
        if action == "handoff" and target is None:
            response = decision.get("response")
            if isinstance(response, str):
                try:
                    parsed = json.loads(response)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    target = parsed.get("target_position_id")
        if target is not None and (isinstance(target, bool) or not isinstance(target, int)):
            raise AleTrainingAdmissionError("conductor handoff target is invalid")

        if action == "complete":
            path.append((action, target, "task_complete", None, ()))
            continue

        route_index: int | None = None
        decision_number = decision.get("decision")
        if (
            isinstance(decision_number, int)
            and not isinstance(decision_number, bool)
            and decision_number in routes_by_runtime_turn
        ):
            route_index = routes_by_runtime_turn[decision_number][0]
        else:
            while positional_route_index in used_route_indices:
                positional_route_index += 1
            if positional_route_index < len(routes):
                route_index = positional_route_index
                positional_route_index += 1
        if route_index is None or route_index in used_route_indices:
            raise AleTrainingAdmissionError(
                "realized coordination divergence cannot be credited: decision has no worker route"
            )
        used_route_indices.add(route_index)
        path.append(
            (
                action,
                target,
                *_structural_route_signature(routes[route_index]),
            )
        )
    return path


def _first_executed_divergence(paths: list[list[tuple[Any, ...]]]) -> int:
    common_length = min(len(path) for path in paths)
    for decision_index in range(common_length):
        if len({path[decision_index] for path in paths}) > 1:
            return decision_index
    if len({len(path) for path in paths}) > 1:
        raise AleTrainingAdmissionError("realized coordination divergence cannot be credited across every arm")
    raise AleTrainingAdmissionError("group has identical realized coordination paths")


def admit_ale_training_episode(
    *,
    run_dir: Path,
    inventory_path: Path,
    source_commit: str,
    expected_runtime_revision: str,
    expected_provider_base: str = "https://yunwu.ai/v1",
    paid_call_limit: int = 120,
    expected_behavior_policy_revision: str | None = None,
    require_fail_closed_provider_errors: bool = False,
) -> dict[str, Any]:
    """Return one identity-free, episode-reward record from a clean train run.

    Whole-task reward remains attached to the episode. It is intentionally not
    copied onto every action; GRPO grouping or causal contrasts must assign
    decision credit later.
    """
    run_dir = run_dir.resolve()
    run_path = run_dir / "run.json"
    eval_path = run_dir / "eval_result.json"
    trajectory_path = run_dir / "trajectory.json"
    context_path = _context_path(run_dir)
    run = _read_json(run_path)
    evaluation = _read_json(eval_path)
    trajectory = _read_json(trajectory_path)
    context = _read_json(context_path)

    run_task = run.get("task") or {}
    run_task_path = run_task.get("path") if isinstance(run_task, dict) else None
    task_id = str(run_task_path).removeprefix("tasks/") if run_task_path else trajectory.get("task_path")
    row = _inventory_rows(inventory_path).get(str(task_id))
    if row is None:
        raise AleTrainingAdmissionError(f"task {task_id!r} is absent from the train inventory")
    if row.get("split") != "train":
        raise AleTrainingAdmissionError(f"task {task_id!r} belongs to {row.get('split')!r}, not train")
    if run.get("status") != "completed" or evaluation.get("eval_status") != "success":
        raise AleTrainingAdmissionError("run and evaluation must both complete successfully")
    score = evaluation.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise AleTrainingAdmissionError("evaluation score must be numeric")
    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise AleTrainingAdmissionError("evaluation score must be within [0, 1]")
    final_metrics = trajectory.get("final_metrics") or {}
    if final_metrics.get("reward") != score:
        raise AleTrainingAdmissionError("trajectory reward differs from eval_result score")

    metadata = context.get("metadata") or {}
    if metadata.get("runtime_revision") != expected_runtime_revision:
        raise AleTrainingAdmissionError("runtime revision differs from the registered collector")
    pool_id = metadata.get("pool_id")
    if not isinstance(pool_id, str) or not pool_id.strip():
        raise AleTrainingAdmissionError("semantic pool ID is missing from runtime metadata")
    pool_binding_revision = metadata.get("pool_binding_revision")
    if not isinstance(pool_binding_revision, str) or not pool_binding_revision.strip():
        raise AleTrainingAdmissionError(
            "semantic pool binding revision is missing from runtime metadata"
        )
    if str(metadata.get("worker_provider_base", "")).rstrip("/") != expected_provider_base:
        raise AleTrainingAdmissionError("worker provider is not the registered Yunwu endpoint")
    attempts = metadata.get("paid_worker_call_attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 0 <= attempts <= paid_call_limit:
        raise AleTrainingAdmissionError("paid-call count is missing or exceeds the ceiling")
    if metadata.get("provider_owner_retry_limit") != 0 or metadata.get("provider_owner_retries") != 0:
        raise AleTrainingAdmissionError("provider retries make the outcome inadmissible")
    if metadata.get("provider_failure_events"):
        raise AleTrainingAdmissionError("provider failure contaminated the whole-task outcome")
    if require_fail_closed_provider_errors and metadata.get("fail_closed_provider_errors") is not True:
        raise AleTrainingAdmissionError("collector did not attest fail-closed provider handling")
    if metadata.get("live_control_failures"):
        raise AleTrainingAdmissionError("live-control failure contaminated the trajectory")
    live_control_temperature = metadata.get("live_control_temperature")
    if (
        isinstance(live_control_temperature, bool)
        or not isinstance(live_control_temperature, (int, float))
        or float(live_control_temperature) != 1.0
    ):
        raise AleTrainingAdmissionError(
            "ALE training episode temperature must be exactly 1.0"
        )
    behavior_policy_revision = metadata.get("typed_conductor_policy_revision")
    if expected_behavior_policy_revision is not None:
        if behavior_policy_revision != expected_behavior_policy_revision:
            raise AleTrainingAdmissionError("behavior-policy revision differs from the registered collector")

    traces = metadata.get("live_control_model_traces")
    decisions = metadata.get("live_control_decisions")
    routes = metadata.get("fugu_routes")
    if not all(isinstance(value, list) and value for value in (traces, decisions, routes)):
        raise AleTrainingAdmissionError("scored episode lacks conductor traces, actions, or worker routes")
    if metadata.get("live_control_records_token_data") is not True:
        raise AleTrainingAdmissionError(
            "ALE training episode was not collected with behavior-policy token evidence"
        )

    admitted_decisions = []
    excluded_normalized_decisions = 0
    for decision in decisions:
        if decision.get("normalization") is not None:
            excluded_normalized_decisions += 1
            continue
        trace_index = decision.get("controller_trace_index")
        if isinstance(trace_index, bool) or not isinstance(trace_index, int):
            raise AleTrainingAdmissionError("accepted action lacks an exact controller trace index")
        if not 0 <= trace_index < len(traces):
            raise AleTrainingAdmissionError("controller trace index is out of range")
        trace = traces[trace_index]
        if not isinstance(trace.get("messages"), list) or not isinstance(trace.get("response"), str):
            raise AleTrainingAdmissionError("controller trace is missing exact messages or response")
        admitted_decisions.append(
            {
                "decision": decision.get("decision"),
                "messages": trace["messages"],
                "response": trace["response"],
                "finish_reason": trace.get("finish_reason"),
                "prompt_tokens": trace.get("prompt_tokens"),
                "action": {
                    key: decision.get(key)
                    for key in (
                        "action",
                        "reason",
                        "target_position_id",
                        "steps",
                        "interrupts_unfinished_position",
                    )
                },
                **_training_token_data(trace),
            }
        )
    if not admitted_decisions:
        raise AleTrainingAdmissionError(
            "scored episode contains no exact unnormalized conductor decisions"
        )

    admitted_routes = [
        {
            key: route.get(key)
            for key in (
                "turn",
                "runtime_turn",
                "worker_id",
                "route_source",
                "workflow_id",
                "workflow_step_index",
                "workflow_step_count",
                "workflow_access",
                "agent_private_turn",
                "subtask",
                "terminal_ready",
                "paid_call_attempt",
                "reported_progress",
                "reported_artifacts",
                "material_progress",
            )
        }
        for route in routes
    ]
    return {
        "schema_version": 3,
        "record_type": "ale_train_episode_outcome",
        "task_id": task_id,
        "task_family": row.get("family"),
        "split": "train",
        "source_commit": source_commit,
        "pool_id": pool_id.strip(),
        "pool_binding_revision": pool_binding_revision.strip(),
        "runtime_revision": expected_runtime_revision,
        "whole_task_reward": score,
        "credit_assignment": "episode_only_pending_grpo_group_or_causal_contrast",
        "paid_worker_call_attempts": attempts,
        "excluded_normalized_decisions": excluded_normalized_decisions,
        "fail_closed_provider_errors": metadata.get("fail_closed_provider_errors", False),
        "behavior_policy": {
            "conductor_model": metadata.get("typed_conductor_model"),
            "conductor_url": metadata.get("typed_conductor_url"),
            "revision": behavior_policy_revision,
            "temperature": metadata.get("live_control_temperature"),
            "seed": metadata.get("live_control_seed"),
            "records_token_data": metadata.get("live_control_records_token_data", False),
            "behavior_likelihood_contract": (
                full_vocabulary_behavior_likelihood_contract()
            ),
        },
        "decisions": admitted_decisions,
        "routes": admitted_routes,
        "source_paths": {
            "run": str(run_path),
            "evaluation": str(eval_path),
            "trajectory": str(trajectory_path),
            "context": str(context_path),
            "inventory": str(inventory_path.resolve()),
        },
    }


def admit_ale_training_group(
    *,
    episode_paths: list[Path],
    group_id: str,
    expected_group_size: int = 2,
) -> dict[str, Any]:
    """Admit one same-task on-policy group with real outcome variation."""
    if not group_id.strip():
        raise AleTrainingAdmissionError("training group ID must be non-empty")
    if expected_group_size < 2:
        raise AleTrainingAdmissionError("GRPO group size must be at least two")
    if len(episode_paths) != expected_group_size:
        raise AleTrainingAdmissionError(f"expected {expected_group_size} episodes, found {len(episode_paths)}")
    episodes = [_read_json(path.resolve()) for path in episode_paths]
    task_ids = {episode.get("task_id") for episode in episodes}
    families = {episode.get("task_family") for episode in episodes}
    source_commits = {episode.get("source_commit") for episode in episodes}
    runtimes = {episode.get("runtime_revision") for episode in episodes}
    pool_ids = {episode.get("pool_id") for episode in episodes}
    pool_binding_revisions = {
        episode.get("pool_binding_revision") for episode in episodes
    }
    resolved_episode_paths = [path.resolve() for path in episode_paths]
    if any(
        episode.get("schema_version") != 3
        or episode.get("record_type") != "ale_train_episode_outcome"
        for episode in episodes
    ):
        raise AleTrainingAdmissionError("group contains a non-ALE training episode")
    if any(episode.get("split") != "train" for episode in episodes):
        raise AleTrainingAdmissionError("group contains held-out ALE evidence")
    for label, values in (
        ("task", task_ids),
        ("family", families),
        ("source commit", source_commits),
        ("runtime", runtimes),
        ("pool ID", pool_ids),
        ("pool binding revision", pool_binding_revisions),
    ):
        if len(values) != 1 or None in values:
            raise AleTrainingAdmissionError(f"group {label} differs across episodes")
    if len(set(resolved_episode_paths)) != len(resolved_episode_paths):
        raise AleTrainingAdmissionError("group episode paths must be distinct")

    rewards: list[float] = []
    behavior_models: set[Any] = set()
    behavior_revisions: set[Any] = set()
    behavior_seeds: set[int] = set()
    temperatures: set[float] = set()
    for episode in episodes:
        reward = episode.get("whole_task_reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise AleTrainingAdmissionError("group reward is invalid")
        rewards.append(float(reward))
        behavior = episode.get("behavior_policy") or {}
        if behavior.get("records_token_data") is not True:
            raise AleTrainingAdmissionError("group episode was not collected with behavior-policy token evidence")
        if not has_full_vocabulary_behavior_likelihood_contract(
            behavior.get("behavior_likelihood_contract")
        ):
            raise AleTrainingAdmissionError(
                "group episode lacks the full-vocabulary behavior-likelihood contract"
            )
        behavior_models.add(behavior.get("conductor_model"))
        behavior_revisions.add(behavior.get("revision"))
        behavior_seed = behavior.get("seed")
        if isinstance(behavior_seed, bool) or not isinstance(behavior_seed, int):
            raise AleTrainingAdmissionError("group behavior-policy seed is invalid")
        if behavior_seed in behavior_seeds:
            raise AleTrainingAdmissionError("group behavior-policy seeds must be distinct")
        behavior_seeds.add(behavior_seed)
        temperature = behavior.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or float(temperature) != 1.0
        ):
            raise AleTrainingAdmissionError(
                "GRPO episode must use conductor sampling temperature exactly 1.0"
            )
        temperatures.add(float(temperature))
        decisions = episode.get("decisions")
        if not isinstance(decisions, list) or not decisions:
            raise AleTrainingAdmissionError("group episode contains no conductor decisions")
        for decision in decisions:
            token_data = _training_token_data(decision)
            if float(token_data["temperature"]) != float(temperature):
                raise AleTrainingAdmissionError("decision temperature differs from its behavior policy")
            if token_data["seed"] != behavior_seed:
                raise AleTrainingAdmissionError("decision seed differs from its behavior policy")
    if len(behavior_models) != 1 or None in behavior_models:
        raise AleTrainingAdmissionError("group behavior model differs across episodes")
    if len(behavior_revisions) != 1 or None in behavior_revisions:
        raise AleTrainingAdmissionError("group behavior-policy revision differs or is unattested")
    if len(temperatures) != 1:
        raise AleTrainingAdmissionError("group sampling temperature differs across episodes")
    if len(set(rewards)) < 2:
        raise AleTrainingAdmissionError("group has zero reward variation")

    realized_paths = [_realized_coordination_path(episode) for episode in episodes]
    credited_decision_index = _first_executed_divergence(realized_paths)
    if any(
        is_correction_attempt(episode["decisions"][credited_decision_index])
        for episode in episodes
    ):
        raise AleTrainingAdmissionError(
            "realized coordination divergence cannot credit a correction attempt"
        )

    mean = statistics.fmean(rewards)
    std = statistics.stdev(rewards)
    advantages = [(reward - mean) / (std + 1.0e-4) for reward in rewards]
    grouped_episodes = [
        {
            "episode_path": str(path.resolve()),
            "whole_task_reward": reward,
            "group_advantage": advantage,
            "decision_count": len(episode["decisions"]),
            "credited_decision_index": credited_decision_index,
            "credited_coordination_signature": list(realized_paths[index][credited_decision_index]),
            "paid_worker_call_attempts": episode["paid_worker_call_attempts"],
        }
        for index, (path, episode, reward, advantage) in enumerate(
            zip(episode_paths, episodes, rewards, advantages, strict=True)
        )
    ]
    record = {
        "schema_version": 4,
        "record_type": "ale_train_grpo_group",
        "group_id": group_id,
        "task_id": next(iter(task_ids)),
        "task_family": next(iter(families)),
        "source_commit": next(iter(source_commits)),
        "runtime_revision": next(iter(runtimes)),
        "pool_id": next(iter(pool_ids)),
        "pool_binding_revision": next(iter(pool_binding_revisions)),
        "behavior_model": next(iter(behavior_models)),
        "behavior_policy_revision": next(iter(behavior_revisions)),
        "behavior_likelihood_contract": (
            full_vocabulary_behavior_likelihood_contract()
        ),
        "sampling_temperature": next(iter(temperatures)),
        "advantage_method": "sample_std_normalized_eps_1e-4",
        "credit_assignment": {
            "mode": "first_executed_coordination_divergence",
            "decision_index": credited_decision_index,
            "policy_attempts": "initial_only",
        },
        "reward_mean": mean,
        "reward_std": std,
        "episodes": grouped_episodes,
    }
    if "worker_model" in json.dumps(record, ensure_ascii=True):
        raise AleTrainingAdmissionError("worker identity leaked into the GRPO group")
    return record
