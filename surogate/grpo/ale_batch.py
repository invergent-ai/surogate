"""Materialize one causally admitted ALE outcome group as one GRPO update.

The paid conductor tokens are never retokenized.  Their sampled token IDs and
behavior-policy log-probabilities are copied exactly from the admitted episode
records.  The complete 27B transfer replay is appended to the same
``TrainingBatch`` so one orchestrator group maps to one replay-anchored native
optimizer update under ``SinglePacker``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import msgspec
from ultra.ale_training import is_correction_attempt
from ultra.behavior_likelihood import (
    FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION,
    full_vocabulary_behavior_likelihood_contract,
    has_full_vocabulary_behavior_likelihood_contract,
)

from surogate.grpo.batch import prepare_sample
from surogate.grpo.transport import TrainingBatch, TrainingSample

# The conductor serves an 8K context, but the current exact policy and replay
# rows fit the proven 2,816-token native optimizer window.
PROVEN_SEQUENCE_LEN = 2_816
CONTROL_ACTIONS = frozenset({"continue", "handoff", "replan", "complete"})
REPLAY_REFERENCE_MODE = "ce_only_no_behavior_ratio_or_kl"
TRAIN_RETENTION_REPLAY_VERSION = "fugu_27b_train_retention_replay_v1"
ACTION_BALANCED_RETENTION_REPLAY_VERSION = "fugu_27b_action_balanced_retention_replay_v2"
ACTION_BALANCED_RETENTION_SAMPLES = 76
ACTION_BALANCED_RETENTION_SELECTED_TOKENS = 17_760
TRAIN_RETENTION_REPLAY_VERSIONS = frozenset({TRAIN_RETENTION_REPLAY_VERSION, ACTION_BALANCED_RETENTION_REPLAY_VERSION})


class AleBatchError(ValueError):
    """An admitted-group artifact cannot safely enter the 27B optimizer."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AleBatchError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AleBatchError(f"{path} must contain one JSON object")
    return value


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _selected_replay_weight_sum(samples: list[TrainingSample]) -> float:
    total = 0.0
    for sample in samples:
        replay_mask = sample.replay_mask or []
        replay_weights = sample.replay_weights or [1.0] * len(replay_mask)
        total += sum(float(weight) for replay, weight in zip(replay_mask, replay_weights) if replay)
    return total


def _load_replay_samples(path: Path, *, expected_samples: int) -> list[TrainingSample]:
    """Load replay directly and validate the fields used by the optimizer."""

    if not path.is_file():
        raise AleBatchError(f"replay batch does not exist: {path}")
    try:
        batch = msgspec.msgpack.decode(path.read_bytes(), type=TrainingBatch)
    except Exception as exc:
        raise AleBatchError(f"replay batch cannot be decoded: {exc}") from exc
    if len(batch.examples) != expected_samples:
        raise AleBatchError(f"replay batch contains {len(batch.examples)} samples, expected {expected_samples}")
    for index, sample in enumerate(batch.examples):
        total_tokens = len(sample.prompt_ids) + len(sample.completion_ids)
        loss_mask = [*sample.prompt_mask, *sample.completion_mask]
        replay_mask = sample.replay_mask
        if (
            len(sample.prompt_mask) != len(sample.prompt_ids)
            or len(sample.completion_mask) != len(sample.completion_ids)
            or replay_mask is None
            or len(replay_mask) != total_tokens
            or not any(replay_mask)
            or any(replay_mask[: len(sample.prompt_ids)])
            or any(replay and not trainable for replay, trainable in zip(replay_mask, loss_mask, strict=True))
        ):
            raise AleBatchError(f"replay sample {index} has an invalid training mask")
        if sample.advantage not in {None, 0} or sample.reward not in {None, 0}:
            raise AleBatchError(f"replay sample {index} carries outcome credit")
        if sample.replay_weights is not None:
            if len(sample.replay_weights) != total_tokens:
                raise AleBatchError(f"replay sample {index} has invalid weights")
            if any(
                not _finite_number(weight) or (replay and float(weight) <= 0.0) or (not replay and float(weight) != 1.0)
                for replay, weight in zip(replay_mask, sample.replay_weights, strict=True)
            ):
                raise AleBatchError(f"replay sample {index} has invalid weights")
    return batch.examples


def _exact_policy_sample(
    decision: dict[str, Any],
    *,
    advantage: float,
    reward: float,
    sampling_temperature: float,
    sequence_len: int,
    completion_advantage_mask: list[bool] | None = None,
) -> TrainingSample:
    prompt_ids = decision.get("prompt_token_ids")
    completion_ids = decision.get("completion_token_ids")
    completion_logprobs = decision.get("completion_logprobs")
    temperature = decision.get("temperature")
    if not has_full_vocabulary_behavior_likelihood_contract(
        decision.get("behavior_likelihood_contract")
    ):
        raise AleBatchError(
            "policy decision lacks the full-vocabulary behavior-likelihood contract"
        )
    if not (
        isinstance(prompt_ids, list)
        and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in prompt_ids)
    ):
        raise AleBatchError("policy decision has invalid exact prompt token IDs")
    if not (
        isinstance(completion_ids, list)
        and completion_ids
        and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in completion_ids)
    ):
        raise AleBatchError("policy decision has invalid exact completion token IDs")
    if not (
        isinstance(completion_logprobs, list)
        and len(completion_logprobs) == len(completion_ids)
        and all(_finite_number(value) for value in completion_logprobs)
    ):
        raise AleBatchError("policy decision has invalid exact completion log-probabilities")
    if (
        not _finite_number(temperature)
        or float(temperature) != 1.0
        or sampling_temperature != 1.0
    ):
        raise AleBatchError("policy decision temperature differs from the group")
    total_tokens = len(prompt_ids) + len(completion_ids)
    if total_tokens > sequence_len:
        raise AleBatchError(f"policy decision has {total_tokens} tokens, exceeding {sequence_len}")
    if decision.get("finish_reason") == "length":
        raise AleBatchError("truncated conductor action cannot enter training")
    if completion_advantage_mask is None:
        exact_advantage_mask = None
    elif (
        not isinstance(completion_advantage_mask, list)
        or len(completion_advantage_mask) != len(completion_ids)
        or any(
            not isinstance(selected, bool)
            for selected in completion_advantage_mask
        )
        or not any(completion_advantage_mask)
    ):
        raise AleBatchError(
            "policy decision has an invalid exact completion advantage mask"
        )
    else:
        exact_advantage_mask = [
            *([False] * len(prompt_ids)),
            *completion_advantage_mask,
        ]
    sample = TrainingSample(
        prompt_ids=list(prompt_ids),
        prompt_mask=[False] * len(prompt_ids),
        completion_ids=list(completion_ids),
        completion_mask=[True] * len(completion_ids),
        completion_logprobs=[float(value) for value in completion_logprobs],
        completion_temperatures=[float(temperature)] * len(completion_ids),
        advantage=advantage,
        reward=reward,
        replay_mask=[False] * total_tokens,
        advantage_mask=exact_advantage_mask,
    )
    prepared = prepare_sample(sample, sequence_len)
    if len(prepared.input_ids) != total_tokens:
        raise AleBatchError("policy sample was truncated during native preparation")
    return sample


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
                raise AleBatchError("policy decision response is not valid JSON") from exc
            if isinstance(parsed, dict):
                value = parsed.get("action")
    if value not in CONTROL_ACTIONS:
        raise AleBatchError("policy decision has an invalid control action")
    return str(value)


def _validate_train_retention_report(
    *,
    report_path: Path,
    replay_path: Path,
    replay_samples: int,
) -> dict[str, Any]:
    if not report_path.is_file():
        raise AleBatchError("train-only retention report is absent")
    report = _read_json(report_path)
    if (
        report.get("version") not in TRAIN_RETENTION_REPLAY_VERSIONS
        or report.get("verdict")
        not in {
            "TRAIN_ONLY_TYPED_CE_RETENTION_REPLAY_READY",
            "TRAIN_ONLY_ACTION_TOKEN_BALANCED_CE_RETENTION_REPLAY_READY",
        }
        or report.get("usage") != "auxiliary_anti_forgetting_replay_only"
        or report.get("reference_mode") != REPLAY_REFERENCE_MODE
        or report.get("typed_contract") != "unified_capability_action_v2"
        or report.get("paid_calls") != 0
        or report.get("optimizer_steps") != 0
    ):
        raise AleBatchError("train-only retention report has an invalid contract")
    counts = report.get("counts") or {}
    source = report.get("source") or {}
    orchestrator = report.get("orchestrator_replay") or {}
    if (
        counts.get("samples") != replay_samples
        or not isinstance(counts.get("train_tasks"), int)
        or counts.get("train_tasks", 0) <= 0
        or not isinstance(counts.get("validation_tasks_excluded"), int)
        or counts.get("validation_tasks_excluded", 0) <= 0
        or counts.get("selected_completion_tokens", 0) <= 0
        or Path(str(orchestrator.get("path", ""))).resolve() != replay_path
        or orchestrator.get("samples_per_step") != replay_samples
    ):
        raise AleBatchError("train-only retention report does not bind its replay")
    if report.get("version") == ACTION_BALANCED_RETENTION_REPLAY_VERSION:
        weighting = report.get("weighting") or {}
        if (
            weighting.get("mode") != "equal_action_completion_token_mass_v1"
            or weighting.get("actions") != sorted(CONTROL_ACTIONS)
            or not _finite_number(weighting.get("target_weighted_tokens_per_action"))
            or not _finite_number(weighting.get("total_weighted_completion_tokens"))
            or not math.isclose(
                float(weighting["total_weighted_completion_tokens"]),
                float(counts["selected_completion_tokens"]),
                rel_tol=1e-6,
                abs_tol=1e-3,
            )
        ):
            raise AleBatchError("action-balanced retention weighting contract changed")
    for path_key in (
        "typed_rows",
        "train",
        "validation_exclusion_inventory",
        "pool_binding",
    ):
        source_path = Path(str(source.get(path_key, ""))).resolve()
        if not source_path.is_file():
            raise AleBatchError(f"train-only retention source is absent: {path_key}")
    return report


def materialize_ale_grpo_update(
    *,
    output_dir: Path,
    expected_behavior_policy_revision: str,
    replay_path: Path,
    group_path: Path | None = None,
    group_paths: list[Path] | None = None,
    sequence_len: int = PROVEN_SEQUENCE_LEN,
    step: int = 0,
    adv_tau: float = 1.0,
    replay_tau: float = 0.05,
    kl_tau: float = 0.001,
    data_parallel_gpus: int = 6,
    credited_actions: tuple[str, ...] | None = None,
    train_retention_replay_path: Path | None = None,
    train_retention_replay_samples: int | None = None,
    train_retention_report_path: Path | None = None,
) -> dict[str, Any]:
    """Write one replay-anchored batch from one or more causal ALE groups."""
    if group_path is not None and group_paths is not None:
        raise AleBatchError("pass either group_path or group_paths, not both")
    selected_group_paths = group_paths if group_paths is not None else ([group_path] if group_path is not None else [])
    if not selected_group_paths:
        raise AleBatchError("at least one admitted ALE group is required")
    resolved_group_paths = [path.resolve() for path in selected_group_paths]
    if len(set(resolved_group_paths)) != len(resolved_group_paths):
        raise AleBatchError("ALE group paths must be unique")
    output_dir = output_dir.resolve()
    replay_path = replay_path.resolve()
    revision = expected_behavior_policy_revision.strip()
    if not revision:
        raise AleBatchError("expected behavior-policy revision must be non-empty")
    if sequence_len != PROVEN_SEQUENCE_LEN:
        raise AleBatchError(f"ALE policy updates require sequence length {PROVEN_SEQUENCE_LEN}")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise AleBatchError("optimizer step must be a non-negative integer")
    if data_parallel_gpus != 6:
        raise AleBatchError("this training path requires six data-parallel GPUs")
    for label, value in (
        ("adv_tau", adv_tau),
        ("replay_tau", replay_tau),
        ("kl_tau", kl_tau),
    ):
        if not _finite_number(value) or float(value) <= 0.0:
            raise AleBatchError(f"{label} must be finite and non-zero")
    required_retention_values = (
        train_retention_replay_path,
        train_retention_replay_samples,
        train_retention_report_path,
    )
    if any(value is not None for value in required_retention_values) and not all(
        value is not None for value in required_retention_values
    ):
        raise AleBatchError("train-only retention replay requires path, count, and report")
    if train_retention_replay_samples is not None and (
        isinstance(train_retention_replay_samples, bool)
        or not isinstance(train_retention_replay_samples, int)
        or train_retention_replay_samples <= 0
    ):
        raise AleBatchError("train-only retention replay sample count must be positive")
    credited_action_set = CONTROL_ACTIONS
    if credited_actions is not None:
        credited_action_set = frozenset(credited_actions)
        if not credited_action_set:
            raise AleBatchError("credited_actions must not be empty")
        invalid_actions = credited_action_set - CONTROL_ACTIONS
        if invalid_actions:
            raise AleBatchError(f"credited_actions contains invalid actions: {sorted(invalid_actions)}")

    credit_assignment: dict[str, Any] = {
        "mode": "first_executed_coordination_divergence",
        "policy_attempts": "initial_only",
        "groups": [],
    }
    if credited_actions is not None:
        credit_assignment["action_allowlist"] = sorted(credited_action_set)

    policy_samples: list[TrainingSample] = []
    group_reports: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    batch_pool_identity: tuple[str, str] | None = None
    signed_credit = {
        action: {
            "negative_samples": 0,
            "negative_tokens": 0,
            "positive_samples": 0,
            "positive_tokens": 0,
        }
        for action in sorted(CONTROL_ACTIONS)
    }
    for resolved_group_path in resolved_group_paths:
        group = _read_json(resolved_group_path)
        if (
            group.get("schema_version") != 4
            or group.get("record_type") != "ale_train_grpo_group"
        ):
            raise AleBatchError("input is not an admitted ALE GRPO group")
        if not has_full_vocabulary_behavior_likelihood_contract(
            group.get("behavior_likelihood_contract")
        ):
            raise AleBatchError(
                "ALE group lacks the full-vocabulary behavior-likelihood contract"
            )
        pool_id = group.get("pool_id")
        pool_binding_revision = group.get("pool_binding_revision")
        if (
            not isinstance(pool_id, str)
            or not pool_id.strip()
            or not isinstance(pool_binding_revision, str)
            or not pool_binding_revision.strip()
        ):
            raise AleBatchError("ALE group lacks semantic pool identity")
        group_pool_identity = (pool_id.strip(), pool_binding_revision.strip())
        if batch_pool_identity is None:
            batch_pool_identity = group_pool_identity
        elif group_pool_identity != batch_pool_identity:
            raise AleBatchError("ALE groups must share one semantic pool identity")
        group_id = group.get("group_id")
        task_id = group.get("task_id")
        if not isinstance(group_id, str) or not group_id or group_id in seen_group_ids:
            raise AleBatchError("ALE group IDs must be unique non-empty strings")
        if not isinstance(task_id, str) or not task_id or task_id in seen_task_ids:
            raise AleBatchError("ALE group task IDs must be unique non-empty strings")
        seen_group_ids.add(group_id)
        seen_task_ids.add(task_id)
        group_credit = group.get("credit_assignment")
        if (
            not isinstance(group_credit, dict)
            or group_credit.get("mode")
            != "first_executed_coordination_divergence"
            or group_credit.get("policy_attempts") != "initial_only"
        ):
            raise AleBatchError("ALE group lacks causal coordination credit")
        group_credit_index = group_credit.get("decision_index")
        if isinstance(group_credit_index, bool) or not isinstance(group_credit_index, int) or group_credit_index < 0:
            raise AleBatchError("ALE group causal decision index is invalid")
        if group.get("behavior_policy_revision") != revision:
            raise AleBatchError("ALE groups must share the behavior-policy revision")
        if group.get("advantage_method") != "sample_std_normalized_eps_1e-4":
            raise AleBatchError("ALE group uses an unregistered advantage method")
        sampling_temperature = group.get("sampling_temperature")
        if (
            not _finite_number(sampling_temperature)
            or float(sampling_temperature) != 1.0
        ):
            raise AleBatchError(
                "ALE group sampling temperature must be exactly 1.0"
            )
        sampling_temperature = float(sampling_temperature)
        grouped_episodes = group.get("episodes")
        if not isinstance(grouped_episodes, list) or len(grouped_episodes) < 2:
            raise AleBatchError("ALE GRPO group must contain at least two episodes")

        decision_indices: list[int] = []
        rewards: list[float] = []
        episode_reports: list[dict[str, Any]] = []
        group_policy_start = len(policy_samples)
        for grouped in grouped_episodes:
            if not isinstance(grouped, dict):
                raise AleBatchError("ALE group episode entry is invalid")
            episode_path = Path(str(grouped.get("episode_path", ""))).resolve()
            episode = _read_json(episode_path)
            if episode.get("schema_version") != 3:
                raise AleBatchError("ALE group contains a non-schema-3 episode")
            for field in (
                "task_id",
                "task_family",
                "source_commit",
                "runtime_revision",
                "pool_id",
                "pool_binding_revision",
            ):
                if episode.get(field) != group.get(field):
                    raise AleBatchError(f"ALE episode {field} differs from its group")
            behavior = episode.get("behavior_policy") or {}
            if behavior.get("revision") != revision:
                raise AleBatchError("ALE episode behavior-policy revision is stale")
            if behavior.get("records_token_data") is not True:
                raise AleBatchError("ALE episode lacks exact behavior-policy token data")
            if not has_full_vocabulary_behavior_likelihood_contract(
                behavior.get("behavior_likelihood_contract")
            ):
                raise AleBatchError(
                    "ALE episode lacks the full-vocabulary behavior-likelihood contract"
                )
            reward = grouped.get("whole_task_reward")
            advantage = grouped.get("group_advantage")
            if not _finite_number(reward) or not 0.0 <= float(reward) <= 1.0:
                raise AleBatchError("ALE episode reward is invalid")
            if not _finite_number(advantage):
                raise AleBatchError("ALE episode advantage is invalid")
            reward = float(reward)
            advantage = float(advantage)
            if float(episode.get("whole_task_reward")) != reward:
                raise AleBatchError("ALE episode reward differs from group admission")
            decisions = episode.get("decisions")
            if not isinstance(decisions, list) or not decisions or len(decisions) != grouped.get("decision_count"):
                raise AleBatchError("ALE episode decision count differs from group admission")
            credited_decision_index = grouped.get("credited_decision_index")
            if (
                isinstance(credited_decision_index, bool)
                or not isinstance(credited_decision_index, int)
                or credited_decision_index != group_credit_index
                or not 0 <= credited_decision_index < len(decisions)
            ):
                raise AleBatchError("ALE episode causal decision cannot be credited")
            decision_indices.append(credited_decision_index)
            before = len(policy_samples)
            action_counts = {action: 0 for action in sorted(CONTROL_ACTIONS)}
            action_tokens = {action: 0 for action in sorted(CONTROL_ACTIONS)}
            credited_action: str | None = None
            credited_conductor_tokens = 0
            for decision_index, decision in enumerate(decisions):
                if not isinstance(decision, dict):
                    raise AleBatchError("ALE episode contains an invalid decision")
                action = _decision_action(decision)
                if (
                    decision_index == credited_decision_index
                    and is_correction_attempt(decision)
                ):
                    raise AleBatchError(
                        "ALE causal decision cannot credit a correction attempt"
                    )
                receives_credit = decision_index == credited_decision_index and action in credited_action_set
                effective_advantage = advantage if receives_credit else 0.0
                sample = _exact_policy_sample(
                    decision,
                    advantage=effective_advantage,
                    reward=reward,
                    sampling_temperature=sampling_temperature,
                    sequence_len=sequence_len,
                )
                policy_samples.append(sample)
                completion_tokens = len(sample.completion_ids)
                action_counts[action] += 1
                action_tokens[action] += completion_tokens
                if receives_credit:
                    credited_action = action
                    credited_conductor_tokens = completion_tokens
                    if advantage < 0.0:
                        signed_credit[action]["negative_samples"] += 1
                        signed_credit[action]["negative_tokens"] += completion_tokens
                    elif advantage > 0.0:
                        signed_credit[action]["positive_samples"] += 1
                        signed_credit[action]["positive_tokens"] += completion_tokens
            rewards.append(reward)
            episode_reports.append(
                {
                    "episode_path": str(episode_path),
                    "reward": reward,
                    "advantage": advantage,
                    "credited_decision_index": credited_decision_index,
                    "credited_action": credited_action,
                    "credited_conductor_tokens": credited_conductor_tokens,
                    "policy_samples": len(policy_samples) - before,
                    "policy_tokens": sum(
                        len(sample.prompt_ids) + len(sample.completion_ids) for sample in policy_samples[before:]
                    ),
                    "conductor_tokens": sum(len(sample.completion_ids) for sample in policy_samples[before:]),
                    "action_counts": action_counts,
                    "action_conductor_tokens": action_tokens,
                }
            )
        if len(set(rewards)) < 2:
            raise AleBatchError("ALE group has no outcome variation")
        credit_assignment["groups"].append({"group_id": group_id, "decision_indices": decision_indices})
        group_reports.append(
            {
                "path": str(resolved_group_path),
                "group_id": group_id,
                "task_id": task_id,
                "pool_id": group_pool_identity[0],
                "pool_binding_revision": group_pool_identity[1],
                "behavior_policy_revision": revision,
                "behavior_likelihood_contract": (
                    full_vocabulary_behavior_likelihood_contract()
                ),
                "policy_samples": len(policy_samples) - group_policy_start,
                "episodes": episode_reports,
            }
        )
    if not any(float(sample.advantage) < 0.0 for sample in policy_samples) or not any(
        float(sample.advantage) > 0.0 for sample in policy_samples
    ):
        raise AleBatchError("credited policy decisions do not preserve signed outcome variation")
    assert batch_pool_identity is not None

    replay_samples = _load_replay_samples(replay_path, expected_samples=52)
    for sample in replay_samples:
        total_tokens = len(sample.prompt_ids) + len(sample.completion_ids)
        if total_tokens > sequence_len:
            raise AleBatchError("mandatory replay exceeds the optimizer window")
        prepared = prepare_sample(sample, sequence_len)
        if len(prepared.input_ids) != total_tokens:
            raise AleBatchError("mandatory replay was truncated during preparation")

    train_retention_report: dict[str, Any] | None = None
    train_retention_samples: list[TrainingSample] = []
    if train_retention_replay_path is not None:
        assert train_retention_replay_samples is not None
        assert train_retention_report_path is not None
        retention_path = train_retention_replay_path.resolve()
        retention_report_path = train_retention_report_path.resolve()
        if retention_path == replay_path:
            raise AleBatchError("train-only retention replay must be distinct from mandatory replay")
        train_retention_report = _validate_train_retention_report(
            report_path=retention_report_path,
            replay_path=retention_path,
            replay_samples=train_retention_replay_samples,
        )
        train_retention_samples = _load_replay_samples(
            retention_path,
            expected_samples=train_retention_replay_samples,
        )
        for sample in train_retention_samples:
            total_tokens = len(sample.prompt_ids) + len(sample.completion_ids)
            if total_tokens > sequence_len:
                raise AleBatchError("train-only retention replay exceeds the optimizer window")
            prepared = prepare_sample(sample, sequence_len)
            if len(prepared.input_ids) != total_tokens:
                raise AleBatchError("train-only retention replay was truncated during preparation")
        expected_weighted_tokens = (
            (train_retention_report.get("weighting") or {}).get("total_weighted_completion_tokens")
            if train_retention_report.get("version") == ACTION_BALANCED_RETENTION_REPLAY_VERSION
            else (train_retention_report.get("counts") or {}).get("selected_completion_tokens")
        )
        if not math.isclose(
            _selected_replay_weight_sum(train_retention_samples),
            float(expected_weighted_tokens),
            rel_tol=1e-6,
            abs_tol=1e-3,
        ):
            raise AleBatchError("train-only retention replay weight mass changed")

    examples = [*policy_samples, *replay_samples, *train_retention_samples]
    batch = TrainingBatch(examples=examples, step=step)
    encoded = msgspec.msgpack.encode(batch)
    decoded = msgspec.msgpack.decode(encoded, type=TrainingBatch)
    if decoded != batch:
        raise AleBatchError("serialized ALE update does not round-trip exactly")
    if output_dir.exists():
        raise AleBatchError(f"refusing to overwrite ALE update directory: {output_dir}")
    output_dir.mkdir(parents=True)
    batch_path = output_dir / "rollouts.bin"
    batch_path.write_bytes(encoded)

    report = {
        "version": "fugu_ale_exact_grpo_batch_v5",
        "verdict": "ALE_EXACT_TOKEN_REPLAY_ANCHORED_GRPO_BATCH_READY",
        "behavior_policy_revision": revision,
        "behavior_likelihood_contract_version": (
            FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION
        ),
        "pool_id": batch_pool_identity[0],
        "pool_binding_revision": batch_pool_identity[1],
        "groups": group_reports,
        "optimizer_contract": {
            "atomic_training_batch": True,
            "sequence_len": sequence_len,
            "data_parallel_gpus": data_parallel_gpus,
            "adv_tau": float(adv_tau),
            "replay_tau": float(replay_tau),
            "kl_tau": float(kl_tau),
            "policy_logprob_source": "exact_behavior_policy_generation",
            "behavior_likelihood_contract": (
                full_vocabulary_behavior_likelihood_contract()
            ),
            "retokenization": False,
            "policy_credit_assignment": credit_assignment,
            "replay_reference_mode": REPLAY_REFERENCE_MODE,
        },
        "policy": {
            "samples": len(policy_samples),
            "credited_samples": sum(1 for sample in policy_samples if float(sample.advantage) != 0.0),
            "tokens": sum(len(sample.prompt_ids) + len(sample.completion_ids) for sample in policy_samples),
            "conductor_tokens": sum(len(sample.completion_ids) for sample in policy_samples),
            "credited_conductor_tokens": sum(
                len(sample.completion_ids) for sample in policy_samples if float(sample.advantage) != 0.0
            ),
            "signed_credit_by_action": signed_credit,
            "max_sequence_tokens": max(
                len(sample.prompt_ids) + len(sample.completion_ids) for sample in policy_samples
            ),
        },
        "mandatory_replay": {
            "path": str(replay_path),
            "samples": len(replay_samples),
            "selected_tokens": sum(sum(sample.replay_mask or []) for sample in replay_samples),
            "selected_weight_sum": _selected_replay_weight_sum(replay_samples),
        },
        "train_only_retention_replay": (
            {
                "path": str(train_retention_replay_path.resolve()),
                "samples": len(train_retention_samples),
                "selected_tokens": sum(sum(sample.replay_mask or []) for sample in train_retention_samples),
                "selected_weight_sum": _selected_replay_weight_sum(train_retention_samples),
                "report": str(train_retention_report_path.resolve()),
                "reference_mode": train_retention_report["reference_mode"],
                "weighting_mode": (train_retention_report.get("weighting") or {}).get("mode", "uniform_per_token"),
                "replay_version": train_retention_report["version"],
                "train_tasks": train_retention_report["counts"]["train_tasks"],
                "validation_tasks_excluded": train_retention_report["counts"]["validation_tasks_excluded"],
            }
            if train_retention_report is not None
            else None
        ),
        "combined_batch": {
            "path": str(batch_path),
            "samples": len(examples),
            "step": step,
        },
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    report_path = output_dir / "prepared_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--behavior-policy-revision", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--sequence-len", type=int, default=PROVEN_SEQUENCE_LEN)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--adv-tau", type=float, default=1.0)
    parser.add_argument("--replay-tau", type=float, default=0.05)
    parser.add_argument("--kl-tau", type=float, default=0.001)
    parser.add_argument("--data-parallel-gpus", type=int, default=6)
    parser.add_argument("--train-retention-replay", type=Path)
    parser.add_argument("--train-retention-replay-samples", type=int)
    parser.add_argument("--train-retention-report", type=Path)
    parser.add_argument(
        "--credited-action",
        action="append",
        choices=sorted(CONTROL_ACTIONS),
        dest="credited_actions",
        help="optional additional action filter on the admitted causal decision",
    )
    args = parser.parse_args()
    report = materialize_ale_grpo_update(
        group_paths=args.group,
        output_dir=args.output_dir,
        expected_behavior_policy_revision=args.behavior_policy_revision,
        replay_path=args.replay,
        sequence_len=args.sequence_len,
        step=args.step,
        adv_tau=args.adv_tau,
        replay_tau=args.replay_tau,
        kl_tau=args.kl_tau,
        data_parallel_gpus=args.data_parallel_gpus,
        credited_actions=(tuple(args.credited_actions) if args.credited_actions is not None else None),
        train_retention_replay_path=args.train_retention_replay,
        train_retention_replay_samples=args.train_retention_replay_samples,
        train_retention_report_path=args.train_retention_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
