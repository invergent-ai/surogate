"""Materialize causal synthetic conductor rollouts as one exact-token GRPO update.

Only tokens sampled by the attested behavior policy enter the policy objective.
The deterministic scenario oracle is used to identify a same-state contrast
between a verified rollout and a rollout's first terminal mistake; oracle text
is never converted into a GRPO sample.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
from ultra.behavior_likelihood import (
    FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION,
    full_vocabulary_behavior_likelihood_contract,
)
from ultra.live_control import (
    ControlContractError,
    capability_reference_map,
    parse_capability_control_action,
    serialize_capability_control_action,
)
from ultra.pool_binding import PoolBinding, load_pool_binding
from ultra.synthetic_collection import SYNTHETIC_COLLECTION_VERSION
from ultra.synthetic_rollouts import (
    SYNTHETIC_CURRICULUM_REVISION,
    SyntheticBoundary,
    SyntheticScenario,
    build_synthetic_curriculum,
)

from surogate.grpo.ale_batch import (
    ACTION_BALANCED_RETENTION_REPLAY_VERSION,
    ACTION_BALANCED_RETENTION_SAMPLES,
    ACTION_BALANCED_RETENTION_SELECTED_TOKENS,
    CONTROL_ACTIONS,
    PROVEN_SEQUENCE_LEN,
    REPLAY_REFERENCE_MODE,
    _exact_policy_sample,
    _load_replay_samples,
    _selected_replay_weight_sum,
    _validate_train_retention_report,
)
from surogate.grpo.batch import prepare_sample
from surogate.grpo.transport import TrainingBatch, TrainingSample

SYNTHETIC_BATCH_VERSION = "fugu_synthetic_exact_grpo_batch_v1"
SYNTHETIC_BATCH_VERDICT = (
    "SYNTHETIC_EXACT_TOKEN_REPLAY_ANCHORED_GRPO_BATCH_READY"
)
SYNTHETIC_CREDIT_MODE = (
    "same_state_first_failure_vs_verified_success_v1"
)
MIN_POLICY_SAMPLES = 16
MAX_POLICY_SAMPLES = 32
MANDATORY_TRANSFER_REPLAY_SAMPLES = 52
MANDATORY_TRANSFER_REPLAY_TOKENS = 2_448
DATA_PARALLEL_GPUS = 6
ADV_TAU = 1.0
REPLAY_TAU = 0.05
KL_TAU = 0.001


class SyntheticBatchError(ValueError):
    """A synthetic collection cannot safely enter the optimizer."""


@dataclass(frozen=True)
class _ObservedDecision:
    scenario_index: int
    scenario_id: str
    motif: str
    boundary_index: int
    boundary_id: str
    sample_index: int
    sampling_seed: int
    reward: float
    action: str
    trace: dict[str, Any]


@dataclass
class _ContrastGroup:
    scenario_index: int
    scenario_id: str
    motif: str
    boundary_index: int
    boundary_id: str
    positives: list[_ObservedDecision]
    negatives: list[_ObservedDecision]
    selected: list[_ObservedDecision]


@dataclass(frozen=True)
class _PolicyMaterialization:
    samples: list[TrainingSample]
    credit_groups: list[dict[str, Any]]
    policy_report: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticBatchError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SyntheticBatchError(f"{path} must contain one JSON object")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise SyntheticBatchError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _messages(trace: dict[str, Any]) -> list[dict[str, str]]:
    messages = trace.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not isinstance(message["content"], str)
            for message in messages
        )
    ):
        raise SyntheticBatchError(
            "synthetic policy trace lacks exact role/content messages"
        )
    return messages


def _validate_trace(
    trace: object,
    *,
    sampling_seed: int,
    reward: float,
    sequence_len: int,
) -> dict[str, Any]:
    if not isinstance(trace, dict):
        raise SyntheticBatchError("synthetic policy trace is not an object")
    if trace.get("seed") != sampling_seed:
        raise SyntheticBatchError(
            "synthetic policy trace seed differs from its rollout"
        )
    if trace.get("correction") is not None:
        raise SyntheticBatchError(
            "synthetic policy trace is a correction attempt"
        )
    if trace.get("finish_reason") != "stop":
        raise SyntheticBatchError(
            "synthetic policy trace did not finish cleanly"
        )
    if not isinstance(trace.get("response"), str):
        raise SyntheticBatchError(
            "synthetic policy trace lacks its exact response"
        )
    _messages(trace)
    prompt_ids = trace.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise SyntheticBatchError(
            "synthetic policy trace has no exact prompt token IDs"
        )
    logprobs = trace.get("completion_logprobs")
    if not isinstance(logprobs, list) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) > 1.0e-6
        for value in logprobs
    ):
        raise SyntheticBatchError(
            "synthetic policy trace has invalid sampled log-probabilities"
        )
    try:
        _exact_policy_sample(
            trace,
            advantage=0.0,
            reward=reward,
            sampling_temperature=1.0,
            sequence_len=sequence_len,
        )
    except ValueError as exc:
        raise SyntheticBatchError(str(exc)) from exc
    return trace


def _canonical_action(
    *,
    trace: dict[str, Any],
    stored_action: object,
    boundary: SyntheticBoundary,
) -> tuple[object | None, str, bool, str]:
    references = capability_reference_map(boundary.state.workers)
    response = trace["response"]
    try:
        action = parse_capability_control_action(response, references)
    except ControlContractError:
        if stored_action is not None:
            raise SyntheticBatchError(
                "invalid raw policy response carries a parsed action"
            )
        return (
            None,
            "invalid",
            False,
            "invalid_policy_output:ControlContractError",
        )

    canonical = json.loads(
        serialize_capability_control_action(action, references)
    )
    if stored_action != canonical:
        raise SyntheticBatchError(
            "serialized synthetic action differs from the raw policy response"
        )
    matched, outcome = boundary.oracle.matches(action, boundary.state)
    return action, action.action, matched, outcome


def _validate_rollout(
    *,
    raw: object,
    scenario: SyntheticScenario,
    scenario_index: int,
    sample_index: int,
    expected_seed: int,
    behavior_policy_revision: str,
    runtime_revision: str,
    binding: PoolBinding,
    sequence_len: int,
) -> tuple[list[_ObservedDecision], bool]:
    if not isinstance(raw, dict):
        raise SyntheticBatchError("synthetic rollout is not an object")
    if raw.get("sample_index") != sample_index:
        raise SyntheticBatchError(
            "synthetic rollout sample indexes are not canonical"
        )
    reward = raw.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or float(reward) not in {0.0, 1.0}
    ):
        raise SyntheticBatchError(
            "synthetic rollout reward must be binary"
        )
    reward = float(reward)
    policy = raw.get("policy")
    expected_policy = {
        "behavior_policy_revision": behavior_policy_revision,
        "runtime_revision": runtime_revision,
        "pool_id": binding.pool_id,
        "pool_binding_revision": binding.binding_revision,
        "sampling_seed": expected_seed,
    }
    if policy != expected_policy:
        raise SyntheticBatchError(
            "synthetic rollout policy attestation changed"
        )
    decisions = raw.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise SyntheticBatchError(
            "synthetic rollout contains no conductor decisions"
        )

    boundaries = scenario.boundary_map()
    boundary_indexes = {
        boundary.boundary_id: index
        for index, boundary in enumerate(scenario.boundaries)
    }
    current_id = scenario.initial_boundary_id
    observed: list[_ObservedDecision] = []
    terminated_by_mistake = False
    for decision_index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise SyntheticBatchError(
                "synthetic rollout decision is not an object"
            )
        if decision.get("boundary_id") != current_id:
            raise SyntheticBatchError(
                "synthetic rollout does not follow its scripted boundary chain"
            )
        boundary = boundaries[current_id]
        trace = _validate_trace(
            decision.get("trace"),
            sampling_seed=expected_seed,
            reward=reward,
            sequence_len=sequence_len,
        )
        _action, action_name, matched, transition = _canonical_action(
            trace=trace,
            stored_action=decision.get("action"),
            boundary=boundary,
        )
        if decision.get("matched_outcome_path") is not matched:
            raise SyntheticBatchError(
                "synthetic rollout match label differs from its scenario"
            )
        if decision.get("transition_outcome") != transition:
            raise SyntheticBatchError(
                "synthetic rollout transition outcome differs from its scenario"
            )
        observed.append(
            _ObservedDecision(
                scenario_index=scenario_index,
                scenario_id=scenario.scenario_id,
                motif=scenario.motif,
                boundary_index=boundary_indexes[current_id],
                boundary_id=current_id,
                sample_index=sample_index,
                sampling_seed=expected_seed,
                reward=reward,
                action=action_name,
                trace=trace,
            )
        )
        is_last = decision_index == len(decisions) - 1
        if not matched:
            if not is_last:
                raise SyntheticBatchError(
                    "synthetic rollout continues after its first mistake"
                )
            terminated_by_mistake = True
            break
        if boundary.next_boundary_id is None:
            if not is_last:
                raise SyntheticBatchError(
                    "synthetic rollout continues after verified completion"
                )
            break
        if is_last:
            raise SyntheticBatchError(
                "synthetic rollout stops before an outcome"
            )
        current_id = boundary.next_boundary_id

    verified = not terminated_by_mistake
    if verified:
        if (
            reward != 1.0
            or raw.get("outcome") != "task_outcome_verified"
        ):
            raise SyntheticBatchError(
                "verified synthetic rollout lacks terminal reward"
            )
        if len(observed) != len(scenario.boundaries):
            raise SyntheticBatchError(
                "verified synthetic rollout skipped a boundary"
            )
    else:
        reported_outcome = raw.get("outcome")
        transition_outcome = decisions[-1].get("transition_outcome")
        if reward != 0.0 or reported_outcome != transition_outcome:
            raise SyntheticBatchError(
                "failed synthetic rollout outcome changed"
            )
    return observed, verified


def _interleaved_groups(
    groups: list[_ContrastGroup],
) -> list[_ContrastGroup]:
    by_scenario: dict[int, list[_ContrastGroup]] = defaultdict(list)
    for group in groups:
        by_scenario[group.scenario_index].append(group)
    for scenario_groups in by_scenario.values():
        scenario_groups.sort(key=lambda group: group.boundary_index)
    ordered: list[_ContrastGroup] = []
    depth = 0
    while True:
        added = False
        for scenario_index in sorted(by_scenario):
            scenario_groups = by_scenario[scenario_index]
            if depth < len(scenario_groups):
                ordered.append(scenario_groups[depth])
                added = True
        if not added:
            return ordered
        depth += 1


def _select_contrasts(
    groups: list[_ContrastGroup],
) -> list[_ContrastGroup]:
    ordered = _interleaved_groups(groups)
    if len({group.scenario_id for group in ordered}) < 2:
        raise SyntheticBatchError(
            "synthetic update requires causal signal from at least two scenarios"
        )

    selected_groups: list[_ContrastGroup] = []
    selected_count = 0
    for group in ordered:
        if selected_count + 2 > MAX_POLICY_SAMPLES:
            break
        group.selected = [group.positives[0], group.negatives[0]]
        selected_groups.append(group)
        selected_count += 2

    remaining: dict[tuple[int, str], list[_ObservedDecision]] = {}
    for group in selected_groups:
        extras = [*group.positives[1:], *group.negatives[1:]]
        extras.sort(
            key=lambda row: (
                row.sample_index,
                row.reward,
                row.sampling_seed,
            )
        )
        remaining[(group.scenario_index, group.boundary_id)] = extras

    while selected_count < MAX_POLICY_SAMPLES:
        added = False
        for group in selected_groups:
            extras = remaining[(group.scenario_index, group.boundary_id)]
            if extras and selected_count < MAX_POLICY_SAMPLES:
                group.selected.append(extras.pop(0))
                selected_count += 1
                added = True
        if not added:
            break

    if selected_count < MIN_POLICY_SAMPLES:
        raise SyntheticBatchError(
            "synthetic collection has fewer than 16 causally contrastive policy rows"
        )
    if len({group.scenario_id for group in selected_groups}) < 2:
        raise SyntheticBatchError(
            "selected synthetic signal comes from only one scenario"
        )
    return selected_groups


def _validated_collection_groups(
    *,
    collection_path: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    sequence_len: int,
) -> list[_ContrastGroup]:
    collection = _read_json(collection_path)
    try:
        binding = load_pool_binding(pool_binding_path)
    except Exception as exc:
        raise SyntheticBatchError(
            f"cannot load pool binding {pool_binding_path}: {exc}"
        ) from exc
    expected_top = {
        "version": SYNTHETIC_COLLECTION_VERSION,
        "verdict": "SYNTHETIC_EXACT_TOKEN_ROLLOUTS_COLLECTED",
        "behavior_policy_revision": expected_behavior_policy_revision,
        "runtime_revision": expected_runtime_revision,
        "pool_id": binding.pool_id,
        "pool_binding_revision": binding.binding_revision,
        "pool_binding": str(pool_binding_path),
        "curriculum_revision": SYNTHETIC_CURRICULUM_REVISION,
        "sampling_temperature": 1.0,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    for key, expected in expected_top.items():
        if collection.get(key) != expected:
            raise SyntheticBatchError(
                f"synthetic collection {key} changed"
            )
    scenario_seed = _require_int(
        collection.get("scenario_seed"),
        "scenario_seed",
    )
    scenario_count = _require_int(
        collection.get("scenario_count"),
        "scenario_count",
        minimum=1,
    )
    samples_per_scenario = _require_int(
        collection.get("samples_per_scenario"),
        "samples_per_scenario",
        minimum=2,
    )
    if collection.get("rollout_count") != (
        scenario_count * samples_per_scenario
    ):
        raise SyntheticBatchError(
            "synthetic collection rollout count changed"
        )
    try:
        expected_scenarios = build_synthetic_curriculum(
            count=scenario_count,
            seed=scenario_seed,
            profile_capabilities=tuple(
                slot.role_prior for slot in binding.slots
            ),
        )
    except Exception as exc:
        raise SyntheticBatchError(
            f"cannot rebuild synthetic curriculum: {exc}"
        ) from exc
    raw_scenarios = collection.get("scenarios")
    if (
        not isinstance(raw_scenarios, list)
        or len(raw_scenarios) != len(expected_scenarios)
    ):
        raise SyntheticBatchError(
            "synthetic collection scenario inventory changed"
        )

    global_seeds: set[int] = set()
    successes: dict[tuple[int, str], list[_ObservedDecision]] = defaultdict(list)
    failures: dict[tuple[int, str], list[_ObservedDecision]] = defaultdict(list)
    for scenario_index, (raw_scenario, scenario) in enumerate(
        zip(raw_scenarios, expected_scenarios, strict=True)
    ):
        if not isinstance(raw_scenario, dict):
            raise SyntheticBatchError(
                "synthetic collection scenario is not an object"
            )
        expected_scenario_fields = {
            "scenario_index": scenario_index,
            "scenario_id": scenario.scenario_id,
            "motif": scenario.motif,
            "evidence_basis": list(scenario.evidence_basis),
            "boundary_count": len(scenario.boundaries),
        }
        for key, expected in expected_scenario_fields.items():
            if raw_scenario.get(key) != expected:
                raise SyntheticBatchError(
                    f"synthetic scenario {scenario_index} {key} changed"
                )
        raw_rollouts = raw_scenario.get("rollouts")
        if (
            not isinstance(raw_rollouts, list)
            or len(raw_rollouts) != samples_per_scenario
        ):
            raise SyntheticBatchError(
                "synthetic scenario rollout inventory changed"
            )
        rewards: list[float] = []
        for sample_index, raw_rollout in enumerate(raw_rollouts):
            expected_seed = (
                scenario_seed
                + 1_000_003 * scenario_index
                + sample_index
            )
            if expected_seed in global_seeds:
                raise SyntheticBatchError(
                    "synthetic collection sampling seeds are not unique"
                )
            global_seeds.add(expected_seed)
            observed, verified = _validate_rollout(
                raw=raw_rollout,
                scenario=scenario,
                scenario_index=scenario_index,
                sample_index=sample_index,
                expected_seed=expected_seed,
                behavior_policy_revision=expected_behavior_policy_revision,
                runtime_revision=expected_runtime_revision,
                binding=binding,
                sequence_len=sequence_len,
            )
            rewards.append(float(raw_rollout["reward"]))
            if verified:
                for decision in observed:
                    successes[
                        (scenario_index, decision.boundary_id)
                    ].append(decision)
            else:
                terminal = observed[-1]
                failures[
                    (scenario_index, terminal.boundary_id)
                ].append(terminal)
        expected_reward_counts = dict(
            sorted(
                Counter(str(float(reward)) for reward in rewards).items()
            )
        )
        if raw_scenario.get("reward_counts") != expected_reward_counts:
            raise SyntheticBatchError(
                "synthetic scenario reward counts changed"
            )

    groups: list[_ContrastGroup] = []
    for scenario_index, scenario in enumerate(expected_scenarios):
        for boundary_index, boundary in enumerate(scenario.boundaries):
            key = (scenario_index, boundary.boundary_id)
            positive = sorted(
                successes.get(key, []),
                key=lambda row: row.sample_index,
            )
            negative = sorted(
                failures.get(key, []),
                key=lambda row: row.sample_index,
            )
            if not positive or not negative:
                continue
            prompt_ids = positive[0].trace["prompt_token_ids"]
            messages = _messages(positive[0].trace)
            for decision in [*positive, *negative]:
                if (
                    decision.trace["prompt_token_ids"] != prompt_ids
                    or _messages(decision.trace) != messages
                ):
                    raise SyntheticBatchError(
                        "same-boundary synthetic contrast has different prompts"
                    )
            groups.append(
                _ContrastGroup(
                    scenario_index=scenario_index,
                    scenario_id=scenario.scenario_id,
                    motif=scenario.motif,
                    boundary_index=boundary_index,
                    boundary_id=boundary.boundary_id,
                    positives=positive,
                    negatives=negative,
                    selected=[],
                )
            )
    if not groups:
        raise SyntheticBatchError(
            "synthetic collection has no same-state outcome contrast"
        )
    return _select_contrasts(groups)


def _materialize_policy(
    groups: list[_ContrastGroup],
    *,
    sequence_len: int,
) -> _PolicyMaterialization:
    samples: list[TrainingSample] = []
    credit_groups: list[dict[str, Any]] = []
    signed_credit = {
        action: {
            "negative_samples": 0,
            "negative_tokens": 0,
            "positive_samples": 0,
            "positive_tokens": 0,
        }
        for action in [*sorted(CONTROL_ACTIONS), "invalid"]
    }
    for group in groups:
        rows = sorted(
            group.selected,
            key=lambda row: (
                row.reward,
                row.sample_index,
                row.sampling_seed,
            ),
        )
        rewards = [row.reward for row in rows]
        if len(set(rewards)) != 2:
            raise SyntheticBatchError(
                "selected synthetic boundary lost signed outcome variation"
            )
        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.stdev(rewards)
        advantages = [
            (reward - reward_mean) / (reward_std + 1.0e-4)
            for reward in rewards
        ]
        group_start = len(samples)
        positive_indices: list[int] = []
        negative_indices: list[int] = []
        source_rows: list[dict[str, Any]] = []
        for row, advantage in zip(rows, advantages, strict=True):
            try:
                sample = _exact_policy_sample(
                    row.trace,
                    advantage=advantage,
                    reward=row.reward,
                    sampling_temperature=1.0,
                    sequence_len=sequence_len,
                )
            except ValueError as exc:
                raise SyntheticBatchError(str(exc)) from exc
            policy_index = len(samples)
            samples.append(sample)
            sign = "positive" if advantage > 0.0 else "negative"
            signed_credit[row.action][f"{sign}_samples"] += 1
            signed_credit[row.action][f"{sign}_tokens"] += len(
                sample.completion_ids
            )
            if row.reward == 1.0:
                positive_indices.append(policy_index)
            else:
                negative_indices.append(policy_index)
            source_rows.append(
                {
                    "policy_sample_index": policy_index,
                    "sample_index": row.sample_index,
                    "sampling_seed": row.sampling_seed,
                    "reward": row.reward,
                    "advantage": advantage,
                    "action": row.action,
                    "conductor_tokens": len(sample.completion_ids),
                }
            )
        credit_groups.append(
            {
                "scenario_index": group.scenario_index,
                "scenario_id": group.scenario_id,
                "motif": group.motif,
                "boundary_index": group.boundary_index,
                "boundary_id": group.boundary_id,
                "policy_sample_start": group_start,
                "policy_sample_count": len(rows),
                "positive_policy_sample_indices": positive_indices,
                "negative_policy_sample_indices": negative_indices,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "advantage_method": (
                    "sample_std_normalized_eps_1e-4"
                ),
                "source_rows": source_rows,
            }
        )
    if not MIN_POLICY_SAMPLES <= len(samples) <= MAX_POLICY_SAMPLES:
        raise SyntheticBatchError(
            "synthetic policy sample count is outside the 16-32 window"
        )
    policy_report = {
        "samples": len(samples),
        "credited_samples": len(samples),
        "tokens": sum(
            len(sample.prompt_ids) + len(sample.completion_ids)
            for sample in samples
        ),
        "conductor_tokens": sum(
            len(sample.completion_ids) for sample in samples
        ),
        "signed_credit_by_action": signed_credit,
        "max_sequence_tokens": max(
            len(sample.prompt_ids) + len(sample.completion_ids)
            for sample in samples
        ),
        "scenario_ids": sorted(
            {group.scenario_id for group in groups}
        ),
        "motifs": sorted({group.motif for group in groups}),
    }
    return _PolicyMaterialization(
        samples=samples,
        credit_groups=credit_groups,
        policy_report=policy_report,
    )


def _load_required_replay(
    *,
    replay_path: Path,
    retention_replay_path: Path,
    retention_report_path: Path,
    sequence_len: int,
) -> tuple[
    list[TrainingSample],
    list[TrainingSample],
    dict[str, Any],
]:
    replay_samples = _load_replay_samples(
        replay_path,
        expected_samples=MANDATORY_TRANSFER_REPLAY_SAMPLES,
    )
    if (
        sum(sum(sample.replay_mask or []) for sample in replay_samples)
        != MANDATORY_TRANSFER_REPLAY_TOKENS
    ):
        raise SyntheticBatchError(
            "mandatory transfer replay selected-token count changed"
        )
    if not math.isclose(
        _selected_replay_weight_sum(replay_samples),
        float(MANDATORY_TRANSFER_REPLAY_TOKENS),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise SyntheticBatchError(
            "mandatory transfer replay weight mass changed"
        )
    retention_report = _validate_train_retention_report(
        report_path=retention_report_path,
        replay_path=retention_replay_path,
        replay_samples=ACTION_BALANCED_RETENTION_SAMPLES,
    )
    if (
        retention_report.get("version")
        != ACTION_BALANCED_RETENTION_REPLAY_VERSION
    ):
        raise SyntheticBatchError(
            "mandatory retention replay is not action-balanced v2"
        )
    retention_samples = _load_replay_samples(
        retention_replay_path,
        expected_samples=ACTION_BALANCED_RETENTION_SAMPLES,
    )
    if (
        sum(sum(sample.replay_mask or []) for sample in retention_samples)
        != ACTION_BALANCED_RETENTION_SELECTED_TOKENS
    ):
        raise SyntheticBatchError(
            "mandatory action-balanced retention token count changed"
        )
    expected_weight = float(
        (retention_report.get("weighting") or {}).get(
            "total_weighted_completion_tokens",
            -1.0,
        )
    )
    if not math.isclose(
        _selected_replay_weight_sum(retention_samples),
        expected_weight,
        rel_tol=1.0e-6,
        abs_tol=1.0e-3,
    ):
        raise SyntheticBatchError(
            "mandatory action-balanced retention weight mass changed"
        )
    for sample in [*replay_samples, *retention_samples]:
        total_tokens = len(sample.prompt_ids) + len(
            sample.completion_ids
        )
        if total_tokens > sequence_len:
            raise SyntheticBatchError(
                "mandatory replay exceeds the optimizer window"
            )
        prepared = prepare_sample(sample, sequence_len)
        if len(prepared.input_ids) != total_tokens:
            raise SyntheticBatchError(
                "mandatory replay was truncated during preparation"
            )
    return replay_samples, retention_samples, retention_report


def _report(
    *,
    collection_path: Path,
    pool_binding_path: Path,
    behavior_policy_revision: str,
    runtime_revision: str,
    policy: _PolicyMaterialization,
    replay_path: Path,
    replay_samples: list[TrainingSample],
    retention_replay_path: Path,
    retention_samples: list[TrainingSample],
    retention_report_path: Path,
    retention_report: dict[str, Any],
    batch_path: Path,
) -> dict[str, Any]:
    return {
        "version": SYNTHETIC_BATCH_VERSION,
        "verdict": SYNTHETIC_BATCH_VERDICT,
        "behavior_policy_revision": behavior_policy_revision,
        "runtime_revision": runtime_revision,
        "pool_id": load_pool_binding(pool_binding_path).pool_id,
        "pool_binding_revision": load_pool_binding(
            pool_binding_path
        ).binding_revision,
        "pool_binding": str(pool_binding_path),
        "curriculum_revision": SYNTHETIC_CURRICULUM_REVISION,
        "behavior_likelihood_contract_version": (
            FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION
        ),
        "source_collection": str(collection_path),
        "credit_groups": policy.credit_groups,
        "optimizer_contract": {
            "atomic_training_batch": True,
            "sequence_len": PROVEN_SEQUENCE_LEN,
            "data_parallel_gpus": DATA_PARALLEL_GPUS,
            "adv_tau": ADV_TAU,
            "replay_tau": REPLAY_TAU,
            "kl_tau": KL_TAU,
            "policy_logprob_source": (
                "exact_behavior_policy_generation"
            ),
            "behavior_likelihood_contract": (
                full_vocabulary_behavior_likelihood_contract()
            ),
            "retokenization": False,
            "policy_credit_assignment": {
                "mode": SYNTHETIC_CREDIT_MODE,
                "policy_attempts": "initial_only_no_corrections",
                "reward": "verified_terminal_binary",
                "same_prompt_required": True,
                "minimum_scenarios": 2,
                "policy_sample_window": [
                    MIN_POLICY_SAMPLES,
                    MAX_POLICY_SAMPLES,
                ],
            },
            "replay_reference_mode": REPLAY_REFERENCE_MODE,
        },
        "policy": policy.policy_report,
        "mandatory_replay": {
            "path": str(replay_path),
            "samples": len(replay_samples),
            "selected_tokens": sum(
                sum(sample.replay_mask or [])
                for sample in replay_samples
            ),
            "selected_weight_sum": _selected_replay_weight_sum(
                replay_samples
            ),
        },
        "train_only_retention_replay": {
            "path": str(retention_replay_path),
            "samples": len(retention_samples),
            "selected_tokens": sum(
                sum(sample.replay_mask or [])
                for sample in retention_samples
            ),
            "selected_weight_sum": _selected_replay_weight_sum(
                retention_samples
            ),
            "report": str(retention_report_path),
            "reference_mode": retention_report["reference_mode"],
            "weighting_mode": (
                retention_report.get("weighting") or {}
            ).get("mode"),
            "replay_version": retention_report["version"],
            "train_tasks": retention_report["counts"]["train_tasks"],
            "validation_tasks_excluded": retention_report["counts"][
                "validation_tasks_excluded"
            ],
        },
        "combined_batch": {
            "path": str(batch_path),
            "samples": (
                len(policy.samples)
                + len(replay_samples)
                + len(retention_samples)
            ),
            "step": 0,
        },
        "paid_calls": 0,
        "optimizer_steps": 0,
    }


def _prepare_expected(
    *,
    collection_path: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    replay_path: Path,
    retention_replay_path: Path,
    retention_report_path: Path,
    batch_path: Path,
) -> tuple[dict[str, Any], TrainingBatch]:
    groups = _validated_collection_groups(
        collection_path=collection_path,
        expected_behavior_policy_revision=(
            expected_behavior_policy_revision
        ),
        expected_runtime_revision=expected_runtime_revision,
        pool_binding_path=pool_binding_path,
        sequence_len=PROVEN_SEQUENCE_LEN,
    )
    policy = _materialize_policy(
        groups,
        sequence_len=PROVEN_SEQUENCE_LEN,
    )
    replay, retention, retention_report = _load_required_replay(
        replay_path=replay_path,
        retention_replay_path=retention_replay_path,
        retention_report_path=retention_report_path,
        sequence_len=PROVEN_SEQUENCE_LEN,
    )
    batch = TrainingBatch(
        examples=[*policy.samples, *replay, *retention],
        step=0,
    )
    report = _report(
        collection_path=collection_path,
        pool_binding_path=pool_binding_path,
        behavior_policy_revision=expected_behavior_policy_revision,
        runtime_revision=expected_runtime_revision,
        policy=policy,
        replay_path=replay_path,
        replay_samples=replay,
        retention_replay_path=retention_replay_path,
        retention_samples=retention,
        retention_report_path=retention_report_path,
        retention_report=retention_report,
        batch_path=batch_path,
    )
    return report, batch


def materialize_synthetic_grpo_update(
    *,
    collection_path: Path,
    output_dir: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    replay_path: Path,
    train_retention_replay_path: Path,
    train_retention_report_path: Path,
) -> dict[str, Any]:
    """Write one bounded exact-token synthetic update with both replay sets."""
    output_dir = output_dir.expanduser().resolve()
    collection_path = collection_path.expanduser().resolve()
    pool_binding_path = pool_binding_path.expanduser().resolve()
    replay_path = replay_path.expanduser().resolve()
    retention_path = train_retention_replay_path.expanduser().resolve()
    retention_report_path = (
        train_retention_report_path.expanduser().resolve()
    )
    if output_dir.exists():
        raise SyntheticBatchError(
            f"refusing to overwrite synthetic update directory: {output_dir}"
        )
    if not expected_behavior_policy_revision.strip():
        raise SyntheticBatchError(
            "expected behavior-policy revision must be non-empty"
        )
    if not expected_runtime_revision.strip():
        raise SyntheticBatchError(
            "expected runtime revision must be non-empty"
        )
    batch_path = output_dir / "rollouts.bin"
    report, batch = _prepare_expected(
        collection_path=collection_path,
        expected_behavior_policy_revision=(
            expected_behavior_policy_revision
        ),
        expected_runtime_revision=expected_runtime_revision,
        pool_binding_path=pool_binding_path,
        replay_path=replay_path,
        retention_replay_path=retention_path,
        retention_report_path=retention_report_path,
        batch_path=batch_path,
    )
    encoded = msgspec.msgpack.encode(batch)
    decoded = msgspec.msgpack.decode(encoded, type=TrainingBatch)
    if decoded != batch:
        raise SyntheticBatchError(
            "serialized synthetic update does not round-trip exactly"
        )
    output_dir.mkdir(parents=True)
    batch_path.write_bytes(encoded)
    (output_dir / "prepared_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def validate_synthetic_prepared_batch(
    *,
    prepared_report_path: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    replay_path: Path,
    train_retention_replay_path: Path,
    train_retention_report_path: Path,
) -> tuple[dict[str, Any], Path, TrainingBatch]:
    """Validate a prepared synthetic batch for optimizer staging."""
    prepared_report_path = prepared_report_path.expanduser().resolve()
    report = _read_json(prepared_report_path)
    collection_path = Path(
        str(report.get("source_collection", ""))
    ).expanduser().resolve()
    batch_path = prepared_report_path.parent / "rollouts.bin"
    combined = report.get("combined_batch")
    if (
        not isinstance(combined, dict)
        or Path(str(combined.get("path", ""))).resolve() != batch_path
        or not batch_path.is_file()
    ):
        raise SyntheticBatchError(
            "prepared synthetic batch path changed"
        )
    expected_report, expected_batch = _prepare_expected(
        collection_path=collection_path,
        expected_behavior_policy_revision=(
            expected_behavior_policy_revision
        ),
        expected_runtime_revision=expected_runtime_revision,
        pool_binding_path=pool_binding_path.expanduser().resolve(),
        replay_path=replay_path.expanduser().resolve(),
        retention_replay_path=(
            train_retention_replay_path.expanduser().resolve()
        ),
        retention_report_path=(
            train_retention_report_path.expanduser().resolve()
        ),
        batch_path=batch_path,
    )
    if report != expected_report:
        raise SyntheticBatchError(
            "prepared synthetic report differs from its source collection"
        )
    try:
        batch = msgspec.msgpack.decode(
            batch_path.read_bytes(),
            type=TrainingBatch,
        )
    except Exception as exc:
        raise SyntheticBatchError(
            f"prepared synthetic batch cannot be decoded: {exc}"
        ) from exc
    if batch != expected_batch:
        raise SyntheticBatchError(
            "prepared synthetic batch differs from its exact source rows"
        )
    return report, batch_path, batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--behavior-policy-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--pool-binding", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--train-retention-replay",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--train-retention-report",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    report = materialize_synthetic_grpo_update(
        collection_path=args.collection,
        output_dir=args.output_dir,
        expected_behavior_policy_revision=(
            args.behavior_policy_revision
        ),
        expected_runtime_revision=args.runtime_revision,
        pool_binding_path=args.pool_binding,
        replay_path=args.replay,
        train_retention_replay_path=args.train_retention_replay,
        train_retention_report_path=args.train_retention_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "MAX_POLICY_SAMPLES",
    "MIN_POLICY_SAMPLES",
    "SYNTHETIC_BATCH_VERDICT",
    "SYNTHETIC_BATCH_VERSION",
    "SYNTHETIC_CREDIT_MODE",
    "SyntheticBatchError",
    "materialize_synthetic_grpo_update",
    "validate_synthetic_prepared_batch",
]


if __name__ == "__main__":
    main()
