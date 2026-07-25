from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import msgspec
import pytest
from ultra.behavior_likelihood import (
    full_vocabulary_behavior_likelihood_contract,
)
from ultra.live_control import (
    capability_reference_map,
    serialize_capability_control_action,
)
from ultra.pool_binding import load_pool_binding
from ultra.synthetic_collection import SYNTHETIC_COLLECTION_VERSION
from ultra.synthetic_rollouts import (
    SYNTHETIC_CURRICULUM_REVISION,
    build_synthetic_curriculum,
)

from surogate.grpo.synthetic_batch import (
    MAX_POLICY_SAMPLES,
    MIN_POLICY_SAMPLES,
    SYNTHETIC_BATCH_VERSION,
    SYNTHETIC_CREDIT_MODE,
    SyntheticBatchError,
    materialize_synthetic_grpo_update,
    validate_synthetic_prepared_batch,
)
from surogate.grpo.transport import TrainingBatch

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "scratchpad/fugu_27b_transfer_replay_v1/replay.bin"
RETENTION_REPLAY = (
    ROOT
    / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/replay.bin"
)
RETENTION_REPORT = (
    ROOT
    / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/report.json"
)
POLICY_REVISION = "fugu-accepted-test-policy"
RUNTIME_REVISION = "runtime-r81-test"
POOL_ID = "anonymous-test-pool-v1"

ROLE_PRIORS = (
    ("reasoner", "verifier", "debugger"),
    ("scientist", "planner", "aggregator"),
    ("mathematician", "coder", "reasoner"),
    ("drafter", "implementer", "fast_pass"),
)


def _write_binding(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "fugu_pool_binding_v1",
                "pool_id": POOL_ID,
                "binding_revision": POOL_ID,
                "provider_base": "https://yunwu.ai/v1",
                "slots": [
                    {
                        "worker_id": index,
                        "training_name": f"bound-worker-{index}",
                        "model_alias": f"alias-{index}",
                        "runtime_model": f"runtime-{index}",
                        "reasoning_effort": "high",
                        "role_prior": list(roles),
                    }
                    for index, roles in enumerate(ROLE_PRIORS)
                ],
                "checkpoint": {
                    "adapter_path": "scratchpad/test-adapter",
                    "base_model_snapshot": "test-model",
                    "trained_control_contract": (
                        "unified_capability_action_v2"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _trace(
    *,
    scenario_index: int,
    boundary_index: int,
    sample_index: int,
    seed: int,
    response: str,
) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Anonymous synthetic conductor.",
            },
            {
                "role": "user",
                "content": (
                    f"scenario={scenario_index};boundary={boundary_index}"
                ),
            },
        ],
        "response": response,
        "finish_reason": "stop",
        "correction": None,
        "prompt_token_ids": [
            100 + scenario_index,
            1_000 + boundary_index,
        ],
        "completion_token_ids": [
            10_000 + scenario_index * 1_000 + sample_index * 20
            + boundary_index * 2,
            10_001 + scenario_index * 1_000 + sample_index * 20
            + boundary_index * 2,
        ],
        "completion_logprobs": [-0.2, -0.1],
        "temperature": 1.0,
        "seed": seed,
        "behavior_likelihood_contract": (
            full_vocabulary_behavior_likelihood_contract()
        ),
    }


def _valid_decision(
    *,
    scenario,
    scenario_index: int,
    boundary_index: int,
    sample_index: int,
    seed: int,
) -> dict:
    boundary = scenario.boundaries[boundary_index]
    action = boundary.oracle.oracle_action(boundary.state)
    response = serialize_capability_control_action(
        action,
        capability_reference_map(boundary.state.workers),
    )
    matched, outcome = boundary.oracle.matches(action, boundary.state)
    assert matched is True
    return {
        "boundary_id": boundary.boundary_id,
        "action": json.loads(response),
        "matched_outcome_path": True,
        "transition_outcome": outcome,
        "trace": _trace(
            scenario_index=scenario_index,
            boundary_index=boundary_index,
            sample_index=sample_index,
            seed=seed,
            response=response,
        ),
    }


def _rollout(
    *,
    scenario,
    scenario_index: int,
    sample_index: int,
    seed: int,
    fail_boundary_index: int | None,
) -> dict:
    stop = (
        len(scenario.boundaries) - 1
        if fail_boundary_index is None
        else fail_boundary_index
    )
    decisions = [
        _valid_decision(
            scenario=scenario,
            scenario_index=scenario_index,
            boundary_index=boundary_index,
            sample_index=sample_index,
            seed=seed,
        )
        for boundary_index in range(stop + 1)
    ]
    if fail_boundary_index is None:
        reward = 1.0
        outcome = "task_outcome_verified"
    else:
        decisions[-1] = {
            "boundary_id": scenario.boundaries[
                fail_boundary_index
            ].boundary_id,
            "action": None,
            "matched_outcome_path": False,
            "transition_outcome": (
                "invalid_policy_output:ControlContractError"
            ),
            "trace": _trace(
                scenario_index=scenario_index,
                boundary_index=fail_boundary_index,
                sample_index=sample_index,
                seed=seed,
                response="not-json",
            ),
        }
        reward = 0.0
        outcome = "invalid_policy_output:ControlContractError"
    return {
        "sample_index": sample_index,
        "reward": reward,
        "outcome": outcome,
        "policy": {
            "behavior_policy_revision": POLICY_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "pool_id": POOL_ID,
            "pool_binding_revision": POOL_ID,
            "sampling_seed": seed,
        },
        "decisions": decisions,
    }


def _write_collection(
    path: Path,
    *,
    binding_path: Path,
    samples_per_scenario: int = 6,
    failure_scenarios: frozenset[int] = frozenset({0, 1}),
) -> Path:
    scenario_seed = 700
    binding = load_pool_binding(binding_path)
    scenarios = build_synthetic_curriculum(
        count=2,
        seed=scenario_seed,
        profile_capabilities=tuple(
            slot.role_prior for slot in binding.slots
        ),
    )
    raw_scenarios = []
    success_count = 2 if samples_per_scenario <= 6 else 4
    for scenario_index, scenario in enumerate(scenarios):
        rollouts = []
        for sample_index in range(samples_per_scenario):
            seed = (
                scenario_seed
                + 1_000_003 * scenario_index
                + sample_index
            )
            if (
                scenario_index not in failure_scenarios
                or sample_index < success_count
            ):
                fail_boundary = None
            else:
                fail_boundary = (
                    sample_index - success_count
                ) % len(scenario.boundaries)
            rollouts.append(
                _rollout(
                    scenario=scenario,
                    scenario_index=scenario_index,
                    sample_index=sample_index,
                    seed=seed,
                    fail_boundary_index=fail_boundary,
                )
            )
        reward_counts = dict(
            sorted(
                Counter(
                    str(float(rollout["reward"]))
                    for rollout in rollouts
                ).items()
            )
        )
        raw_scenarios.append(
            {
                "scenario_index": scenario_index,
                "scenario_id": scenario.scenario_id,
                "motif": scenario.motif,
                "evidence_basis": list(scenario.evidence_basis),
                "boundary_count": len(scenario.boundaries),
                "reward_counts": reward_counts,
                "rollouts": rollouts,
            }
        )
    collection = {
        "version": SYNTHETIC_COLLECTION_VERSION,
        "verdict": "SYNTHETIC_EXACT_TOKEN_ROLLOUTS_COLLECTED",
        "behavior_policy_revision": POLICY_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "pool_id": POOL_ID,
        "pool_binding_revision": POOL_ID,
        "pool_binding": str(binding_path),
        "curriculum_revision": SYNTHETIC_CURRICULUM_REVISION,
        "sampling_temperature": 1.0,
        "scenario_seed": scenario_seed,
        "scenario_count": 2,
        "samples_per_scenario": samples_per_scenario,
        "rollout_count": 2 * samples_per_scenario,
        "scenarios": raw_scenarios,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    path.write_text(json.dumps(collection), encoding="utf-8")
    return path.resolve()


def _materialize(
    tmp_path: Path,
    *,
    collection_path: Path,
    binding_path: Path,
    name: str = "prepared",
):
    return materialize_synthetic_grpo_update(
        collection_path=collection_path,
        output_dir=tmp_path / name,
        expected_behavior_policy_revision=POLICY_REVISION,
        expected_runtime_revision=RUNTIME_REVISION,
        pool_binding_path=binding_path,
        replay_path=REPLAY,
        train_retention_replay_path=RETENTION_REPLAY,
        train_retention_report_path=RETENTION_REPORT,
    )


def test_materializes_same_state_first_failure_contrasts_with_replay(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )

    report = _materialize(
        tmp_path,
        collection_path=collection_path,
        binding_path=binding_path,
    )

    batch_path = Path(report["combined_batch"]["path"])
    batch = msgspec.msgpack.decode(
        batch_path.read_bytes(),
        type=TrainingBatch,
    )
    policy_count = report["policy"]["samples"]
    assert report["version"] == SYNTHETIC_BATCH_VERSION
    assert report["optimizer_contract"][
        "policy_credit_assignment"
    ]["mode"] == SYNTHETIC_CREDIT_MODE
    assert MIN_POLICY_SAMPLES <= policy_count <= MAX_POLICY_SAMPLES
    assert len(report["policy"]["scenario_ids"]) == 2
    assert all(
        group["positive_policy_sample_indices"]
        and group["negative_policy_sample_indices"]
        for group in report["credit_groups"]
    )
    assert len(batch.examples) == policy_count + 52 + 76
    assert all(
        not any(sample.replay_mask or [])
        for sample in batch.examples[:policy_count]
    )
    assert sum(
        sum(sample.replay_mask or [])
        for sample in batch.examples[policy_count : policy_count + 52]
    ) == 2_448
    assert sum(
        sum(sample.replay_mask or [])
        for sample in batch.examples[policy_count + 52 :]
    ) == 17_760
    assert report["policy"]["signed_credit_by_action"]["invalid"][
        "negative_samples"
    ] > 0

    validated, validated_path, validated_batch = (
        validate_synthetic_prepared_batch(
            prepared_report_path=(
                tmp_path / "prepared/prepared_report.json"
            ),
            expected_behavior_policy_revision=POLICY_REVISION,
            expected_runtime_revision=RUNTIME_REVISION,
            pool_binding_path=binding_path,
            replay_path=REPLAY,
            train_retention_replay_path=RETENTION_REPLAY,
            train_retention_report_path=RETENTION_REPORT,
        )
    )
    assert validated == report
    assert validated_path == batch_path
    assert validated_batch == batch


def test_caps_deterministic_policy_selection_at_32_rows(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
        samples_per_scenario=12,
    )

    report = _materialize(
        tmp_path,
        collection_path=collection_path,
        binding_path=binding_path,
    )

    assert report["policy"]["samples"] == MAX_POLICY_SAMPLES
    policy_indices = [
        row["policy_sample_index"]
        for group in report["credit_groups"]
        for row in group["source_rows"]
    ]
    assert policy_indices == list(range(MAX_POLICY_SAMPLES))


def test_rejects_same_boundary_prompt_mismatch(tmp_path: Path) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["scenarios"][0]["rollouts"][2]["decisions"][-1][
        "trace"
    ]["prompt_token_ids"] = [999, 999]
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(
        SyntheticBatchError,
        match="different prompts",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_rejects_signal_from_only_one_scenario(tmp_path: Path) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
        failure_scenarios=frozenset({0}),
    )

    with pytest.raises(
        SyntheticBatchError,
        match="at least two scenarios",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_rejects_non_sampled_logprob_evidence(tmp_path: Path) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["scenarios"][0]["rollouts"][0]["decisions"][0][
        "trace"
    ]["completion_logprobs"][0] = 0.5
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(
        SyntheticBatchError,
        match="invalid sampled log-probabilities",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )
