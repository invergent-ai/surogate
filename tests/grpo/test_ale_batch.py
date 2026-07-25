from __future__ import annotations

import json
from pathlib import Path

import msgspec
import pytest
from ultra.ale_training import admit_ale_training_group
from ultra.behavior_likelihood import (
    FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION,
    full_vocabulary_behavior_likelihood_contract,
)

from surogate.grpo.ale_batch import AleBatchError, materialize_ale_grpo_update
from surogate.grpo.transport import TrainingBatch

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "scratchpad/fugu_27b_transfer_replay_v1/replay.bin"
RETENTION_REPLAY = ROOT / "scratchpad/fugu_27b_train_retention_replay_v1/replay.bin"
RETENTION_REPORT = ROOT / "scratchpad/fugu_27b_train_retention_replay_v1/report.json"
BALANCED_RETENTION_REPLAY = ROOT / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/replay.bin"
BALANCED_RETENTION_REPORT = ROOT / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/report.json"
REVISION = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"


def _episode(
    path: Path,
    *,
    reward: float,
    seed: int,
    task_id: str = "domain/task",
    prompt_tokens: int = 3,
    actions: tuple[str, ...] = ("continue",),
) -> Path:
    value = {
        "schema_version": 3,
        "record_type": "ale_train_episode_outcome",
        "task_id": task_id,
        "task_family": "family",
        "split": "train",
        "source_commit": "ale-commit",
        "pool_id": "yunwu-test-pool-v1",
        "pool_binding_revision": "yunwu-test-pool-v1",
        "runtime_revision": "runtime-r1",
        "whole_task_reward": reward,
        "paid_worker_call_attempts": 2,
        "behavior_policy": {
            "conductor_model": "fugu-27b-conductor",
            "conductor_url": "http://localhost:8010/v1",
            "revision": REVISION,
            "temperature": 1.0,
            "seed": seed,
            "records_token_data": True,
            "behavior_likelihood_contract": (
                full_vocabulary_behavior_likelihood_contract()
            ),
        },
        "decisions": [
            {
                "decision": index,
                "messages": [{"role": "user", "content": "anonymous state"}],
                "response": json.dumps({"action": action}),
                "finish_reason": "stop",
                "prompt_token_ids": list(range(prompt_tokens)),
                "completion_token_ids": [100 + index * 2 - 1, 100 + index * 2],
                "completion_logprobs": [-0.1, -0.2],
                "temperature": 1.0,
                "seed": seed,
                "behavior_likelihood_contract": (
                    full_vocabulary_behavior_likelihood_contract()
                ),
                "action": {
                    "action": action,
                    "target_position_id": index if action == "handoff" else None,
                    "steps": [],
                },
            }
            for index, action in enumerate(actions, start=1)
        ],
        "routes": [
            {
                "turn": index,
                "route_source": ("workflow_agent_continuation" if action == "continue" else "workflow_step_start"),
                "workflow_step_index": 0 if action == "continue" else index,
                "workflow_access": list(range(index - 1)) if action != "continue" else [],
                "worker_id": seed,
                "workflow_step_count": len(actions),
                "subtask": f"wording-{seed}-{index}",
            }
            for index, action in enumerate(actions, start=1)
            if action != "complete"
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _group(
    tmp_path: Path,
    *,
    oversized: bool = False,
    group_id: str = "group-1",
    task_id: str = "domain/task",
    low_action: str = "continue",
    high_action: str = "handoff",
    seed_offset: int = 0,
) -> Path:
    low = _episode(
        tmp_path / "low.json",
        reward=0.0,
        seed=10 + seed_offset,
        task_id=task_id,
        prompt_tokens=8_191 if oversized else 3,
        actions=(low_action,),
    )
    high = _episode(
        tmp_path / "high.json",
        reward=1.0,
        seed=11 + seed_offset,
        task_id=task_id,
        actions=(high_action,),
    )
    group = admit_ale_training_group(
        episode_paths=[low, high],
        group_id=group_id,
    )
    path = tmp_path / "group.json"
    path.write_text(json.dumps(group), encoding="utf-8")
    return path


def test_materializes_exact_policy_tokens_with_all_mandatory_replay(tmp_path: Path):
    report = materialize_ale_grpo_update(
        group_path=_group(tmp_path),
        output_dir=tmp_path / "update",
        expected_behavior_policy_revision=REVISION,
        replay_path=REPLAY,
    )

    batch_path = Path(report["combined_batch"]["path"])
    batch = msgspec.msgpack.decode(batch_path.read_bytes(), type=TrainingBatch)
    assert report["version"] == "fugu_ale_exact_grpo_batch_v5"
    assert report["verdict"] == "ALE_EXACT_TOKEN_REPLAY_ANCHORED_GRPO_BATCH_READY"
    assert report["pool_id"] == "yunwu-test-pool-v1"
    assert report["pool_binding_revision"] == "yunwu-test-pool-v1"
    assert report["behavior_likelihood_contract_version"] == (
        FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION
    )
    assert report["optimizer_contract"]["behavior_likelihood_contract"] == (
        full_vocabulary_behavior_likelihood_contract()
    )
    assert report["groups"][0]["pool_id"] == report["pool_id"]
    assert report["groups"][0]["pool_binding_revision"] == report["pool_binding_revision"]
    assert report["mandatory_replay"] == {
        "path": str(REPLAY.resolve()),
        "samples": 52,
        "selected_tokens": 2_448,
        "selected_weight_sum": 2_448.0,
    }
    assert len(batch.examples) == 54
    assert batch.examples[0].prompt_ids == [0, 1, 2]
    assert batch.examples[0].completion_ids == [101, 102]
    assert batch.examples[0].completion_logprobs == [-0.1, -0.2]
    assert batch.examples[0].completion_mask == [True, True]
    assert batch.examples[0].advantage < 0
    assert batch.examples[1].advantage > 0
    assert report["optimizer_contract"]["policy_credit_assignment"] == {
        "mode": "first_executed_coordination_divergence",
        "policy_attempts": "initial_only",
        "groups": [{"group_id": "group-1", "decision_indices": [0, 0]}],
    }
    assert "sha256" not in report["groups"][0]
    assert "sha256" not in report["combined_batch"]
    assert all(len(sample.prompt_ids) + len(sample.completion_ids) <= 2_816 for sample in batch.examples)


def test_aggregates_causal_groups_and_attaches_each_replay_once(tmp_path: Path) -> None:
    first = _group(
        tmp_path / "first",
        group_id="group-1",
        task_id="domain/task-1",
    )
    second = _group(
        tmp_path / "second",
        group_id="group-2",
        task_id="domain/task-2",
        low_action="handoff",
        high_action="continue",
        seed_offset=10,
    )

    report = materialize_ale_grpo_update(
        group_paths=[first, second],
        output_dir=tmp_path / "multi-group",
        expected_behavior_policy_revision=REVISION,
        replay_path=REPLAY,
        train_retention_replay_path=BALANCED_RETENTION_REPLAY,
        train_retention_replay_samples=76,
        train_retention_report_path=BALANCED_RETENTION_REPORT,
    )

    batch = msgspec.msgpack.decode(Path(report["combined_batch"]["path"]).read_bytes(), type=TrainingBatch)
    assert len(report["groups"]) == 2
    assert report["policy"]["samples"] == 4
    assert len(batch.examples) == 4 + 52 + 76
    assert sum(sum(sample.replay_mask or []) for sample in batch.examples[4:56]) == 2_448
    assert sum(sum(sample.replay_mask or []) for sample in batch.examples[56:]) == 17_760
    assert report["policy"]["signed_credit_by_action"]["continue"] == {
        "negative_samples": 1,
        "negative_tokens": 2,
        "positive_samples": 1,
        "positive_tokens": 2,
    }
    assert report["policy"]["signed_credit_by_action"]["handoff"] == {
        "negative_samples": 1,
        "negative_tokens": 2,
        "positive_samples": 1,
        "positive_tokens": 2,
    }


def test_rejects_duplicate_group_or_task_records(tmp_path: Path) -> None:
    first = _group(
        tmp_path / "first",
        group_id="group-1",
        task_id="domain/task-1",
    )
    duplicate_task = _group(
        tmp_path / "second",
        group_id="group-2",
        task_id="domain/task-1",
        seed_offset=10,
    )
    with pytest.raises(AleBatchError, match="task IDs must be unique"):
        materialize_ale_grpo_update(
            group_paths=[first, duplicate_task],
            output_dir=tmp_path / "duplicate-task",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )


def test_requires_current_schemas(tmp_path: Path) -> None:
    stale_group = _group(tmp_path / "stale-group")
    stale_group_record = json.loads(stale_group.read_text(encoding="utf-8"))
    stale_group_record["schema_version"] = 3
    stale_group.write_text(json.dumps(stale_group_record), encoding="utf-8")
    with pytest.raises(AleBatchError, match="not an admitted ALE GRPO group"):
        materialize_ale_grpo_update(
            group_path=stale_group,
            output_dir=tmp_path / "stale-group-update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )

    stale_episode_group = _group(tmp_path / "stale-episode")
    stale_episode_record = json.loads(stale_episode_group.read_text(encoding="utf-8"))
    stale_episode_path = Path(stale_episode_record["episodes"][0]["episode_path"])
    stale_episode = json.loads(stale_episode_path.read_text(encoding="utf-8"))
    stale_episode["schema_version"] = 2
    stale_episode_path.write_text(json.dumps(stale_episode), encoding="utf-8")
    with pytest.raises(AleBatchError, match="non-schema-3 episode"):
        materialize_ale_grpo_update(
            group_path=stale_episode_group,
            output_dir=tmp_path / "stale-episode-update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )


def test_rejects_correction_attempt_credit_even_if_group_was_already_admitted(
    tmp_path: Path,
) -> None:
    group_path = _group(tmp_path)
    group = json.loads(group_path.read_text(encoding="utf-8"))
    episode_path = Path(group["episodes"][0]["episode_path"])
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["decisions"][0]["messages"] = [
        {"role": "system", "content": "control contract"},
        {"role": "user", "content": "live state"},
        {
            "role": "user",
            "content": (
                "Correction attempt 1 of 2.\n"
                "Your previous conductor action was rejected."
            ),
        },
    ]
    episode_path.write_text(json.dumps(episode), encoding="utf-8")

    with pytest.raises(AleBatchError, match="cannot credit a correction attempt"):
        materialize_ale_grpo_update(
            group_path=group_path,
            output_dir=tmp_path / "correction-credit-update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )


def test_rejects_old_constrained_rollout_contract_at_group_and_decision(
    tmp_path: Path,
) -> None:
    missing_group_contract = _group(tmp_path / "missing-group-contract")
    group_record = json.loads(missing_group_contract.read_text(encoding="utf-8"))
    group_record.pop("behavior_likelihood_contract")
    missing_group_contract.write_text(json.dumps(group_record), encoding="utf-8")
    with pytest.raises(AleBatchError, match="group lacks the full-vocabulary"):
        materialize_ale_grpo_update(
            group_path=missing_group_contract,
            output_dir=tmp_path / "missing-group-contract-update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )

    missing_decision_contract = _group(tmp_path / "missing-decision-contract")
    group_record = json.loads(missing_decision_contract.read_text(encoding="utf-8"))
    episode_path = Path(group_record["episodes"][0]["episode_path"])
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["decisions"][0].pop("behavior_likelihood_contract")
    episode_path.write_text(json.dumps(episode), encoding="utf-8")
    with pytest.raises(AleBatchError, match="policy decision lacks the full-vocabulary"):
        materialize_ale_grpo_update(
            group_path=missing_decision_contract,
            output_dir=tmp_path / "missing-decision-contract-update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )


def test_requires_one_semantic_pool_identity(tmp_path: Path) -> None:
    missing_identity = _group(tmp_path / "missing-identity")
    missing_record = json.loads(missing_identity.read_text(encoding="utf-8"))
    missing_record.pop("pool_binding_revision")
    missing_identity.write_text(json.dumps(missing_record), encoding="utf-8")
    with pytest.raises(AleBatchError, match="lacks semantic pool identity"):
        materialize_ale_grpo_update(
            group_path=missing_identity,
            output_dir=tmp_path / "missing-identity-update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )

    first = _group(
        tmp_path / "pool-first",
        group_id="pool-group-1",
        task_id="domain/pool-task-1",
    )
    second = _group(
        tmp_path / "pool-second",
        group_id="pool-group-2",
        task_id="domain/pool-task-2",
        seed_offset=10,
    )
    second_record = json.loads(second.read_text(encoding="utf-8"))
    second_record["pool_binding_revision"] = "different-pool-v2"
    second.write_text(json.dumps(second_record), encoding="utf-8")
    with pytest.raises(AleBatchError, match="share one semantic pool identity"):
        materialize_ale_grpo_update(
            group_paths=[first, second],
            output_dir=tmp_path / "mixed-pool-update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )
    with pytest.raises(AleBatchError, match="group paths must be unique"):
        materialize_ale_grpo_update(
            group_paths=[first, first],
            output_dir=tmp_path / "duplicate-group",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )


def test_appends_train_only_retention_after_mandatory_replay(tmp_path: Path) -> None:
    report = materialize_ale_grpo_update(
        group_path=_group(tmp_path),
        output_dir=tmp_path / "update-retention",
        expected_behavior_policy_revision=REVISION,
        replay_path=REPLAY,
        train_retention_replay_path=RETENTION_REPLAY,
        train_retention_replay_samples=76,
        train_retention_report_path=RETENTION_REPORT,
    )

    batch = msgspec.msgpack.decode(Path(report["combined_batch"]["path"]).read_bytes(), type=TrainingBatch)
    assert len(batch.examples) == 130
    assert batch.examples[:2][0].replay_mask == [False] * 5
    assert sum(sum(sample.replay_mask or []) for sample in batch.examples[2:54]) == 2_448
    assert sum(sum(sample.replay_mask or []) for sample in batch.examples[54:]) == 17_760
    assert all(sample.completion_logprobs == [0.0] * len(sample.completion_ids) for sample in batch.examples[54:])
    assert report["optimizer_contract"]["replay_reference_mode"] == ("ce_only_no_behavior_ratio_or_kl")
    assert report["train_only_retention_replay"] == {
        "path": str(RETENTION_REPLAY.resolve()),
        "samples": 76,
        "selected_tokens": 17_760,
        "selected_weight_sum": 17_760.0,
        "report": str(RETENTION_REPORT.resolve()),
        "reference_mode": "ce_only_no_behavior_ratio_or_kl",
        "weighting_mode": "uniform_per_token",
        "replay_version": "fugu_27b_train_retention_replay_v1",
        "train_tasks": 49,
        "validation_tasks_excluded": 11,
    }


def test_rejects_partial_train_only_retention_contract(tmp_path: Path) -> None:
    with pytest.raises(AleBatchError, match="requires path, count, and report"):
        materialize_ale_grpo_update(
            group_path=_group(tmp_path),
            output_dir=tmp_path / "partial-retention",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
            train_retention_replay_path=RETENTION_REPLAY,
        )


def test_appends_explicit_action_token_balanced_retention_weights(tmp_path: Path) -> None:
    report = materialize_ale_grpo_update(
        group_path=_group(tmp_path),
        output_dir=tmp_path / "update-balanced-retention",
        expected_behavior_policy_revision=REVISION,
        replay_path=REPLAY,
        train_retention_replay_path=BALANCED_RETENTION_REPLAY,
        train_retention_replay_samples=76,
        train_retention_report_path=BALANCED_RETENTION_REPORT,
    )

    batch = msgspec.msgpack.decode(Path(report["combined_batch"]["path"]).read_bytes(), type=TrainingBatch)
    retention = batch.examples[54:]
    mass_by_weight: dict[float, float] = {}
    for sample in retention:
        assert sample.replay_mask is not None and sample.replay_weights is not None
        weight = next(weight for replay, weight in zip(sample.replay_mask, sample.replay_weights) if replay)
        mass_by_weight[weight] = mass_by_weight.get(weight, 0.0) + weight * sum(sample.replay_mask)
    assert len(mass_by_weight) == 4
    assert list(mass_by_weight.values()) == pytest.approx([4_440.0] * 4, abs=1e-3)
    assert report["train_only_retention_replay"]["selected_tokens"] == 17_760
    assert report["train_only_retention_replay"]["selected_weight_sum"] == pytest.approx(17_760.0, abs=1e-3)
    assert report["train_only_retention_replay"]["weighting_mode"] == ("equal_action_completion_token_mass_v1")
    assert report["train_only_retention_replay"]["replay_version"] == ("fugu_27b_action_balanced_retention_replay_v2")


def test_rejects_policy_revision_drift(tmp_path: Path):
    with pytest.raises(AleBatchError, match="share the behavior-policy revision"):
        materialize_ale_grpo_update(
            group_path=_group(tmp_path),
            output_dir=tmp_path / "update",
            expected_behavior_policy_revision="different",
            replay_path=REPLAY,
        )


def test_rejects_policy_decision_outside_optimizer_window(tmp_path: Path):
    with pytest.raises(AleBatchError, match="exceeding 2816"):
        materialize_ale_grpo_update(
            group_path=_group(tmp_path, oversized=True),
            output_dir=tmp_path / "update",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
        )


def test_rejects_disabled_kl_or_replay_anchor(tmp_path: Path):
    group = _group(tmp_path)
    with pytest.raises(AleBatchError, match="kl_tau"):
        materialize_ale_grpo_update(
            group_path=group,
            output_dir=tmp_path / "update-kl",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
            kl_tau=0.0,
        )
    with pytest.raises(AleBatchError, match="replay_tau"):
        materialize_ale_grpo_update(
            group_path=group,
            output_dir=tmp_path / "update-replay",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
            replay_tau=0.0,
        )


def test_first_divergence_credit_keeps_exact_tokens_and_zeros_shared_decisions(
    tmp_path: Path,
) -> None:
    low = _episode(
        tmp_path / "low-actions.json",
        reward=0.0,
        seed=20,
        actions=("continue", "replan", "continue"),
    )
    high = _episode(
        tmp_path / "high-actions.json",
        reward=1.0,
        seed=21,
        actions=("continue", "handoff", "replan"),
    )
    group = admit_ale_training_group(
        episode_paths=[low, high],
        group_id="action-credit-group",
    )
    group_path = tmp_path / "action-credit-group.json"
    group_path.write_text(json.dumps(group), encoding="utf-8")

    report = materialize_ale_grpo_update(
        group_path=group_path,
        output_dir=tmp_path / "causal-credit",
        expected_behavior_policy_revision=REVISION,
        replay_path=REPLAY,
    )
    batch = msgspec.msgpack.decode(
        Path(report["combined_batch"]["path"]).read_bytes(),
        type=TrainingBatch,
    )

    assert len(batch.examples) == 58
    assert [sample.advantage for sample in batch.examples[:6]] == [
        0.0,
        pytest.approx(group["episodes"][0]["group_advantage"]),
        0.0,
        0.0,
        pytest.approx(group["episodes"][1]["group_advantage"]),
        0.0,
    ]
    assert report["optimizer_contract"]["policy_credit_assignment"] == {
        "mode": "first_executed_coordination_divergence",
        "policy_attempts": "initial_only",
        "groups": [{"group_id": "action-credit-group", "decision_indices": [1, 1]}],
    }
    assert report["policy"]["credited_samples"] == 2
    assert report["policy"]["credited_conductor_tokens"] == 4


def test_rejects_empty_or_one_sided_manual_action_filter(tmp_path: Path) -> None:
    group = _group(tmp_path)
    with pytest.raises(AleBatchError, match="must not be empty"):
        materialize_ale_grpo_update(
            group_path=group,
            output_dir=tmp_path / "empty-credit",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
            credited_actions=(),
        )
    with pytest.raises(AleBatchError, match="signed outcome variation"):
        materialize_ale_grpo_update(
            group_path=group,
            output_dir=tmp_path / "no-signal-credit",
            expected_behavior_policy_revision=REVISION,
            replay_path=REPLAY,
            credited_actions=("continue",),
        )
