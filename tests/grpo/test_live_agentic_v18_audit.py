from __future__ import annotations

import json
from pathlib import Path

import msgspec

from scratchpad import audit_fugu_live_agentic_grpo_v18 as audit
from surogate.grpo.transport.types import TrainingBatch, TrainingSample


def _result(rollout_id: str, reward: float, *, provider: str | None = None) -> dict:
    metadata = {
        "grpo_bridge_version": audit.env_mod.BRIDGE_VERSION,
        "grpo_rollout_id": rollout_id,
        "grpo_external_live_controller": True,
        "grpo_control_protocol": "compact_live_decision_v1",
        "grpo_control_requests": 1,
        "grpo_control_responses": 1,
        "worker_provider_base": provider or audit.env_mod.YUNWU_API_BASE,
        "worker_models": list(audit.env_mod.DEFAULT_WORKER_MODELS),
        "provider_owner_retry_limit": 0,
        "provider_owner_retries": 0,
        "max_agent_turns": audit.env_mod.MAX_AGENT_TURNS,
        "fair_position_call_budget": None,
        "paid_worker_call_attempts": 3,
        "runtime_revision": audit.env_mod.PRODUCT_RUNTIME_REVISION,
        "workspace_snapshot_ready": True,
        "workspace_root": "/testbed",
        "live_control_failures": 0,
        "grpo_worker_timeout_s": 600.0,
        "protected_test_restore_policy": audit.env_mod.PROTECTED_TEST_POLICY,
        "protected_test_repo": "/testbed",
        "protected_test_snapshot_entries": 2,
        "protected_test_restores": [],
    }
    return {
        "agent_result": {"metadata": metadata},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": None,
    }


def _rollout(root: Path, rollout_id: str, reward: float, **kwargs) -> None:
    rollout = root / rollout_id
    control = rollout / "control"
    result_dir = rollout / "harbor/rollout/trial"
    control.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    (rollout / "harbor.log").write_text("finished\n", encoding="utf-8")
    for prefix in ("request", "response"):
        payload = {
            "version": audit.env_mod.BRIDGE_VERSION,
            "rollout_id": rollout_id,
            "request_id": 1,
        }
        if prefix == "response":
            payload["completion"] = '{"action":"complete"}'
        (control / f"{prefix}_0001.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    (result_dir / "result.json").write_text(
        json.dumps(_result(rollout_id, reward, **kwargs)), encoding="utf-8"
    )


def _sample(reward: float, advantage: float) -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3],
        completion_mask=[True],
        completion_logprobs=[-0.1],
        completion_temperatures=[1.0],
        teacher_logprobs=[-0.2, -0.2, -0.2],
        advantage=advantage,
        reward=reward,
    )


def _batch(path: Path, rewards: tuple[float, float]) -> None:
    advantages = (-0.707, 0.707) if len(set(rewards)) == 2 else (0.0, 0.0)
    batch = TrainingBatch(
        examples=[
            _sample(rewards[0], advantages[0]),
            _sample(rewards[1], advantages[1]),
        ],
        step=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(msgspec.msgpack.encode(batch))


def test_group_audit_allows_one_step_only_for_two_valid_different_rewards(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _rollout(artifacts, "rollout-a", 0.0)
    _rollout(artifacts, "rollout-b", 1.0)
    batch = tmp_path / "rollouts.bin"
    _batch(batch, (0.0, 1.0))

    report = audit.audit_group(artifact_root=artifacts, batch_path=batch)

    assert report["verdict"] == "train_one_step"
    assert report["eligible_for_optimizer_step"] is True
    assert report["total_paid_calls"] == 6


def test_group_audit_stops_valid_equal_rewards_before_training(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _rollout(artifacts, "rollout-a", 1.0)
    _rollout(artifacts, "rollout-b", 1.0)
    batch = tmp_path / "rollouts.bin"
    _batch(batch, (1.0, 1.0))

    report = audit.audit_group(artifact_root=artifacts, batch_path=batch)

    assert report["verdict"] == "stop_zero_variance"
    assert report["valid_zero_variance"] is True
    assert report["eligible_for_optimizer_step"] is False


def test_group_audit_rejects_non_yunwu_evidence(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _rollout(artifacts, "rollout-a", 0.0, provider="https://other.invalid/v1")
    _rollout(artifacts, "rollout-b", 1.0)
    batch = tmp_path / "rollouts.bin"
    _batch(batch, (0.0, 1.0))

    report = audit.audit_group(artifact_root=artifacts, batch_path=batch)

    assert report["verdict"] == "stop_invalid_group"
    assert report["eligible_for_optimizer_step"] is False
    assert report["checks"]["both_rollouts_trainable"] is False
