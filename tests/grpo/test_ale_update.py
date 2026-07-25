from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from ultra.ale_training import admit_ale_training_group
from ultra.behavior_likelihood import (
    full_vocabulary_behavior_likelihood_contract,
)

from surogate.grpo.ale_batch import materialize_ale_grpo_update
from surogate.grpo.ale_update import (
    AleUpdateError,
    prepare_ale_optimizer_run,
    register_trained_policy_adapter,
)

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "scratchpad/fugu_27b_transfer_replay_v1/replay.bin"
RETENTION_REPLAY = ROOT / "scratchpad/fugu_27b_train_retention_replay_v1/replay.bin"
RETENTION_REPORT = ROOT / "scratchpad/fugu_27b_train_retention_replay_v1/report.json"
BALANCED_RETENTION_REPLAY = ROOT / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/replay.bin"
BALANCED_RETENTION_REPORT = ROOT / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/report.json"
REVISION = "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"


def _episode(path: Path, *, reward: float, seed: int, action: str, task_id: str) -> Path:
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
        "paid_worker_call_attempts": 1,
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
                "decision": 1,
                "messages": [{"role": "user", "content": "anonymous state"}],
                "response": json.dumps({"action": action}),
                "finish_reason": "stop",
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4, 5],
                "completion_logprobs": [-0.1, -0.2],
                "temperature": 1.0,
                "seed": seed,
                "behavior_likelihood_contract": (
                    full_vocabulary_behavior_likelihood_contract()
                ),
                "action": {
                    "action": action,
                    "target_position_id": 1 if action == "handoff" else None,
                    "steps": [],
                },
            }
        ],
        "routes": [
            {
                "turn": 1,
                "runtime_turn": 1,
                "route_source": ("workflow_agent_continuation" if action == "continue" else "workflow_step_start"),
                "workflow_step_index": 0 if action == "continue" else 1,
                "workflow_access": [] if action == "continue" else [0],
                "worker_id": seed,
                "workflow_step_count": 99,
            }
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _group(
    root: Path,
    *,
    group_id: str,
    task_id: str,
    low_action: str,
    high_action: str,
    seed_offset: int,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    group = admit_ale_training_group(
        episode_paths=[
            _episode(
                root / "low.json",
                reward=0.0,
                seed=10 + seed_offset,
                action=low_action,
                task_id=task_id,
            ),
            _episode(
                root / "high.json",
                reward=1.0,
                seed=11 + seed_offset,
                action=high_action,
                task_id=task_id,
            ),
        ],
        group_id=group_id,
    )
    group_path = root / "group.json"
    group_path.write_text(json.dumps(group), encoding="utf-8")
    return group_path


def _prepared(
    tmp_path: Path,
    *,
    with_retention: bool = False,
    with_balanced_retention: bool = False,
    group_count: int = 2,
    directional_balance: bool = True,
) -> Path:
    group_paths = [
        _group(
            tmp_path / "group-1",
            group_id="group-1",
            task_id="domain/task-1",
            low_action="continue",
            high_action="handoff",
            seed_offset=0,
        )
    ]
    if group_count == 2:
        group_paths.append(
            _group(
                tmp_path / "group-2",
                group_id="group-2",
                task_id="domain/task-2",
                low_action="handoff" if directional_balance else "continue",
                high_action="continue" if directional_balance else "handoff",
                seed_offset=10,
            )
        )
    if with_retention and with_balanced_retention:
        raise ValueError("select only one retention replay")
    retention = (
        {
            "train_retention_replay_path": RETENTION_REPLAY,
            "train_retention_replay_samples": 76,
            "train_retention_report_path": RETENTION_REPORT,
        }
        if with_retention
        else {}
    )
    if with_balanced_retention:
        retention = {
            "train_retention_replay_path": BALANCED_RETENTION_REPLAY,
            "train_retention_replay_samples": 76,
            "train_retention_report_path": BALANCED_RETENTION_REPORT,
        }
    materialize_ale_grpo_update(
        group_paths=group_paths,
        output_dir=tmp_path / "prepared",
        expected_behavior_policy_revision=REVISION,
        replay_path=REPLAY,
        **retention,
    )
    return tmp_path / "prepared/prepared_report.json"


def _fake_loaded_config() -> SimpleNamespace:
    return SimpleNamespace(
        max_steps=1,
        gpus=6,
        sequence_len=2_816,
        resume_from_checkpoint=False,
        lora_dtype="fp32",
        master_dtype="bf16",
        gradient_dtype="bf16",
        loss=SimpleNamespace(adv_tau=1.0, replay_tau=0.05, kl_tau=0.001),
    )


def test_rejects_optimizer_staging_without_mandatory_action_balanced_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(AleUpdateError, match="mandatory action-balanced train-retention replay is absent"):
        prepare_ale_optimizer_run(
            prepared_report_path=_prepared(tmp_path),
            output_dir=tmp_path / "optimizer",
            model_path=model,
            behavior_policy_revision=REVISION,
            require_six_gpus=False,
            require_no_stale_process=False,
        )


def test_rejects_legacy_train_only_retention_for_optimizer_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(AleUpdateError, match="mandatory action-balanced train-retention replay changed"):
        prepare_ale_optimizer_run(
            prepared_report_path=_prepared(tmp_path, with_retention=True),
            output_dir=tmp_path / "optimizer-retention",
            model_path=model,
            behavior_policy_revision=REVISION,
            require_six_gpus=False,
            require_no_stale_process=False,
        )


def test_stages_action_balanced_retention_with_exact_weight_mass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    report = prepare_ale_optimizer_run(
        prepared_report_path=_prepared(tmp_path, with_balanced_retention=True),
        output_dir=tmp_path / "optimizer-balanced-retention",
        model_path=model,
        behavior_policy_revision=REVISION,
        require_six_gpus=False,
        require_no_stale_process=False,
    )

    retention = report["optimizer_contract"]["train_only_retention_replay"]
    assert report["version"] == "fugu_ale_optimizer_run_v3"
    assert report["pool_id"] == "yunwu-test-pool-v1"
    assert report["pool_binding_revision"] == "yunwu-test-pool-v1"
    assert retention["weighting_mode"] == "equal_action_completion_token_mass_v1"
    assert retention["selected_weight_sum"] == pytest.approx(17_760.0, abs=1e-3)
    assert report["optimizer_contract"]["policy_credit_assignment"] == {
        "mode": "first_executed_coordination_divergence",
        "policy_attempts": "initial_only",
        "groups": [
            {"group_id": "group-1", "decision_indices": [0, 0]},
            {"group_id": "group-2", "decision_indices": [0, 0]},
        ],
    }
    assert report["optimizer_contract"]["signed_credit_by_action"]["continue"] == {
        "negative_samples": 1,
        "negative_tokens": 2,
        "positive_samples": 1,
        "positive_tokens": 2,
    }
    assert report["optimizer_contract"]["lora_dtype"] == "fp32"
    assert report["optimizer_contract"]["master_dtype"] == "bf16"
    assert report["optimizer_contract"]["gradient_dtype"] == "bf16"
    train = yaml.safe_load(
        (tmp_path / "optimizer-balanced-retention/train.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert train["lora_dtype"] == "fp32"
    assert train["master_dtype"] == "bf16"
    assert train["gradient_dtype"] == "bf16"
    assert report["verdict"] == "READY_TO_RUN"
    assert (tmp_path / "optimizer-balanced-retention/optimizer_run.json").is_file()


def test_optimizer_requires_prepared_semantic_pool_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )
    for case in ("missing", "mismatch"):
        root = tmp_path / case
        prepared = _prepared(root, with_balanced_retention=True)
        prepared_record = json.loads(prepared.read_text(encoding="utf-8"))
        if case == "missing":
            prepared_record.pop("pool_id")
            expected = "lacks semantic pool identity"
        else:
            prepared_record["groups"][1]["pool_binding_revision"] = "different-pool-v2"
            expected = "one semantic pool identity"
        prepared.write_text(json.dumps(prepared_record), encoding="utf-8")
        model = root / REVISION
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")

        with pytest.raises(AleUpdateError, match=expected):
            prepare_ale_optimizer_run(
                prepared_report_path=prepared,
                output_dir=root / "optimizer",
                model_path=model,
                behavior_policy_revision=REVISION,
                require_six_gpus=False,
                require_no_stale_process=False,
            )


def test_optimizer_rejects_stale_behavior_likelihood_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared(tmp_path, with_balanced_retention=True)
    prepared_record = json.loads(prepared.read_text(encoding="utf-8"))
    prepared_record["groups"][0].pop("behavior_likelihood_contract")
    prepared.write_text(json.dumps(prepared_record), encoding="utf-8")
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(AleUpdateError, match="behavior-likelihood contract changed"):
        prepare_ale_optimizer_run(
            prepared_report_path=prepared,
            output_dir=tmp_path / "optimizer",
            model_path=model,
            behavior_policy_revision=REVISION,
            require_six_gpus=False,
            require_no_stale_process=False,
        )


def test_rejects_single_group_optimizer_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(AleUpdateError, match="at least two causal ALE groups"):
        prepare_ale_optimizer_run(
            prepared_report_path=_prepared(
                tmp_path,
                with_balanced_retention=True,
                group_count=1,
            ),
            output_dir=tmp_path / "optimizer-single-group",
            model_path=model,
            behavior_policy_revision=REVISION,
            require_six_gpus=False,
            require_no_stale_process=False,
        )


def test_rejects_one_sided_noncomplete_action_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(
        AleUpdateError,
        match="insufficient directional evidence.*continue, handoff",
    ):
        prepare_ale_optimizer_run(
            prepared_report_path=_prepared(
                tmp_path,
                with_balanced_retention=True,
                directional_balance=False,
            ),
            output_dir=tmp_path / "optimizer-one-sided",
            model_path=model,
            behavior_policy_revision=REVISION,
            require_six_gpus=False,
            require_no_stale_process=False,
        )


def test_registers_trained_adapter_with_explicit_semantic_revision(
    tmp_path: Path,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    optimizer_run = tmp_path / "optimizer_run.json"
    optimizer_run.write_text(
        json.dumps(
            {
                "version": "fugu_ale_optimizer_run_v3",
                "verdict": "READY_TO_RUN",
                "pool_id": "yunwu-test-pool-v1",
                "pool_binding_revision": "yunwu-test-pool-v1",
            }
        ),
        encoding="utf-8",
    )

    manifest = register_trained_policy_adapter(
        adapter_dir=adapter,
        policy_revision="fugu-ale-r2",
        parent_policy_revision="fugu-ale-r1",
        optimizer_run_path=optimizer_run,
        optimizer_step=2,
    )

    assert manifest["policy_revision"] == "fugu-ale-r2"
    assert manifest["parent_policy_revision"] == "fugu-ale-r1"
    assert manifest["optimizer_step"] == 2
    assert "sha256" not in json.dumps(manifest).lower()


def test_rejects_reused_policy_revision(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    optimizer_run = tmp_path / "optimizer_run.json"
    optimizer_run.write_text(json.dumps({"verdict": "READY_TO_RUN"}), encoding="utf-8")

    with pytest.raises(AleUpdateError, match="new explicit fugu-ale revision"):
        register_trained_policy_adapter(
            adapter_dir=adapter,
            policy_revision="fugu-ale-r1",
            parent_policy_revision="fugu-ale-r1",
            optimizer_run_path=optimizer_run,
            optimizer_step=2,
        )


def test_rejects_noncausal_prepared_credit_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared(tmp_path, with_balanced_retention=True)
    report = json.loads(prepared.read_text(encoding="utf-8"))
    report["optimizer_contract"]["policy_credit_assignment"] = {"mode": "whole_trajectory"}
    prepared.write_text(json.dumps(report), encoding="utf-8")
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(AleUpdateError, match="policy credit assignment"):
        prepare_ale_optimizer_run(
            prepared_report_path=prepared,
            output_dir=tmp_path / "optimizer-noncausal",
            model_path=model,
            behavior_policy_revision=REVISION,
            require_six_gpus=False,
            require_no_stale_process=False,
        )


def test_rejects_prepared_credit_without_initial_attempt_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path, with_balanced_retention=True)
    report = json.loads(prepared.read_text(encoding="utf-8"))
    report["optimizer_contract"]["policy_credit_assignment"].pop("policy_attempts")
    prepared.write_text(json.dumps(report), encoding="utf-8")
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(AleUpdateError, match="policy credit assignment"):
        prepare_ale_optimizer_run(
            prepared_report_path=prepared,
            output_dir=tmp_path / "optimizer-correction-credit",
            model_path=model,
            behavior_policy_revision=REVISION,
            require_six_gpus=False,
            require_no_stale_process=False,
        )


def _prepared_report_text(tmp_path: Path) -> str:
    return (tmp_path / "prepared/prepared_report.json").read_text(encoding="utf-8")


def test_rejects_base_revision_or_learning_rate_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared(tmp_path)
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "surogate.grpo.ale_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )

    with pytest.raises(AleUpdateError, match="base behavior-policy revision"):
        prepare_ale_optimizer_run(
            prepared_report_path=prepared,
            output_dir=tmp_path / "wrong-revision",
            model_path=model,
            behavior_policy_revision="wrong",
            require_six_gpus=False,
            require_no_stale_process=False,
        )
    with pytest.raises(AleUpdateError, match="learning rate"):
        prepare_ale_optimizer_run(
            prepared_report_path=prepared,
            output_dir=tmp_path / "high-lr",
            model_path=model,
            behavior_policy_revision=REVISION,
            learning_rate=2.0e-4,
            require_six_gpus=False,
            require_no_stale_process=False,
        )
