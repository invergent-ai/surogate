"""Prepare one conservative replay-anchored ALE optimizer update."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec
import torch
import yaml

from surogate.core.config.loader import load_config
from surogate.grpo.ale_batch import (
    ACTION_BALANCED_RETENTION_REPLAY_VERSION,
    ACTION_BALANCED_RETENTION_SAMPLES,
    ACTION_BALANCED_RETENTION_SELECTED_TOKENS,
    CONTROL_ACTIONS,
    FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION,
    PROVEN_SEQUENCE_LEN,
    REPLAY_REFERENCE_MODE,
    TRAIN_RETENTION_REPLAY_VERSIONS,
    full_vocabulary_behavior_likelihood_contract,
    has_full_vocabulary_behavior_likelihood_contract,
)
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.transport import TrainingBatch, TrainingSample

DATA_PARALLEL_GPUS = 6
MAX_LEARNING_RATE = 1.0e-4
POLICY_MANIFEST = "fugu_policy_revision.json"


class AleUpdateError(ValueError):
    """An ALE optimizer update is not safe to stage or promote."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AleUpdateError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AleUpdateError(f"{path} must contain one JSON object")
    return value


def _selected_replay_weight_sum(samples: list[TrainingSample]) -> float:
    total = 0.0
    for sample in samples:
        replay_mask = sample.replay_mask or []
        replay_weights = sample.replay_weights or [1.0] * len(replay_mask)
        total += sum(float(weight) for replay, weight in zip(replay_mask, replay_weights) if replay)
    return total


def _cuda_inventory() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
        for index in range(torch.cuda.device_count())
    ]


def _assert_no_stale_grpo_process() -> None:
    result = subprocess.run(
        ["pgrep", "-af", r"surogate\.cli\.grpo_(orch|train)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise AleUpdateError("cannot audit active GRPO processes")
    if result.returncode == 0 and result.stdout.strip():
        raise AleUpdateError(f"another GRPO process is active: {result.stdout.strip()}")


def _validate_parent_policy(
    *,
    model_path: Path,
    behavior_policy_revision: str,
    parent_adapter: Path | None,
) -> dict[str, Any]:
    if parent_adapter is None:
        if behavior_policy_revision != model_path.name:
            raise AleUpdateError("base behavior-policy revision does not match the model snapshot")
        return {"kind": "base", "revision": behavior_policy_revision}

    manifest_path = parent_adapter / POLICY_MANIFEST
    manifest = _read_json(manifest_path)
    if manifest.get("version") not in {
        "fugu_ale_policy_revision_v1",
        "fugu_ale_policy_revision_v2",
        "fugu_conductor_policy_revision_v1",
    }:
        raise AleUpdateError("parent adapter policy manifest has the wrong version")
    if manifest.get("policy_revision") != behavior_policy_revision:
        raise AleUpdateError("parent adapter policy revision differs from collection")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = parent_adapter / name
        if not path.is_file():
            raise AleUpdateError(f"parent adapter is missing {name}")
    return {
        "kind": "adapter",
        "path": str(parent_adapter),
        "revision": behavior_policy_revision,
    }


def _validate_prepared_batch(
    *, prepared_report_path: Path, behavior_policy_revision: str
) -> tuple[dict[str, Any], Path, TrainingBatch]:
    report = _read_json(prepared_report_path)
    if (
        report.get("version") != "fugu_ale_exact_grpo_batch_v5"
        or report.get("verdict") != "ALE_EXACT_TOKEN_REPLAY_ANCHORED_GRPO_BATCH_READY"
        or report.get("behavior_likelihood_contract_version")
        != FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION
    ):
        raise AleUpdateError("input is not a ready exact-token ALE update")
    groups = report.get("groups")
    contract = report.get("optimizer_contract") or {}
    replay = report.get("mandatory_replay") or {}
    combined = report.get("combined_batch") or {}
    if report.get("behavior_policy_revision") != behavior_policy_revision:
        raise AleUpdateError("prepared batch behavior-policy revision is stale")
    pool_id = report.get("pool_id")
    pool_binding_revision = report.get("pool_binding_revision")
    if (
        not isinstance(pool_id, str)
        or not pool_id.strip()
        or not isinstance(pool_binding_revision, str)
        or not pool_binding_revision.strip()
    ):
        raise AleUpdateError("prepared batch lacks semantic pool identity")
    pool_identity = (pool_id.strip(), pool_binding_revision.strip())
    if not isinstance(groups, list) or len(groups) < 2:
        raise AleUpdateError("optimizer updates require at least two causal ALE groups")
    group_ids: set[str] = set()
    task_ids: set[str] = set()
    group_paths: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise AleUpdateError("prepared batch contains an invalid causal group")
        if not has_full_vocabulary_behavior_likelihood_contract(
            group.get("behavior_likelihood_contract")
        ):
            raise AleUpdateError(
                "prepared causal group behavior-likelihood contract changed"
            )
        group_id = group.get("group_id")
        task_id = group.get("task_id")
        group_path = group.get("path")
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in group_ids
            or not isinstance(task_id, str)
            or not task_id
            or task_id in task_ids
            or not isinstance(group_path, str)
            or not group_path
            or group_path in group_paths
            or not Path(group_path).is_file()
            or group.get("behavior_policy_revision") != behavior_policy_revision
            or group.get("pool_id") != pool_identity[0]
            or group.get("pool_binding_revision") != pool_identity[1]
        ):
            raise AleUpdateError(
                "prepared causal groups must have unique IDs, tasks, paths, and one semantic pool identity"
            )
        group_ids.add(group_id)
        task_ids.add(task_id)
        group_paths.add(group_path)
    expected_contract = {
        "atomic_training_batch": True,
        "sequence_len": PROVEN_SEQUENCE_LEN,
        "data_parallel_gpus": DATA_PARALLEL_GPUS,
        "adv_tau": 1.0,
        "replay_tau": 0.05,
        "kl_tau": 0.001,
        "policy_logprob_source": "exact_behavior_policy_generation",
        "behavior_likelihood_contract": (
            full_vocabulary_behavior_likelihood_contract()
        ),
        "retokenization": False,
        "replay_reference_mode": REPLAY_REFERENCE_MODE,
    }
    credit_assignment = contract.get("policy_credit_assignment")
    contract_without_credit = {key: value for key, value in contract.items() if key != "policy_credit_assignment"}
    if contract_without_credit != expected_contract:
        raise AleUpdateError("prepared batch optimizer contract changed")
    if (
        not isinstance(credit_assignment, dict)
        or credit_assignment.get("mode")
        != "first_executed_coordination_divergence"
        or credit_assignment.get("policy_attempts") != "initial_only"
    ):
        raise AleUpdateError("prepared batch policy credit assignment changed")
    credit_groups = credit_assignment.get("groups")
    if not isinstance(credit_groups, list) or len(credit_groups) != len(groups):
        raise AleUpdateError("prepared batch causal group credit changed")
    action_allowlist = credit_assignment.get("action_allowlist")
    if action_allowlist is not None and (
        not isinstance(action_allowlist, list)
        or not action_allowlist
        or action_allowlist != sorted(set(action_allowlist))
        or any(action not in CONTROL_ACTIONS for action in action_allowlist)
    ):
        raise AleUpdateError("prepared batch policy credit assignment changed")
    replay_path = Path(str(replay.get("path", ""))).resolve()
    if replay.get("samples") != 52 or replay.get("selected_tokens") != 2_448 or not replay_path.is_file():
        raise AleUpdateError("mandatory 27B transfer replay is absent or changed")
    retention = report.get("train_only_retention_replay")
    if not isinstance(retention, dict):
        raise AleUpdateError("mandatory action-balanced train-retention replay is absent")
    retention_samples = retention.get("samples")
    retention_tokens = retention.get("selected_tokens")
    retention_weight_sum = retention.get("selected_weight_sum")
    retention_report_path = Path(str(retention.get("report", ""))).resolve()
    retention_replay_path = Path(str(retention.get("path", ""))).resolve()
    if (
        retention.get("replay_version") != ACTION_BALANCED_RETENTION_REPLAY_VERSION
        or retention_samples != ACTION_BALANCED_RETENTION_SAMPLES
        or retention_tokens != ACTION_BALANCED_RETENTION_SELECTED_TOKENS
    ):
        raise AleUpdateError("mandatory action-balanced train-retention replay changed")
    if (
        isinstance(retention_samples, bool)
        or not isinstance(retention_samples, int)
        or retention_samples <= 0
        or isinstance(retention_tokens, bool)
        or not isinstance(retention_tokens, int)
        or retention_tokens <= 0
        or not isinstance(retention_weight_sum, (int, float))
        or isinstance(retention_weight_sum, bool)
        or not math.isfinite(float(retention_weight_sum))
        or float(retention_weight_sum) <= 0.0
        or retention.get("reference_mode") != REPLAY_REFERENCE_MODE
        or not retention_report_path.is_file()
        or not retention_replay_path.is_file()
    ):
        raise AleUpdateError("train-only retention replay is absent or changed")
    retention_report = _read_json(retention_report_path)
    if (
        retention_report.get("version") not in TRAIN_RETENTION_REPLAY_VERSIONS
        or retention_report.get("reference_mode") != REPLAY_REFERENCE_MODE
        or (retention_report.get("counts") or {}).get("samples") != retention_samples
        or (retention_report.get("counts") or {}).get("selected_completion_tokens") != retention_tokens
    ):
        raise AleUpdateError("train-only retention report contract changed")
    if retention.get("replay_version") != retention_report.get("version"):
        raise AleUpdateError("train-only retention replay version changed")
    weighting = retention_report.get("weighting") or {}
    if (
        retention.get("weighting_mode") != "equal_action_completion_token_mass_v1"
        or weighting.get("mode") != retention.get("weighting_mode")
        or not math.isclose(
            float(retention_weight_sum),
            float(weighting.get("total_weighted_completion_tokens", -1.0)),
            rel_tol=1e-6,
            abs_tol=1e-3,
        )
    ):
        raise AleUpdateError("train-only retention action weighting changed")
    batch_path = Path(str(combined.get("path", ""))).resolve()
    if not batch_path.is_file():
        raise AleUpdateError("prepared ALE batch is missing")
    try:
        batch = msgspec.msgpack.decode(batch_path.read_bytes(), type=TrainingBatch)
    except msgspec.DecodeError as exc:
        raise AleUpdateError("prepared ALE batch cannot be decoded") from exc
    if batch.step != 0 or combined.get("step") != 0:
        raise AleUpdateError("one-step ALE updates must start at optimizer step zero")
    if len(batch.examples) != combined.get("samples"):
        raise AleUpdateError("prepared ALE batch sample count changed")
    policy_count = (report.get("policy") or {}).get("samples")
    if isinstance(policy_count, bool) or not isinstance(policy_count, int) or policy_count <= 0:
        raise AleUpdateError("prepared ALE batch policy count changed")
    policy_samples = batch.examples[:policy_count]
    mandatory_samples = batch.examples[policy_count : policy_count + 52]
    retention_batch_samples = batch.examples[policy_count + 52 :]
    replay_samples = [*mandatory_samples, *retention_batch_samples]
    if (
        len(mandatory_samples) != 52
        or len(retention_batch_samples) != retention_samples
        or len(policy_samples) != policy_count
        or any(any(sample.replay_mask or []) for sample in policy_samples)
        or any(not any(sample.replay_mask or []) for sample in replay_samples)
    ):
        raise AleUpdateError("prepared ALE batch lost policy or replay samples")
    expected_credited_samples: set[int] = set()
    cursor = 0
    computed_signed_credit = {
        action: {
            "negative_samples": 0,
            "negative_tokens": 0,
            "positive_samples": 0,
            "positive_tokens": 0,
        }
        for action in sorted(CONTROL_ACTIONS)
    }
    for group, credit_group in zip(groups, credit_groups, strict=True):
        if not isinstance(credit_group, dict) or credit_group.get("group_id") != group.get("group_id"):
            raise AleUpdateError("prepared batch causal group credit changed")
        decision_indices = credit_group.get("decision_indices")
        episode_reports = group.get("episodes")
        if (
            not isinstance(decision_indices, list)
            or not isinstance(episode_reports, list)
            or len(decision_indices) != len(episode_reports)
            or len(decision_indices) < 2
            or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in decision_indices)
        ):
            raise AleUpdateError("prepared batch lost causal episode credit")
        group_policy_samples = 0
        for episode, decision_index in zip(episode_reports, decision_indices, strict=True):
            if not isinstance(episode, dict):
                raise AleUpdateError("prepared ALE causal episode report is invalid")
            sample_count = episode.get("policy_samples")
            credited_action = episode.get("credited_action")
            credited_tokens = episode.get("credited_conductor_tokens")
            if (
                isinstance(sample_count, bool)
                or not isinstance(sample_count, int)
                or sample_count <= 0
                or episode.get("credited_decision_index") != decision_index
                or decision_index >= sample_count
                or credited_action not in {*CONTROL_ACTIONS, None}
                or isinstance(credited_tokens, bool)
                or not isinstance(credited_tokens, int)
                or credited_tokens < 0
                or cursor + sample_count > len(policy_samples)
            ):
                raise AleUpdateError("prepared ALE causal decision cannot be credited")
            advantage = episode.get("advantage")
            sample = policy_samples[cursor + decision_index]
            expected_advantage = float(advantage) if credited_action is not None else 0.0
            if (
                isinstance(advantage, bool)
                or not isinstance(advantage, (int, float))
                or not math.isfinite(float(advantage))
                or not math.isclose(
                    float(sample.advantage),
                    expected_advantage,
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
                or credited_tokens != (len(sample.completion_ids) if credited_action is not None else 0)
            ):
                raise AleUpdateError("prepared ALE causal decision advantage changed")
            if expected_advantage != 0.0:
                expected_credited_samples.add(cursor + decision_index)
                sign = "negative" if expected_advantage < 0.0 else "positive"
                computed_signed_credit[credited_action][f"{sign}_samples"] += 1
                computed_signed_credit[credited_action][f"{sign}_tokens"] += credited_tokens
            cursor += sample_count
            group_policy_samples += sample_count
        if group.get("policy_samples") != group_policy_samples:
            raise AleUpdateError("prepared ALE causal group sample layout changed")
    if cursor != len(policy_samples):
        raise AleUpdateError("prepared ALE causal sample layout changed")
    actual_credited_samples = {index for index, sample in enumerate(policy_samples) if float(sample.advantage) != 0.0}
    if actual_credited_samples != expected_credited_samples:
        raise AleUpdateError("prepared ALE batch credits shared or downstream decisions")
    reported_signed_credit = (report.get("policy") or {}).get("signed_credit_by_action")
    if reported_signed_credit != computed_signed_credit:
        raise AleUpdateError("prepared ALE signed action credit changed")
    insufficient_actions = [
        action
        for action, counts in computed_signed_credit.items()
        if action != "complete"
        and counts["negative_samples"] + counts["positive_samples"] > 0
        and (counts["negative_samples"] == 0 or counts["positive_samples"] == 0)
    ]
    if insufficient_actions:
        raise AleUpdateError(
            "insufficient directional evidence for credited action(s): " + ", ".join(insufficient_actions)
        )
    if sum(sum(sample.replay_mask or []) for sample in mandatory_samples) != 2_448:
        raise AleUpdateError("prepared ALE batch mandatory replay mask changed")
    if sum(sum(sample.replay_mask or []) for sample in retention_batch_samples) != retention_tokens:
        raise AleUpdateError("prepared ALE batch retention replay mask changed")
    if not math.isclose(
        _selected_replay_weight_sum(mandatory_samples),
        float((report.get("mandatory_replay") or {}).get("selected_weight_sum", -1.0)),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise AleUpdateError("prepared ALE batch mandatory replay weights changed")
    if not math.isclose(
        _selected_replay_weight_sum(retention_batch_samples),
        float(retention_weight_sum),
        rel_tol=1e-6,
        abs_tol=1e-3,
    ):
        raise AleUpdateError("prepared ALE batch retention replay weights changed")
    if not any(float(sample.advantage) < 0.0 for sample in policy_samples) or not any(
        float(sample.advantage) > 0.0 for sample in policy_samples
    ):
        raise AleUpdateError("prepared ALE batch has no signed group advantage")
    return report, batch_path, batch


def _train_config(
    *,
    model_path: Path,
    output_dir: Path,
    parent_adapter: Path | None,
    learning_rate: float,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model": str(model_path),
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_len": PROVEN_SEQUENCE_LEN,
        "max_steps": 1,
        "logging_steps": 1,
        "learning_rate": learning_rate,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "weight_decay": 0.0,
        "optimizer": "adamw",
        "gpus": DATA_PARALLEL_GPUS,
        "recipe": "fp8_hybrid",
        "recompute": True,
        "cpu_training": True,
        "offload_residual": True,
        "lmhead_chunks": 8,
        "lora": True,
        "lora_rank": 16,
        "lora_alpha": 16.0,
        "lora_dtype": "fp32",
        "train_router": False,
        "lora_target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "loss": {
            "ipo_mask_low": 0.2,
            "ipo_mask_high": 0.2,
            "adv_tau": 1.0,
            "teacher_tau": 0.0,
            "opd_tau": 0.0,
            "opd_beta": 1.0,
            "replay_tau": 0.05,
            "kl_tau": 0.001,
        },
        "save_steps": 0,
        "checkpoint_dir": str(output_dir),
        "resume_from_checkpoint": False,
        "report_to": ["surogate"],
        "surogate_metrics_path": str(output_dir / "surogate_metrics.jsonl"),
        "offload_optimizer": True,
        "offload_grads": True,
        # Keep the frozen base and graph working tensors BF16. lora_dtype
        # controls the separate FP32 trainable LoRA master and LoRA gradients;
        # its work copy remains BF16.
        "master_dtype": "bf16",
        "gradient_dtype": "bf16",
        "doc_masking": False,
        "train_seed": 20_260_721,
    }
    if parent_adapter is not None:
        config["adapter_path"] = str(parent_adapter)
        config["adapter_init_mode"] = "trainable"
    return config


def _control_config(
    *, output_dir: Path, model_path: Path, behavior_policy_revision: str, samples: int
) -> dict[str, Any]:
    run_dir = output_dir / "run_default"
    return {
        "batch_size": samples + samples % 2,
        "env": [],
        "max_async_level": 0,
        "max_steps": 1,
        "model": {
            "lora_adapter": behavior_policy_revision,
            "name": str(model_path),
        },
        "num_train_workers": 1,
        "output_dir": str(run_dir),
        "rollout_transport": {"type": "filesystem"},
        "rollouts_per_example": 2,
        "sequence_len": PROVEN_SEQUENCE_LEN,
        "strict_async_level": True,
        "verification": {"enabled": True},
        "weight_broadcast": {"type": "filesystem"},
    }


def prepare_ale_optimizer_run(
    *,
    prepared_report_path: Path,
    output_dir: Path,
    model_path: Path,
    behavior_policy_revision: str,
    parent_adapter: Path | None = None,
    learning_rate: float = MAX_LEARNING_RATE,
    require_six_gpus: bool = True,
    require_no_stale_process: bool = True,
) -> dict[str, Any]:
    """Prepare one filesystem-transport optimizer update for execution."""
    prepared_report_path = prepared_report_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    parent_adapter = parent_adapter.expanduser().resolve() if parent_adapter is not None else None
    revision = behavior_policy_revision.strip()
    if not revision:
        raise AleUpdateError("behavior-policy revision must be non-empty")
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise AleUpdateError(f"27B model snapshot is incomplete: {model_path}")
    if output_dir.exists():
        raise AleUpdateError(f"refusing to overwrite optimizer output: {output_dir}")
    if not math.isfinite(learning_rate) or not 0.0 < learning_rate <= MAX_LEARNING_RATE:
        raise AleUpdateError(f"learning rate must be within (0, {MAX_LEARNING_RATE:.1e}]")
    if require_no_stale_process:
        _assert_no_stale_grpo_process()
    gpu_inventory = _cuda_inventory()
    if require_six_gpus and len(gpu_inventory) != DATA_PARALLEL_GPUS:
        raise AleUpdateError(f"expected exactly six visible GPUs, found {len(gpu_inventory)}")
    parent = _validate_parent_policy(
        model_path=model_path,
        behavior_policy_revision=revision,
        parent_adapter=parent_adapter,
    )
    prepared, batch_path, batch = _validate_prepared_batch(
        prepared_report_path=prepared_report_path,
        behavior_policy_revision=revision,
    )
    train = _train_config(
        model_path=model_path,
        output_dir=output_dir,
        parent_adapter=parent_adapter,
        learning_rate=learning_rate,
    )
    control = _control_config(
        output_dir=output_dir,
        model_path=model_path,
        behavior_policy_revision=revision,
        samples=len(batch.examples),
    )

    run_dir = output_dir / "run_default"
    rollout_path = run_dir / "rollouts/step_0/rollouts.bin"
    control_path = run_dir / "control/orch.yaml"
    train_path = output_dir / "train.yaml"
    rollout_path.parent.mkdir(parents=True)
    control_path.parent.mkdir(parents=True)
    shutil.copy2(batch_path, rollout_path)
    control_path.write_text(yaml.safe_dump(control, sort_keys=True), encoding="utf-8")
    train_path.write_text(yaml.safe_dump(train, sort_keys=False), encoding="utf-8")
    try:
        loaded = load_config(GRPOTrainConfig, str(train_path))
    except Exception as exc:
        shutil.rmtree(output_dir)
        raise AleUpdateError(f"native trainer rejected staged config: {exc}") from exc
    if (
        loaded.max_steps != 1
        or loaded.gpus != DATA_PARALLEL_GPUS
        or loaded.sequence_len != PROVEN_SEQUENCE_LEN
        or loaded.resume_from_checkpoint is not False
        or loaded.lora_dtype != "fp32"
        or loaded.master_dtype != "bf16"
        or loaded.gradient_dtype != "bf16"
        or loaded.loss.adv_tau != 1.0
        or loaded.loss.replay_tau != 0.05
        or loaded.loss.kl_tau != 0.001
    ):
        shutil.rmtree(output_dir)
        raise AleUpdateError("native trainer parsed a different optimizer contract")

    report = {
        "version": "fugu_ale_optimizer_run_v3",
        "verdict": "READY_TO_RUN",
        "created_at": datetime.now(UTC).isoformat(),
        "behavior_policy_revision": revision,
        "pool_id": prepared["pool_id"],
        "pool_binding_revision": prepared["pool_binding_revision"],
        "parent_policy": parent,
        "model": {"path": str(model_path), "snapshot_revision": model_path.name},
        "source": {
            "prepared_report": str(prepared_report_path),
            "groups": [group["path"] for group in prepared["groups"]],
            "batch": prepared["combined_batch"]["path"],
        },
        "staged": {
            "output_dir": str(output_dir),
            "train_config": str(train_path),
            "control_config": str(control_path),
            "rollouts": str(rollout_path),
            "samples": len(batch.examples),
        },
        "optimizer_contract": {
            "max_steps": 1,
            "sequence_len": PROVEN_SEQUENCE_LEN,
            "gpus": DATA_PARALLEL_GPUS,
            "learning_rate": learning_rate,
            "lora_dtype": "fp32",
            "master_dtype": "bf16",
            "gradient_dtype": "bf16",
            "adv_tau": 1.0,
            "replay_tau": 0.05,
            "kl_tau": 0.001,
            "replay_reference_mode": REPLAY_REFERENCE_MODE,
            "resume": False,
            "mandatory_replay": prepared.get("mandatory_replay"),
            "train_only_retention_replay": prepared.get("train_only_retention_replay"),
            "policy_credit_assignment": prepared["optimizer_contract"].get("policy_credit_assignment"),
            "signed_credit_by_action": prepared["policy"].get("signed_credit_by_action"),
        },
        "gpu_inventory": gpu_inventory,
        "external_calls": 0,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    report_path = output_dir / "optimizer_run.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def register_trained_policy_adapter(
    *,
    adapter_dir: Path,
    policy_revision: str,
    parent_policy_revision: str,
    optimizer_run_path: Path,
    optimizer_step: int,
) -> dict[str, Any]:
    """Assign an explicit revision to the adapter used by the next collection."""
    adapter_dir = adapter_dir.expanduser().resolve()
    optimizer_run_path = optimizer_run_path.expanduser().resolve()
    policy_revision = policy_revision.strip()
    if (
        not policy_revision.startswith(
            ("fugu-ale-", "fugu-conductor-")
        )
        or policy_revision == parent_policy_revision
    ):
        raise AleUpdateError(
            "policy_revision must be a new explicit fugu-ale revision "
            "or fugu-conductor revision"
        )
    if optimizer_step < 1:
        raise AleUpdateError("a policy adapter cannot be registered before an optimizer step")
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise AleUpdateError("trained adapter export is incomplete")
    optimizer_run = _read_json(optimizer_run_path)
    optimizer_run_version = optimizer_run.get("version")
    supported_run = (
        optimizer_run_version == "fugu_ale_optimizer_run_v3"
        or (
            optimizer_run_version
            == "fugu_conductor_optimizer_run_v1"
            and optimizer_run.get("source_kind") == "synthetic"
        )
    )
    if (
        not supported_run
        or optimizer_run.get("verdict") != "READY_TO_RUN"
        or not isinstance(optimizer_run.get("pool_id"), str)
        or not optimizer_run["pool_id"].strip()
        or not isinstance(optimizer_run.get("pool_binding_revision"), str)
        or not optimizer_run["pool_binding_revision"].strip()
    ):
        raise AleUpdateError(
            "optimizer run record lacks a supported semantic pool contract"
        )
    manifest = {
        "version": (
            "fugu_conductor_policy_revision_v1"
            if optimizer_run_version
            == "fugu_conductor_optimizer_run_v1"
            else "fugu_ale_policy_revision_v2"
        ),
        "policy_revision": policy_revision,
        "adapter_config": config_path.name,
        "adapter_model": weights_path.name,
        "optimizer_run": str(optimizer_run_path),
        "optimizer_step": optimizer_step,
        "parent_policy_revision": parent_policy_revision,
    }
    path = adapter_dir / POLICY_MANIFEST
    if path.exists():
        raise AleUpdateError("trained adapter already has a policy revision manifest")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--behavior-policy-revision", required=True)
    parser.add_argument("--parent-adapter", type=Path)
    parser.add_argument("--learning-rate", type=float, default=MAX_LEARNING_RATE)
    args = parser.parse_args()
    report = prepare_ale_optimizer_run(
        prepared_report_path=args.prepared_report,
        output_dir=args.output_dir,
        model_path=args.model,
        behavior_policy_revision=args.behavior_policy_revision,
        parent_adapter=args.parent_adapter,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
