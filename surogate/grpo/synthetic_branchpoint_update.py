"""Stage one conservative fixed-continuation branchpoint optimizer update."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from ultra.pool_binding import load_pool_binding

from surogate.core.config.loader import load_config
from surogate.grpo.ale_batch import PROVEN_SEQUENCE_LEN, REPLAY_REFERENCE_MODE
from surogate.grpo.ale_update import (
    DATA_PARALLEL_GPUS,
    MAX_LEARNING_RATE,
    AleUpdateError,
    _assert_no_stale_grpo_process,
    _control_config,
    _cuda_inventory,
    _train_config,
    _validate_parent_policy,
)
from surogate.grpo.config import GRPOTrainConfig
from surogate.grpo.synthetic_branchpoint_batch import (
    SYNTHETIC_BRANCHPOINT_BATCH_VERSION,
    SYNTHETIC_BRANCHPOINT_CREDIT_MODE,
    SyntheticBranchpointBatchError,
    validate_synthetic_branchpoint_prepared_batch,
)

CONDUCTOR_OPTIMIZER_RUN_VERSION = "fugu_conductor_optimizer_run_v1"
SYNTHETIC_BRANCHPOINT_SOURCE_KIND = "synthetic"
SYNTHETIC_BRANCHPOINT_SOURCE_VARIANT = SYNTHETIC_BRANCHPOINT_CREDIT_MODE
DEFAULT_BRANCHPOINT_LEARNING_RATE = 1.0e-4


class SyntheticBranchpointUpdateError(ValueError):
    """A branchpoint optimizer update is not safe to stage."""


def _helper_error(exc: Exception) -> SyntheticBranchpointUpdateError:
    return SyntheticBranchpointUpdateError(str(exc))


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticBranchpointUpdateError(
            f"cannot read {label} {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SyntheticBranchpointUpdateError(
            f"{label} {path} must contain one JSON object"
        )
    return value


def _resolve_bound_adapter(
    *,
    pool_binding_path: Path,
    adapter_path: str,
) -> Path:
    configured = Path(adapter_path).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    candidates = [Path.cwd(), *pool_binding_path.parents]
    for base in candidates:
        candidate = (base / configured).resolve()
        if candidate.is_dir():
            return candidate
    raise SyntheticBranchpointUpdateError(
        "pool binding checkpoint adapter cannot be resolved: "
        f"{adapter_path}"
    )


def _validate_binding_checkpoint(
    *,
    pool_binding_path: Path,
    model_path: Path,
    parent_adapter: Path | None,
) -> dict[str, Any]:
    """Bind the optimizer inputs to the campaign's accepted checkpoint."""
    try:
        binding = load_pool_binding(pool_binding_path)
    except Exception as exc:
        raise SyntheticBranchpointUpdateError(
            f"cannot load pool binding {pool_binding_path}: {exc}"
        ) from exc
    bound_adapter = _resolve_bound_adapter(
        pool_binding_path=pool_binding_path,
        adapter_path=binding.checkpoint.adapter_path,
    )
    if parent_adapter is None or parent_adapter != bound_adapter:
        raise SyntheticBranchpointUpdateError(
            "parent adapter differs from the pool-bound accepted checkpoint"
        )
    manifest_path = parent_adapter / "fugu_policy_revision.json"
    manifest = _read_object(manifest_path, "parent policy manifest")
    optimizer_run_value = manifest.get("optimizer_run")
    optimizer_model_path: Path | None = None
    if isinstance(optimizer_run_value, str) and optimizer_run_value.strip():
        optimizer_run_path = Path(
            optimizer_run_value
        ).expanduser().resolve()
        if optimizer_run_path.is_file():
            optimizer_run = _read_object(
                optimizer_run_path,
                "parent optimizer run",
            )
            source_model = (optimizer_run.get("model") or {}).get("path")
            if not isinstance(source_model, str) or not source_model.strip():
                raise SyntheticBranchpointUpdateError(
                    "parent optimizer run lacks its base model path"
                )
            optimizer_model_path = Path(
                source_model
            ).expanduser().resolve()
            if model_path != optimizer_model_path:
                raise SyntheticBranchpointUpdateError(
                    "selected model differs from the parent optimizer base"
                )
    model_config = _read_object(model_path / "config.json", "model config")
    if "FP8" in binding.checkpoint.base_model_snapshot.upper():
        quantization = model_config.get("quantization_config")
        if (
            not isinstance(quantization, dict)
            or str(quantization.get("quant_method", "")).casefold()
            != "fp8"
        ):
            raise SyntheticBranchpointUpdateError(
                "pool binding requires the FP8 base model"
            )
    return {
        "adapter_path": str(bound_adapter),
        "base_model_snapshot": binding.checkpoint.base_model_snapshot,
        "trained_control_contract": (
            binding.checkpoint.trained_control_contract
        ),
        "parent_optimizer_model": (
            str(optimizer_model_path)
            if optimizer_model_path is not None
            else None
        ),
    }


def prepare_synthetic_branchpoint_optimizer_run(
    *,
    prepared_report_path: Path,
    output_dir: Path,
    model_path: Path,
    behavior_policy_revision: str,
    runtime_revision: str,
    pool_binding_path: Path,
    replay_path: Path,
    train_retention_replay_path: Path,
    train_retention_report_path: Path,
    parent_adapter: Path | None = None,
    learning_rate: float = DEFAULT_BRANCHPOINT_LEARNING_RATE,
    require_six_gpus: bool = True,
    require_no_stale_process: bool = True,
) -> dict[str, Any]:
    """Prepare one atomic six-GPU update from an exact branchpoint batch."""
    prepared_report_path = prepared_report_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    pool_binding_path = pool_binding_path.expanduser().resolve()
    replay_path = replay_path.expanduser().resolve()
    retention_replay_path = (
        train_retention_replay_path.expanduser().resolve()
    )
    retention_report_path = (
        train_retention_report_path.expanduser().resolve()
    )
    parent_adapter = (
        parent_adapter.expanduser().resolve()
        if parent_adapter is not None
        else None
    )
    revision = behavior_policy_revision.strip()
    runtime_revision = runtime_revision.strip()
    if not revision:
        raise SyntheticBranchpointUpdateError(
            "behavior-policy revision must be non-empty"
        )
    if not runtime_revision:
        raise SyntheticBranchpointUpdateError(
            "runtime revision must be non-empty"
        )
    if (
        not model_path.is_dir()
        or not (model_path / "config.json").is_file()
    ):
        raise SyntheticBranchpointUpdateError(
            f"27B model snapshot is incomplete: {model_path}"
        )
    if output_dir.exists():
        raise SyntheticBranchpointUpdateError(
            f"refusing to overwrite optimizer output: {output_dir}"
        )
    if (
        not math.isfinite(learning_rate)
        or not 0.0 < learning_rate <= MAX_LEARNING_RATE
    ):
        raise SyntheticBranchpointUpdateError(
            f"learning rate must be within (0, {MAX_LEARNING_RATE:.1e}]"
        )
    if require_no_stale_process:
        try:
            _assert_no_stale_grpo_process()
        except AleUpdateError as exc:
            raise _helper_error(exc) from exc
    gpu_inventory = _cuda_inventory()
    if require_six_gpus and len(gpu_inventory) != DATA_PARALLEL_GPUS:
        raise SyntheticBranchpointUpdateError(
            f"expected exactly six visible GPUs, found {len(gpu_inventory)}"
        )
    binding_checkpoint = _validate_binding_checkpoint(
        pool_binding_path=pool_binding_path,
        model_path=model_path,
        parent_adapter=parent_adapter,
    )
    try:
        parent = _validate_parent_policy(
            model_path=model_path,
            behavior_policy_revision=revision,
            parent_adapter=parent_adapter,
        )
        prepared, batch_path, batch = (
            validate_synthetic_branchpoint_prepared_batch(
                prepared_report_path=prepared_report_path,
                expected_behavior_policy_revision=revision,
                expected_runtime_revision=runtime_revision,
                pool_binding_path=pool_binding_path,
                tokenizer_model_path=model_path,
                replay_path=replay_path,
                train_retention_replay_path=retention_replay_path,
                train_retention_report_path=retention_report_path,
            )
        )
    except (AleUpdateError, SyntheticBranchpointBatchError) as exc:
        raise _helper_error(exc) from exc

    if prepared.get("version") != SYNTHETIC_BRANCHPOINT_BATCH_VERSION:
        raise SyntheticBranchpointUpdateError(
            "prepared source is not a fixed-continuation branchpoint batch"
        )
    credit_contract = (
        prepared.get("optimizer_contract") or {}
    ).get("policy_credit_assignment")
    if (
        not isinstance(credit_contract, dict)
        or credit_contract.get("mode")
        != SYNTHETIC_BRANCHPOINT_CREDIT_MODE
    ):
        raise SyntheticBranchpointUpdateError(
            "branchpoint policy credit assignment changed"
        )

    train = _train_config(
        model_path=model_path,
        output_dir=output_dir,
        parent_adapter=parent_adapter,
        learning_rate=learning_rate,
    )
    # Make the no-packing contract explicit in both the staged YAML and the
    # parsed native configuration. Exact behavior-policy rows are indivisible.
    train["sample_packing"] = False
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
    control_path.write_text(
        yaml.safe_dump(control, sort_keys=True),
        encoding="utf-8",
    )
    train_path.write_text(
        yaml.safe_dump(train, sort_keys=False),
        encoding="utf-8",
    )
    try:
        loaded = load_config(GRPOTrainConfig, str(train_path))
    except Exception as exc:
        shutil.rmtree(output_dir)
        raise SyntheticBranchpointUpdateError(
            f"native trainer rejected staged config: {exc}"
        ) from exc
    if (
        loaded.max_steps != 1
        or loaded.gpus != DATA_PARALLEL_GPUS
        or loaded.sequence_len != PROVEN_SEQUENCE_LEN
        or loaded.resume_from_checkpoint is not False
        or loaded.sample_packing is not False
        or loaded.lora_dtype != "fp32"
        or loaded.master_dtype != "bf16"
        or loaded.gradient_dtype != "bf16"
        or loaded.loss.adv_tau != 1.0
        or loaded.loss.replay_tau != 0.05
        or loaded.loss.kl_tau != 0.001
    ):
        shutil.rmtree(output_dir)
        raise SyntheticBranchpointUpdateError(
            "native trainer parsed a different optimizer contract"
        )

    report = {
        "version": CONDUCTOR_OPTIMIZER_RUN_VERSION,
        "verdict": "READY_TO_RUN",
        "source_kind": SYNTHETIC_BRANCHPOINT_SOURCE_KIND,
        "source_variant": SYNTHETIC_BRANCHPOINT_SOURCE_VARIANT,
        "created_at": datetime.now(UTC).isoformat(),
        "behavior_policy_revision": revision,
        "runtime_revision": runtime_revision,
        "pool_id": prepared["pool_id"],
        "pool_binding_revision": prepared[
            "pool_binding_revision"
        ],
        "pool_binding": str(pool_binding_path),
        "binding_checkpoint": binding_checkpoint,
        "parent_policy": parent,
        "model": {
            "path": str(model_path),
            "snapshot_revision": model_path.name,
        },
        "source": {
            "prepared_report": str(prepared_report_path),
            "collection": prepared["source_collection"],
            "batch": prepared["combined_batch"]["path"],
            "prepared_batch_version": prepared["version"],
            "tokenizer_model": prepared["tokenizer_model"],
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
            "sample_packing": False,
            "resume": False,
            "policy_credit_assignment": credit_contract,
            "credit_groups": prepared["credit_groups"],
            "signed_credit_by_action": prepared["policy"][
                "signed_credit_by_action"
            ],
            "mandatory_replay": prepared["mandatory_replay"],
            "train_only_retention_replay": prepared[
                "train_only_retention_replay"
            ],
        },
        "gpu_inventory": gpu_inventory,
        "external_calls": 0,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    (output_dir / "optimizer_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--behavior-policy-revision",
        required=True,
    )
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
    parser.add_argument("--parent-adapter", type=Path)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_BRANCHPOINT_LEARNING_RATE,
    )
    args = parser.parse_args()
    report = prepare_synthetic_branchpoint_optimizer_run(
        prepared_report_path=args.prepared_report,
        output_dir=args.output_dir,
        model_path=args.model,
        behavior_policy_revision=args.behavior_policy_revision,
        runtime_revision=args.runtime_revision,
        pool_binding_path=args.pool_binding,
        replay_path=args.replay,
        train_retention_replay_path=args.train_retention_replay,
        train_retention_report_path=args.train_retention_report,
        parent_adapter=args.parent_adapter,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "CONDUCTOR_OPTIMIZER_RUN_VERSION",
    "DEFAULT_BRANCHPOINT_LEARNING_RATE",
    "SYNTHETIC_BRANCHPOINT_SOURCE_KIND",
    "SYNTHETIC_BRANCHPOINT_SOURCE_VARIANT",
    "SyntheticBranchpointUpdateError",
    "prepare_synthetic_branchpoint_optimizer_run",
]


if __name__ == "__main__":
    main()
