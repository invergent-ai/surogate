"""Stage one conservative optimizer update from a synthetic conductor batch."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from surogate.core.config.loader import load_config
from surogate.grpo.ale_batch import (
    PROVEN_SEQUENCE_LEN,
    REPLAY_REFERENCE_MODE,
)
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
from surogate.grpo.synthetic_batch import (
    SYNTHETIC_BATCH_VERSION,
    SYNTHETIC_CREDIT_MODE,
    SyntheticBatchError,
    validate_synthetic_prepared_batch,
)

CONDUCTOR_OPTIMIZER_RUN_VERSION = "fugu_conductor_optimizer_run_v1"


class SyntheticUpdateError(ValueError):
    """A synthetic optimizer update is not safe to stage."""


def _helper_error(exc: Exception) -> SyntheticUpdateError:
    return SyntheticUpdateError(str(exc))


def prepare_synthetic_optimizer_run(
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
    learning_rate: float = MAX_LEARNING_RATE,
    require_six_gpus: bool = True,
    require_no_stale_process: bool = True,
) -> dict[str, Any]:
    """Prepare one filesystem-transport synthetic conductor update."""
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
        raise SyntheticUpdateError(
            "behavior-policy revision must be non-empty"
        )
    if not runtime_revision:
        raise SyntheticUpdateError(
            "runtime revision must be non-empty"
        )
    if (
        not model_path.is_dir()
        or not (model_path / "config.json").is_file()
    ):
        raise SyntheticUpdateError(
            f"27B model snapshot is incomplete: {model_path}"
        )
    if output_dir.exists():
        raise SyntheticUpdateError(
            f"refusing to overwrite optimizer output: {output_dir}"
        )
    if (
        not math.isfinite(learning_rate)
        or not 0.0 < learning_rate <= MAX_LEARNING_RATE
    ):
        raise SyntheticUpdateError(
            f"learning rate must be within (0, {MAX_LEARNING_RATE:.1e}]"
        )
    if require_no_stale_process:
        try:
            _assert_no_stale_grpo_process()
        except AleUpdateError as exc:
            raise _helper_error(exc) from exc
    gpu_inventory = _cuda_inventory()
    if require_six_gpus and len(gpu_inventory) != DATA_PARALLEL_GPUS:
        raise SyntheticUpdateError(
            f"expected exactly six visible GPUs, found {len(gpu_inventory)}"
        )
    try:
        parent = _validate_parent_policy(
            model_path=model_path,
            behavior_policy_revision=revision,
            parent_adapter=parent_adapter,
        )
        prepared, batch_path, batch = (
            validate_synthetic_prepared_batch(
                prepared_report_path=prepared_report_path,
                expected_behavior_policy_revision=revision,
                expected_runtime_revision=runtime_revision,
                pool_binding_path=pool_binding_path,
                replay_path=replay_path,
                train_retention_replay_path=retention_replay_path,
                train_retention_report_path=retention_report_path,
            )
        )
    except (AleUpdateError, SyntheticBatchError) as exc:
        raise _helper_error(exc) from exc

    if prepared.get("version") != SYNTHETIC_BATCH_VERSION:
        raise SyntheticUpdateError(
            "prepared source is not a synthetic exact-token batch"
        )
    credit_contract = (
        prepared.get("optimizer_contract") or {}
    ).get("policy_credit_assignment")
    if (
        not isinstance(credit_contract, dict)
        or credit_contract.get("mode") != SYNTHETIC_CREDIT_MODE
    ):
        raise SyntheticUpdateError(
            "synthetic policy credit assignment changed"
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
        raise SyntheticUpdateError(
            f"native trainer rejected staged config: {exc}"
        ) from exc
    if (
        loaded.max_steps != 1
        or loaded.gpus != DATA_PARALLEL_GPUS
        or loaded.sequence_len != PROVEN_SEQUENCE_LEN
        or loaded.resume_from_checkpoint is not False
        or loaded.loss.adv_tau != 1.0
        or loaded.loss.replay_tau != 0.05
        or loaded.loss.kl_tau != 0.001
    ):
        shutil.rmtree(output_dir)
        raise SyntheticUpdateError(
            "native trainer parsed a different optimizer contract"
        )

    report = {
        "version": CONDUCTOR_OPTIMIZER_RUN_VERSION,
        "verdict": "READY_TO_RUN",
        "source_kind": "synthetic",
        "created_at": datetime.now(UTC).isoformat(),
        "behavior_policy_revision": revision,
        "runtime_revision": runtime_revision,
        "pool_id": prepared["pool_id"],
        "pool_binding_revision": prepared[
            "pool_binding_revision"
        ],
        "pool_binding": str(pool_binding_path),
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
            "adv_tau": 1.0,
            "replay_tau": 0.05,
            "kl_tau": 0.001,
            "replay_reference_mode": REPLAY_REFERENCE_MODE,
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
        default=MAX_LEARNING_RATE,
    )
    args = parser.parse_args()
    report = prepare_synthetic_optimizer_run(
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
    "SyntheticUpdateError",
    "prepare_synthetic_optimizer_run",
]


if __name__ == "__main__":
    main()
