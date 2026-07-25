from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import msgspec

from surogate.grpo.ale_update import register_trained_policy_adapter
from surogate.grpo.synthetic_batch import (
    SYNTHETIC_BATCH_VERSION,
    SYNTHETIC_CREDIT_MODE,
)
from surogate.grpo.synthetic_update import (
    CONDUCTOR_OPTIMIZER_RUN_VERSION,
    prepare_synthetic_optimizer_run,
)
from surogate.grpo.transport import TrainingBatch, TrainingSample

POLICY_REVISION = "fugu-conductor-test-parent"
RUNTIME_REVISION = "runtime-r81-test"
POOL_ID = "anonymous-test-pool-v1"


def _fake_loaded_config() -> SimpleNamespace:
    return SimpleNamespace(
        max_steps=1,
        gpus=6,
        sequence_len=2_816,
        resume_from_checkpoint=False,
        loss=SimpleNamespace(
            adv_tau=1.0,
            replay_tau=0.05,
            kl_tau=0.001,
        ),
    )


def _sample() -> TrainingSample:
    return TrainingSample(
        prompt_ids=[1, 2],
        prompt_mask=[False, False],
        completion_ids=[3, 4],
        completion_mask=[True, True],
        completion_logprobs=[-0.2, -0.1],
        completion_temperatures=[1.0, 1.0],
        advantage=0.7,
        reward=1.0,
        replay_mask=[False, False, False, False],
    )


def _validated_source(
    tmp_path: Path,
    *,
    policy_revision: str,
) -> tuple[dict, Path, TrainingBatch]:
    batch = TrainingBatch(examples=[_sample()], step=0)
    batch_path = tmp_path / "source-rollouts.bin"
    batch_path.write_bytes(msgspec.msgpack.encode(batch))
    report = {
        "version": SYNTHETIC_BATCH_VERSION,
        "pool_id": POOL_ID,
        "pool_binding_revision": POOL_ID,
        "behavior_policy_revision": policy_revision,
        "runtime_revision": RUNTIME_REVISION,
        "source_collection": str(tmp_path / "collection.json"),
        "combined_batch": {
            "path": str(batch_path),
            "samples": 1,
            "step": 0,
        },
        "credit_groups": [
            {
                "scenario_id": "scenario-a",
                "boundary_id": "boundary-a",
            }
        ],
        "optimizer_contract": {
            "policy_credit_assignment": {
                "mode": SYNTHETIC_CREDIT_MODE,
            }
        },
        "policy": {
            "signed_credit_by_action": {
                "continue": {
                    "positive_samples": 1,
                    "negative_samples": 1,
                }
            }
        },
        "mandatory_replay": {
            "samples": 52,
            "selected_tokens": 2_448,
        },
        "train_only_retention_replay": {
            "samples": 76,
            "selected_tokens": 17_760,
        },
    }
    return report, batch_path, batch


def _patch_staging(
    monkeypatch,
    *,
    prepared: tuple[dict, Path, TrainingBatch],
) -> None:
    monkeypatch.setattr(
        "surogate.grpo.synthetic_update."
        "validate_synthetic_prepared_batch",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        "surogate.grpo.synthetic_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(),
    )
    monkeypatch.setattr(
        "surogate.grpo.synthetic_update._cuda_inventory",
        lambda: [],
    )


def test_stages_generic_synthetic_conductor_optimizer_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = tmp_path / POLICY_REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    prepared_report = tmp_path / "prepared_report.json"
    prepared_report.write_text("{}", encoding="utf-8")
    prepared = _validated_source(
        tmp_path,
        policy_revision=POLICY_REVISION,
    )
    _patch_staging(monkeypatch, prepared=prepared)

    report = prepare_synthetic_optimizer_run(
        prepared_report_path=prepared_report,
        output_dir=tmp_path / "optimizer",
        model_path=model,
        behavior_policy_revision=POLICY_REVISION,
        runtime_revision=RUNTIME_REVISION,
        pool_binding_path=tmp_path / "binding.json",
        replay_path=tmp_path / "transfer.bin",
        train_retention_replay_path=tmp_path / "retention.bin",
        train_retention_report_path=tmp_path / "retention.json",
        learning_rate=5.0e-7,
        require_six_gpus=False,
        require_no_stale_process=False,
    )

    assert report["version"] == CONDUCTOR_OPTIMIZER_RUN_VERSION
    assert report["source_kind"] == "synthetic"
    assert report["runtime_revision"] == RUNTIME_REVISION
    assert report["pool_id"] == POOL_ID
    assert report["optimizer_contract"][
        "policy_credit_assignment"
    ]["mode"] == SYNTHETIC_CREDIT_MODE
    assert report["optimizer_contract"]["mandatory_replay"][
        "samples"
    ] == 52
    assert report["optimizer_contract"][
        "train_only_retention_replay"
    ]["samples"] == 76
    assert Path(report["staged"]["rollouts"]).read_bytes() == Path(
        prepared[1]
    ).read_bytes()
    assert (tmp_path / "optimizer/train.yaml").is_file()
    assert (tmp_path / "optimizer/optimizer_run.json").is_file()


def test_registers_generic_policy_and_accepts_it_as_next_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    optimizer_run_path = tmp_path / "optimizer_run.json"
    optimizer_run_path.write_text(
        json.dumps(
            {
                "version": CONDUCTOR_OPTIMIZER_RUN_VERSION,
                "verdict": "READY_TO_RUN",
                "source_kind": "synthetic",
                "pool_id": POOL_ID,
                "pool_binding_revision": POOL_ID,
            }
        ),
        encoding="utf-8",
    )
    revision = "fugu-conductor-synthetic-r1"
    manifest = register_trained_policy_adapter(
        adapter_dir=adapter,
        policy_revision=revision,
        parent_policy_revision="fugu-ale-r2",
        optimizer_run_path=optimizer_run_path,
        optimizer_step=1,
    )

    assert manifest["version"] == (
        "fugu_conductor_policy_revision_v1"
    )
    assert manifest["policy_revision"] == revision

    model = tmp_path / "base-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    prepared_report = tmp_path / "prepared_report.json"
    prepared_report.write_text("{}", encoding="utf-8")
    prepared = _validated_source(
        tmp_path,
        policy_revision=revision,
    )
    _patch_staging(monkeypatch, prepared=prepared)

    report = prepare_synthetic_optimizer_run(
        prepared_report_path=prepared_report,
        output_dir=tmp_path / "next-optimizer",
        model_path=model,
        behavior_policy_revision=revision,
        runtime_revision=RUNTIME_REVISION,
        pool_binding_path=tmp_path / "binding.json",
        replay_path=tmp_path / "transfer.bin",
        train_retention_replay_path=tmp_path / "retention.bin",
        train_retention_report_path=tmp_path / "retention.json",
        parent_adapter=adapter,
        require_six_gpus=False,
        require_no_stale_process=False,
    )

    assert report["parent_policy"] == {
        "kind": "adapter",
        "path": str(adapter.resolve()),
        "revision": revision,
    }
