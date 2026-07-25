from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
import yaml

from surogate.grpo.synthetic_branchpoint_batch import (
    SYNTHETIC_BRANCHPOINT_BATCH_VERSION,
    SYNTHETIC_BRANCHPOINT_CREDIT_MODE,
)
from surogate.grpo.synthetic_branchpoint_update import (
    CONDUCTOR_OPTIMIZER_RUN_VERSION,
    DEFAULT_BRANCHPOINT_LEARNING_RATE,
    SyntheticBranchpointUpdateError,
    _validate_binding_checkpoint,
    prepare_synthetic_branchpoint_optimizer_run,
)
from surogate.grpo.transport import TrainingBatch, TrainingSample

POLICY_REVISION = "fugu-conductor-branchpoint-parent"
RUNTIME_REVISION = "runtime-r84-test"
POOL_ID = "anonymous-test-pool-v1"


def _fake_loaded_config(*, sample_packing: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        max_steps=1,
        gpus=6,
        sequence_len=2_816,
        resume_from_checkpoint=False,
        sample_packing=sample_packing,
        lora_dtype="fp32",
        master_dtype="bf16",
        gradient_dtype="bf16",
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
) -> tuple[dict, Path, TrainingBatch]:
    batch = TrainingBatch(examples=[_sample()], step=0)
    batch_path = tmp_path / "source-rollouts.bin"
    batch_path.write_bytes(msgspec.msgpack.encode(batch))
    report = {
        "version": SYNTHETIC_BRANCHPOINT_BATCH_VERSION,
        "pool_id": POOL_ID,
        "pool_binding_revision": POOL_ID,
        "behavior_policy_revision": POLICY_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "tokenizer_model": str(tmp_path / POLICY_REVISION),
        "source_collection": str(tmp_path / "collection.json"),
        "combined_batch": {
            "path": str(batch_path),
            "samples": 1,
            "step": 0,
        },
        "credit_groups": [
            {
                "scenario_id": "scenario-a",
                "positive_policy_sample_indices": [1],
                "negative_policy_sample_indices": [0],
            }
        ],
        "optimizer_contract": {
            "sample_packing": False,
            "policy_credit_assignment": {
                "mode": SYNTHETIC_BRANCHPOINT_CREDIT_MODE,
            },
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
    monkeypatch: pytest.MonkeyPatch,
    *,
    prepared: tuple[dict, Path, TrainingBatch],
    sample_packing: bool = False,
) -> None:
    monkeypatch.setattr(
        "surogate.grpo.synthetic_branchpoint_update."
        "validate_synthetic_branchpoint_prepared_batch",
        lambda **kwargs: prepared,
    )
    monkeypatch.setattr(
        "surogate.grpo.synthetic_branchpoint_update.load_config",
        lambda *args, **kwargs: _fake_loaded_config(
            sample_packing=sample_packing
        ),
    )
    monkeypatch.setattr(
        "surogate.grpo.synthetic_branchpoint_update._cuda_inventory",
        lambda: [],
    )
    monkeypatch.setattr(
        "surogate.grpo.synthetic_branchpoint_update."
        "_validate_binding_checkpoint",
        lambda **kwargs: {
            "adapter_path": "bound-adapter",
            "base_model_snapshot": "Qwen/Qwen3.6-27B-FP8",
            "trained_control_contract": "unified_capability_action_v2",
            "parent_optimizer_model": "model",
        },
    )


def _stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    sample_packing: bool = False,
) -> dict:
    model = tmp_path / POLICY_REVISION
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    prepared_report = tmp_path / "prepared_report.json"
    prepared_report.write_text("{}", encoding="utf-8")
    prepared = _validated_source(tmp_path)
    _patch_staging(
        monkeypatch,
        prepared=prepared,
        sample_packing=sample_packing,
    )
    return prepare_synthetic_branchpoint_optimizer_run(
        prepared_report_path=prepared_report,
        output_dir=tmp_path / "optimizer",
        model_path=model,
        behavior_policy_revision=POLICY_REVISION,
        runtime_revision=RUNTIME_REVISION,
        pool_binding_path=tmp_path / "binding.json",
        replay_path=tmp_path / "transfer.bin",
        train_retention_replay_path=tmp_path / "retention.bin",
        train_retention_report_path=tmp_path / "retention.json",
        require_six_gpus=False,
        require_no_stale_process=False,
    )


def test_stages_atomic_no_packing_branchpoint_optimizer_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _stage(tmp_path, monkeypatch)

    assert report["version"] == CONDUCTOR_OPTIMIZER_RUN_VERSION
    assert report["source_kind"] == "synthetic"
    assert report["source_variant"] == SYNTHETIC_BRANCHPOINT_CREDIT_MODE
    assert report["runtime_revision"] == RUNTIME_REVISION
    assert report["pool_id"] == POOL_ID
    contract = report["optimizer_contract"]
    assert contract["max_steps"] == 1
    assert contract["resume"] is False
    assert contract["sample_packing"] is False
    assert contract["learning_rate"] == DEFAULT_BRANCHPOINT_LEARNING_RATE
    assert contract["lora_dtype"] == "fp32"
    assert contract["master_dtype"] == "bf16"
    assert contract["gradient_dtype"] == "bf16"
    assert contract["policy_credit_assignment"]["mode"] == (
        SYNTHETIC_BRANCHPOINT_CREDIT_MODE
    )
    assert contract["mandatory_replay"]["samples"] == 52
    assert contract["train_only_retention_replay"]["samples"] == 76
    train = yaml.safe_load(
        (tmp_path / "optimizer/train.yaml").read_text(encoding="utf-8")
    )
    assert train["sample_packing"] is False
    assert train["max_steps"] == 1
    assert train["resume_from_checkpoint"] is False
    assert train["lora_dtype"] == "fp32"
    assert train["master_dtype"] == "bf16"
    assert train["gradient_dtype"] == "bf16"
    assert (tmp_path / "optimizer/optimizer_run.json").is_file()


def test_rejects_native_config_that_enables_sample_packing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        SyntheticBranchpointUpdateError,
        match="different optimizer contract",
    ):
        _stage(tmp_path, monkeypatch, sample_packing=True)

    assert not (tmp_path / "optimizer").exists()


def test_binding_checkpoint_requires_bound_adapter_and_fp8_parent_model(
    tmp_path: Path,
) -> None:
    model = tmp_path / "fp8-model"
    model.mkdir()
    (model / "config.json").write_text(
        '{"quantization_config":{"quant_method":"fp8"}}',
        encoding="utf-8",
    )
    adapter = tmp_path / "accepted-adapter"
    adapter.mkdir()
    optimizer_run = tmp_path / "parent-optimizer.json"
    optimizer_run.write_text(
        (
            '{"model":{"path":'
            f'{msgspec.json.encode(str(model.resolve())).decode()}'
            "}}"
        ),
        encoding="utf-8",
    )
    (adapter / "fugu_policy_revision.json").write_text(
        (
            '{"optimizer_run":'
            f'{msgspec.json.encode(str(optimizer_run.resolve())).decode()}'
            "}"
        ),
        encoding="utf-8",
    )
    binding = tmp_path / "binding.json"
    binding.write_text(
        msgspec.json.encode(
            {
                "schema_version": "fugu_pool_binding_v1",
                "pool_id": POOL_ID,
                "binding_revision": POOL_ID,
                "provider_base": "https://yunwu.ai/v1",
                "slots": [
                    {
                        "worker_id": 0,
                        "training_name": "worker-0",
                        "model_alias": "alias-0",
                        "runtime_model": "runtime-0",
                        "reasoning_effort": "high",
                        "role_prior": ["reasoner"],
                    }
                ],
                "checkpoint": {
                    "adapter_path": str(adapter.resolve()),
                    "base_model_snapshot": "Qwen/Qwen3.6-27B-FP8",
                    "trained_control_contract": (
                        "unified_capability_action_v2"
                    ),
                },
            }
        ).decode(),
        encoding="utf-8",
    )

    report = _validate_binding_checkpoint(
        pool_binding_path=binding.resolve(),
        model_path=model.resolve(),
        parent_adapter=adapter.resolve(),
    )
    assert report["adapter_path"] == str(adapter.resolve())
    assert report["parent_optimizer_model"] == str(model.resolve())

    bf16 = tmp_path / "bf16-model"
    bf16.mkdir()
    (bf16 / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        SyntheticBranchpointUpdateError,
        match="selected model differs",
    ):
        _validate_binding_checkpoint(
            pool_binding_path=binding.resolve(),
            model_path=bf16.resolve(),
            parent_adapter=adapter.resolve(),
        )
    with pytest.raises(
        SyntheticBranchpointUpdateError,
        match="parent adapter differs",
    ):
        _validate_binding_checkpoint(
            pool_binding_path=binding.resolve(),
            model_path=model.resolve(),
            parent_adapter=None,
        )
