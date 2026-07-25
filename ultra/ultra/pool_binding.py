"""Versioned worker-pool bindings for pool-specific conductor checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POOL_BINDING_SCHEMA = "fugu_pool_binding_v1"
CHECKPOINT_BINDING_SCHEMA = "fugu_checkpoint_pool_binding_v1"


class PoolBindingError(ValueError):
    """A checkpoint, manifest, or runtime pool does not match its binding."""


@dataclass(frozen=True)
class PoolSlot:
    worker_id: int
    training_name: str
    model_alias: str
    runtime_model: str
    reasoning_effort: str
    role_prior: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointArtifacts:
    adapter_path: str
    base_model_snapshot: str
    trained_control_contract: str


@dataclass(frozen=True)
class PoolBinding:
    pool_id: str
    binding_revision: str
    provider_base: str
    slots: tuple[PoolSlot, ...]
    checkpoint: CheckpointArtifacts

    @property
    def pool_fingerprint(self) -> str:
        """Legacy name for consumers that have not moved to semantic revisions."""
        return self.binding_revision

    @property
    def runtime_models(self) -> tuple[str, ...]:
        return tuple(slot.runtime_model for slot in self.slots)

    @property
    def reasoning_efforts(self) -> tuple[str, ...]:
        return tuple(slot.reasoning_effort for slot in self.slots)


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PoolBindingError(f"{label} must be non-empty text")
    return value.strip()


def pool_fingerprint(
    *,
    pool_id: str,
    provider_base: str,
    slots: tuple[PoolSlot, ...],
) -> str:
    """Return the semantic revision used by legacy pool-fingerprint callers."""
    del provider_base, slots
    return _require_text(pool_id, "pool_id")


def load_pool_binding(path: Path) -> PoolBinding:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != POOL_BINDING_SCHEMA:
        raise PoolBindingError(f"{path} is not a {POOL_BINDING_SCHEMA} manifest")
    required_fields = {
        "schema_version",
        "pool_id",
        "provider_base",
        "slots",
        "checkpoint",
    }
    if not required_fields.issubset(raw):
        raise PoolBindingError("pool binding is missing required fields")

    pool_id = _require_text(raw.get("pool_id"), "pool_id")
    binding_revision = _require_text(
        raw.get("binding_revision", pool_id),
        "binding_revision",
    )
    provider_base = _require_text(raw.get("provider_base"), "provider_base").rstrip("/")
    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        raise PoolBindingError("slots must be a non-empty list")
    slots: list[PoolSlot] = []
    for index, row in enumerate(raw_slots):
        if not isinstance(row, dict) or set(row) != {
            "worker_id",
            "training_name",
            "model_alias",
            "runtime_model",
            "reasoning_effort",
            "role_prior",
        }:
            raise PoolBindingError(f"slots[{index}] has an invalid schema")
        worker_id = row.get("worker_id")
        if isinstance(worker_id, bool) or not isinstance(worker_id, int):
            raise PoolBindingError(f"slots[{index}].worker_id must be an integer")
        roles = row.get("role_prior")
        if not isinstance(roles, list) or any(
            not isinstance(role, str) or not role.strip() for role in roles
        ):
            raise PoolBindingError(f"slots[{index}].role_prior must be a list of text")
        slots.append(
            PoolSlot(
                worker_id=worker_id,
                training_name=_require_text(row.get("training_name"), "training_name"),
                model_alias=_require_text(row.get("model_alias"), "model_alias"),
                runtime_model=_require_text(row.get("runtime_model"), "runtime_model"),
                reasoning_effort=_require_text(row.get("reasoning_effort"), "reasoning_effort"),
                role_prior=tuple(role.strip() for role in roles),
            )
        )
    slot_tuple = tuple(slots)
    if tuple(slot.worker_id for slot in slot_tuple) != tuple(range(len(slot_tuple))):
        raise PoolBindingError("worker_id values must be consecutive stable ordinals starting at zero")
    if len({slot.training_name for slot in slot_tuple}) != len(slot_tuple):
        raise PoolBindingError("training_name values must be unique")

    checkpoint = raw.get("checkpoint")
    required_checkpoint_fields = {
        "adapter_path",
        "base_model_snapshot",
        "trained_control_contract",
    }
    if (
        not isinstance(checkpoint, dict)
        or not required_checkpoint_fields.issubset(checkpoint)
    ):
        raise PoolBindingError("checkpoint has an invalid schema")
    artifacts = CheckpointArtifacts(
        adapter_path=_require_text(checkpoint.get("adapter_path"), "adapter_path"),
        base_model_snapshot=_require_text(
            checkpoint.get("base_model_snapshot"),
            "base_model_snapshot",
        ),
        trained_control_contract=_require_text(
            checkpoint.get("trained_control_contract"),
            "trained_control_contract",
        ),
    )
    return PoolBinding(
        pool_id=pool_id,
        binding_revision=binding_revision,
        provider_base=provider_base,
        slots=slot_tuple,
        checkpoint=artifacts,
    )


def verify_runtime_pool(
    binding: PoolBinding,
    *,
    runtime_models: tuple[str, ...],
    reasoning_efforts: tuple[str, ...],
    provider_base: str,
) -> None:
    if runtime_models != binding.runtime_models:
        raise PoolBindingError(
            f"runtime models do not match bound pool {binding.pool_id}: "
            f"expected {binding.runtime_models!r}, got {runtime_models!r}"
        )
    if reasoning_efforts != binding.reasoning_efforts:
        raise PoolBindingError(
            f"reasoning efforts do not match bound pool {binding.pool_id}"
        )
    if provider_base.rstrip("/") != binding.provider_base:
        raise PoolBindingError(
            f"provider does not match bound pool {binding.pool_id}: expected {binding.provider_base}"
        )


def verify_checkpoint_artifacts(binding: PoolBinding, *, repo_root: Path) -> None:
    adapter = repo_root / binding.checkpoint.adapter_path
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    for path in (config, weights):
        if not path.is_file():
            raise PoolBindingError(f"bound checkpoint artifact is missing: {path}")


def checkpoint_sidecar(binding: PoolBinding) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_BINDING_SCHEMA,
        "pool_id": binding.pool_id,
        "binding_revision": binding.binding_revision,
        "trained_control_contract": binding.checkpoint.trained_control_contract,
    }


def verify_checkpoint_sidecar(path: Path, binding: PoolBinding) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = checkpoint_sidecar(binding)
    if raw != expected:
        raise PoolBindingError(f"checkpoint sidecar does not match bound pool {binding.pool_id}")
