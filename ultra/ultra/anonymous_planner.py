"""Anonymous planner views derived from versioned worker-pool bindings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

from .pool_binding import PoolBinding, PoolBindingError


ANONYMOUS_PLANNER_CONFIG_SCHEMA = "fugu_anonymous_planner_config_v1"
CAPABILITY_SET_PLANNER_CONFIG_SCHEMA = "fugu_capability_set_planner_config_v1"


def normalize_capability_tags(tags: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({" ".join(tag.lower().split()) for tag in tags}))
    if not normalized:
        raise PoolBindingError("anonymous capability descriptors must be non-empty")
    return normalized


def capability_profile_ref(tags: Sequence[str]) -> str:
    roles = normalize_capability_tags(tags)
    encoded_roles = "_".join(f"{len(role)}:{role}" for role in roles)
    return f"profile_{encoded_roles}"


def _normalized_roles(binding: PoolBinding) -> dict[int, tuple[str, ...]]:
    normalized: dict[int, tuple[str, ...]] = {}
    for slot in binding.slots:
        roles = normalize_capability_tags(slot.role_prior)
        normalized[slot.worker_id] = roles
    return normalized


@dataclass(frozen=True)
class CapabilitySetPlannerView:
    """Permutation-invariant learned view of a concrete runtime pool."""

    config: dict[str, Any]
    profile_ref_to_worker_id: dict[str, int]
    worker_id_to_profile_ref: dict[int, str]

    def profile_refs_for_workers(self, worker_ids: tuple[int, ...]) -> tuple[str, ...]:
        try:
            return tuple(sorted(self.worker_id_to_profile_ref[item] for item in worker_ids))
        except KeyError as exc:
            raise PoolBindingError(f"unknown runtime worker id: {exc.args[0]}") from exc


def capability_set_planner_view(
    binding: PoolBinding,
    *,
    max_workflow_steps: int = 5,
) -> CapabilitySetPlannerView:
    """Build stable anonymous profile references independent of runtime slot order."""
    if max_workflow_steps <= 0:
        raise PoolBindingError("max_workflow_steps must be positive")
    roles_by_worker = _normalized_roles(binding)
    worker_to_ref = {
        worker_id: capability_profile_ref(roles)
        for worker_id, roles in roles_by_worker.items()
    }
    if len(set(worker_to_ref.values())) != len(worker_to_ref):
        raise PoolBindingError(
            "capability profiles must be distinct; enrich anonymous calibration "
            "metadata before binding interchangeable role descriptors"
        )
    ref_to_worker = {ref: worker_id for worker_id, ref in worker_to_ref.items()}
    refs = sorted(ref_to_worker)
    config = {
        "schema_version": CAPABILITY_SET_PLANNER_CONFIG_SCHEMA,
        "selector_field": "profile_ref",
        "worker_pool_names": refs,
        "lane_worker_masks": {"single_turn": refs},
        "worker_pool": {
            ref: {
                "name": ref,
                "profile_ref": ref,
                "role_prior": list(roles_by_worker[ref_to_worker[ref]]),
            }
            for ref in refs
        },
        "workflow_policy": {"max_workflow_steps": max_workflow_steps},
    }
    return CapabilitySetPlannerView(
        config=config,
        profile_ref_to_worker_id=ref_to_worker,
        worker_id_to_profile_ref=worker_to_ref,
    )


def anonymous_planner_config(
    binding: PoolBinding,
    *,
    max_workflow_steps: int = 5,
) -> dict[str, Any]:
    """Render pool-specific capabilities without exposing concrete model identities."""
    if max_workflow_steps <= 0:
        raise PoolBindingError("max_workflow_steps must be positive")
    names = [f"worker_{slot.worker_id}" for slot in binding.slots]
    return {
        "schema_version": ANONYMOUS_PLANNER_CONFIG_SCHEMA,
        "worker_pool_names": names,
        "lane_worker_masks": {"single_turn": names},
        "worker_pool": {
            name: {
                "name": name,
                "worker_id": slot.worker_id,
                "role_prior": list(slot.role_prior),
            }
            for name, slot in zip(names, binding.slots, strict=True)
        },
        "workflow_policy": {"max_workflow_steps": max_workflow_steps},
    }


def verify_anonymous_planner_config(
    config: dict[str, Any],
    binding: PoolBinding,
    *,
    max_workflow_steps: int = 5,
) -> None:
    """Fail closed if slot semantics differ from the supplied versioned binding."""
    expected = anonymous_planner_config(
        binding,
        max_workflow_steps=max_workflow_steps,
    )
    if config != expected:
        raise PoolBindingError(
            "anonymous planner config does not match the supplied pool binding"
        )


def verify_capability_set_planner_config(
    config: dict[str, Any],
    binding: PoolBinding,
    *,
    max_workflow_steps: int = 5,
) -> None:
    expected = capability_set_planner_view(
        binding,
        max_workflow_steps=max_workflow_steps,
    ).config
    if config != expected:
        raise PoolBindingError(
            "capability-set planner config does not match the supplied pool binding"
        )
