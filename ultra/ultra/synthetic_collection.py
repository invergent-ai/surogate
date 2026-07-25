"""Parallel exact-token collection over synthetic conductor rollouts."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .live_control import (
    capability_reference_map,
    serialize_capability_control_action,
)
from .pool_binding import load_pool_binding
from .synthetic_rollouts import (
    SYNTHETIC_CURRICULUM_REVISION,
    SyntheticController,
    SyntheticPolicyAttestation,
    SyntheticSampledRollout,
    SyntheticScenario,
    build_synthetic_curriculum,
    sample_synthetic_rollout,
)


SYNTHETIC_COLLECTION_VERSION = "fugu_synthetic_rollout_collection_v1"


class SyntheticCollectionError(ValueError):
    """A synthetic rollout collection cannot safely enter training."""


ControllerFactory = Callable[
    [SyntheticScenario, SyntheticPolicyAttestation],
    SyntheticController,
]


def _serialize_rollout(
    rollout: SyntheticSampledRollout,
    *,
    scenario: SyntheticScenario,
    sample_index: int,
) -> dict[str, Any]:
    if len(rollout.decisions) != len(rollout.model_traces):
        raise SyntheticCollectionError(
            "synthetic rollout decisions and exact traces are misaligned"
        )
    boundaries = scenario.boundary_map()
    decisions: list[dict[str, Any]] = []
    for decision, trace in zip(
        rollout.decisions,
        rollout.model_traces,
        strict=True,
    ):
        boundary = boundaries.get(decision.boundary_id)
        if boundary is None:
            raise SyntheticCollectionError(
                f"rollout references unknown boundary {decision.boundary_id!r}"
            )
        action = (
            json.loads(
                serialize_capability_control_action(
                    decision.action,
                    capability_reference_map(boundary.state.workers),
                )
            )
            if decision.action is not None
            else None
        )
        decisions.append(
            {
                "boundary_id": decision.boundary_id,
                "action": action,
                "matched_outcome_path": decision.matched_oracle,
                "transition_outcome": decision.outcome,
                "trace": trace,
            }
        )
    return {
        "sample_index": sample_index,
        "reward": rollout.reward,
        "outcome": rollout.outcome,
        "policy": {
            "behavior_policy_revision": rollout.policy.behavior_policy_revision,
            "runtime_revision": rollout.policy.runtime_revision,
            "pool_id": rollout.policy.pool_id,
            "pool_binding_revision": rollout.policy.pool_binding_revision,
            "sampling_seed": rollout.policy.sampling_seed,
        },
        "decisions": decisions,
    }


async def collect_synthetic_rollouts(
    *,
    output_dir: Path,
    behavior_policy_revision: str,
    runtime_revision: str,
    pool_binding_path: Path,
    scenario_count: int,
    samples_per_scenario: int,
    seed: int,
    concurrency: int,
    controller_factory: ControllerFactory,
) -> dict[str, Any]:
    """Collect distinct policy samples over a deterministic synthetic curriculum."""
    if output_dir.exists():
        raise SyntheticCollectionError(
            f"refusing to overwrite synthetic collection: {output_dir}"
        )
    for label, value in (
        ("behavior_policy_revision", behavior_policy_revision),
        ("runtime_revision", runtime_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise SyntheticCollectionError(f"{label} must be non-empty")
    for label, value in (
        ("scenario_count", scenario_count),
        ("samples_per_scenario", samples_per_scenario),
        ("concurrency", concurrency),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SyntheticCollectionError(f"{label} must be a positive integer")
    if samples_per_scenario < 2:
        raise SyntheticCollectionError(
            "GRPO collection requires at least two samples per scenario"
        )

    pool_binding_path = pool_binding_path.expanduser().resolve()
    binding = load_pool_binding(pool_binding_path)
    scenarios = build_synthetic_curriculum(
        count=scenario_count,
        seed=seed,
        profile_capabilities=tuple(slot.role_prior for slot in binding.slots),
    )
    jobs: list[
        tuple[
            int,
            int,
            SyntheticScenario,
            SyntheticPolicyAttestation,
            SyntheticController,
        ]
    ] = []
    controller_ids: set[int] = set()
    sampling_seeds: set[int] = set()
    for scenario_index, scenario in enumerate(scenarios):
        for sample_index in range(samples_per_scenario):
            sampling_seed = seed + 1_000_003 * scenario_index + sample_index
            if sampling_seed in sampling_seeds:
                raise SyntheticCollectionError("sampling seeds are not unique")
            sampling_seeds.add(sampling_seed)
            policy = SyntheticPolicyAttestation(
                behavior_policy_revision=behavior_policy_revision,
                runtime_revision=runtime_revision,
                pool_id=binding.pool_id,
                pool_binding_revision=binding.binding_revision,
                sampling_seed=sampling_seed,
            )
            controller = controller_factory(scenario, policy)
            if id(controller) in controller_ids:
                raise SyntheticCollectionError(
                    "parallel synthetic rollouts require one controller per sample"
                )
            controller_ids.add(id(controller))
            jobs.append(
                (
                    scenario_index,
                    sample_index,
                    scenario,
                    policy,
                    controller,
                )
            )

    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(
        scenario_index: int,
        sample_index: int,
        scenario: SyntheticScenario,
        policy: SyntheticPolicyAttestation,
        controller: SyntheticController,
    ) -> tuple[int, int, SyntheticSampledRollout]:
        async with semaphore:
            rollout = await sample_synthetic_rollout(
                controller,
                scenario,
                policy=policy,
            )
        return scenario_index, sample_index, rollout

    tasks: list[asyncio.Task[tuple[int, int, SyntheticSampledRollout]]] = []
    async with asyncio.TaskGroup() as task_group:
        for job in jobs:
            tasks.append(task_group.create_task(run_one(*job)))
    sampled = sorted((task.result() for task in tasks), key=lambda row: row[:2])

    grouped: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_rollouts = [
            (sample_index, rollout)
            for collected_index, sample_index, rollout in sampled
            if collected_index == scenario_index
        ]
        reward_counts = Counter(
            str(float(rollout.reward))
            for _, rollout in scenario_rollouts
        )
        grouped.append(
            {
                "scenario_index": scenario_index,
                "scenario_id": scenario.scenario_id,
                "motif": scenario.motif,
                "evidence_basis": list(scenario.evidence_basis),
                "boundary_count": len(scenario.boundaries),
                "reward_counts": dict(sorted(reward_counts.items())),
                "rollouts": [
                    _serialize_rollout(
                        rollout,
                        scenario=scenario,
                        sample_index=sample_index,
                    )
                    for sample_index, rollout in scenario_rollouts
                ],
            }
        )

    report = {
        "version": SYNTHETIC_COLLECTION_VERSION,
        "verdict": "SYNTHETIC_EXACT_TOKEN_ROLLOUTS_COLLECTED",
        "behavior_policy_revision": behavior_policy_revision,
        "runtime_revision": runtime_revision,
        "pool_id": binding.pool_id,
        "pool_binding_revision": binding.binding_revision,
        "pool_binding": str(pool_binding_path),
        "curriculum_revision": SYNTHETIC_CURRICULUM_REVISION,
        "sampling_temperature": 1.0,
        "scenario_seed": seed,
        "scenario_count": scenario_count,
        "samples_per_scenario": samples_per_scenario,
        "rollout_count": len(sampled),
        "scenarios": grouped,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    output_dir.mkdir(parents=True)
    report_path = output_dir / "collection.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "SYNTHETIC_COLLECTION_VERSION",
    "SyntheticCollectionError",
    "collect_synthetic_rollouts",
]
