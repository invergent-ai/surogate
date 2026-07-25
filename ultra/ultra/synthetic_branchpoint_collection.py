"""Parallel exact-token collection for one-call synthetic branchpoints."""

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
from .synthetic_branchpoints import (
    BRANCHPOINT_CURRICULUM_REVISION,
    FIXED_CONTINUATION_MODE,
    FIXED_CONTINUATION_REVISION,
    BranchpointController,
    BranchpointPolicyAttestation,
    SyntheticBranchpointSample,
    SyntheticBranchpointScenario,
    build_synthetic_branchpoint_curriculum,
    sample_synthetic_branchpoint,
)


SYNTHETIC_BRANCHPOINT_COLLECTION_VERSION = (
    "fugu_synthetic_branchpoint_collection_v1"
)


class SyntheticBranchpointCollectionError(ValueError):
    """A branchpoint collection cannot safely enter training."""


BranchpointControllerFactory = Callable[
    [SyntheticBranchpointScenario, BranchpointPolicyAttestation],
    BranchpointController,
]


def _serialize_sample(
    sample: SyntheticBranchpointSample,
    *,
    scenario: SyntheticBranchpointScenario,
    sample_index: int,
) -> dict[str, Any]:
    action = (
        json.loads(
            serialize_capability_control_action(
                sample.action,
                capability_reference_map(scenario.state.workers),
            )
        )
        if sample.action is not None
        else None
    )
    return {
        "sample_index": sample_index,
        "sample_id": (
            f"{scenario.scenario_id}:sample-{sample_index:03d}"
        ),
        "policy": {
            "behavior_policy_revision": (
                sample.policy.behavior_policy_revision
            ),
            "runtime_revision": sample.policy.runtime_revision,
            "pool_id": sample.policy.pool_id,
            "pool_binding_revision": sample.policy.pool_binding_revision,
            "sampling_seed": sample.policy.sampling_seed,
        },
        "action": action,
        "disposition": sample.disposition,
        "training_eligible": sample.training_eligible,
        "reward": sample.reward,
        "outcome": sample.outcome,
        "events": list(sample.events),
        "evidence": sample.evidence,
        "trace": sample.trace,
    }


def _assert_same_prompt(
    scenario: SyntheticBranchpointScenario,
    samples: list[tuple[int, SyntheticBranchpointSample]],
) -> None:
    if not samples:
        raise SyntheticBranchpointCollectionError(
            f"scenario {scenario.scenario_id} has no samples"
        )
    reference_messages = samples[0][1].trace.get("messages")
    reference_prompt_ids = samples[0][1].trace.get("prompt_token_ids")
    for _, sample in samples[1:]:
        if (
            sample.trace.get("messages") != reference_messages
            or sample.trace.get("prompt_token_ids") != reference_prompt_ids
        ):
            raise SyntheticBranchpointCollectionError(
                "samples from one scenario must have exactly identical prompts"
            )


async def collect_synthetic_branchpoints(
    *,
    output_dir: Path,
    behavior_policy_revision: str,
    runtime_revision: str,
    pool_binding_path: Path,
    scenario_count: int,
    samples_per_scenario: int,
    seed: int,
    concurrency: int,
    controller_factory: BranchpointControllerFactory,
) -> dict[str, Any]:
    """Collect one fresh controller call per scenario/sample in parallel."""
    if output_dir.exists():
        raise SyntheticBranchpointCollectionError(
            f"refusing to overwrite branchpoint collection: {output_dir}"
        )
    for label, value in (
        ("behavior_policy_revision", behavior_policy_revision),
        ("runtime_revision", runtime_revision),
    ):
        if not isinstance(value, str) or not value.strip():
            raise SyntheticBranchpointCollectionError(
                f"{label} must be non-empty"
            )
    for label, value in (
        ("scenario_count", scenario_count),
        ("samples_per_scenario", samples_per_scenario),
        ("concurrency", concurrency),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SyntheticBranchpointCollectionError(
                f"{label} must be a positive integer"
            )
    if samples_per_scenario < 2:
        raise SyntheticBranchpointCollectionError(
            "branchpoint GRPO requires at least two samples per scenario"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SyntheticBranchpointCollectionError("seed must be an integer")

    binding_path = pool_binding_path.expanduser().resolve()
    binding = load_pool_binding(binding_path)
    scenarios = build_synthetic_branchpoint_curriculum(
        count=scenario_count,
        seed=seed,
        profile_capabilities=tuple(slot.role_prior for slot in binding.slots),
    )

    jobs: list[
        tuple[
            int,
            int,
            SyntheticBranchpointScenario,
            BranchpointPolicyAttestation,
            BranchpointController,
        ]
    ] = []
    controller_ids: set[int] = set()
    sampling_seeds: set[int] = set()
    for scenario_index, scenario in enumerate(scenarios):
        for sample_index in range(samples_per_scenario):
            job_index = scenario_index * samples_per_scenario + sample_index
            sampling_seed = seed + 10_000_019 + job_index
            if sampling_seed in sampling_seeds:
                raise SyntheticBranchpointCollectionError(
                    "branchpoint sampling seeds are not unique"
                )
            sampling_seeds.add(sampling_seed)
            policy = BranchpointPolicyAttestation(
                behavior_policy_revision=behavior_policy_revision,
                runtime_revision=runtime_revision,
                pool_id=binding.pool_id,
                pool_binding_revision=binding.binding_revision,
                sampling_seed=sampling_seed,
            )
            controller = controller_factory(scenario, policy)
            if id(controller) in controller_ids:
                raise SyntheticBranchpointCollectionError(
                    "parallel branchpoints require one fresh controller per sample"
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
        scenario: SyntheticBranchpointScenario,
        policy: BranchpointPolicyAttestation,
        controller: BranchpointController,
    ) -> tuple[int, int, SyntheticBranchpointSample]:
        async with semaphore:
            sample = await sample_synthetic_branchpoint(
                controller,
                scenario,
                policy=policy,
            )
        return scenario_index, sample_index, sample

    tasks: list[
        asyncio.Task[tuple[int, int, SyntheticBranchpointSample]]
    ] = []
    async with asyncio.TaskGroup() as task_group:
        for job in jobs:
            tasks.append(task_group.create_task(run_one(*job)))
    sampled = sorted(
        (task.result() for task in tasks),
        key=lambda row: row[:2],
    )

    grouped: list[dict[str, Any]] = []
    all_dispositions: Counter[str] = Counter()
    all_rewards: Counter[str] = Counter()
    for scenario_index, scenario in enumerate(scenarios):
        scenario_samples = [
            (sample_index, sample)
            for collected_index, sample_index, sample in sampled
            if collected_index == scenario_index
        ]
        _assert_same_prompt(scenario, scenario_samples)
        dispositions = Counter(
            sample.disposition for _, sample in scenario_samples
        )
        rewards = Counter(
            str(float(sample.reward))
            for _, sample in scenario_samples
            if sample.reward is not None
        )
        all_dispositions.update(dispositions)
        all_rewards.update(rewards)
        grouped.append(
            {
                "scenario_index": scenario_index,
                "scenario_id": scenario.scenario_id,
                "motif": scenario.motif,
                "evidence_basis": list(scenario.evidence_basis),
                "sample_count": len(scenario_samples),
                "disposition_counts": dict(sorted(dispositions.items())),
                "reward_counts": dict(sorted(rewards.items())),
                "samples": [
                    _serialize_sample(
                        sample,
                        scenario=scenario,
                        sample_index=sample_index,
                    )
                    for sample_index, sample in scenario_samples
                ],
            }
        )

    report: dict[str, Any] = {
        "version": SYNTHETIC_BRANCHPOINT_COLLECTION_VERSION,
        "verdict": "SYNTHETIC_BRANCHPOINTS_COLLECTED",
        "behavior_policy_revision": behavior_policy_revision,
        "runtime_revision": runtime_revision,
        "pool_id": binding.pool_id,
        "pool_binding_revision": binding.binding_revision,
        "pool_binding": str(binding_path),
        "curriculum_revision": BRANCHPOINT_CURRICULUM_REVISION,
        "fixed_continuation": {
            "revision": FIXED_CONTINUATION_REVISION,
            "mode": FIXED_CONTINUATION_MODE,
        },
        "sampling_temperature": 1.0,
        "scenario_seed": seed,
        "scenario_count": scenario_count,
        "samples_per_scenario": samples_per_scenario,
        "sample_count": len(sampled),
        "eligible_count": all_dispositions["eligible"],
        "disposition_counts": dict(sorted(all_dispositions.items())),
        "reward_counts": dict(sorted(all_rewards.items())),
        "scenarios": grouped,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "collection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "SYNTHETIC_BRANCHPOINT_COLLECTION_VERSION",
    "SyntheticBranchpointCollectionError",
    "collect_synthetic_branchpoints",
]
