from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import msgspec
import pytest
from ultra.behavior_likelihood import (
    full_vocabulary_behavior_likelihood_contract,
)
from ultra.live_control import (
    ControlAction,
    ControlContractError,
    ControlStep,
    capability_reference_map,
    serialize_capability_control_action,
    validate_control_action,
)
from ultra.pool_binding import load_pool_binding
from ultra.synthetic_branchpoint_collection import (
    SYNTHETIC_BRANCHPOINT_COLLECTION_VERSION,
)
from ultra.synthetic_branchpoints import (
    BRANCHPOINT_CURRICULUM_REVISION,
    FIXED_CONTINUATION_MODE,
    FIXED_CONTINUATION_REVISION,
    build_synthetic_branchpoint_curriculum,
    evaluate_synthetic_branchpoint_action,
)

from surogate.grpo.synthetic_branchpoint_batch import (
    MIN_CONTRAST_GROUPS,
    MIN_POLICY_SAMPLES,
    REQUIRED_CONTRAST_BY_MOTIF,
    REQUIRED_GROUPS_BY_CONTRAST,
    SYNTHETIC_BRANCHPOINT_BATCH_VERSION,
    SYNTHETIC_BRANCHPOINT_CREDIT_MODE,
    SyntheticBranchpointBatchError,
    _exact_completion_span_mask,
    _json_value_span,
    materialize_synthetic_branchpoint_grpo_update,
    validate_synthetic_branchpoint_prepared_batch,
)
from surogate.grpo.transport import TrainingBatch

ROOT = Path(__file__).resolve().parents[2]
REPLAY = ROOT / "scratchpad/fugu_27b_transfer_replay_v1/replay.bin"
RETENTION_REPLAY = (
    ROOT
    / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/replay.bin"
)
RETENTION_REPORT = (
    ROOT
    / "scratchpad/fugu_27b_action_balanced_retention_replay_v2/report.json"
)
POLICY_REVISION = "fugu-conductor-branchpoint-test-parent"
RUNTIME_REVISION = "20260724-r84-completion-attested-finalization"
POOL_ID = "anonymous-branchpoint-test-pool-v1"
SCENARIO_SEED = 700
SAMPLES_PER_SCENARIO = 5

ROLE_PRIORS = (
    ("reasoner", "verifier", "debugger"),
    ("scientist", "planner", "aggregator"),
    ("mathematician", "coder", "reasoner"),
    ("drafter", "implementer", "fast_pass"),
)


def _text_token_ids(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _prompt_token_ids(messages: list[dict[str, str]]) -> list[int]:
    return _text_token_ids(
        json.dumps(
            messages,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


class _LocalTestTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        return {"input_ids": _prompt_token_ids(messages)}

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return bytes(token_ids).decode("utf-8")


@pytest.fixture(autouse=True)
def _patch_local_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "surogate.grpo.synthetic_branchpoint_batch."
        "AutoTokenizer.from_pretrained",
        lambda path, *, local_files_only: _LocalTestTokenizer(),
    )


def _write_binding(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "fugu_pool_binding_v1",
                "pool_id": POOL_ID,
                "binding_revision": POOL_ID,
                "provider_base": "https://yunwu.ai/v1",
                "slots": [
                    {
                        "worker_id": index,
                        "training_name": f"bound-worker-{index}",
                        "model_alias": f"alias-{index}",
                        "runtime_model": f"runtime-{index}",
                        "reasoning_effort": "high",
                        "role_prior": list(roles),
                    }
                    for index, roles in enumerate(ROLE_PRIORS)
                ],
                "checkpoint": {
                    "adapter_path": "scratchpad/test-adapter",
                    "base_model_snapshot": "Qwen/Qwen3.6-27B-FP8",
                    "trained_control_contract": (
                        "unified_capability_action_v2"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    return path.resolve()


def _sampling_seed(
    scenario_index: int,
    sample_index: int,
) -> int:
    return (
        SCENARIO_SEED
        + 10_000_019
        + scenario_index * SAMPLES_PER_SCENARIO
        + sample_index
    )


def _trace(
    *,
    scenario_index: int,
    sample_index: int,
    response: str,
) -> dict:
    seed = _sampling_seed(scenario_index, sample_index)
    messages = [
        {
            "role": "system",
            "content": "Anonymous fixed-continuation conductor.",
        },
        {
            "role": "user",
            "content": f"branchpoint scenario={scenario_index}",
        },
    ]
    completion_ids = _text_token_ids(response)
    return {
        "messages": messages,
        "response": response,
        "finish_reason": "stop",
        "correction": None,
        "prompt_token_ids": _prompt_token_ids(messages),
        "completion_token_ids": completion_ids,
        "completion_logprobs": [-0.1] * len(completion_ids),
        "temperature": 1.0,
        "seed": seed,
        "behavior_likelihood_contract": (
            full_vocabulary_behavior_likelihood_contract()
        ),
    }


def _candidate_actions(scenario) -> list[ControlAction]:
    reason = "Use observed workflow evidence and verification."
    candidates = [
        ControlAction(action="continue", reason=reason),
        ControlAction(action="complete", reason=reason),
    ]
    candidates.extend(
        ControlAction(
            action="handoff",
            reason=reason,
            target_position_id=position.position_id,
        )
        for position in scenario.state.positions
        if position.status == "pending"
    )
    legal: list[ControlAction] = []
    for action in candidates:
        try:
            validate_control_action(action, scenario.state)
        except ControlContractError:
            continue
        legal.append(action)
    return legal


def _required_pair(
    scenario,
) -> tuple[ControlAction, ControlAction] | None:
    if scenario.motif not in REQUIRED_CONTRAST_BY_MOTIF:
        return None
    return _replan_only_pair(scenario)


def _replan_only_pair(
    scenario,
) -> tuple[ControlAction, ControlAction]:
    implementer = next(
        worker.worker_id
        for worker in scenario.state.workers
        if "implementer" in worker.capability_tags
    )
    verifier = next(
        worker.worker_id
        for worker in scenario.state.workers
        if "verifier" in worker.capability_tags
    )
    if scenario.motif == "unverified_completion_pending_verifier":
        positive_steps = (
            ControlStep(
                worker_id=verifier,
                subtask=(
                    f"Independently run the {scenario.check_label}."
                ),
            ),
        )
        negative_steps = (
            ControlStep(
                worker_id=verifier,
                subtask=f"Inspect the {scenario.artifact_label}.",
            ),
        )
    else:
        material_verb = (
            "Repair"
            if scenario.motif
            in {
                "failed_verification_repair_and_reverify",
                "active_private_loop_continue_before_handoff",
            }
            else "Implement"
        )
        material_step = ControlStep(
            worker_id=implementer,
            subtask=(
                f"{material_verb} the {scenario.artifact_label}."
            ),
        )
        positive_steps = (
            material_step,
            ControlStep(
                worker_id=verifier,
                subtask=(
                    f"Independently run the {scenario.check_label}."
                ),
                access=(0,),
            ),
        )
        negative_steps = (material_step,)
    positive = ControlAction(
        action="replan",
        reason=(
            "Replan to satisfy the artifact and independent verification "
            "obligations."
        ),
        steps=positive_steps,
    )
    negative = ControlAction(
        action="replan",
        reason="Replan from the observed workflow evidence.",
        steps=negative_steps,
    )
    assert evaluate_synthetic_branchpoint_action(
        scenario,
        positive,
    ).reward == 1.0
    assert evaluate_synthetic_branchpoint_action(
        scenario,
        negative,
    ).reward == 0.0
    return positive, negative


def _semantic_sample(
    *,
    scenario,
    scenario_index: int,
    sample_index: int,
    action: ControlAction,
) -> dict:
    evaluation = evaluate_synthetic_branchpoint_action(scenario, action)
    response = serialize_capability_control_action(
        action,
        capability_reference_map(scenario.state.workers),
    )
    return {
        "sample_index": sample_index,
        "sample_id": (
            f"{scenario.scenario_id}:sample-{sample_index:03d}"
        ),
        "policy": {
            "behavior_policy_revision": POLICY_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "pool_id": POOL_ID,
            "pool_binding_revision": POOL_ID,
            "sampling_seed": _sampling_seed(
                scenario_index,
                sample_index,
            ),
        },
        "action": json.loads(response),
        "disposition": evaluation.disposition,
        "training_eligible": evaluation.training_eligible,
        "reward": evaluation.reward,
        "outcome": evaluation.outcome,
        "events": list(evaluation.events),
        "evidence": evaluation.evidence,
        "trace": _trace(
            scenario_index=scenario_index,
            sample_index=sample_index,
            response=response,
        ),
    }


def _protocol_sample(
    *,
    scenario,
    scenario_index: int,
    sample_index: int,
) -> dict:
    outcome = "protocol_only:ControlContractError"
    return {
        "sample_index": sample_index,
        "sample_id": (
            f"{scenario.scenario_id}:sample-{sample_index:03d}"
        ),
        "policy": {
            "behavior_policy_revision": POLICY_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "pool_id": POOL_ID,
            "pool_binding_revision": POOL_ID,
            "sampling_seed": _sampling_seed(
                scenario_index,
                sample_index,
            ),
        },
        "action": None,
        "disposition": "protocol_only",
        "training_eligible": False,
        "reward": None,
        "outcome": outcome,
        "events": [],
        "evidence": {
            "exclusion_reason": outcome,
            "semantic_reward_assigned": False,
        },
        "trace": _trace(
            scenario_index=scenario_index,
            sample_index=sample_index,
            response="not-json",
        ),
    }


def _write_collection(
    path: Path,
    *,
    binding_path: Path,
    scenario_count: int = 24,
) -> Path:
    binding = load_pool_binding(binding_path)
    scenarios = build_synthetic_branchpoint_curriculum(
        count=scenario_count,
        seed=SCENARIO_SEED,
        profile_capabilities=tuple(
            slot.role_prior for slot in binding.slots
        ),
    )
    raw_scenarios: list[dict] = []
    top_dispositions: Counter[str] = Counter()
    top_rewards: Counter[str] = Counter()
    eligible_count = 0
    for scenario_index, scenario in enumerate(scenarios):
        required_pair = _required_pair(scenario)
        if required_pair is None:
            negative = _candidate_actions(scenario)[0]
            actions = [negative] * 4
        else:
            positive, negative = required_pair
            actions = [positive, negative, positive, negative]
        samples = [
            _semantic_sample(
                scenario=scenario,
                scenario_index=scenario_index,
                sample_index=sample_index,
                action=action,
            )
            for sample_index, action in enumerate(actions)
        ]
        samples.append(
            _protocol_sample(
                scenario=scenario,
                scenario_index=scenario_index,
                sample_index=4,
            )
        )
        dispositions = Counter(
            sample["disposition"] for sample in samples
        )
        rewards = Counter(
            str(float(sample["reward"]))
            for sample in samples
            if sample["reward"] is not None
        )
        top_dispositions.update(dispositions)
        top_rewards.update(rewards)
        eligible_count += dispositions["eligible"]
        raw_scenarios.append(
            {
                "scenario_index": scenario_index,
                "scenario_id": scenario.scenario_id,
                "motif": scenario.motif,
                "evidence_basis": list(scenario.evidence_basis),
                "sample_count": len(samples),
                "disposition_counts": dict(
                    sorted(dispositions.items())
                ),
                "reward_counts": dict(sorted(rewards.items())),
                "samples": samples,
            }
        )
    report = {
        "version": SYNTHETIC_BRANCHPOINT_COLLECTION_VERSION,
        "verdict": "SYNTHETIC_BRANCHPOINTS_COLLECTED",
        "behavior_policy_revision": POLICY_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "pool_id": POOL_ID,
        "pool_binding_revision": POOL_ID,
        "pool_binding": str(binding_path),
        "curriculum_revision": BRANCHPOINT_CURRICULUM_REVISION,
        "fixed_continuation": {
            "revision": FIXED_CONTINUATION_REVISION,
            "mode": FIXED_CONTINUATION_MODE,
        },
        "sampling_temperature": 1.0,
        "scenario_seed": SCENARIO_SEED,
        "scenario_count": scenario_count,
        "samples_per_scenario": SAMPLES_PER_SCENARIO,
        "sample_count": scenario_count * SAMPLES_PER_SCENARIO,
        "eligible_count": eligible_count,
        "disposition_counts": dict(sorted(top_dispositions.items())),
        "reward_counts": dict(sorted(top_rewards.items())),
        "scenarios": raw_scenarios,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path.resolve()


def _materialize(
    tmp_path: Path,
    *,
    collection_path: Path,
    binding_path: Path,
    name: str = "prepared",
) -> dict:
    tokenizer_model_path = tmp_path / "tokenizer-model"
    tokenizer_model_path.mkdir(exist_ok=True)
    return materialize_synthetic_branchpoint_grpo_update(
        collection_path=collection_path,
        output_dir=tmp_path / name,
        expected_behavior_policy_revision=POLICY_REVISION,
        expected_runtime_revision=RUNTIME_REVISION,
        pool_binding_path=binding_path,
        tokenizer_model_path=tokenizer_model_path,
        replay_path=REPLAY,
        train_retention_replay_path=RETENTION_REPLAY,
        train_retention_report_path=RETENTION_REPORT,
    )


def test_materializes_low_budget_topology_contrasts_and_replay(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )

    report = _materialize(
        tmp_path,
        collection_path=collection_path,
        binding_path=binding_path,
    )

    batch = msgspec.msgpack.decode(
        Path(report["combined_batch"]["path"]).read_bytes(),
        type=TrainingBatch,
    )
    policy_count = report["policy"]["samples"]
    assert report["version"] == SYNTHETIC_BRANCHPOINT_BATCH_VERSION
    assert report["tokenizer_model"] == str(
        (tmp_path / "tokenizer-model").resolve()
    )
    assert report["optimizer_contract"][
        "exact_token_semantic_binding"
    ]["validated_rows"] == "all_collection_traces"
    assert len(report["credit_groups"]) == MIN_CONTRAST_GROUPS
    assert policy_count == MIN_POLICY_SAMPLES
    assert report["policy"]["positive_samples"] == MIN_CONTRAST_GROUPS
    assert report["policy"]["negative_samples"] == MIN_CONTRAST_GROUPS
    assert report["policy"]["coverage_counts"] == {
        kind: count
        for kind, count in REQUIRED_GROUPS_BY_CONTRAST.items()
    }
    assert len(
        {group["context_key"] for group in report["credit_groups"]}
    ) == MIN_CONTRAST_GROUPS
    assert report["optimizer_contract"]["sample_packing"] is False
    assert report["optimizer_contract"][
        "policy_credit_assignment"
    ]["mode"] == SYNTHETIC_BRANCHPOINT_CREDIT_MODE
    assert report["optimizer_contract"][
        "policy_credit_assignment"
    ]["uncertainty_admission"] == {
        "minimum_eligible_outcomes_per_arm": 2,
        "success_fraction_window": [0.2, 0.8],
        "ranking": "closest_to_half_then_strongest_balanced_support",
    }
    assert report["optimizer_contract"][
        "policy_credit_assignment"
    ]["excluded_dispositions"] == [
        "protocol_only",
        "unmodeled",
    ]
    assert all(
        group["positive_policy_sample_indices"]
        and group["negative_policy_sample_indices"]
        for group in report["credit_groups"]
    )
    for group in report["credit_groups"]:
        assert group["contrast_kind"] == "low_budget_replan_topology"
        assert group["eligible_topology_positive_outcomes"] >= 2
        assert group["eligible_topology_negative_outcomes"] >= 2
        assert 0.2 <= group["eligible_topology_success_fraction"] <= 0.8
        assert {
            row["reward"]: row["action"]
            for row in group["source_rows"]
        } == {0.0: "replan", 1.0: "replan"}
    assert all(
        row["target_position_id"] is None
        for group in report["credit_groups"]
        for row in group["source_rows"]
    )
    assert len(batch.examples) == policy_count + 52 + 76
    assert all(
        not any(sample.replay_mask or [])
        for sample in batch.examples[:policy_count]
    )
    assert all(
        all(sample.completion_mask)
        and sample.advantage_mask is not None
        and sum(sample.advantage_mask) == 1
        and not any(
            sample.advantage_mask[: len(sample.prompt_ids)]
        )
        for sample in batch.examples[:policy_count]
    )
    assert report["policy"]["credited_tokens"] == policy_count
    assert report["policy"]["absolute_outcome_seed_mass"] == (
        pytest.approx(
            sum(
                abs(float(sample.advantage or 0.0))
                * sum(sample.advantage_mask or [])
                for sample in batch.examples[:policy_count]
            )
        )
    )
    for group in report["credit_groups"]:
        for row in group["source_rows"]:
            sample = batch.examples[row["policy_sample_index"]]
            selected = [
                index - len(sample.prompt_ids)
                for index, value in enumerate(
                    sample.advantage_mask or []
                )
                if value
            ]
            assert len(selected) == 1
            selected_text = chr(sample.completion_ids[selected[0]])
            assert row["credit_field"] == "steps_continuation_delimiter"
            assert selected_text == ("," if row["reward"] == 1.0 else "]")
    assert all(
        sample.advantage_mask is None
        for sample in batch.examples[policy_count:]
    )
    assert sum(
        sum(sample.replay_mask or [])
        for sample in batch.examples[
            policy_count : policy_count + 52
        ]
    ) == 2_448
    assert sum(
        sum(sample.replay_mask or [])
        for sample in batch.examples[policy_count + 52 :]
    ) == 17_760

    validated, validated_path, validated_batch = (
        validate_synthetic_branchpoint_prepared_batch(
            prepared_report_path=(
                tmp_path / "prepared/prepared_report.json"
            ),
            expected_behavior_policy_revision=POLICY_REVISION,
            expected_runtime_revision=RUNTIME_REVISION,
            pool_binding_path=binding_path,
            tokenizer_model_path=tmp_path / "tokenizer-model",
            replay_path=REPLAY,
            train_retention_replay_path=RETENTION_REPLAY,
            train_retention_report_path=RETENTION_REPORT,
        )
    )
    assert validated == report
    assert validated_path == Path(report["combined_batch"]["path"])
    assert validated_batch == batch


def test_rejects_tampered_fixed_continuation_report(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    tampered = False
    for scenario in collection["scenarios"]:
        for sample in scenario["samples"]:
            for event in reversed(sample["events"]):
                if isinstance(event.get("passed"), bool):
                    event["passed"] = not event["passed"]
                    tampered = True
                    break
            if tampered:
                break
        if tampered:
            break
    assert tampered
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="differs from replay",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_unmodeled_and_protocol_only_rows_are_both_excluded(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    binding = load_pool_binding(binding_path)
    scenario = build_synthetic_branchpoint_curriculum(
        count=10,
        seed=SCENARIO_SEED,
        profile_capabilities=tuple(
            slot.role_prior for slot in binding.slots
        ),
    )[0]
    unmodeled_action = ControlAction(
        action="replan",
        reason="Use observed workflow evidence.",
        steps=(
            ControlStep(
                worker_id=0,
                subtask="Coordinate workflow.",
            ),
        ),
    )
    unmodeled = _semantic_sample(
        scenario=scenario,
        scenario_index=0,
        sample_index=4,
        action=unmodeled_action,
    )
    assert unmodeled["disposition"] == "unmodeled"
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["scenarios"][0]["samples"][4] = unmodeled
    collection["scenarios"][0]["disposition_counts"] = {
        "eligible": 4,
        "unmodeled": 1,
    }
    original_protocol = collection["disposition_counts"]["protocol_only"]
    collection["disposition_counts"] = {
        "eligible": collection["eligible_count"],
        "protocol_only": original_protocol - 1,
        "unmodeled": 1,
    }
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    report = _materialize(
        tmp_path,
        collection_path=collection_path,
        binding_path=binding_path,
    )
    batch = msgspec.msgpack.decode(
        Path(report["combined_batch"]["path"]).read_bytes(),
        type=TrainingBatch,
    )
    policy = batch.examples[: report["policy"]["samples"]]
    excluded_completion = unmodeled["trace"]["completion_token_ids"]
    assert all(
        sample.completion_ids != excluded_completion
        for sample in policy
    )
    # Protocol-only sample 4 from every other scenario is excluded as well.
    assert all(
        sample.completion_ids
        != _trace(
            scenario_index=1,
            sample_index=4,
            response="not-json",
        )["completion_token_ids"]
        for sample in policy
    )


def test_parseable_length_truncation_is_excluded_without_replay_failure(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    binding = load_pool_binding(binding_path)
    scenario_index = 3
    scenario = build_synthetic_branchpoint_curriculum(
        count=20,
        seed=SCENARIO_SEED,
        profile_capabilities=tuple(
            slot.role_prior for slot in binding.slots
        ),
    )[scenario_index]
    action = _candidate_actions(scenario)[0]
    response = serialize_capability_control_action(
        action,
        capability_reference_map(scenario.state.workers),
    )
    trace = _trace(
        scenario_index=scenario_index,
        sample_index=4,
        response=response,
    )
    trace["finish_reason"] = "length"
    outcome = "protocol_only:length_truncated"
    truncated = {
        "sample_index": 4,
        "sample_id": f"{scenario.scenario_id}:sample-004",
        "policy": {
            "behavior_policy_revision": POLICY_REVISION,
            "runtime_revision": RUNTIME_REVISION,
            "pool_id": POOL_ID,
            "pool_binding_revision": POOL_ID,
            "sampling_seed": _sampling_seed(scenario_index, 4),
        },
        "action": None,
        "disposition": "protocol_only",
        "training_eligible": False,
        "reward": None,
        "outcome": outcome,
        "events": [],
        "evidence": {
            "exclusion_reason": outcome,
            "semantic_reward_assigned": False,
        },
        "trace": trace,
    }
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["scenarios"][scenario_index]["samples"][4] = truncated
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    report = _materialize(
        tmp_path,
        collection_path=collection_path,
        binding_path=binding_path,
    )
    scenario_group = next(
        group
        for group in report["credit_groups"]
        if group["scenario_index"] == scenario_index
    )
    assert all(
        row["sample_index"] != 4
        for row in scenario_group["source_rows"]
    )


def test_rejects_same_scenario_prompt_drift(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    trace = collection["scenarios"][0]["samples"][2]["trace"]
    trace["messages"][1]["content"] += " drift"
    trace["prompt_token_ids"] = _prompt_token_ids(trace["messages"])
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="different prompts",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_rejects_prompt_token_ids_not_encoding_messages(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["scenarios"][0]["samples"][0]["trace"][
        "prompt_token_ids"
    ][0] += 1
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="prompt token IDs do not encode trace messages",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_rejects_completion_token_ids_not_decoding_to_response(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["scenarios"][0]["samples"][0]["trace"][
        "completion_token_ids"
    ][0] = ord("[")
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="completion token IDs do not decode to trace response",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_rejects_missing_required_control_contrast(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
        scenario_count=6,
    )

    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="lacks required eligible low-budget replan topology contrasts",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_rejects_replan_pairs_that_change_the_producer_target(
    tmp_path: Path,
) -> None:
    binding_path = _write_binding(tmp_path / "binding.json")
    collection_path = _write_collection(
        tmp_path / "collection.json",
        binding_path=binding_path,
    )
    binding = load_pool_binding(binding_path)
    scenarios = build_synthetic_branchpoint_curriculum(
        count=24,
        seed=SCENARIO_SEED,
        profile_capabilities=tuple(
            slot.role_prior for slot in binding.slots
        ),
    )
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    for scenario_index, scenario in enumerate(scenarios):
        if scenario.motif not in REQUIRED_CONTRAST_BY_MOTIF:
            continue
        positive, negative = _replan_only_pair(scenario)
        alternate_producer = next(
            worker.worker_id
            for worker in scenario.state.workers
            if (
                worker.worker_id != negative.steps[0].worker_id
                and "coder" in worker.capability_tags
            )
        )
        negative = ControlAction(
            action="replan",
            reason=negative.reason,
            steps=(
                ControlStep(
                    worker_id=alternate_producer,
                    subtask=negative.steps[0].subtask,
                ),
            ),
        )
        assert evaluate_synthetic_branchpoint_action(
            scenario,
            negative,
        ).reward == 0.0
        for sample_index, action in enumerate(
            (positive, negative, positive, negative)
        ):
            collection["scenarios"][scenario_index]["samples"][
                sample_index
            ] = _semantic_sample(
                scenario=scenario,
                scenario_index=scenario_index,
                sample_index=sample_index,
                action=action,
            )
    collection_path.write_text(json.dumps(collection), encoding="utf-8")

    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="lacks required eligible low-budget replan topology contrasts",
    ):
        _materialize(
            tmp_path,
            collection_path=collection_path,
            binding_path=binding_path,
        )


def test_exact_span_mapping_accepts_zero_width_eos_without_retokenizing() -> None:
    class Tokenizer:
        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens,
            clean_up_tokenization_spaces,
        ):
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is False
            return bytes(token for token in token_ids if token != 0).decode()

    response = '{"target_position_id": 115}'
    ids = [*response.encode(), 0]
    start = response.index("5")
    mask = _exact_completion_span_mask(
        tokenizer=Tokenizer(),
        completion_ids=ids,
        response=response,
        span=(start, start + 1),
    )

    assert sum(mask) == 1
    assert ids[mask.index(True)] == ord("5")
    assert mask[-1] is False


def test_exact_span_mapping_rejects_nonmonotonic_prefix_decode() -> None:
    class Tokenizer:
        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens,
            clean_up_tokenization_spaces,
        ):
            return "a" if len(token_ids) == 1 else "b"

    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="not monotonic",
    ):
        _exact_completion_span_mask(
            tokenizer=Tokenizer(),
            completion_ids=[1, 2],
            response="bb",
            span=(0, 1),
        )


def test_json_value_span_rejects_duplicate_field_occurrences() -> None:
    response = (
        '{"reason":"target_position_id",'
        '"target_position_id":115,"target_position_id":118}'
    )
    with pytest.raises(
        SyntheticBranchpointBatchError,
        match="exactly one target_position_id field",
    ):
        _json_value_span(
            response,
            field="target_position_id",
            expected_value=118,
        )
