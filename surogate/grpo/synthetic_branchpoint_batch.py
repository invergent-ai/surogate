"""Materialize fixed-continuation synthetic branchpoints as exact-token GRPO.

Each credited row is one exact action sampled by the attested parent policy.
The deterministic environment supplies only outcome credit: continuation
events, evidence, scripted text, and oracle actions never become trainable
tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import msgspec
from transformers import AutoTokenizer
from ultra.behavior_likelihood import (
    FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION,
    full_vocabulary_behavior_likelihood_contract,
    has_full_vocabulary_behavior_likelihood_contract,
)
from ultra.live_control import (
    ControlContractError,
    capability_reference_map,
    parse_capability_control_action,
    serialize_capability_control_action,
    validate_control_action,
)
from ultra.pool_binding import PoolBinding, load_pool_binding
from ultra.synthetic_branchpoints import (
    BRANCHPOINT_CURRICULUM_REVISION,
    FIXED_CONTINUATION_MODE,
    FIXED_CONTINUATION_REVISION,
    SyntheticBranchpointScenario,
    build_synthetic_branchpoint_curriculum,
    evaluate_synthetic_branchpoint_action,
)
from ultra.synthetic_branchpoint_collection import (
    SYNTHETIC_BRANCHPOINT_COLLECTION_VERSION,
)

from surogate.grpo.ale_batch import (
    ACTION_BALANCED_RETENTION_REPLAY_VERSION,
    ACTION_BALANCED_RETENTION_SAMPLES,
    ACTION_BALANCED_RETENTION_SELECTED_TOKENS,
    CONTROL_ACTIONS,
    PROVEN_SEQUENCE_LEN,
    REPLAY_REFERENCE_MODE,
    _exact_policy_sample,
    _load_replay_samples,
    _selected_replay_weight_sum,
    _validate_train_retention_report,
)
from surogate.grpo.batch import prepare_sample
from surogate.grpo.transport import TrainingBatch, TrainingSample

SYNTHETIC_BRANCHPOINT_BATCH_VERSION = (
    "fugu_synthetic_branchpoint_grpo_batch_v6"
)
SYNTHETIC_BRANCHPOINT_BATCH_VERDICT = (
    "SYNTHETIC_BRANCHPOINT_REPLAY_ANCHORED_GRPO_BATCH_READY"
)
SYNTHETIC_BRANCHPOINT_CREDIT_MODE = (
    "same_prompt_low_budget_replan_topology_v6"
)
MIN_CONTRAST_GROUPS = 4
MAX_CONTRAST_GROUPS = 4
MIN_POLICY_SAMPLES = 8
MAX_POLICY_SAMPLES = 8
MANDATORY_TRANSFER_REPLAY_SAMPLES = 52
MANDATORY_TRANSFER_REPLAY_TOKENS = 2_448
DATA_PARALLEL_GPUS = 6
ADV_TAU = 1.0
REPLAY_TAU = 0.05
KL_TAU = 0.001
CAUSAL_CREDIT_MASS_PER_ROW = 8.0
MIN_ELIGIBLE_OUTCOMES_PER_ARM = 2
MIN_SUCCESS_FRACTION = 0.2
MAX_SUCCESS_FRACTION = 0.8
REQUIRED_CONTRAST_BY_MOTIF = {
    "low_budget_deliverable_first": "low_budget_replan_topology",
}
REQUIRED_GROUPS_BY_CONTRAST = {
    "low_budget_replan_topology": 4,
}


class SyntheticBranchpointBatchError(ValueError):
    """A synthetic branchpoint collection cannot safely enter training."""


@dataclass(frozen=True)
class _ObservedBranchpoint:
    scenario_index: int
    scenario_id: str
    motif: str
    sample_index: int
    sampling_seed: int
    reward: float
    action: str
    target_position_id: int | None
    action_payload: dict[str, Any]
    step_operations: tuple[str | None, ...]
    trace: dict[str, Any]


@dataclass(frozen=True)
class _ContrastGroup:
    scenario_index: int
    scenario_id: str
    motif: str
    context_key: str
    contrast_kind: str
    positives: tuple[_ObservedBranchpoint, ...]
    negatives: tuple[_ObservedBranchpoint, ...]
    selected: tuple[_ObservedBranchpoint, ...]


@dataclass(frozen=True)
class _PolicyMaterialization:
    samples: list[TrainingSample]
    credit_groups: list[dict[str, Any]]
    policy_report: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticBranchpointBatchError(
            f"cannot read {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SyntheticBranchpointBatchError(
            f"{path} must contain one JSON object"
        )
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise SyntheticBranchpointBatchError(
            f"{label} must be an integer greater than or equal to {minimum}"
        )
    return value


def _messages(trace: dict[str, Any]) -> list[dict[str, str]]:
    messages = trace.get("messages")
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or not message["role"]
            or not isinstance(message["content"], str)
            or not message["content"]
            for message in messages
        )
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint trace lacks exact non-empty role/content messages"
        )
    return messages


def _load_local_tokenizer(tokenizer_model_path: Path) -> Any:
    if not tokenizer_model_path.is_dir():
        raise SyntheticBranchpointBatchError(
            "tokenizer model snapshot is not a local directory: "
            f"{tokenizer_model_path}"
        )
    try:
        return AutoTokenizer.from_pretrained(
            str(tokenizer_model_path),
            local_files_only=True,
        )
    except Exception as exc:
        raise SyntheticBranchpointBatchError(
            "cannot load tokenizer from local model snapshot "
            f"{tokenizer_model_path}: {exc}"
        ) from exc


def _template_token_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
) -> list[int]:
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:
        raise SyntheticBranchpointBatchError(
            f"cannot apply the collection chat template: {exc}"
        ) from exc
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if (
        not isinstance(encoded, list)
        or not encoded
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in encoded
        )
    ):
        raise SyntheticBranchpointBatchError(
            "local chat template did not produce one exact token-ID list"
        )
    return encoded


def _validate_token_semantic_binding(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    response: str,
    prompt_ids: list[int],
    completion_ids: list[int],
) -> None:
    if _template_token_ids(tokenizer, messages) != prompt_ids:
        raise SyntheticBranchpointBatchError(
            "branchpoint prompt token IDs do not encode trace messages"
        )
    try:
        decoded = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise SyntheticBranchpointBatchError(
            f"cannot decode branchpoint completion token IDs: {exc}"
        ) from exc
    if not isinstance(decoded, str) or decoded != response:
        raise SyntheticBranchpointBatchError(
            "branchpoint completion token IDs do not decode to trace response"
        )


def _json_value_span(
    response: str,
    *,
    field: str,
    expected_value: object,
) -> tuple[int, int]:
    matches = list(
        re.finditer(rf'"{re.escape(field)}"\s*:', response)
    )
    if len(matches) != 1:
        raise SyntheticBranchpointBatchError(
            f"branchpoint response must contain exactly one {field} field"
        )
    value_start = matches[0].end()
    while (
        value_start < len(response)
        and response[value_start].isspace()
    ):
        value_start += 1
    try:
        value, relative_end = json.JSONDecoder().raw_decode(
            response[value_start:]
        )
    except json.JSONDecodeError as exc:
        raise SyntheticBranchpointBatchError(
            f"cannot locate exact {field} value span"
        ) from exc
    if value != expected_value:
        raise SyntheticBranchpointBatchError(
            f"branchpoint response {field} differs from its parsed action"
        )
    return value_start, value_start + relative_end


def _exact_completion_span_mask(
    *,
    tokenizer: Any,
    completion_ids: list[int],
    response: str,
    span: tuple[int, int],
) -> list[bool]:
    start, end = span
    if not 0 <= start < end <= len(response):
        raise SyntheticBranchpointBatchError(
            "causal response span is empty or out of bounds"
        )
    boundaries = [0]
    previous = ""
    for token_end in range(1, len(completion_ids) + 1):
        try:
            decoded = tokenizer.decode(
                completion_ids[:token_end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as exc:
            raise SyntheticBranchpointBatchError(
                "cannot map causal response span to original token IDs"
            ) from exc
        if (
            not isinstance(decoded, str)
            or not decoded.startswith(previous)
            or not response.startswith(decoded)
        ):
            raise SyntheticBranchpointBatchError(
                "completion prefix decoding is not monotonic"
            )
        boundaries.append(len(decoded))
        previous = decoded
    if previous != response:
        raise SyntheticBranchpointBatchError(
            "completion prefix mapping does not end at the exact response"
        )
    mask = [
        boundaries[index] < end
        and boundaries[index + 1] > start
        for index in range(len(completion_ids))
    ]
    selected = [index for index, value in enumerate(mask) if value]
    if (
        not selected
        or selected != list(range(selected[0], selected[-1] + 1))
    ):
        raise SyntheticBranchpointBatchError(
            "causal response span does not map to one contiguous token span"
        )
    return mask


def _target_divergence_spans(
    positive: _ObservedBranchpoint,
    negative: _ObservedBranchpoint,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if (
        positive.target_position_id is None
        or negative.target_position_id is None
    ):
        raise SyntheticBranchpointBatchError(
            "handoff target contrast lacks its selected position"
        )
    positive_response = positive.trace["response"]
    negative_response = negative.trace["response"]
    positive_span = _json_value_span(
        positive_response,
        field="target_position_id",
        expected_value=positive.target_position_id,
    )
    negative_span = _json_value_span(
        negative_response,
        field="target_position_id",
        expected_value=negative.target_position_id,
    )
    positive_text = positive_response[slice(*positive_span)]
    negative_text = negative_response[slice(*negative_span)]
    if (
        positive_text != str(positive.target_position_id)
        or negative_text != str(negative.target_position_id)
    ):
        raise SyntheticBranchpointBatchError(
            "handoff target is not one canonical integer lexeme"
        )
    divergence = 0
    while (
        divergence < min(len(positive_text), len(negative_text))
        and positive_text[divergence] == negative_text[divergence]
    ):
        divergence += 1
    if divergence == len(positive_text) == len(negative_text):
        raise SyntheticBranchpointBatchError(
            "handoff target contrast has no paired divergence character"
        )

    def branch_span(
        response: str,
        value_span: tuple[int, int],
        value_text: str,
    ) -> tuple[int, int]:
        if divergence < len(value_text):
            start = value_span[0] + divergence
            return start, start + 1
        cursor = value_span[1]
        while cursor < len(response) and response[cursor].isspace():
            cursor += 1
        if (
            cursor >= len(response)
            or response[cursor] not in {",", "}"}
        ):
            raise SyntheticBranchpointBatchError(
                "shorter handoff target has no structural termination token"
            )
        return cursor, cursor + 1

    return (
        branch_span(positive_response, positive_span, positive_text),
        branch_span(negative_response, negative_span, negative_text),
    )


def _action_divergence_spans(
    positive: _ObservedBranchpoint,
    negative: _ObservedBranchpoint,
) -> tuple[tuple[int, int], tuple[int, int]]:
    positive_response = positive.trace["response"]
    negative_response = negative.trace["response"]
    positive_span = _json_value_span(
        positive_response,
        field="action",
        expected_value=positive.action,
    )
    negative_span = _json_value_span(
        negative_response,
        field="action",
        expected_value=negative.action,
    )
    positive_text = positive_response[slice(*positive_span)]
    negative_text = negative_response[slice(*negative_span)]
    if (
        positive_text != json.dumps(positive.action)
        or negative_text != json.dumps(negative.action)
    ):
        raise SyntheticBranchpointBatchError(
            "control action is not one canonical JSON string lexeme"
        )
    divergence = 0
    while (
        divergence < min(len(positive_text), len(negative_text))
        and positive_text[divergence] == negative_text[divergence]
    ):
        divergence += 1
    if (
        divergence >= len(positive_text)
        or divergence >= len(negative_text)
    ):
        raise SyntheticBranchpointBatchError(
            "control actions do not have a paired divergence character"
        )
    return (
        (
            positive_span[0] + divergence,
            positive_span[0] + divergence + 1,
        ),
        (
            negative_span[0] + divergence,
            negative_span[0] + divergence + 1,
        ),
    )


def _replan_branch_span(
    row: _ObservedBranchpoint,
) -> tuple[int, int]:
    response = row.trace["response"]
    steps = _replan_steps(row)
    if len(steps) not in {1, 2}:
        raise SyntheticBranchpointBatchError(
            "replan topology row must contain one or two steps"
        )
    steps_start, steps_end = _json_value_span(
        response,
        field="steps",
        expected_value=steps,
    )
    if response[steps_start] != "[":
        raise SyntheticBranchpointBatchError(
            "replan steps value is not an array"
        )
    cursor = steps_start + 1
    while cursor < steps_end and response[cursor].isspace():
        cursor += 1
    try:
        first_step, relative_end = json.JSONDecoder().raw_decode(
            response[cursor:steps_end]
        )
    except json.JSONDecodeError as exc:
        raise SyntheticBranchpointBatchError(
            "cannot locate the first replan topology branch"
        ) from exc
    if first_step != steps[0]:
        raise SyntheticBranchpointBatchError(
            "first replan step differs from its parsed action"
        )
    cursor += relative_end
    while cursor < steps_end and response[cursor].isspace():
        cursor += 1
    expected_delimiter = "," if len(steps) == 2 else "]"
    if (
        cursor >= steps_end
        or response[cursor] != expected_delimiter
    ):
        raise SyntheticBranchpointBatchError(
            "replan topology branch delimiter is not canonical"
        )
    return cursor, cursor + 1


def _causal_advantage_masks(
    *,
    rows: list[_ObservedBranchpoint],
    contrast_kind: str,
    tokenizer: Any,
) -> list[tuple[list[bool], str]]:
    if len(rows) != 2 or [row.reward for row in rows] != [0.0, 1.0]:
        raise SyntheticBranchpointBatchError(
            "causal advantage masks require one negative and one positive row"
        )
    negative, positive = rows
    if contrast_kind == "low_budget_replan_topology":
        spans = [
            _replan_branch_span(negative),
            _replan_branch_span(positive),
        ]
        credit_field = "steps_continuation_delimiter"
    else:
        raise SyntheticBranchpointBatchError(
            f"unsupported causal contrast kind: {contrast_kind}"
        )
    return [
        (
            _exact_completion_span_mask(
                tokenizer=tokenizer,
                completion_ids=row.trace["completion_token_ids"],
                response=row.trace["response"],
                span=span,
            ),
            credit_field,
        )
        for row, span in zip(rows, spans, strict=True)
    ]


def _validate_trace(
    trace: object,
    *,
    sampling_seed: int,
    reward: float | None,
    sequence_len: int,
    tokenizer: Any,
) -> dict[str, Any]:
    if not isinstance(trace, dict):
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace is not an object"
        )
    if trace.get("seed") != sampling_seed:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace seed differs from its sample"
        )
    if trace.get("correction") is not None:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace is a correction attempt"
        )
    if trace.get("finish_reason") not in {"stop", "length"}:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace has an unknown finish reason"
        )
    response = trace.get("response")
    if not isinstance(response, str):
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace lacks its exact response"
        )
    messages = _messages(trace)
    prompt_ids = trace.get("prompt_token_ids")
    completion_ids = trace.get("completion_token_ids")
    logprobs = trace.get("completion_logprobs")
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in prompt_ids
        )
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace has no exact prompt token IDs"
        )
    if (
        not isinstance(completion_ids, list)
        or not completion_ids
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in completion_ids
        )
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace has no exact completion token IDs"
        )
    if (
        not isinstance(logprobs, list)
        or len(logprobs) != len(completion_ids)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) > 1.0e-6
            for value in logprobs
        )
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace has invalid sampled log-probabilities"
        )
    if (
        trace.get("temperature") != 1.0
        or not has_full_vocabulary_behavior_likelihood_contract(
            trace.get("behavior_likelihood_contract")
        )
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace lacks exact behavior likelihood"
        )
    if len(prompt_ids) + len(completion_ids) > sequence_len:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy trace exceeds the optimizer window"
        )
    _validate_token_semantic_binding(
        tokenizer=tokenizer,
        messages=messages,
        response=response,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
    )
    return trace


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return msgspec.to_builtins(value)
    except (TypeError, ValueError) as exc:
        raise SyntheticBranchpointBatchError(
            f"fixed-continuation result is not JSON-compatible: {exc}"
        ) from exc


def _result_field(result: object, name: str) -> Any:
    if isinstance(result, dict):
        if name not in result:
            raise SyntheticBranchpointBatchError(
                f"fixed-continuation result lacks {name}"
            )
        return result[name]
    if not hasattr(result, name):
        raise SyntheticBranchpointBatchError(
            f"fixed-continuation result lacks {name}"
        )
    return getattr(result, name)


def _scenario_state(
    scenario: SyntheticBranchpointScenario,
) -> Any:
    state = getattr(scenario, "state", None)
    if state is None:
        state = getattr(scenario, "branch_state", None)
    if state is None:
        raise SyntheticBranchpointBatchError(
            "branchpoint scenario lacks its live control state"
        )
    return state


def _replay_sample(
    *,
    raw: object,
    scenario: SyntheticBranchpointScenario,
    scenario_index: int,
    sample_index: int,
    expected_seed: int,
    behavior_policy_revision: str,
    runtime_revision: str,
    binding: PoolBinding,
    sequence_len: int,
    tokenizer: Any,
) -> tuple[_ObservedBranchpoint | None, str]:
    if not isinstance(raw, dict):
        raise SyntheticBranchpointBatchError(
            "branchpoint sample is not an object"
        )
    if raw.get("sample_index") != sample_index:
        raise SyntheticBranchpointBatchError(
            "branchpoint sample indexes are not canonical"
        )
    if raw.get("sample_id") != (
        f"{scenario.scenario_id}:sample-{sample_index:03d}"
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint sample identity changed"
        )
    policy = raw.get("policy")
    expected_policy = {
        "behavior_policy_revision": behavior_policy_revision,
        "runtime_revision": runtime_revision,
        "pool_id": binding.pool_id,
        "pool_binding_revision": binding.binding_revision,
        "sampling_seed": expected_seed,
    }
    if policy != expected_policy:
        raise SyntheticBranchpointBatchError(
            "branchpoint sample policy attestation changed"
        )
    disposition = raw.get("disposition")
    if disposition not in {"eligible", "protocol_only", "unmodeled"}:
        raise SyntheticBranchpointBatchError(
            "branchpoint sample disposition is unknown"
        )
    reward_value = raw.get("reward")
    if reward_value is not None and (
        isinstance(reward_value, bool)
        or not isinstance(reward_value, (int, float))
        or float(reward_value) not in {0.0, 1.0}
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint sample reward must be binary or null"
        )
    reward = (
        None if reward_value is None else float(reward_value)
    )
    trace = _validate_trace(
        raw.get("trace"),
        sampling_seed=expected_seed,
        reward=reward,
        sequence_len=sequence_len,
        tokenizer=tokenizer,
    )
    state = _scenario_state(scenario)
    references = capability_reference_map(state.workers)
    response = trace["response"]
    action = None
    canonical_action = None
    parse_error: ControlContractError | None = None
    legal_error: ControlContractError | None = None
    try:
        action = parse_capability_control_action(response, references)
        canonical_action = json.loads(
            serialize_capability_control_action(action, references)
        )
    except ControlContractError as exc:
        parse_error = exc
    if action is not None:
        try:
            validate_control_action(action, state)
        except ControlContractError as exc:
            legal_error = exc
    if trace.get("finish_reason") == "length":
        if raw.get("action") is not None:
            raise SyntheticBranchpointBatchError(
                "length-truncated branchpoint serialized an action"
            )
    elif raw.get("action") != canonical_action:
        raise SyntheticBranchpointBatchError(
            "serialized branchpoint action differs from the raw response"
        )
    if trace.get("finish_reason") == "length":
        protocol_outcome = "protocol_only:length_truncated"
        expected_result = {
            "disposition": "protocol_only",
            "training_eligible": False,
            "reward": None,
            "outcome": protocol_outcome,
            "events": [],
            "evidence": {
                "exclusion_reason": protocol_outcome,
                "semantic_reward_assigned": False,
            },
        }
    elif parse_error is not None:
        protocol_outcome = (
            f"protocol_only:{type(parse_error).__name__}"
        )
        expected_result = {
            "disposition": "protocol_only",
            "training_eligible": False,
            "reward": None,
            "outcome": protocol_outcome,
            "events": [],
            "evidence": {
                "exclusion_reason": protocol_outcome,
                "semantic_reward_assigned": False,
            },
        }
    elif legal_error is not None:
        protocol_outcome = "protocol_only:invalid_control_action"
        expected_result = {
            "disposition": "protocol_only",
            "training_eligible": False,
            "reward": None,
            "outcome": protocol_outcome,
            "events": [],
            "evidence": {
                "exclusion_reason": protocol_outcome,
                "semantic_reward_assigned": False,
            },
        }
    else:
        try:
            reconstructed = evaluate_synthetic_branchpoint_action(
                scenario,
                action,
            )
        except Exception as exc:
            raise SyntheticBranchpointBatchError(
                f"cannot replay fixed continuation: {exc}"
            ) from exc
        expected_result = {
            "disposition": _result_field(
                reconstructed,
                "disposition",
            ),
            "training_eligible": _result_field(
                reconstructed,
                "training_eligible",
            ),
            "reward": _result_field(reconstructed, "reward"),
            "outcome": _result_field(reconstructed, "outcome"),
            "events": _jsonable(
                _result_field(reconstructed, "events")
            ),
            "evidence": _jsonable(
                _result_field(reconstructed, "evidence")
            ),
        }
    observed_result = {
        "disposition": disposition,
        "training_eligible": raw.get("training_eligible"),
        "reward": reward,
        "outcome": raw.get("outcome"),
        "events": raw.get("events"),
        "evidence": raw.get("evidence"),
    }
    if observed_result != expected_result:
        raise SyntheticBranchpointBatchError(
            "branchpoint fixed-continuation report differs from replay"
        )
    expected_eligible = disposition == "eligible"
    if raw.get("training_eligible") is not expected_eligible:
        raise SyntheticBranchpointBatchError(
            "branchpoint training eligibility differs from disposition"
        )
    if not expected_eligible:
        return None, disposition
    if action is None or canonical_action is None:
        raise SyntheticBranchpointBatchError(
            "eligible branchpoint action is not parsed and legal"
        )
    if reward not in {0.0, 1.0}:
        raise SyntheticBranchpointBatchError(
            "eligible branchpoint sample lacks binary terminal reward"
        )
    if trace.get("finish_reason") != "stop":
        raise SyntheticBranchpointBatchError(
            "eligible branchpoint sample is truncated"
        )
    if reward == 1.0:
        coverage = (
            raw.get("evidence") or {}
        ).get("executor_evidence_coverage")
        if (
            not isinstance(coverage, dict)
            or coverage.get("all_sampled_text_supported") is not True
            or coverage.get("reason_supported") is not True
            or coverage.get("basis")
            != "emitted_executor_events_and_observed_live_state"
            or any(
                not isinstance(item, dict)
                or item.get("supported") is not True
                for item in coverage.get("step_coverage", [])
            )
        ):
            raise SyntheticBranchpointBatchError(
                "positive branchpoint text lacks executor evidence coverage"
            )
    step_coverage = (
        (raw.get("evidence") or {})
        .get("executor_evidence_coverage", {})
        .get("step_coverage", [])
    )
    if not isinstance(step_coverage, list) or any(
        not isinstance(item, dict)
        or item.get("step_index") != index
        or (
            item.get("operation") is not None
            and item.get("operation")
            not in {"produce", "repair", "verify", "inspect"}
        )
        for index, item in enumerate(step_coverage)
    ):
        raise SyntheticBranchpointBatchError(
            "eligible branchpoint step coverage is malformed"
        )
    return (
        _ObservedBranchpoint(
            scenario_index=scenario_index,
            scenario_id=scenario.scenario_id,
            motif=scenario.motif,
            sample_index=sample_index,
            sampling_seed=expected_seed,
            reward=reward,
            action=action.action,
            target_position_id=action.target_position_id,
            action_payload=canonical_action,
            step_operations=tuple(
                item.get("operation") for item in step_coverage
            ),
            trace=trace,
        ),
        disposition,
    )


def _replan_steps(row: _ObservedBranchpoint) -> list[dict[str, Any]]:
    steps = row.action_payload.get("steps")
    if not isinstance(steps, list) or any(
        not isinstance(step, dict) for step in steps
    ):
        return []
    return steps


def _is_replan_topology_pair(
    positive: _ObservedBranchpoint,
    negative: _ObservedBranchpoint,
) -> bool:
    positive_steps = _replan_steps(positive)
    negative_steps = _replan_steps(negative)
    return (
        _is_positive_replan_topology(positive)
        and _is_negative_replan_topology(negative)
        and positive_steps[0].get("profile_ref")
        == negative_steps[0].get("profile_ref")
        and positive_steps[0].get("access_positions")
        == negative_steps[0].get("access_positions")
        and positive_steps[1].get("access_positions") == [0]
    )


def _is_positive_replan_topology(row: _ObservedBranchpoint) -> bool:
    steps = _replan_steps(row)
    return (
        row.reward == 1.0
        and row.action == "replan"
        and row.target_position_id is None
        and row.step_operations == ("produce", "verify")
        and len(steps) == 2
        and steps[1].get("access_positions") == [0]
    )


def _is_negative_replan_topology(row: _ObservedBranchpoint) -> bool:
    steps = _replan_steps(row)
    return (
        row.reward == 0.0
        and row.action == "replan"
        and row.target_position_id is None
        and row.step_operations == ("produce",)
        and len(steps) == 1
    )


def _topology_outcomes(
    group: _ContrastGroup,
) -> tuple[
    tuple[_ObservedBranchpoint, ...],
    tuple[_ObservedBranchpoint, ...],
]:
    return (
        tuple(
            row for row in group.positives
            if _is_positive_replan_topology(row)
        ),
        tuple(
            row for row in group.negatives
            if _is_negative_replan_topology(row)
        ),
    )


def _compatible_contrast_pair(
    group: _ContrastGroup,
) -> tuple[_ObservedBranchpoint, _ObservedBranchpoint] | None:
    """Choose the shortest exact same-state pair with one causal difference."""
    topology_positives, topology_negatives = _topology_outcomes(group)
    positives = sorted(
        topology_positives,
        key=lambda row: (row.sample_index, row.sampling_seed),
    )
    negatives = sorted(
        topology_negatives,
        key=lambda row: (row.sample_index, row.sampling_seed),
    )
    candidates: list[
        tuple[_ObservedBranchpoint, _ObservedBranchpoint]
    ] = []
    for positive in positives:
        for negative in negatives:
            if (
                group.contrast_kind == "low_budget_replan_topology"
                and _is_replan_topology_pair(positive, negative)
            ):
                candidates.append((positive, negative))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda pair: (
            len(pair[0].trace["completion_token_ids"])
            + len(pair[1].trace["completion_token_ids"]),
            abs(
                len(pair[0].trace["completion_token_ids"])
                - len(pair[1].trace["completion_token_ids"])
            ),
            pair[0].sample_index,
            pair[1].sample_index,
            pair[0].sampling_seed,
            pair[1].sampling_seed,
        ),
    )


def _uncertainty_rank(group: _ContrastGroup) -> tuple[float, int, int, int, str]:
    positive_count = len(group.positives)
    negative_count = len(group.negatives)
    total = positive_count + negative_count
    success_fraction = positive_count / total
    return (
        abs(success_fraction - 0.5),
        -min(positive_count, negative_count),
        -total,
        group.scenario_index,
        group.scenario_id,
    )


def _select_contrasts(
    groups: list[_ContrastGroup],
) -> list[_ContrastGroup]:
    ordered = sorted(
        groups,
        key=lambda group: (
            group.scenario_index,
            group.scenario_id,
        ),
    )
    compatible: list[_ContrastGroup] = []
    for group in ordered:
        topology_positives, topology_negatives = _topology_outcomes(group)
        total = len(topology_positives) + len(topology_negatives)
        if (
            len(topology_positives) < MIN_ELIGIBLE_OUTCOMES_PER_ARM
            or len(topology_negatives) < MIN_ELIGIBLE_OUTCOMES_PER_ARM
            or total == 0
        ):
            continue
        success_fraction = len(topology_positives) / total
        if not (
            MIN_SUCCESS_FRACTION
            <= success_fraction
            <= MAX_SUCCESS_FRACTION
        ):
            continue
        pair = _compatible_contrast_pair(group)
        if pair is None:
            continue
        selected_group = _ContrastGroup(
            scenario_index=group.scenario_index,
            scenario_id=group.scenario_id,
            motif=group.motif,
            context_key=group.context_key,
            contrast_kind=group.contrast_kind,
            positives=topology_positives,
            negatives=topology_negatives,
            selected=pair,
        )
        compatible.append(selected_group)

    compatible_by_context: dict[str, _ContrastGroup] = {}
    for group in sorted(compatible, key=_uncertainty_rank):
        compatible_by_context.setdefault(group.context_key, group)

    required = REQUIRED_GROUPS_BY_CONTRAST[
        "low_budget_replan_topology"
    ]
    if len(compatible_by_context) < required:
        raise SyntheticBranchpointBatchError(
            "branchpoint update lacks required eligible low-budget replan "
            "topology contrasts: "
            f"unique_contexts={len(compatible_by_context)}/{required}"
        )
    selected = sorted(
        compatible_by_context.values(),
        key=_uncertainty_rank,
    )[:required]
    selected.sort(
        key=lambda group: (group.scenario_index, group.scenario_id)
    )
    if len(selected) != MIN_CONTRAST_GROUPS:
        raise SyntheticBranchpointBatchError(
            "branchpoint required coverage did not produce exactly four groups"
        )
    sample_count = 2 * len(selected)
    if not MIN_POLICY_SAMPLES <= sample_count <= MAX_POLICY_SAMPLES:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy sample count must be exactly 8"
        )
    return selected


def _validated_collection_groups(
    *,
    collection_path: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    sequence_len: int,
    tokenizer: Any,
) -> list[_ContrastGroup]:
    collection = _read_json(collection_path)
    try:
        binding = load_pool_binding(pool_binding_path)
    except Exception as exc:
        raise SyntheticBranchpointBatchError(
            f"cannot load pool binding {pool_binding_path}: {exc}"
        ) from exc
    expected_top = {
        "version": SYNTHETIC_BRANCHPOINT_COLLECTION_VERSION,
        "behavior_policy_revision": expected_behavior_policy_revision,
        "runtime_revision": expected_runtime_revision,
        "pool_id": binding.pool_id,
        "pool_binding_revision": binding.binding_revision,
        "pool_binding": str(pool_binding_path),
        "curriculum_revision": BRANCHPOINT_CURRICULUM_REVISION,
        "fixed_continuation": {
            "revision": FIXED_CONTINUATION_REVISION,
            "mode": FIXED_CONTINUATION_MODE,
        },
        "sampling_temperature": 1.0,
        "paid_calls": 0,
        "optimizer_steps": 0,
    }
    for key, expected in expected_top.items():
        if collection.get(key) != expected:
            raise SyntheticBranchpointBatchError(
                f"branchpoint collection {key} changed"
            )
    if collection.get("verdict") != "SYNTHETIC_BRANCHPOINTS_COLLECTED":
        raise SyntheticBranchpointBatchError(
            "branchpoint collection verdict changed"
        )
    scenario_seed = _require_int(
        collection.get("scenario_seed"),
        "scenario_seed",
    )
    scenario_count = _require_int(
        collection.get("scenario_count"),
        "scenario_count",
        minimum=1,
    )
    samples_per_scenario = _require_int(
        collection.get("samples_per_scenario"),
        "samples_per_scenario",
        minimum=2,
    )
    if collection.get("sample_count") != (
        scenario_count * samples_per_scenario
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint collection sample count changed"
        )
    try:
        expected_scenarios = build_synthetic_branchpoint_curriculum(
            count=scenario_count,
            seed=scenario_seed,
            profile_capabilities=tuple(
                slot.role_prior for slot in binding.slots
            ),
        )
    except Exception as exc:
        raise SyntheticBranchpointBatchError(
            f"cannot rebuild branchpoint curriculum: {exc}"
        ) from exc
    raw_scenarios = collection.get("scenarios")
    if (
        not isinstance(raw_scenarios, list)
        or len(raw_scenarios) != len(expected_scenarios)
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint scenario inventory changed"
        )

    global_seeds: set[int] = set()
    all_dispositions: Counter[str] = Counter()
    eligible_count = 0
    groups: list[_ContrastGroup] = []
    for scenario_index, (raw_scenario, scenario) in enumerate(
        zip(raw_scenarios, expected_scenarios, strict=True)
    ):
        if not isinstance(raw_scenario, dict):
            raise SyntheticBranchpointBatchError(
                "branchpoint scenario is not an object"
            )
        expected_scenario_fields = {
            "scenario_index": scenario_index,
            "scenario_id": scenario.scenario_id,
            "motif": scenario.motif,
            "evidence_basis": list(scenario.evidence_basis),
        }
        for key, expected in expected_scenario_fields.items():
            if raw_scenario.get(key) != expected:
                raise SyntheticBranchpointBatchError(
                    f"branchpoint scenario {scenario_index} {key} changed"
                )
        raw_samples = raw_scenario.get("samples")
        if (
            not isinstance(raw_samples, list)
            or len(raw_samples) != samples_per_scenario
            or raw_scenario.get("sample_count")
            != samples_per_scenario
        ):
            raise SyntheticBranchpointBatchError(
                "branchpoint scenario sample inventory changed"
            )
        scenario_dispositions: Counter[str] = Counter()
        scenario_rewards: Counter[str] = Counter()
        positives: list[_ObservedBranchpoint] = []
        negatives: list[_ObservedBranchpoint] = []
        prompt_ids: list[int] | None = None
        messages: list[dict[str, str]] | None = None
        for sample_index, raw_sample in enumerate(raw_samples):
            expected_seed = (
                scenario_seed
                + 10_000_019
                + scenario_index * samples_per_scenario
                + sample_index
            )
            if expected_seed in global_seeds:
                raise SyntheticBranchpointBatchError(
                    "branchpoint sampling seeds are not unique"
                )
            global_seeds.add(expected_seed)
            observed, disposition = _replay_sample(
                raw=raw_sample,
                scenario=scenario,
                scenario_index=scenario_index,
                sample_index=sample_index,
                expected_seed=expected_seed,
                behavior_policy_revision=(
                    expected_behavior_policy_revision
                ),
                runtime_revision=expected_runtime_revision,
                binding=binding,
                sequence_len=sequence_len,
                tokenizer=tokenizer,
            )
            scenario_dispositions[disposition] += 1
            all_dispositions[disposition] += 1
            trace = raw_sample["trace"]
            if prompt_ids is None:
                prompt_ids = trace["prompt_token_ids"]
                messages = _messages(trace)
            elif (
                trace["prompt_token_ids"] != prompt_ids
                or _messages(trace) != messages
            ):
                raise SyntheticBranchpointBatchError(
                    "same-scenario branchpoint samples have different prompts"
                )
            raw_reward = raw_sample.get("reward")
            if raw_reward is not None:
                scenario_rewards[str(float(raw_reward))] += 1
            if observed is None:
                continue
            eligible_count += 1
            if observed.reward == 1.0:
                positives.append(observed)
            else:
                negatives.append(observed)
        if raw_scenario.get("disposition_counts") != dict(
            sorted(scenario_dispositions.items())
        ):
            raise SyntheticBranchpointBatchError(
                "branchpoint scenario disposition counts changed"
            )
        if raw_scenario.get("reward_counts") != dict(
            sorted(scenario_rewards.items())
        ):
            raise SyntheticBranchpointBatchError(
                "branchpoint scenario reward counts changed"
            )
        if positives and negatives:
            contrast_kind = REQUIRED_CONTRAST_BY_MOTIF.get(
                scenario.motif,
                "not_admitted",
            )
            groups.append(
                _ContrastGroup(
                    scenario_index=scenario_index,
                    scenario_id=scenario.scenario_id,
                    motif=scenario.motif,
                    context_key=scenario.artifact_label,
                    contrast_kind=contrast_kind,
                    positives=tuple(positives),
                    negatives=tuple(negatives),
                    selected=(),
                )
            )
    if collection.get("disposition_counts") != dict(
        sorted(all_dispositions.items())
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint collection disposition counts changed"
        )
    if collection.get("eligible_count") != eligible_count:
        raise SyntheticBranchpointBatchError(
            "branchpoint collection eligible count changed"
        )
    top_rewards: Counter[str] = Counter()
    for raw_scenario in raw_scenarios:
        top_rewards.update(raw_scenario["reward_counts"])
    if collection.get("reward_counts") != dict(
        sorted(top_rewards.items())
    ):
        raise SyntheticBranchpointBatchError(
            "branchpoint collection reward counts changed"
        )
    return _select_contrasts(groups)


def _materialize_policy(
    groups: list[_ContrastGroup],
    *,
    sequence_len: int,
    tokenizer: Any,
) -> _PolicyMaterialization:
    samples: list[TrainingSample] = []
    credit_groups: list[dict[str, Any]] = []
    signed_credit = {
        action: {
            "negative_samples": 0,
            "negative_tokens": 0,
            "positive_samples": 0,
            "positive_tokens": 0,
        }
        for action in sorted(CONTROL_ACTIONS)
    }
    for group in groups:
        rows = sorted(
            group.selected,
            key=lambda row: (
                row.reward,
                row.sample_index,
                row.sampling_seed,
            ),
        )
        rewards = [row.reward for row in rows]
        if rewards != [0.0, 1.0]:
            raise SyntheticBranchpointBatchError(
                "selected branchpoint group is not one balanced outcome pair"
            )
        reward_mean = statistics.fmean(rewards)
        reward_std = statistics.stdev(rewards)
        advantages = [
            (reward - reward_mean) / (reward_std + 1.0e-4)
            for reward in rewards
        ]
        causal_masks = _causal_advantage_masks(
            rows=rows,
            contrast_kind=group.contrast_kind,
            tokenizer=tokenizer,
        )
        group_start = len(samples)
        positive_indices: list[int] = []
        negative_indices: list[int] = []
        source_rows: list[dict[str, Any]] = []
        for row, group_advantage, causal in zip(
            rows,
            advantages,
            causal_masks,
            strict=True,
        ):
            causal_mask, credit_field = causal
            credited_tokens = sum(causal_mask)
            optimizer_advantage = (
                group_advantage
                * CAUSAL_CREDIT_MASS_PER_ROW
                / credited_tokens
            )
            try:
                sample = _exact_policy_sample(
                    row.trace,
                    advantage=optimizer_advantage,
                    reward=row.reward,
                    sampling_temperature=1.0,
                    sequence_len=sequence_len,
                    completion_advantage_mask=causal_mask,
                )
            except ValueError as exc:
                raise SyntheticBranchpointBatchError(str(exc)) from exc
            policy_index = len(samples)
            samples.append(sample)
            sign = (
                "positive"
                if optimizer_advantage > 0.0
                else "negative"
            )
            signed_credit[row.action][f"{sign}_samples"] += 1
            signed_credit[row.action][
                f"{sign}_tokens"
            ] += credited_tokens
            if row.reward == 1.0:
                positive_indices.append(policy_index)
            else:
                negative_indices.append(policy_index)
            source_rows.append(
                {
                    "policy_sample_index": policy_index,
                    "sample_index": row.sample_index,
                    "sampling_seed": row.sampling_seed,
                    "reward": row.reward,
                    "group_advantage": group_advantage,
                    "optimizer_advantage": optimizer_advantage,
                    "action": row.action,
                    "target_position_id": row.target_position_id,
                    "generated_tokens": len(sample.completion_ids),
                    "credited_tokens": credited_tokens,
                    "credit_field": credit_field,
                }
            )
        credit_groups.append(
            {
                "scenario_index": group.scenario_index,
                "scenario_id": group.scenario_id,
                "motif": group.motif,
                "context_key": group.context_key,
                "contrast_kind": group.contrast_kind,
                "policy_sample_start": group_start,
                "policy_sample_count": len(rows),
                "positive_policy_sample_indices": positive_indices,
                "negative_policy_sample_indices": negative_indices,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "advantage_method": (
                    "sample_std_normalized_eps_1e-4_then_"
                    "fixed_causal_row_mass"
                ),
                "causal_credit_mass_per_row": (
                    CAUSAL_CREDIT_MASS_PER_ROW
                ),
                "eligible_topology_positive_outcomes": len(
                    group.positives
                ),
                "eligible_topology_negative_outcomes": len(
                    group.negatives
                ),
                "eligible_topology_success_fraction": (
                    len(group.positives)
                    / (len(group.positives) + len(group.negatives))
                ),
                "source_rows": source_rows,
            }
        )
    if not MIN_POLICY_SAMPLES <= len(samples) <= MAX_POLICY_SAMPLES:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy sample count must be exactly 8"
        )
    if len(credit_groups) < MIN_CONTRAST_GROUPS:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy materialization lost contrast groups"
        )
    positives = sum(
        1 for sample in samples if float(sample.reward or 0.0) == 1.0
    )
    negatives = len(samples) - positives
    if positives != negatives:
        raise SyntheticBranchpointBatchError(
            "branchpoint policy rows are not outcome-balanced"
        )
    return _PolicyMaterialization(
        samples=samples,
        credit_groups=credit_groups,
        policy_report={
            "samples": len(samples),
            "credited_samples": len(samples),
            "positive_samples": positives,
            "negative_samples": negatives,
            "tokens": sum(
                len(sample.prompt_ids) + len(sample.completion_ids)
                for sample in samples
            ),
            "conductor_tokens": sum(
                len(sample.completion_ids) for sample in samples
            ),
            "credited_tokens": sum(
                sum(sample.advantage_mask or [])
                for sample in samples
            ),
            "absolute_outcome_seed_mass": sum(
                abs(float(sample.advantage or 0.0))
                * sum(sample.advantage_mask or [])
                for sample in samples
            ),
            "causal_credit_mass_per_row": (
                CAUSAL_CREDIT_MASS_PER_ROW
            ),
            "signed_credit_by_action": signed_credit,
            "max_sequence_tokens": max(
                len(sample.prompt_ids) + len(sample.completion_ids)
                for sample in samples
            ),
            "scenario_ids": [
                group.scenario_id for group in groups
            ],
            "motifs": sorted({group.motif for group in groups}),
            "coverage_counts": dict(
                sorted(
                    Counter(
                        group.contrast_kind for group in groups
                    ).items()
                )
            ),
        },
    )


def _load_required_replay(
    *,
    replay_path: Path,
    retention_replay_path: Path,
    retention_report_path: Path,
    sequence_len: int,
) -> tuple[
    list[TrainingSample],
    list[TrainingSample],
    dict[str, Any],
]:
    try:
        replay_samples = _load_replay_samples(
            replay_path,
            expected_samples=MANDATORY_TRANSFER_REPLAY_SAMPLES,
        )
        retention_report = _validate_train_retention_report(
            report_path=retention_report_path,
            replay_path=retention_replay_path,
            replay_samples=ACTION_BALANCED_RETENTION_SAMPLES,
        )
        retention_samples = _load_replay_samples(
            retention_replay_path,
            expected_samples=ACTION_BALANCED_RETENTION_SAMPLES,
        )
    except ValueError as exc:
        raise SyntheticBranchpointBatchError(str(exc)) from exc
    transfer_tokens = sum(
        sum(sample.replay_mask or []) for sample in replay_samples
    )
    if transfer_tokens != MANDATORY_TRANSFER_REPLAY_TOKENS:
        raise SyntheticBranchpointBatchError(
            "mandatory transfer replay selected-token count changed"
        )
    if not math.isclose(
        _selected_replay_weight_sum(replay_samples),
        float(MANDATORY_TRANSFER_REPLAY_TOKENS),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        raise SyntheticBranchpointBatchError(
            "mandatory transfer replay weight mass changed"
        )
    if (
        retention_report.get("version")
        != ACTION_BALANCED_RETENTION_REPLAY_VERSION
    ):
        raise SyntheticBranchpointBatchError(
            "mandatory retention replay is not action-balanced v2"
        )
    retention_tokens = sum(
        sum(sample.replay_mask or []) for sample in retention_samples
    )
    if retention_tokens != ACTION_BALANCED_RETENTION_SELECTED_TOKENS:
        raise SyntheticBranchpointBatchError(
            "mandatory action-balanced retention token count changed"
        )
    expected_weight = float(
        (retention_report.get("weighting") or {}).get(
            "total_weighted_completion_tokens",
            -1.0,
        )
    )
    if not math.isclose(
        _selected_replay_weight_sum(retention_samples),
        expected_weight,
        rel_tol=1.0e-6,
        abs_tol=1.0e-3,
    ):
        raise SyntheticBranchpointBatchError(
            "mandatory action-balanced retention weight mass changed"
        )
    for sample in [*replay_samples, *retention_samples]:
        total_tokens = len(sample.prompt_ids) + len(sample.completion_ids)
        if total_tokens > sequence_len:
            raise SyntheticBranchpointBatchError(
                "mandatory replay exceeds the optimizer window"
            )
        if len(prepare_sample(sample, sequence_len).input_ids) != total_tokens:
            raise SyntheticBranchpointBatchError(
                "mandatory replay was truncated during preparation"
            )
    return replay_samples, retention_samples, retention_report


def _report(
    *,
    collection_path: Path,
    pool_binding_path: Path,
    tokenizer_model_path: Path,
    behavior_policy_revision: str,
    runtime_revision: str,
    policy: _PolicyMaterialization,
    replay_path: Path,
    replay_samples: list[TrainingSample],
    retention_replay_path: Path,
    retention_samples: list[TrainingSample],
    retention_report_path: Path,
    retention_report: dict[str, Any],
    batch_path: Path,
) -> dict[str, Any]:
    binding = load_pool_binding(pool_binding_path)
    return {
        "version": SYNTHETIC_BRANCHPOINT_BATCH_VERSION,
        "verdict": SYNTHETIC_BRANCHPOINT_BATCH_VERDICT,
        "behavior_policy_revision": behavior_policy_revision,
        "runtime_revision": runtime_revision,
        "pool_id": binding.pool_id,
        "pool_binding_revision": binding.binding_revision,
        "pool_binding": str(pool_binding_path),
        "tokenizer_model": str(tokenizer_model_path),
        "curriculum_revision": BRANCHPOINT_CURRICULUM_REVISION,
        "fixed_continuation": {
            "revision": FIXED_CONTINUATION_REVISION,
            "mode": FIXED_CONTINUATION_MODE,
        },
        "behavior_likelihood_contract_version": (
            FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION
        ),
        "source_collection": str(collection_path),
        "credit_groups": policy.credit_groups,
        "optimizer_contract": {
            "atomic_training_batch": True,
            "sequence_len": PROVEN_SEQUENCE_LEN,
            "data_parallel_gpus": DATA_PARALLEL_GPUS,
            "sample_packing": False,
            "adv_tau": ADV_TAU,
            "replay_tau": REPLAY_TAU,
            "kl_tau": KL_TAU,
            "policy_logprob_source": (
                "exact_behavior_policy_generation"
            ),
            "behavior_likelihood_contract": (
                full_vocabulary_behavior_likelihood_contract()
            ),
            "retokenization": False,
            "exact_token_semantic_binding": {
                "tokenizer_model": str(tokenizer_model_path),
                "prompt": (
                    "apply_chat_template_messages_tokenize_true_"
                    "add_generation_prompt_true_enable_thinking_false"
                ),
                "completion": (
                    "decode_completion_ids_skip_special_tokens_true"
                ),
                "validated_rows": "all_collection_traces",
            },
            "policy_credit_assignment": {
                "mode": SYNTHETIC_BRANCHPOINT_CREDIT_MODE,
                "policy_attempts": "initial_only_no_corrections",
                "reward": "fixed_continuation_terminal_binary",
                "same_prompt_required": True,
                "minimum_contrast_groups": MIN_CONTRAST_GROUPS,
                "required_scenarios": MIN_CONTRAST_GROUPS,
                "policy_sample_window": [
                    MIN_POLICY_SAMPLES,
                    MAX_POLICY_SAMPLES,
                ],
                "selection": (
                    "four_unique_context_same_prompt_natural_"
                    "produce_verify_vs_produce_replan_topologies"
                ),
                "required_coverage": dict(
                    sorted(REQUIRED_GROUPS_BY_CONTRAST.items())
                ),
                "uncertainty_admission": {
                    "minimum_eligible_outcomes_per_arm": (
                        MIN_ELIGIBLE_OUTCOMES_PER_ARM
                    ),
                    "success_fraction_window": [
                        MIN_SUCCESS_FRACTION,
                        MAX_SUCCESS_FRACTION,
                    ],
                    "ranking": (
                        "closest_to_half_then_strongest_balanced_support"
                    ),
                },
                "advantage_surface": (
                    "exact_original_generated_causal_branch_tokens"
                ),
                "causal_credit_mass_per_row": (
                    CAUSAL_CREDIT_MASS_PER_ROW
                ),
                "row_mass_normalization": (
                    "optimizer_advantage_equals_group_advantage_times_"
                    "causal_credit_mass_per_row_divided_by_credited_tokens"
                ),
                "full_completion_kl": True,
                "excluded_dispositions": [
                    "protocol_only",
                    "unmodeled",
                ],
                "oracle_or_script_tokens": False,
            },
            "replay_reference_mode": REPLAY_REFERENCE_MODE,
        },
        "policy": policy.policy_report,
        "mandatory_replay": {
            "path": str(replay_path),
            "samples": len(replay_samples),
            "selected_tokens": sum(
                sum(sample.replay_mask or [])
                for sample in replay_samples
            ),
            "selected_weight_sum": _selected_replay_weight_sum(
                replay_samples
            ),
        },
        "train_only_retention_replay": {
            "path": str(retention_replay_path),
            "samples": len(retention_samples),
            "selected_tokens": sum(
                sum(sample.replay_mask or [])
                for sample in retention_samples
            ),
            "selected_weight_sum": _selected_replay_weight_sum(
                retention_samples
            ),
            "report": str(retention_report_path),
            "reference_mode": retention_report["reference_mode"],
            "weighting_mode": (
                retention_report.get("weighting") or {}
            ).get("mode"),
            "replay_version": retention_report["version"],
            "train_tasks": retention_report["counts"]["train_tasks"],
            "validation_tasks_excluded": retention_report["counts"][
                "validation_tasks_excluded"
            ],
        },
        "combined_batch": {
            "path": str(batch_path),
            "samples": (
                len(policy.samples)
                + len(replay_samples)
                + len(retention_samples)
            ),
            "step": 0,
        },
        "paid_calls": 0,
        "optimizer_steps": 0,
    }


def _prepare_expected(
    *,
    collection_path: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    tokenizer_model_path: Path,
    replay_path: Path,
    retention_replay_path: Path,
    retention_report_path: Path,
    batch_path: Path,
) -> tuple[dict[str, Any], TrainingBatch]:
    tokenizer = _load_local_tokenizer(tokenizer_model_path)
    groups = _validated_collection_groups(
        collection_path=collection_path,
        expected_behavior_policy_revision=(
            expected_behavior_policy_revision
        ),
        expected_runtime_revision=expected_runtime_revision,
        pool_binding_path=pool_binding_path,
        sequence_len=PROVEN_SEQUENCE_LEN,
        tokenizer=tokenizer,
    )
    policy = _materialize_policy(
        groups,
        sequence_len=PROVEN_SEQUENCE_LEN,
        tokenizer=tokenizer,
    )
    replay, retention, retention_report = _load_required_replay(
        replay_path=replay_path,
        retention_replay_path=retention_replay_path,
        retention_report_path=retention_report_path,
        sequence_len=PROVEN_SEQUENCE_LEN,
    )
    batch = TrainingBatch(
        examples=[*policy.samples, *replay, *retention],
        step=0,
    )
    report = _report(
        collection_path=collection_path,
        pool_binding_path=pool_binding_path,
        tokenizer_model_path=tokenizer_model_path,
        behavior_policy_revision=expected_behavior_policy_revision,
        runtime_revision=expected_runtime_revision,
        policy=policy,
        replay_path=replay_path,
        replay_samples=replay,
        retention_replay_path=retention_replay_path,
        retention_samples=retention,
        retention_report_path=retention_report_path,
        retention_report=retention_report,
        batch_path=batch_path,
    )
    return report, batch


def materialize_synthetic_branchpoint_grpo_update(
    *,
    collection_path: Path,
    output_dir: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    tokenizer_model_path: Path,
    replay_path: Path,
    train_retention_replay_path: Path,
    train_retention_report_path: Path,
) -> dict[str, Any]:
    """Write one bounded exact-token branchpoint update with both replays."""
    output_dir = output_dir.expanduser().resolve()
    collection_path = collection_path.expanduser().resolve()
    pool_binding_path = pool_binding_path.expanduser().resolve()
    tokenizer_model_path = tokenizer_model_path.expanduser().resolve()
    replay_path = replay_path.expanduser().resolve()
    retention_path = train_retention_replay_path.expanduser().resolve()
    retention_report_path = (
        train_retention_report_path.expanduser().resolve()
    )
    if output_dir.exists():
        raise SyntheticBranchpointBatchError(
            "refusing to overwrite branchpoint update directory: "
            f"{output_dir}"
        )
    if not expected_behavior_policy_revision.strip():
        raise SyntheticBranchpointBatchError(
            "expected behavior-policy revision must be non-empty"
        )
    if not expected_runtime_revision.strip():
        raise SyntheticBranchpointBatchError(
            "expected runtime revision must be non-empty"
        )
    batch_path = output_dir / "rollouts.bin"
    report, batch = _prepare_expected(
        collection_path=collection_path,
        expected_behavior_policy_revision=(
            expected_behavior_policy_revision
        ),
        expected_runtime_revision=expected_runtime_revision,
        pool_binding_path=pool_binding_path,
        tokenizer_model_path=tokenizer_model_path,
        replay_path=replay_path,
        retention_replay_path=retention_path,
        retention_report_path=retention_report_path,
        batch_path=batch_path,
    )
    encoded = msgspec.msgpack.encode(batch)
    decoded = msgspec.msgpack.decode(encoded, type=TrainingBatch)
    if decoded != batch:
        raise SyntheticBranchpointBatchError(
            "serialized branchpoint update does not round-trip exactly"
        )
    output_dir.mkdir(parents=True)
    batch_path.write_bytes(encoded)
    (output_dir / "prepared_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def validate_synthetic_branchpoint_prepared_batch(
    *,
    prepared_report_path: Path,
    expected_behavior_policy_revision: str,
    expected_runtime_revision: str,
    pool_binding_path: Path,
    tokenizer_model_path: Path,
    replay_path: Path,
    train_retention_replay_path: Path,
    train_retention_report_path: Path,
) -> tuple[dict[str, Any], Path, TrainingBatch]:
    """Reconstruct and validate an exact branchpoint batch before staging."""
    prepared_report_path = prepared_report_path.expanduser().resolve()
    report = _read_json(prepared_report_path)
    collection_path = Path(
        str(report.get("source_collection", ""))
    ).expanduser().resolve()
    batch_path = prepared_report_path.parent / "rollouts.bin"
    combined = report.get("combined_batch")
    if (
        not isinstance(combined, dict)
        or Path(str(combined.get("path", ""))).resolve() != batch_path
        or not batch_path.is_file()
    ):
        raise SyntheticBranchpointBatchError(
            "prepared branchpoint batch path changed"
        )
    expected_report, expected_batch = _prepare_expected(
        collection_path=collection_path,
        expected_behavior_policy_revision=(
            expected_behavior_policy_revision
        ),
        expected_runtime_revision=expected_runtime_revision,
        pool_binding_path=pool_binding_path.expanduser().resolve(),
        tokenizer_model_path=tokenizer_model_path.expanduser().resolve(),
        replay_path=replay_path.expanduser().resolve(),
        retention_replay_path=(
            train_retention_replay_path.expanduser().resolve()
        ),
        retention_report_path=(
            train_retention_report_path.expanduser().resolve()
        ),
        batch_path=batch_path,
    )
    if report != expected_report:
        raise SyntheticBranchpointBatchError(
            "prepared branchpoint report differs from its source collection"
        )
    try:
        batch = msgspec.msgpack.decode(
            batch_path.read_bytes(),
            type=TrainingBatch,
        )
    except Exception as exc:
        raise SyntheticBranchpointBatchError(
            f"prepared branchpoint batch cannot be decoded: {exc}"
        ) from exc
    if batch != expected_batch:
        raise SyntheticBranchpointBatchError(
            "prepared branchpoint batch differs from exact source rows"
        )
    return report, batch_path, batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--behavior-policy-revision", required=True)
    parser.add_argument("--runtime-revision", required=True)
    parser.add_argument("--pool-binding", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
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
    args = parser.parse_args()
    report = materialize_synthetic_branchpoint_grpo_update(
        collection_path=args.collection,
        output_dir=args.output_dir,
        expected_behavior_policy_revision=args.behavior_policy_revision,
        expected_runtime_revision=args.runtime_revision,
        pool_binding_path=args.pool_binding,
        tokenizer_model_path=args.tokenizer_model,
        replay_path=args.replay,
        train_retention_replay_path=args.train_retention_replay,
        train_retention_report_path=args.train_retention_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "MAX_CONTRAST_GROUPS",
    "MAX_POLICY_SAMPLES",
    "MIN_CONTRAST_GROUPS",
    "MIN_POLICY_SAMPLES",
    "SYNTHETIC_BRANCHPOINT_BATCH_VERDICT",
    "SYNTHETIC_BRANCHPOINT_BATCH_VERSION",
    "SYNTHETIC_BRANCHPOINT_CREDIT_MODE",
    "SyntheticBranchpointBatchError",
    "materialize_synthetic_branchpoint_grpo_update",
    "validate_synthetic_branchpoint_prepared_batch",
]


if __name__ == "__main__":
    main()
