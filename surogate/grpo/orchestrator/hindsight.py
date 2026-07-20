"""Exact sampled-token rescoring under a training-only hindsight context."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

OPD_REFERENCE_LOGPROBS_KEY = "seed_opd_reference_completion_logprobs"
HINDSIGHT_LOGPROBS_KEY = "seed_hindsight_completion_logprobs"
HINDSIGHT_MASK_KEY = "seed_hindsight_completion_mask"
TRAINING_CONTEXT_PREFIX = "TRAINING-ONLY HINDSIGHT."


class HindsightRescoreError(ValueError):
    """The privileged and ordinary branches cannot be aligned exactly."""


@dataclass(frozen=True)
class HindsightPrefillRequest:
    token_ids: tuple[int, ...]
    completion_start: int
    completion_ids: tuple[int, ...]


def _message_dict(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        try:
            message = dict(message)
        except (TypeError, ValueError) as exc:
            raise HindsightRescoreError("prompt message is not mapping-like") from exc
    return copy.deepcopy(message)


def _token_list(value: Any, *, label: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        value = value.get("input_ids")
        if hasattr(value, "tolist"):
            value = value.tolist()
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise HindsightRescoreError(f"{label} tokenizer output is not a token-id list")
    return list(value)


def _render_prompt_ids(
    tokenizer: Any,
    messages: Sequence[dict[str, Any]],
    *,
    chat_template_kwargs: dict[str, Any] | None,
) -> list[int]:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    except Exception as exc:
        raise HindsightRescoreError("chat-template rendering failed") from exc
    return _token_list(rendered, label="chat template")


def augment_prompt_messages(prompt: Sequence[Any], *, training_context: str) -> list[dict[str, Any]]:
    """Append privileged guidance to the final user message without changing roles."""
    context = training_context.strip()
    if not context.startswith(TRAINING_CONTEXT_PREFIX):
        raise HindsightRescoreError("hindsight context lacks the training-only prefix")
    messages = [_message_dict(message) for message in prompt]
    if not messages:
        raise HindsightRescoreError("cannot augment an empty prompt")
    user_indices = [index for index, message in enumerate(messages) if message.get("role") == "user"]
    if not user_indices:
        raise HindsightRescoreError("prompt has no user message for hindsight augmentation")
    index = user_indices[-1]
    content = messages[index].get("content")
    if not isinstance(content, str):
        raise HindsightRescoreError("final user message content must be plain text")
    messages[index]["content"] = f"{content.rstrip()}\n\n{context}"
    return messages


def build_hindsight_prefill_request(
    step: dict[str, Any],
    *,
    tokenizer: Any,
    training_context: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> HindsightPrefillRequest:
    """Build an augmented prefill containing the exact sampled completion tokens."""
    tokens = step.get("tokens")
    prompt = step.get("prompt")
    if not isinstance(tokens, dict) or not isinstance(prompt, list):
        raise HindsightRescoreError("trajectory step lacks prompt or token data")
    ordinary_prompt_ids = _token_list(tokens.get("prompt_ids"), label="ordinary prompt")
    completion_ids = _token_list(tokens.get("completion_ids"), label="completion")
    completion_mask = tokens.get("completion_mask")
    if len(completion_ids) == 0:
        raise HindsightRescoreError("trajectory step has no sampled action tokens")
    if not isinstance(completion_mask, list) or len(completion_mask) != len(completion_ids):
        raise HindsightRescoreError("completion mask is not aligned to sampled tokens")

    ordinary_messages = [_message_dict(message) for message in prompt]
    reconstructed = _render_prompt_ids(
        tokenizer,
        ordinary_messages,
        chat_template_kwargs=chat_template_kwargs,
    )
    if reconstructed != ordinary_prompt_ids:
        raise HindsightRescoreError("ordinary prompt token reconstruction does not match rollout tokens")
    augmented_messages = augment_prompt_messages(
        ordinary_messages,
        training_context=training_context,
    )
    augmented_prompt_ids = _render_prompt_ids(
        tokenizer,
        augmented_messages,
        chat_template_kwargs=chat_template_kwargs,
    )
    if augmented_prompt_ids == ordinary_prompt_ids:
        raise HindsightRescoreError("hindsight augmentation did not change prompt tokens")
    return HindsightPrefillRequest(
        token_ids=tuple([*augmented_prompt_ids, *completion_ids]),
        completion_start=len(augmented_prompt_ids),
        completion_ids=tuple(completion_ids),
    )


def extract_hindsight_completion_logprobs(
    response: Any,
    *,
    request: HindsightPrefillRequest,
) -> list[float]:
    """Extract finite logprobs for exactly the fixed completion token span."""
    prompt_logprobs = (
        response.get("prompt_logprobs") if isinstance(response, dict) else getattr(response, "prompt_logprobs", None)
    )
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(request.token_ids):
        raise HindsightRescoreError("prefill logprobs do not align to submitted tokens")
    selected = prompt_logprobs[request.completion_start : request.completion_start + len(request.completion_ids)]
    values: list[float] = []
    for token_id, entry in zip(request.completion_ids, selected, strict=True):
        if not isinstance(entry, dict) or not entry:
            raise HindsightRescoreError("sampled action token lacks a prefill logprob")
        payload = entry.get(str(token_id), entry.get(token_id))
        if payload is None:
            raise HindsightRescoreError(f"prefill logprobs omit submitted action token {token_id}")
        value = payload.get("logprob") if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HindsightRescoreError("sampled action logprob is not numeric")
        value = float(value)
        if not math.isfinite(value) or value > 1e-6:
            raise HindsightRescoreError("sampled action logprob is not finite and non-positive")
        values.append(value)
    if len(values) != len(request.completion_ids):
        raise HindsightRescoreError("sampled action logprob span length drift")
    return values


def attach_hindsight_completion_scores(
    step: dict[str, Any],
    *,
    reference_logprobs: Sequence[float],
    hindsight_logprobs: Sequence[float],
) -> None:
    """Attach one deterministic matched-branch result for interleaving."""
    tokens = step.get("tokens")
    if not isinstance(tokens, dict):
        raise HindsightRescoreError("trajectory step lacks token data")
    completion_ids = tokens.get("completion_ids")
    completion_mask = tokens.get("completion_mask")
    if (
        not isinstance(completion_ids, list)
        or not isinstance(completion_mask, list)
        or len(completion_ids) != len(completion_mask)
        or len(reference_logprobs) != len(completion_ids)
        or len(hindsight_logprobs) != len(completion_ids)
    ):
        raise HindsightRescoreError("matched OPD scores do not align to completion tokens")
    reference_values = [float(value) for value in reference_logprobs]
    hindsight_values = [float(value) for value in hindsight_logprobs]
    if any(not math.isfinite(value) or value > 1e-6 for value in [*reference_values, *hindsight_values]):
        raise HindsightRescoreError("matched OPD scores must be finite logprobs")
    extras = step.setdefault("extras", {})
    if not isinstance(extras, dict):
        raise HindsightRescoreError("trajectory step extras must be an object")
    extras[OPD_REFERENCE_LOGPROBS_KEY] = reference_values
    extras[HINDSIGHT_LOGPROBS_KEY] = hindsight_values
    extras[HINDSIGHT_MASK_KEY] = [bool(value) for value in completion_mask]


def compute_matched_hindsight_logprobs(
    *,
    scorer: Any,
    tokenizer: Any,
    rollouts: Sequence[dict[str, Any]],
    training_contexts: Sequence[str | None],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    """Score ordinary and hindsight branches together using one direct scorer."""
    if len(rollouts) != len(training_contexts):
        raise HindsightRescoreError("rollout and hindsight-context counts differ")

    scored_steps = 0
    for rollout, training_context in zip(rollouts, training_contexts, strict=True):
        if training_context is None:
            continue
        trajectory = rollout.get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise HindsightRescoreError("hindsight rollout has no completed trajectory")
        for step in trajectory:
            request = build_hindsight_prefill_request(
                step,
                tokenizer=tokenizer,
                training_context=training_context,
                chat_template_kwargs=chat_template_kwargs,
            )
            tokens = step.get("tokens")
            if not isinstance(tokens, dict):
                raise HindsightRescoreError("trajectory step lacks token data")
            ordinary_prompt_ids = _token_list(tokens.get("prompt_ids"), label="ordinary prompt")
            matched = scorer.score_matched(
                ordinary_prompt_ids=ordinary_prompt_ids,
                hindsight_prompt_ids=list(request.token_ids[: request.completion_start]),
                completion_ids=list(request.completion_ids),
            )
            attach_hindsight_completion_scores(
                step,
                reference_logprobs=matched.reference_logprobs,
                hindsight_logprobs=matched.hindsight_logprobs,
            )
            scored_steps += 1

    return scored_steps


async def compute_hindsight_logprobs(**_: Any) -> int:
    """Reject the retired endpoint scorer before it can create training data."""
    raise HindsightRescoreError(
        "endpoint-based hindsight scoring is disabled; use "
        "compute_matched_hindsight_logprobs with a deterministic direct scorer"
    )
