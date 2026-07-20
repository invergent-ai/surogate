"""Exact sampled-token rescoring under a training-only hindsight context."""

from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Sequence

import verifiers as vf
from openai.types.chat.chat_completion import ChatCompletion
from verifiers.utils.client_utils import setup_openai_client

from surogate.grpo.orchestrator.utils import get_semaphore


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
    if (
        not isinstance(value, (list, tuple))
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
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


def augment_prompt_messages(
    prompt: Sequence[Any], *, training_context: str
) -> list[dict[str, Any]]:
    """Append privileged guidance to the final user message without changing roles."""
    context = training_context.strip()
    if not context.startswith(TRAINING_CONTEXT_PREFIX):
        raise HindsightRescoreError("hindsight context lacks the training-only prefix")
    messages = [_message_dict(message) for message in prompt]
    if not messages:
        raise HindsightRescoreError("cannot augment an empty prompt")
    user_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "user"
    ]
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
        raise HindsightRescoreError(
            "ordinary prompt token reconstruction does not match rollout tokens"
        )
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
        response.get("prompt_logprobs")
        if isinstance(response, dict)
        else getattr(response, "prompt_logprobs", None)
    )
    if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) != len(
        request.token_ids
    ):
        raise HindsightRescoreError("prefill logprobs do not align to submitted tokens")
    selected = prompt_logprobs[
        request.completion_start : request.completion_start + len(request.completion_ids)
    ]
    values: list[float] = []
    for entry in selected:
        if not isinstance(entry, dict) or not entry:
            raise HindsightRescoreError("sampled action token lacks a prefill logprob")
        payload = next(iter(entry.values()))
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
    step: dict[str, Any], *, logprobs: Sequence[float]
) -> None:
    """Attach an exact scorer result for later trajectory interleaving."""
    tokens = step.get("tokens")
    if not isinstance(tokens, dict):
        raise HindsightRescoreError("trajectory step lacks token data")
    completion_ids = tokens.get("completion_ids")
    completion_mask = tokens.get("completion_mask")
    if (
        not isinstance(completion_ids, list)
        or not isinstance(completion_mask, list)
        or len(completion_ids) != len(completion_mask)
        or len(logprobs) != len(completion_ids)
    ):
        raise HindsightRescoreError("hindsight scores do not align to completion tokens")
    values = [float(value) for value in logprobs]
    if any(not math.isfinite(value) or value > 1e-6 for value in values):
        raise HindsightRescoreError("hindsight scores must be finite logprobs")
    extras = step.setdefault("extras", {})
    if not isinstance(extras, dict):
        raise HindsightRescoreError("trajectory step extras must be an object")
    extras[HINDSIGHT_LOGPROBS_KEY] = values
    extras[HINDSIGHT_MASK_KEY] = [bool(value) for value in completion_mask]


async def compute_hindsight_logprobs(
    *,
    clients: list[vf.ClientConfig],
    model_name: str,
    tokenizer: Any,
    rollouts: Sequence[dict[str, Any]],
    training_contexts: Sequence[str | None],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> int:
    """Re-score rollout actions in place with the same inference-policy endpoint."""
    if len(rollouts) != len(training_contexts):
        raise HindsightRescoreError("rollout and hindsight-context counts differ")
    if not clients:
        raise HindsightRescoreError("at least one inference client is required")

    requests: list[tuple[vf.ClientConfig, dict[str, Any], HindsightPrefillRequest]] = []
    client_cycle = cycle(clients)
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
            requests.append((next(client_cycle), step, request))

    async def score_one(
        client_config: vf.ClientConfig,
        step: dict[str, Any],
        request: HindsightPrefillRequest,
    ) -> None:
        client = setup_openai_client(client_config)
        async with await get_semaphore():
            response = await client.post(
                "/chat/completions/tokens",
                body={
                    "model": model_name,
                    "messages": [{"role": "user", "content": ""}],
                    "tokens": list(request.token_ids),
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "skip_special_tokens": False,
                    "prompt_logprobs": True,
                },
                cast_to=ChatCompletion,
            )
        attach_hindsight_completion_scores(
            step,
            logprobs=extract_hindsight_completion_logprobs(
                response,
                request=request,
            ),
        )

    await asyncio.gather(*(score_one(*item) for item in requests))
    return len(requests)
