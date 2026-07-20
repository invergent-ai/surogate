from __future__ import annotations

from types import SimpleNamespace

import pytest

from surogate.grpo.orchestrator.hindsight import (
    HINDSIGHT_LOGPROBS_KEY,
    HINDSIGHT_MASK_KEY,
    HindsightRescoreError,
    attach_hindsight_completion_scores,
    augment_prompt_messages,
    build_hindsight_prefill_request,
    extract_hindsight_completion_logprobs,
)
from surogate.grpo.orchestrator.trajectories import interleave_rollout


CONTEXT = (
    "TRAINING-ONLY HINDSIGHT. Do not quote this context. Use independent "
    "verification capability before completion."
)


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **chat_template_kwargs
    ):
        assert chat_template_kwargs in ({}, {"enable_thinking": False})
        assert tokenize is True
        assert add_generation_prompt is True
        tokens = [101]
        roles = {"system": 11, "user": 12, "assistant": 13}
        for message in messages:
            tokens.append(roles[message["role"]])
            tokens.extend(ord(char) + 200 for char in message["content"])
        tokens.append(102)
        return tokens


def _scored_step() -> tuple[dict, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    prompt = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "Choose the next action."},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
    )
    return (
        {
            "prompt": prompt,
            "tokens": {
                "prompt_ids": prompt_ids,
                "prompt_mask": [False] * len(prompt_ids),
                "completion_ids": [701, 702, 703],
                "completion_mask": [True, True, False],
                "completion_logprobs": [-0.3, -0.2, -0.1],
            },
            "extras": {},
        },
        tokenizer,
    )


def test_prefill_request_keeps_exact_sampled_action_tokens() -> None:
    step, tokenizer = _scored_step()
    ordinary_ids = list(step["tokens"]["prompt_ids"])

    request = build_hindsight_prefill_request(
        step,
        tokenizer=tokenizer,
        training_context=CONTEXT,
    )

    assert request.completion_ids == (701, 702, 703)
    assert request.token_ids[-3:] == request.completion_ids
    assert list(request.token_ids[: request.completion_start]) != ordinary_ids
    augmented = augment_prompt_messages(step["prompt"], training_context=CONTEXT)
    assert augmented[-1]["content"].endswith(CONTEXT)
    assert step["prompt"][-1]["content"] == "Choose the next action."


def test_prefill_response_attaches_only_completion_aligned_scores() -> None:
    step, tokenizer = _scored_step()
    request = build_hindsight_prefill_request(
        step,
        tokenizer=tokenizer,
        training_context=CONTEXT,
    )
    prompt_logprobs = [None] + [
        {str(index): {"logprob": -1.0}}
        for index in range(1, len(request.token_ids))
    ]
    prompt_logprobs[-3:] = [
        {"701": {"logprob": -0.4}},
        {"702": {"logprob": -0.5}},
        {"703": {"logprob": -0.6}},
    ]
    values = extract_hindsight_completion_logprobs(
        SimpleNamespace(prompt_logprobs=prompt_logprobs),
        request=request,
    )
    attach_hindsight_completion_scores(step, logprobs=values)

    assert values == [-0.4, -0.5, -0.6]
    assert step["extras"][HINDSIGHT_LOGPROBS_KEY] == values
    assert step["extras"][HINDSIGHT_MASK_KEY] == [True, True, False]


def test_prompt_reconstruction_mismatch_fails_closed() -> None:
    step, tokenizer = _scored_step()
    step["tokens"]["prompt_ids"][-1] = 999

    with pytest.raises(HindsightRescoreError, match="reconstruction"):
        build_hindsight_prefill_request(
            step,
            tokenizer=tokenizer,
            training_context=CONTEXT,
        )


def test_interleave_preserves_hindsight_alignment_and_environment_mask() -> None:
    steps = [
        {
            "tokens": {
                "prompt_ids": [1, 2],
                "prompt_mask": [False, False],
                "completion_ids": [3, 4],
                "completion_mask": [True, False],
                "completion_logprobs": [-0.1, -0.2],
            },
            "extras": {
                HINDSIGHT_LOGPROBS_KEY: [-0.3, -0.4],
                HINDSIGHT_MASK_KEY: [True, True],
            },
        },
        {
            "tokens": {
                "prompt_ids": [1, 2, 3, 4, 5],
                "prompt_mask": [False] * 5,
                "completion_ids": [6, 7],
                "completion_mask": [True, True],
                "completion_logprobs": [-0.5, -0.6],
            },
            "extras": {
                HINDSIGHT_LOGPROBS_KEY: [-0.7, -0.8],
                HINDSIGHT_MASK_KEY: [True, True],
            },
        },
    ]
    samples = interleave_rollout(
        {
            "trajectory": steps,
            "error": None,
            "example_id": "seed-alignment",
            "sampling_args": {"temperature": 0.7},
        }
    )

    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.prompt_ids == [1, 2]
    assert sample.completion_ids == [3, 4, 5, 6, 7]
    assert sample.hindsight_logprobs == [0.0, 0.0, -0.3, -0.4, 0.0, -0.7, -0.8]
    assert sample.hindsight_mask == [False, False, True, False, False, True, True]
