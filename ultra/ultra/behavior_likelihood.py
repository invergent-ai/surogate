"""Behavior-policy sampling contract for exact-likelihood conductor training."""

from __future__ import annotations

from typing import Any

FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION = (
    "fugu_full_vocabulary_behavior_likelihood_v2"
)


def full_vocabulary_behavior_likelihood_contract() -> dict[str, Any]:
    """Return the exact neutral-sampling contract used for paid policy tokens."""

    return {
        "version": FULL_VOCABULARY_BEHAVIOR_LIKELIHOOD_CONTRACT_VERSION,
        "training_eligible": True,
        "sampling_support": "full_vocabulary",
        "response_format": "omitted",
        "logprobs_mode": "processed_logprobs",
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }


def has_full_vocabulary_behavior_likelihood_contract(value: object) -> bool:
    """Return whether *value* exactly attests the current training contract."""

    if not isinstance(value, dict):
        return False
    expected = full_vocabulary_behavior_likelihood_contract()
    return value.keys() == expected.keys() and all(
        type(value[key]) is type(expected_value) and value[key] == expected_value
        for key, expected_value in expected.items()
    )
