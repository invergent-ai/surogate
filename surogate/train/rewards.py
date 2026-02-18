# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Rule-based reward functions for GRPO training.
#
# Rewards are computed entirely in Python — no neural reward model is needed.
# Multiple rewards can be combined by listing them in GRPOConfig.reward_functions;
# the GRPO trainer sums them at each position.

import re
from abc import ABC, abstractmethod
from typing import List, Optional


class RewardFunction(ABC):
    """Base class for GRPO reward functions.

    Reward functions are called after generation with the full list of
    prompts, completions, and (optionally) ground-truth answers.

    Args:
        prompts:       List of prompt strings (one per sample).
        completions:   List of completion strings (one per sample).
        ground_truths: Optional list of ground-truth strings aligned with
                       prompts.  Not all rewards use this.

    Returns:
        List of scalar reward values, one per completion.
    """

    @abstractmethod
    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[float]:
        raise NotImplementedError


class AccuracyReward(RewardFunction):
    """Exact-match reward: +1 if the completion matches the ground truth.

    Matching is case-insensitive after stripping leading/trailing whitespace.
    The comparison is done against the answer extracted from the completion
    when an ``<answer>...</answer>`` tag is present; otherwise the full
    stripped completion is compared.

    Returns 0.0 for every completion when ground_truths is None.
    """

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[float]:
        if ground_truths is None:
            return [0.0] * len(completions)

        rewards = []
        for completion, gt in zip(completions, ground_truths):
            answer = _extract_answer(completion)
            match = answer.strip().lower() == str(gt).strip().lower()
            rewards.append(1.0 if match else 0.0)
        return rewards


class FormatReward(RewardFunction):
    """Structural format reward for chain-of-thought completions.

    Awards +0.5 if the completion contains ``<think>...</think>`` and
    +0.5 if it also contains ``<answer>...</answer>`` (total +1 if both).
    """

    _THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
    _ANSWER_RE = re.compile(r'<answer>.*?</answer>', re.DOTALL)

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[float]:
        rewards = []
        for completion in completions:
            score = 0.0
            if self._THINK_RE.search(completion):
                score += 0.5
            if self._ANSWER_RE.search(completion):
                score += 0.5
            rewards.append(score)
        return rewards


class LengthPenaltyReward(RewardFunction):
    """Soft length penalty: discourages very long completions.

    Returns a reward in [-penalty_scale, 0] proportional to how much the
    completion exceeds ``target_length`` tokens (split by whitespace).
    Completions at or below ``target_length`` receive 0.
    """

    def __init__(self, target_length: int = 256, penalty_scale: float = 0.5):
        self.target_length = target_length
        self.penalty_scale = penalty_scale

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[float]:
        rewards = []
        for completion in completions:
            n_tokens = len(completion.split())
            if n_tokens <= self.target_length:
                rewards.append(0.0)
            else:
                excess = n_tokens - self.target_length
                penalty = -self.penalty_scale * (excess / self.target_length)
                rewards.append(max(penalty, -self.penalty_scale))
        return rewards


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BUILTIN_REWARDS = {
    'accuracy': AccuracyReward,
    'format': FormatReward,
    'length_penalty': LengthPenaltyReward,
}


def build_reward_functions(names: List[str]) -> List[RewardFunction]:
    """Instantiate reward functions by name.

    Args:
        names: List of reward function names (see _BUILTIN_REWARDS keys).

    Returns:
        List of instantiated RewardFunction objects.

    Raises:
        ValueError: If an unknown reward function name is given.
    """
    fns = []
    for name in names:
        if name not in _BUILTIN_REWARDS:
            raise ValueError(
                f"Unknown reward function '{name}'. "
                f"Available: {sorted(_BUILTIN_REWARDS.keys())}"
            )
        fns.append(_BUILTIN_REWARDS[name]())
    return fns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_answer(completion: str) -> str:
    """Return text inside <answer>...</answer> tags, or the full completion."""
    m = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if m:
        return m.group(1)
    return completion
