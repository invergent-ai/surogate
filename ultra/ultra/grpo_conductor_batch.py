"""Turn collected conductor rollout groups into an exact-token GRPO batch.

Bridges `grpo_rollout_collector` output to the surogate trainer's
`TrainingBatch`. Three rules, each load-bearing:

* EXACT TOKENS. The behavior policy's sampled completion ids are the training
  tokens; they are never re-tokenized from text. Rollouts that carry the
  serving stack's own ids (`token_ids` / `prompt_token_ids`, via vLLM
  `return_token_ids`) use those DIRECTLY — grammar-constrained decoding can
  sample non-canonical splits, so re-encoding text is not equivalent
  (measured: 0/480 clean round-trips on r4 plans). The injected tokenizers
  are the fallback for legacy records only. When the sampler recorded
  serving logprobs, those are carried through as `completion_logprobs`.
* GROUP-NORMALIZED ADVANTAGE. A_i = (r_i - mean) / std within the question's
  group (Conductor eq. 7). std == 0 means the group is uninformative — every
  rollout agrees — and the whole group is DROPPED rather than trained on a
  zero (or worse, divide-by-zero) signal.
* PROMPT TOKENS CARRY NO LOSS. Only completion tokens are trainable, so the
  prompt mask is all-False and the completion mask all-True.

Infra failures (reward None) are excluded before advantages are computed, so
a provider outage can never masquerade as a policy failure.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

MIN_GROUP_STD = 1e-6


@dataclass(frozen=True)
class BatchStats:
    groups_in: int
    groups_used: int
    groups_uniform: int          # dropped: no reward variance
    rollouts_used: int
    rollouts_infra_excluded: int
    trainable_tokens: int
    # Samples whose serving logprobs were missing/misaligned and fell back to
    # zeros. The trainer's IPO mask then drops those tokens (inference_prob
    # reads as 1.0), so these samples are effectively DISCARDED — paid
    # rollouts buying nothing. Must stay ~0 now that ids+logprobs come from
    # vLLM return_token_ids; a rise means the serving path regressed.
    rollouts_logprob_fallback: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups_in": self.groups_in,
            "groups_used": self.groups_used,
            "groups_uniform_dropped": self.groups_uniform,
            "rollouts_used": self.rollouts_used,
            "rollouts_infra_excluded": self.rollouts_infra_excluded,
            "trainable_tokens": self.trainable_tokens,
            "rollouts_logprob_fallback": self.rollouts_logprob_fallback,
        }


def group_advantages(rewards: list[float]) -> list[float] | None:
    """(r - mean)/std, or None when the group carries no gradient."""
    if len(rewards) < 2:
        return None
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std < MIN_GROUP_STD:
        return None
    return [(r - mean) / std for r in rewards]


def build_conductor_batch(
    groups: list[dict[str, Any]],
    tokenize_prompt,
    tokenize_completion,
    step: int = 0,
    temperature: float = 1.0,
):
    """Materialize groups as a TrainingBatch of exact-token samples.

    `tokenize_prompt(question) -> list[int]` renders the conductor prompt for
    a question with the SAME template the policy was served with;
    `tokenize_completion(raw) -> list[int]` returns the sampled completion's
    ids. Both are injected so this module stays offline-testable and free of
    tokenizer/model assumptions.
    """
    from surogate.grpo.transport.types import TrainingBatch, TrainingSample

    samples: list[TrainingSample] = []
    groups_used = groups_uniform = 0
    rollouts_used = infra_excluded = trainable_tokens = 0
    logprob_fallbacks = 0

    for group in groups:
        scored = [r for r in group.get("rollouts", [])
                  if r.get("reward") is not None]
        infra_excluded += len(group.get("rollouts", [])) - len(scored)
        rewards = [float(r["reward"]) for r in scored]
        advantages = group_advantages(rewards)
        if advantages is None:
            groups_uniform += 1
            continue
        groups_used += 1
        fallback_prompt_ids = None
        for rollout, advantage, reward in zip(scored, advantages, rewards):
            completion_ids = rollout.get("token_ids") or tokenize_completion(
                rollout.get("raw", ""))
            if not completion_ids:
                continue
            prompt_ids = rollout.get("prompt_token_ids")
            if prompt_ids is None:
                if fallback_prompt_ids is None:
                    fallback_prompt_ids = tokenize_prompt(group["question"])
                prompt_ids = fallback_prompt_ids
            logprobs = _completion_logprobs(rollout, len(completion_ids))
            if logprobs == [0.0] * len(completion_ids):
                logprob_fallbacks += 1
            samples.append(TrainingSample(
                prompt_ids=list(prompt_ids),
                prompt_mask=[False] * len(prompt_ids),
                completion_ids=list(completion_ids),
                completion_mask=[True] * len(completion_ids),
                completion_logprobs=logprobs,
                completion_temperatures=[temperature] * len(completion_ids),
                advantage=float(advantage),
                reward=float(reward),
            ))
            rollouts_used += 1
            trainable_tokens += len(completion_ids)

    stats = BatchStats(
        groups_in=len(groups),
        groups_used=groups_used,
        groups_uniform=groups_uniform,
        rollouts_used=rollouts_used,
        rollouts_infra_excluded=infra_excluded,
        trainable_tokens=trainable_tokens,
        rollouts_logprob_fallback=logprob_fallbacks,
    )
    return TrainingBatch(examples=samples, step=step), stats


def _completion_logprobs(rollout: dict[str, Any], n_tokens: int) -> list[float]:
    """Serving logprobs when present, aligned to the completion length.

    A length mismatch means the server's logprob view and our token view
    disagree; rather than silently mis-aligning per-token credit, fall back to
    zeros (the trainer treats these as unavailable behavior likelihoods).
    """
    raw = rollout.get("logprobs")
    values: list[float] = []
    if isinstance(raw, dict):
        content = raw.get("content")
        if isinstance(content, list):
            for entry in content:
                if isinstance(entry, dict) and isinstance(
                    entry.get("logprob"), (int, float)
                ):
                    values.append(float(entry["logprob"]))
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, (int, float)):
                values.append(float(entry))
    if len(values) != n_tokens:
        return [0.0] * n_tokens
    return values
