"""Exclude non-trainable rollouts (e.g. infrastructure failures) from GRPO updates.

Some environments mark a rollout as not valid for training -- for example a
provider, harness, or grader failure that yields reward 0.0 but must not be
learned as a genuine negative (see the Fugu-Ultra failure taxonomy). Such a
rollout is flagged by a per-rollout rubric metric whose value is 1.0 when the
rollout is trainable and 0.0 when it is not.

A non-trainable rollout must be removed from BOTH:

  - the loss, by clearing its completion mask (no token contributes a gradient),
  - the per-group advantage baseline, otherwise its reward (typically 0.0) drags
    the group mean down and inflates the advantage of its trainable siblings.

For plain-mean GRPO, substituting a non-trainable rollout's reward with the
group's mean over trainable rollouts is exact: the substituted entry then has
zero advantage and leaves the baseline seen by trainable siblings equal to the
trainable mean. This module performs that substitution (returning a separate
advantage-reward list, so true rewards remain intact for logging) and clears the
masks, and reports how many rollouts and groups were affected.
"""

from __future__ import annotations

from typing import Any

TRAINABLE_THRESHOLD = 0.5


def _is_trainable(rollout: dict[str, Any], metric_name: str) -> bool:
    metrics = rollout.get("metrics") or {}
    return float(metrics.get(metric_name, 1.0)) >= TRAINABLE_THRESHOLD


def _clear_completion_mask(rollout: dict[str, Any]) -> None:
    for step in rollout.get("trajectory", []) or []:
        tokens = step.get("tokens")
        if tokens is not None:
            tokens["completion_mask"] = [0] * len(tokens["completion_mask"])
    rollout["stop_condition"] = "non_trainable"


def exclude_non_trainable_rollouts(
    rollouts: list[dict[str, Any]],
    rewards: list[float],
    samples_per_problem: int,
    metric_name: str,
) -> tuple[list[float], dict[str, float]]:
    """Mask non-trainable rollouts and return (advantage_rewards, metrics).

    Rollouts are grouped into consecutive blocks of ``samples_per_problem`` (the
    same grouping ``compute_advantages`` applies). Within each group, every
    rollout whose ``metric_name`` metric is below ``TRAINABLE_THRESHOLD`` has its
    completion mask cleared and its reward in the returned advantage-reward list
    replaced by the group's mean reward over trainable rollouts. The input
    ``rewards`` list and ``rollouts[*]["reward"]`` are left unchanged.

    A missing metric defaults to trainable (1.0), so environments that do not
    emit the flag are unaffected.
    """
    n = len(rollouts)
    advantage_rewards = list(rewards)
    if n == 0 or samples_per_problem <= 0:
        return advantage_rewards, {}

    non_trainable = 0
    degenerate_groups = 0
    num_groups = 0

    for start in range(0, n, samples_per_problem):
        group = list(range(start, min(start + samples_per_problem, n)))
        if not group:
            continue
        num_groups += 1
        trainable = {i for i in group if _is_trainable(rollouts[i], metric_name)}
        # Baseline over trainable rollouts only; fall back to the whole group when
        # none are trainable (all entries get masked, so the value is inert).
        ref = trainable if trainable else set(group)
        baseline = sum(rewards[i] for i in ref) / len(ref)
        if len(trainable) < 2:
            degenerate_groups += 1
        for i in group:
            if i not in trainable:
                _clear_completion_mask(rollouts[i])
                advantage_rewards[i] = baseline
                non_trainable += 1

    metrics = {
        "trainability/non_trainable_count": float(non_trainable),
        "trainability/non_trainable_rate": non_trainable / n,
        "trainability/degenerate_group_count": float(degenerate_groups),
        "trainability/degenerate_group_rate": degenerate_groups / num_groups if num_groups else 0.0,
    }
    return advantage_rewards, metrics
