from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from surogate.core.config.grpo_orch_config import AdvantageConfigType, GRPOCustomAdvantageConfig
from surogate.grpo.utils.utils import import_object


@dataclass
class AdvantageInputs:
    """Inputs for advantage computation."""

    rewards: Float[Tensor, "num_problems rollouts_per_example"]
    completion_lengths: Int[Tensor, "num_problems rollouts_per_example"]


@dataclass
class AdvantageOutputs:
    """Outputs from advantage computation."""

    advantages: Float[Tensor, "num_problems rollouts_per_example"]


AdvantageFn = Callable[..., AdvantageOutputs]
"""Type for an advantage function.

Expected signature:
    def my_advantage(inputs: AdvantageInputs, **kwargs) -> AdvantageOutputs:
        ...
"""


def default_advantage_fn(inputs: AdvantageInputs, length_weighted_mean: bool = False) -> AdvantageOutputs:
    """Default GRPO advantage: reward minus per-problem baseline."""
    if length_weighted_mean:
        baseline = (inputs.rewards * inputs.completion_lengths).sum(
            dim=1, keepdim=True
        ) / inputs.completion_lengths.sum(dim=1, keepdim=True)
    else:
        baseline = inputs.rewards.mean(dim=1, keepdim=True)

    return AdvantageOutputs(advantages=inputs.rewards - baseline)


def std_normalized_advantage(inputs: AdvantageInputs, eps: float = 1e-4) -> AdvantageOutputs:
    """GRPO advantage with per-group std normalization: A = (r - mean) / (std + eps).

    Matches Shao et al. (2024) eq. 4 as used by the Conductor (Nielsen et al., 2025, eq. 2):
    the group baseline is the mean and the residual is scaled by the group's reward std,
    so nearly-uniform groups still yield unit-scale advantages. Zero-variance groups get
    zero advantage from the mean subtraction; eps only guards the division.
    """
    mean = inputs.rewards.mean(dim=1, keepdim=True)
    std = inputs.rewards.std(dim=1, keepdim=True)
    return AdvantageOutputs(advantages=(inputs.rewards - mean) / (std + eps))


def setup_advantage_fn(config: AdvantageConfigType) -> AdvantageFn:
    """Setup advantage function from config."""
    if isinstance(config, GRPOCustomAdvantageConfig):
        custom_fn = import_object(config.import_path)
        kwargs = config.kwargs

        def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
            return custom_fn(inputs, **kwargs)

        return advantage_fn

    def advantage_fn(inputs: AdvantageInputs) -> AdvantageOutputs:
        return default_advantage_fn(inputs, length_weighted_mean=config.length_weighted_mean)

    return advantage_fn


def compute_advantages(
    rewards: list[float],
    completion_lengths: list[int],
    samples_per_problem: int,
    advantage_config: AdvantageConfigType | None,
) -> list[float]:
    """
    Computes advantages from a flattened list of rewards, grouped by problem.

    Args:
        rewards: Flattened list of rewards where first `samples_per_problem` rewards are for the first problem
        completion_lengths: List of completion lengths for each reward
        samples_per_problem: Number of samples (and thus, rewards) per problem
        advantage_config: Configuration for advantage computation (AdvantageConfig or CustomAdvantageConfig)
    """
    if not advantage_config:
        return rewards

    advantage_fn = setup_advantage_fn(advantage_config)

    inputs = AdvantageInputs(
        rewards=torch.tensor(rewards).view(-1, samples_per_problem),
        completion_lengths=torch.tensor(completion_lengths).view(-1, samples_per_problem),
    )

    result = advantage_fn(inputs)
    return result.advantages.flatten().tolist()
