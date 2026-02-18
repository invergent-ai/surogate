from dataclasses import dataclass
from typing import Optional, List

from surogate.core.config.sft_config import SFTConfig
from surogate.utils.dict import DictDefault


@dataclass
class GRPOConfig(SFTConfig):
    """
    Configuration for GRPO (Group Relative Policy Optimization) training.

    Extends SFTConfig with online-generation and RL-specific parameters.

    Args:
        grpo_datasets (Optional[str]):
            Path to a JSONL file with prompt/ground-truth pairs.
            Format: {"prompt": "...", "ground_truth": "..."}
            The "ground_truth" field is passed to reward functions (e.g., AccuracyReward).
        num_generations (int, defaults to 8):
            Number of completions to generate per prompt (G in GRPO).
            Advantages are computed group-relative within these G samples.
        max_completion_length (int, defaults to 512):
            Maximum number of tokens to generate per completion.
        temperature (float, defaults to 1.0):
            Sampling temperature for generation. 0.0 → greedy argmax.
        top_p (float, defaults to 1.0):
            Nucleus (top-p) sampling threshold. 1.0 disables nucleus filtering.
        top_k (int, defaults to -1):
            Top-k sampling. -1 disables top-k filtering.
        grpo_beta (float, defaults to 0.0):
            KL-divergence penalty coefficient (β). 0.0 disables the KL term.
            When > 0, adds β × KL(policy ‖ reference) to the GRPO loss.
        grpo_epsilon (float, defaults to 0.2):
            PPO-style clipping range (ε). The importance-sampling ratio is
            clipped to [1-ε, 1+ε] × advantage.
        reward_functions (List[str], defaults to ['accuracy']):
            Names of reward functions to apply. Built-in options:
            - 'accuracy': exact-match vs ground_truth (case-insensitive strip)
            - 'format': checks for <think>...</think><answer>...</answer> structure
            Multiple rewards are summed.
    """

    # Dataset
    grpo_datasets: Optional[str] = None

    # Generation
    num_generations: Optional[int] = 8
    max_completion_length: Optional[int] = 512

    # Sampling
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = -1

    # GRPO loss
    grpo_beta: Optional[float] = 0.0
    grpo_epsilon: Optional[float] = 0.2

    # Reward
    reward_functions: Optional[List[str]] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)

        self.grpo_datasets = cfg.get('grpo_datasets', self.grpo_datasets)
        self.num_generations = cfg.get('num_generations', self.num_generations)
        self.max_completion_length = cfg.get('max_completion_length', self.max_completion_length)
        self.temperature = float(cfg.get('temperature', self.temperature))
        self.top_p = float(cfg.get('top_p', self.top_p))
        self.top_k = int(cfg.get('top_k', self.top_k))
        self.grpo_beta = float(cfg.get('grpo_beta', self.grpo_beta))
        self.grpo_epsilon = float(cfg.get('grpo_epsilon', self.grpo_epsilon))

        reward_fns = cfg.get('reward_functions', None)
        if reward_fns is None:
            self.reward_functions = ['accuracy']
        elif isinstance(reward_fns, str):
            self.reward_functions = [reward_fns]
        else:
            self.reward_functions = list(reward_fns)
