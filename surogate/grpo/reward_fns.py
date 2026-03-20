"""Built-in reward functions for native GRPO callback mode."""


def length_diversity_reward(
    prompts: list[str], completions: list[str]
) -> list[float]:
    """Simple reward: prefer longer, non-repetitive completions.

    Score = unique_word_ratio * length_bonus
    - unique_word_ratio: fraction of unique words in the completion
    - length_bonus: min(word_count / 20, 1.0)
    """
    rewards = []
    for completion in completions:
        words = completion.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        length_bonus = min(len(words) / 20.0, 1.0)
        rewards.append(unique_ratio * length_bonus)
    return rewards
