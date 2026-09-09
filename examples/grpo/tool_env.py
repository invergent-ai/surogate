"""Local Verifiers function-tool task; no service or dataset download needed."""

import random

import verifiers as vf
from datasets import Dataset


def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.
    """
    return str(a + b)


def correct(completion, answer, **kwargs) -> float:
    """Reward a correct final answer after receiving a real tool result."""
    if not completion or not any(message.role == "tool" for message in completion):
        return 0.0
    return float((completion[-1].content or "").strip() == answer)


def load_environment(num_examples: int = 128, seed: int = 42, max_turns: int = 3, **kwargs):
    rng = random.Random(seed)
    rows = []
    for _ in range(num_examples):
        a, b = rng.randrange(10000), rng.randrange(10000)
        rows.append(
            dict(
                prompt=[
                    dict(
                        role="user",
                        content=(
                            f"First call the add tool with a={a} and b={b}. Wait for its result before answering. "
                            "Then reply with only the resulting number."
                        ),
                    )
                ],
                answer=str(a + b),
            )
        )
    return vf.ToolEnv(
        dataset=Dataset.from_list(rows), tools=[add], max_turns=max_turns, rubric=vf.Rubric(funcs=[correct]), **kwargs
    )
