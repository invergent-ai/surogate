"""Reward providers for native GRPO training.

Abstracts reward computation behind a common interface, supporting both
verifiers environments (with multi-turn rollouts) and simple callback functions.
"""

from __future__ import annotations

import asyncio
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Callable

from surogate.grpo.native_config import NativeGRPORewardConfig
from surogate.utils.logger import get_logger

logger = get_logger()


class RewardProvider(ABC):
    """Abstract base class for reward computation."""

    @abstractmethod
    def score(
        self,
        prompts: list[str],
        completions: list[str],
        num_completions: int,
    ) -> list[float]:
        """Score completions.

        Args:
            prompts: Prompt strings (one per completion, expanded by num_completions).
            completions: Decoded completion strings (len = num_prompts * num_completions).
            num_completions: Number of completions per unique prompt.

        Returns:
            Flat list of float rewards (same length as completions).
        """
        ...

    @abstractmethod
    def get_next_batch(self, batch_size: int) -> list[str]:
        """Return the next batch of prompt strings for generation."""
        ...

    @abstractmethod
    def has_data(self) -> bool:
        """Whether there is data remaining."""
        ...

    def get_rollout_data(self) -> list[dict] | None:
        """Return rollout data from the last score() call, if available.

        Used by the trainer to extract per-token logprobs and trajectory data
        from multi-turn verifiers rollouts.
        """
        return None


class CallbackRewardProvider(RewardProvider):
    """Wraps a user-provided reward function.

    The reward function signature is:
        fn(prompts: list[str], completions: list[str]) -> list[float]
    """

    def __init__(
        self,
        reward_fn: Callable[[list[str], list[str]], list[float]],
        dataset_path: str | None = None,
        prompts: list[str] | None = None,
    ):
        self.reward_fn = reward_fn
        self._prompts: list[str] = []
        self._index = 0

        if dataset_path is not None:
            self._prompts = self._load_dataset(dataset_path)
        elif prompts is not None:
            self._prompts = list(prompts)

    @staticmethod
    def _load_dataset(path: str) -> list[str]:
        """Load prompts from JSONL file. Each line must have a 'prompt' field."""
        prompts = []
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["prompt"])
        return prompts

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        num_completions: int,
    ) -> list[float]:
        return self.reward_fn(prompts, completions)

    def get_next_batch(self, batch_size: int) -> list[str]:
        if not self._prompts:
            raise RuntimeError("No prompts available. Provide dataset_path or prompts.")
        batch = []
        for _ in range(batch_size):
            batch.append(self._prompts[self._index % len(self._prompts)])
            self._index += 1
        return batch

    def has_data(self) -> bool:
        return len(self._prompts) > 0


class VerifiersRewardProvider(RewardProvider):
    """Uses verifiers environments for reward computation with multi-turn support.

    Loads vf.Environment instances and uses a NativeClient to run full
    multi-turn rollouts through the environment's rollout loop. This means
    the environment controls the conversation flow (tool calls, environment
    responses, stop conditions) while generation happens natively via C++.
    """

    def __init__(
        self,
        env_configs: list[Any],
        tokenizer: Any,
    ):
        import verifiers as vf

        self._envs: list[vf.Environment] = []
        self._env_names: list[str] = []
        self._datasets: list[list[dict]] = []
        self._tokenizer = tokenizer
        self._index = 0
        self._native_client = None  # Set via set_trainer()
        self._last_rollout_outputs: list[Any] = []

        for env_cfg in env_configs:
            env_id = env_cfg.id.split("@")[0] if "@" in env_cfg.id else env_cfg.id
            env_args = env_cfg.args or {}
            env = vf.load_environment(env_id, **env_args)
            env_name = env_cfg.name or env_id
            self._envs.append(env)
            self._env_names.append(env_name)

            # Extract dataset from environment
            ds = env.get_dataset(n=-1)
            self._datasets.append(ds)
            logger.info(f"Loaded verifiers env '{env_name}' with {len(ds)} examples")

        # Flatten all examples with env reference
        self._all_examples: list[tuple[int, dict]] = []
        for env_idx, ds in enumerate(self._datasets):
            for example in ds:
                self._all_examples.append((env_idx, example))
        random.shuffle(self._all_examples)

    def set_trainer(self, trainer: Any, max_gen_len: int = 512, use_lora: bool = True):
        """Initialize the NativeClient with the C++ trainer.

        Must be called before score() to enable multi-turn rollouts.
        """
        from surogate.grpo.native_client import NativeClient

        self._native_client = NativeClient(
            trainer=trainer,
            tokenizer=self._tokenizer,
            max_gen_len=max_gen_len,
            use_lora=use_lora,
        )

    def _run_rollouts(
        self,
        examples: list[dict],
        env_indices: list[int],
        num_completions: int,
        sampling_args: dict | None = None,
    ) -> list[Any]:
        """Run multi-turn rollouts through verifiers environments.

        Returns list of RolloutOutput dicts (one per prompt * num_completions).
        """
        if self._native_client is None:
            raise RuntimeError(
                "NativeClient not initialized. Call set_trainer() first."
            )

        async def _run():
            # Run rollouts sequentially — the NativeClient's generate() is a
            # blocking synchronous GPU call that cannot run concurrently.
            results = []
            for example, env_idx in zip(examples, env_indices):
                env = self._envs[env_idx]
                for _ in range(num_completions):
                    rollout_input = {
                        "prompt": example.get("prompt", []),
                        "example_id": example.get("example_id", 0),
                        "task": example.get("task", self._env_names[env_idx]),
                        "answer": example.get("answer", ""),
                        "info": example.get("info", {}),
                    }
                    try:
                        output = await env.run_rollout(
                            input=rollout_input,
                            client=self._native_client,
                            model="native",
                            sampling_args=sampling_args or {},
                            state_columns=["trajectory", "sampling_args"],
                        )
                        results.append(output)
                    except Exception as e:
                        logger.warning(f"Rollout failed: {e}")
                        results.append(e)
            return results

        return asyncio.run(_run())

    def score(
        self,
        prompts: list[str],
        completions: list[str],
        num_completions: int,
        use_rollouts: bool = False,
    ) -> list[float]:
        """Score completions using verifiers.

        Args:
            use_rollouts: When True, runs full multi-turn rollouts via
                NativeClient (generation + scoring together). When False
                (default), scores pre-generated completions using rubric.
        """
        if use_rollouts and self._native_client is not None:
            return self._score_with_rollouts(num_completions)
        return self._score_with_rubric(prompts, completions, num_completions)

    def _score_with_rollouts(self, num_completions: int) -> list[float]:
        """Score by running full multi-turn rollouts."""
        examples = self._current_examples
        env_indices = self._current_env_indices

        # Pass temperature in sampling_args — required by interleave_rollout
        sampling_args = {"temperature": self._native_client.default_temperature}

        rollout_outputs = self._run_rollouts(
            examples=examples,
            env_indices=env_indices,
            num_completions=num_completions,
            sampling_args=sampling_args,
        )

        # Extract rewards, handling exceptions
        rewards = []
        valid_outputs = []
        for output in rollout_outputs:
            if isinstance(output, Exception):
                logger.warning(f"Rollout failed: {output}")
                rewards.append(0.0)
            else:
                reward = output.get("reward", 0.0) if isinstance(output, dict) else getattr(output, "reward", 0.0)
                rewards.append(float(reward))
                valid_outputs.append(output)

        self._last_rollout_outputs = rollout_outputs
        return rewards

    def _score_with_rubric(
        self,
        prompts: list[str],
        completions: list[str],
        num_completions: int,
    ) -> list[float]:
        """Fallback: score pre-generated completions using rubric directly."""
        if not self._envs:
            return [0.0] * len(completions)

        env_idx = 0
        env = self._envs[env_idx]
        n_prompts = len(completions) // num_completions
        examples = self._current_examples[:n_prompts] if hasattr(self, "_current_examples") else [{}] * n_prompts

        states = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            example_idx = i // num_completions
            example = examples[example_idx]
            state = {
                "example_id": example.get("example_id", example_idx),
                "task": example.get("task", self._env_names[env_idx]),
                "prompt": example.get("prompt", [{"role": "user", "content": prompt}]),
                "completion": [{"role": "assistant", "content": completion}],
                "answer": example.get("answer", ""),
                "info": example.get("info", {}),
                "reward": 0.0,
                "metrics": {},
                "timing": {"generation_ms": 0.0, "scoring_ms": 0.0, "total_ms": 0.0},
                "is_completed": True,
                "is_truncated": False,
                "trajectory": [],
                "stop_condition": None,
                "error": None,
            }
            states.append(state)

        async def _score():
            await env.rubric.score_group(states)
            return [s.get("reward", 0.0) for s in states]

        return asyncio.run(_score())

    def get_rollout_data(self) -> list[Any] | None:
        """Return rollout outputs from the last multi-turn score() call."""
        return self._last_rollout_outputs if self._last_rollout_outputs else None

    def get_next_batch(self, batch_size: int) -> list[str]:
        if not self._all_examples:
            raise RuntimeError("No examples available from verifiers environments.")

        batch_prompts = []
        batch_examples = []
        batch_env_indices = []
        for _ in range(batch_size):
            env_idx, example = self._all_examples[self._index % len(self._all_examples)]
            self._index += 1

            # Extract prompt text from messages
            prompt_messages = example.get("prompt", [])
            if isinstance(prompt_messages, list) and prompt_messages:
                try:
                    prompt_text = self._tokenizer.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt_text = "\n".join(
                        m.get("content", "") for m in prompt_messages if isinstance(m, dict)
                    )
            elif isinstance(prompt_messages, str):
                prompt_text = prompt_messages
            else:
                prompt_text = str(prompt_messages)

            batch_prompts.append(prompt_text)
            batch_examples.append(example)
            batch_env_indices.append(env_idx)

        # Store for scoring reference
        self._current_examples = batch_examples
        self._current_env_indices = batch_env_indices
        return batch_prompts

    def has_data(self) -> bool:
        return len(self._all_examples) > 0


def setup_reward_provider(
    config: NativeGRPORewardConfig | None,
    tokenizer: Any,
    reward_fn: Callable | None = None,
) -> RewardProvider:
    """Factory to create a RewardProvider from config.

    Args:
        config: Reward configuration from NativeGRPOConfig.
        tokenizer: HuggingFace tokenizer.
        reward_fn: Optional reward function for callback mode. If provided,
            overrides config.mode.
    """
    if reward_fn is not None:
        return CallbackRewardProvider(
            reward_fn=reward_fn,
            dataset_path=config.dataset_path if config else None,
            prompts=config.prompts if config else None,
        )

    if config is None:
        raise ValueError("Either reward_fn or config.reward must be provided.")

    if config.mode == "callback":
        if config.reward_fn_import_path is None:
            raise ValueError("reward_fn_import_path required for callback mode.")

        from surogate.grpo.utils.utils import import_object

        fn = import_object(config.reward_fn_import_path)
        return CallbackRewardProvider(
            reward_fn=fn,
            dataset_path=config.dataset_path,
            prompts=config.prompts,
        )

    elif config.mode == "verifiers":
        if not config.env:
            raise ValueError("reward.env must be configured for verifiers mode.")
        return VerifiersRewardProvider(
            env_configs=config.env,
            tokenizer=tokenizer,
        )

    raise ValueError(f"Unknown reward mode: {config.mode}")
