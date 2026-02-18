# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# GRPO prompt dataset loader.
#
# GRPO training uses prompt-only data (no teacher completions).  Each sample
# carries a prompt and an optional ground-truth answer used by rule-based
# reward functions.
#
# JSONL format (one JSON object per line):
#   {"prompt": "Solve: 2+2=?", "ground_truth": "4"}
#   {"prompt": "What is the capital of France?", "ground_truth": "Paris"}
#
# The "ground_truth" field is optional; omit it if your reward functions
# don't need it (e.g., FormatReward).

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional


class GRPOSample:
    """A single GRPO training prompt."""

    __slots__ = ('prompt', 'ground_truth')

    def __init__(self, prompt: str, ground_truth: Optional[str] = None):
        self.prompt = prompt
        self.ground_truth = ground_truth

    def __repr__(self) -> str:
        gt = repr(self.ground_truth[:40]) if self.ground_truth else 'None'
        return f"GRPOSample(prompt={repr(self.prompt[:60])!s}, ground_truth={gt})"


class GRPODataset:
    """Iterable dataset of prompts for GRPO training.

    Loads all samples into memory from a JSONL file and provides a
    :py:meth:`sample_batch` method that randomly selects a mini-batch.

    Args:
        path:   Path to the JSONL file.
        seed:   Random seed for reproducible shuffling.
    """

    def __init__(self, path: str, seed: int = 42):
        self.path = path
        self.seed = seed
        self.samples: List[GRPOSample] = _load_jsonl(path)
        self._rng = random.Random(seed)

        if not self.samples:
            raise ValueError(f"GRPO dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def sample_batch(self, batch_size: int) -> List[GRPOSample]:
        """Return a random mini-batch of prompts (with replacement if needed).

        Args:
            batch_size: Number of prompts to return.

        Returns:
            List of :class:`GRPOSample` objects.
        """
        if batch_size >= len(self.samples):
            # Shuffle all and repeat as needed
            pool = list(self.samples)
            self._rng.shuffle(pool)
            result = []
            while len(result) < batch_size:
                result.extend(pool)
            return result[:batch_size]
        return self._rng.sample(self.samples, batch_size)

    def iter_epochs(self, batch_size: int):
        """Yield batches over one full pass through the dataset (shuffled).

        Args:
            batch_size: Number of prompts per batch.

        Yields:
            List[GRPOSample] — one batch at a time.
        """
        pool = list(self.samples)
        self._rng.shuffle(pool)
        for i in range(0, len(pool), batch_size):
            yield pool[i:i + batch_size]


def load_grpo_dataset(path: str, seed: int = 42) -> GRPODataset:
    """Load a GRPO dataset from a JSONL file.

    Args:
        path: Path to the JSONL file.
        seed: Random seed.

    Returns:
        :class:`GRPODataset` instance.
    """
    return GRPODataset(path, seed=seed)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[GRPOSample]:
    samples = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GRPO dataset file not found: {path}")

    with open(path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error in {path}:{lineno}: {exc}") from exc

            prompt = obj.get('prompt')
            if not prompt:
                raise ValueError(
                    f"Missing 'prompt' field in {path}:{lineno}. "
                    "Each line must have at least {\"prompt\": \"...\"}"
                )

            ground_truth = obj.get('ground_truth', None)
            samples.append(GRPOSample(prompt=str(prompt), ground_truth=ground_truth))

    return samples
