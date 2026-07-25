"""Deterministic, hash-bound replay samples for on-policy training updates."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import msgspec

from surogate.core.config.grpo_orch_config import GRPOReplayConfig
from surogate.grpo.transport import TrainingBatch, TrainingSample


class ReplayBatchError(ValueError):
    """A frozen replay batch is missing or violates its training contract."""


def _validate_sample(sample: TrainingSample, index: int) -> None:
    total_tokens = len(sample.prompt_ids) + len(sample.completion_ids)
    loss_mask = [*sample.prompt_mask, *sample.completion_mask]
    if len(sample.prompt_mask) != len(sample.prompt_ids):
        raise ReplayBatchError(f"replay sample {index} has an invalid prompt mask")
    if len(sample.completion_mask) != len(sample.completion_ids):
        raise ReplayBatchError(f"replay sample {index} has an invalid completion mask")
    if len(sample.completion_logprobs) != len(sample.completion_ids):
        raise ReplayBatchError(f"replay sample {index} has invalid completion scores")
    if len(sample.completion_temperatures) != len(sample.completion_ids):
        raise ReplayBatchError(f"replay sample {index} has invalid temperatures")
    if sample.replay_mask is None or len(sample.replay_mask) != total_tokens:
        raise ReplayBatchError(f"replay sample {index} has an invalid replay mask")
    if not any(sample.replay_mask):
        raise ReplayBatchError(f"replay sample {index} selects no replay tokens")
    if any(sample.replay_mask[: len(sample.prompt_ids)]):
        raise ReplayBatchError(f"replay sample {index} selects prompt tokens")
    if any(replay and not trainable for replay, trainable in zip(sample.replay_mask, loss_mask)):
        raise ReplayBatchError(f"replay sample {index} selects a non-trainable token")
    if sample.replay_weights is not None:
        if len(sample.replay_weights) != total_tokens:
            raise ReplayBatchError(f"replay sample {index} has invalid replay weights")
        for replay, weight in zip(sample.replay_mask, sample.replay_weights):
            if not isinstance(weight, (int, float)) or not math.isfinite(weight):
                raise ReplayBatchError(f"replay sample {index} has non-finite replay weights")
            if replay and weight <= 0.0:
                raise ReplayBatchError(f"replay sample {index} has non-positive replay weights")
            if not replay and weight != 1.0:
                raise ReplayBatchError(f"replay sample {index} weights a non-replay token")
    if sample.advantage not in {None, 0} or sample.reward not in {None, 0}:
        raise ReplayBatchError(f"replay sample {index} must not carry outcome advantage or reward")
    if any(
        value is not None
        for value in (
            sample.hindsight_logprobs,
            sample.hindsight_mask,
            sample.opd_reference_logprobs,
        )
    ):
        raise ReplayBatchError(f"replay sample {index} cannot also carry hindsight/OPD credit")


@dataclass(frozen=True)
class ReplayBatchSource:
    """Validated replay source with deterministic per-step subsampling."""

    samples: tuple[TrainingSample, ...]
    samples_per_step: int
    seed: int

    @classmethod
    def load(cls, config: GRPOReplayConfig) -> ReplayBatchSource:
        path = Path(config.path).expanduser().resolve()
        if not path.is_file():
            raise ReplayBatchError(f"replay batch does not exist: {path}")
        payload = path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != config.sha256:
            raise ReplayBatchError(f"replay batch SHA-256 mismatch: expected {config.sha256}, got {actual_hash}")
        try:
            batch = msgspec.msgpack.decode(payload, type=TrainingBatch)
        except Exception as exc:
            raise ReplayBatchError(f"replay batch is not a TrainingBatch: {exc}") from exc
        if not batch.examples:
            raise ReplayBatchError("replay batch is empty")
        for index, sample in enumerate(batch.examples):
            _validate_sample(sample, index)
        samples_per_step = config.samples_per_step or len(batch.examples)
        if samples_per_step > len(batch.examples):
            raise ReplayBatchError("replay.samples_per_step exceeds the frozen replay sample count")
        return cls(
            samples=tuple(batch.examples),
            samples_per_step=samples_per_step,
            seed=config.seed,
        )

    def examples_for_step(self, step: int) -> list[TrainingSample]:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ReplayBatchError("training step must be a non-negative integer")
        ranked = sorted(
            range(len(self.samples)),
            key=lambda index: hashlib.sha256(f"{self.seed}:{step}:{index}".encode("ascii")).digest(),
        )
        return [copy.deepcopy(self.samples[index]) for index in ranked[: self.samples_per_step]]
