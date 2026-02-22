"""Packer: packs TrainingSamples into fixed-length MicroBatches.

The packer receives TrainingBatch from the orchestrator, packs samples into
fixed-length MicroBatches (with sample packing + padding), and distributes
them via the transport layer to each data-parallel trainer rank.

Only the master rank (rank 0) runs the packer. Other ranks just receive
the packed micro-batches from the transport layer.
"""

import shutil
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import torch

from surogate.grpo.batch import prepare_batch
from surogate.grpo.runs import get_multi_run_manager, setup_multi_run_manager
from surogate.grpo.transport import (
    MicroBatchSender,
    setup_micro_batch_sender,
    setup_training_batch_receiver,
)
from surogate.grpo.transport.types import MicroBatch, TrainingSample
from surogate.grpo.utils.logger import get_logger
from surogate.grpo.utils.pathing import get_rollout_dir

logger = get_logger()

TIMEOUT_SECONDS = 0.1


# ---------------------------------------------------------------------------
# Packer classes
# ---------------------------------------------------------------------------


class BasePacker(ABC):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer,
        config,
        start_step: int = 0,
    ):
        self.multi_run_manager = get_multi_run_manager()
        self.dp_world_size = dp_world_size
        self.seq_len = seq_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.tokenizer = tokenizer
        self.receiver = setup_training_batch_receiver(config)
        shutil.rmtree(get_rollout_dir(self.multi_run_manager.output_dir), ignore_errors=True)
        self.sender: MicroBatchSender = setup_micro_batch_sender(
            self.multi_run_manager.output_dir, dp_world_size, start_step, config
        )

    @abstractmethod
    def pack(self) -> None:
        """Pack samples for the next step."""
        pass


class SinglePacker(BasePacker):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer,
        config,
        start_step: int = 0,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, start_step)
        assert self.multi_run_manager.max_runs == 1, "SinglePacker only supports one run"

    def pack(self):
        batches = []
        while len(batches) == 0:
            self.multi_run_manager.discover_runs()
            batches = self.receiver.receive()
            time.sleep(0.2)

        assert len(batches) == 1, "SinglePacker only supports one batch per step"
        batch = batches[0]

        self.multi_run_manager.ready_to_update[0] = True
        self.multi_run_manager.progress[0].step += 1
        micro_batch_grid = prepare_batch(
            rollouts=batch.examples,
            seq_len=self.seq_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            num_train_workers=self.dp_world_size,
            idxs=[0] * len(batch.examples),
            num_loras=self.multi_run_manager.max_runs,
        )

        self.sender.send(micro_batch_grid)


class MultiPacker(BasePacker):
    def __init__(
        self,
        dp_world_size: int,
        seq_len: int,
        pad_to_multiple_of: int,
        tokenizer,
        config,
        start_step: int = 0,
    ):
        super().__init__(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, config, start_step)
        self.buffers: list[deque[tuple[TrainingSample, int]]] = [
            deque() for _ in range(self.multi_run_manager.max_runs)
        ]
        self._round_robin_position: int = 0
        self.multi_run_manager.register_forgotten_hook(self._on_run_data_deleted)

    def _on_run_data_deleted(self, idx: int, run_id: str) -> None:
        logger.debug(f"Packing is resetting run state for deleted run {idx}")
        self.receiver.reset_run(idx)
        self.buffers[idx].clear()

    def _validate_sample(self, sample: TrainingSample) -> tuple[bool, str | None]:
        sample_length = len(sample.prompt_ids) + len(sample.completion_ids)
        if len(sample.prompt_mask) != len(sample.prompt_ids):
            return (False, f"prompt mask length != prompt ids length ({len(sample.prompt_mask)} != {len(sample.prompt_ids)})")
        if len(sample.completion_mask) != len(sample.completion_ids):
            return (False, f"completion mask length != completion ids length ({len(sample.completion_mask)} != {len(sample.completion_ids)})")
        if len(sample.completion_logprobs) != len(sample.completion_ids):
            return (False, f"completion logprobs length != completion ids length ({len(sample.completion_logprobs)} != {len(sample.completion_ids)})")
        if len(sample.completion_temperatures) != len(sample.completion_ids):
            return (False, f"completion temperatures length != completion ids length ({len(sample.completion_temperatures)} != {len(sample.completion_ids)})")
        if sample_length == 0:
            return False, "sample with no tokens"
        if sample_length > self.seq_len:
            return (False, f"sample length {sample_length} exceeds max sequence length {self.seq_len}")
        if sample.teacher_logprobs is not None and len(sample.teacher_logprobs) != sample_length:
            return (False, f"teacher logprobs length != sample length ({len(sample.teacher_logprobs)} != {sample_length})")
        return True, None

    def _get_batch(self) -> None:
        self.multi_run_manager.discover_runs()
        batches = self.receiver.receive()

        for batch in batches:
            if batch.run_idx is None:
                logger.warning("Received batch with no run index")
                continue
            if len(batch.examples) == 0:
                self.multi_run_manager.evict_run(batch.run_idx, "Run wrote a batch with no samples")
                continue
            for sample in batch.examples:
                valid, reason = self._validate_sample(sample)
                if not valid:
                    self.multi_run_manager.evict_run(batch.run_idx, f"Invalid sample: {reason}")
                    break
                self.buffers[batch.run_idx].append((sample, batch.step))

        # Forget evicted runs
        self.multi_run_manager.discover_runs()

    def _count_tokens(self, threshold: int | None = None) -> int:
        tokens = 0
        for run_idx in self.multi_run_manager.used_idxs:
            buffer = self.buffers[run_idx]
            current_step = self.multi_run_manager.progress[run_idx].step
            for sample, step in buffer:
                if step > current_step:
                    break
                tokens += len(sample.prompt_ids) + len(sample.completion_ids)
                if threshold is not None and tokens >= threshold:
                    return tokens
        return tokens

    def _has_enough_tokens(self) -> bool:
        threshold = self.seq_len * self.dp_world_size
        return self._count_tokens(threshold) >= threshold

    def _select_samples_round_robin(self, token_budget: int) -> list[tuple[int, TrainingSample, int]]:
        selected: list[tuple[int, TrainingSample, int]] = []
        tokens_collected = 0

        while tokens_collected < token_budget:
            for _ in range(len(self.buffers)):
                if len(self.buffers[self._round_robin_position]) > 0:
                    _, step = self.buffers[self._round_robin_position][0]
                    if step <= self.multi_run_manager.progress[self._round_robin_position].step:
                        break
                self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            else:
                break

            run_idx = self._round_robin_position
            self._round_robin_position = (self._round_robin_position + 1) % len(self.buffers)
            current_step = self.multi_run_manager.progress[run_idx].step

            while len(self.buffers[run_idx]) > 0:
                sample, step = self.buffers[run_idx][0]
                if step > current_step:
                    break
                tokens_collected += len(sample.prompt_ids) + len(sample.completion_ids)
                if tokens_collected > token_budget:
                    if tokens_collected == (len(sample.prompt_ids) + len(sample.completion_ids)):
                        tokens_collected -= len(sample.prompt_ids) + len(sample.completion_ids)
                        self.buffers[run_idx].popleft()
                        continue
                    return selected
                selected.append((run_idx, sample, step))
                self.buffers[run_idx].popleft()

        return selected

    def _update_run_progress(self, run_idx: int, num_samples: int, num_tokens: int) -> None:
        if (
            len(self.buffers[run_idx]) == 0
            or self.buffers[run_idx][0][1] > self.multi_run_manager.progress[run_idx].step
        ):
            self.multi_run_manager.progress[run_idx].step += 1
            self.multi_run_manager.ready_to_update[run_idx] = True

        self.multi_run_manager.progress[run_idx].total_tokens += num_tokens
        self.multi_run_manager.progress[run_idx].total_samples += num_samples

    def pack(self):
        self._get_batch()
        start_time = time.time()

        while not self._has_enough_tokens():
            if time.time() - start_time > TIMEOUT_SECONDS and self._count_tokens() > 0:
                logger.warning("Timeout waiting for enough tokens to pack")
                break
            time.sleep(1)
            self._get_batch()

        token_budget = self.seq_len * self.dp_world_size
        selected_samples = self._select_samples_round_robin(token_budget)
        assert selected_samples, "No samples selected"

        # Group by run_idx — each micro-batch must contain samples from only ONE run
        samples_by_run: dict[int, list[TrainingSample]] = {}
        per_run_stats: dict[int, tuple[int, int]] = {}
        for run_idx, sample, step in selected_samples:
            if run_idx not in samples_by_run:
                samples_by_run[run_idx] = []
            samples_by_run[run_idx].append(sample)

            num_tokens = len(sample.prompt_ids) + len(sample.completion_ids)
            if run_idx in per_run_stats:
                cur_samples, cur_tokens = per_run_stats[run_idx]
                per_run_stats[run_idx] = (cur_samples + 1, cur_tokens + num_tokens)
            else:
                per_run_stats[run_idx] = (1, num_tokens)

        for run_idx, (num_samples, num_tokens) in per_run_stats.items():
            self._update_run_progress(run_idx, num_samples, num_tokens)

        all_micro_batches: list[list[MicroBatch]] = [[] for _ in range(self.dp_world_size)]
        for run_idx in sorted(samples_by_run.keys()):
            run_samples = samples_by_run[run_idx]
            run_micro_batch_grid = prepare_batch(
                rollouts=run_samples,
                seq_len=self.seq_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                num_train_workers=self.dp_world_size,
                idxs=[run_idx] * len(run_samples),
                num_loras=self.multi_run_manager.max_runs,
            )
            for worker_idx, worker_batches in enumerate(run_micro_batch_grid):
                all_micro_batches[worker_idx].extend(worker_batches)

        self.sender.send(all_micro_batches)


# ---------------------------------------------------------------------------
# Public API (unchanged interface for callers)
# ---------------------------------------------------------------------------


def init_multi_run_manager(output_dir: str, max_runs: int = 1):
    """Initialize the MultiRunManager singleton.

    Must be called before setup_grpo_packer(). Surogate uses max_runs=1
    since it handles a single training run (multi-GPU is internal).
    """
    setup_multi_run_manager(
        output_dir=Path(output_dir),
        max_runs=max_runs,
        device=torch.device("cuda", 0),
        lora_config=None,
    )
    logger.info(f"MultiRunManager initialized (output_dir={output_dir}, max_runs={max_runs})")


def setup_grpo_packer(
    dp_world_size: int,
    seq_len: int,
    pad_to_multiple_of: int,
    tokenizer,
    transport_config,
    start_step: int = 0,
) -> BasePacker:
    """Create a packer for GRPO training.

    Requires init_multi_run_manager() to have been called first.

    Returns:
        SinglePacker (for 1 run) or MultiPacker (for multi-run).
    """
    multi_run_manager = get_multi_run_manager()
    if multi_run_manager.max_runs == 1:
        packer = SinglePacker(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, start_step)
    else:
        packer = MultiPacker(dp_world_size, seq_len, pad_to_multiple_of, tokenizer, transport_config, start_step)

    logger.info(f"GRPO packer initialized (dp_world_size={dp_world_size}, seq_len={seq_len})")
    return packer
