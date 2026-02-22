"""Data adapter: converts MicroBatch to numpy arrays for Surogate."""

import numpy as np
from pathlib import Path

from surogate.utils.logger import get_logger
from surogate.grpo.transport import setup_micro_batch_receiver

logger = get_logger()


def microbatch_to_numpy(micro_batch) -> dict[str, np.ndarray]:
    """Convert a MicroBatch (msgspec struct with lists) to numpy arrays.

    Returns dict with:
        input_ids: int32 [1, T]
        targets: int32 [1, T]  (input_ids shifted left by 1)
        position_ids: int32 [1, T]
        advantages: float32 [1, T]
        inference_logprobs: float32 [1, T]
        loss_mask: bool [1, T]
        temperatures: float32 [1, T]
        teacher_logprobs: float32 [1, T] or None
    """
    T = len(micro_batch.input_ids)

    input_ids = np.array(micro_batch.input_ids, dtype=np.int32).reshape(1, T)

    # Targets = input_ids shifted left by 1 (predict next token)
    targets = np.empty_like(input_ids)
    targets[0, :-1] = input_ids[0, 1:]
    targets[0, -1] = 0  # Last position doesn't matter (will be masked)

    position_ids = np.array(micro_batch.position_ids, dtype=np.int32).reshape(1, T)
    advantages = np.array(micro_batch.advantages, dtype=np.float32).reshape(1, T)
    inference_logprobs = np.array(micro_batch.inference_logprobs, dtype=np.float32).reshape(1, T)
    loss_mask = np.array(micro_batch.loss_mask, dtype=np.bool_).reshape(1, T)
    temperatures = np.array(micro_batch.temperatures, dtype=np.float32).reshape(1, T)

    teacher_logprobs = None
    if micro_batch.teacher_logprobs is not None:
        teacher_logprobs = np.array(micro_batch.teacher_logprobs, dtype=np.float32).reshape(1, T)

    return {
        "input_ids": input_ids,
        "targets": targets,
        "position_ids": position_ids,
        "advantages": advantages,
        "inference_logprobs": inference_logprobs,
        "loss_mask": loss_mask,
        "temperatures": temperatures,
        "teacher_logprobs": teacher_logprobs,
    }


class GRPODataLoader:
    """Wraps the transport layer to deliver numpy micro-batches."""

    def __init__(self, output_dir: str, dp_rank: int, start_step: int, transport_config):
        self.receiver = setup_micro_batch_receiver(Path(output_dir), dp_rank, start_step, transport_config)
        logger.info(f"GRPO data loader initialized (dp_rank={dp_rank}, start_step={start_step})")

    def wait_for_batch(self):
        """Block until the next batch is available."""
        self.receiver.wait()

    def get_batch(self) -> list[dict[str, np.ndarray]]:
        """Receive micro-batches and convert to numpy."""
        micro_batches = self.receiver.receive()
        return [microbatch_to_numpy(mb) for mb in micro_batches]
