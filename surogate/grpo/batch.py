"""Batch preparation: converts TrainingSamples into packed MicroBatches.

Handles sample packing (First Fit Decreasing), padding, and distribution
across data-parallel workers.
"""

import copy

from surogate.grpo.routing_replay import no_route_bytes
from surogate.grpo.transport.types import MicroBatch, TrainingSample
from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_sample(training_example: TrainingSample, seq_len: int) -> MicroBatch:
    """Prepare a single training example for sequence packing.

    Concatenates prompt and completion tokens, builds position IDs,
    and truncates to seq_len if needed.
    """
    input_ids = training_example.prompt_ids + training_example.completion_ids
    loss_mask = training_example.prompt_mask + training_example.completion_mask
    inference_logprobs = [0.0] * len(training_example.prompt_ids) + training_example.completion_logprobs
    advantages = [training_example.advantage] * len(input_ids)
    position_ids = list(range(len(input_ids)))

    # Per-token temperatures: prompt tokens use first completion temp (masked out anyway)
    prompt_temp = training_example.completion_temperatures[0] if training_example.completion_temperatures else 1.0
    temperatures = [prompt_temp] * len(training_example.prompt_ids) + training_example.completion_temperatures

    teacher_logprobs = training_example.teacher_logprobs

    # Turn ids: prompt tokens are not part of any model turn (-1).
    turn_ids = None
    if training_example.completion_turn_ids is not None:
        turn_ids = [-1] * len(training_example.prompt_ids) + training_example.completion_turn_ids

    # Routing replay rows ride along token for token ([len, layers, top_k] int16 bytes).
    routed, dims = training_example.routed_experts, training_example.routed_experts_dims
    if routed is not None:
        row_bytes = 2 * int(dims[0]) * int(dims[1]) if dims else 0
        if not row_bytes or len(routed) != row_bytes * len(input_ids):
            logger.warning(
                f"routed_experts length {len(routed)} does not match {len(input_ids)} tokens x dims {dims}; "
                "dropping replay rows for this sample")
            routed, dims = None, None

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        loss_mask = loss_mask[:seq_len]
        inference_logprobs = inference_logprobs[:seq_len]
        position_ids = position_ids[:seq_len]
        advantages = advantages[:seq_len]
        temperatures = temperatures[:seq_len]
        if teacher_logprobs is not None:
            teacher_logprobs = teacher_logprobs[:seq_len]
        if turn_ids is not None:
            turn_ids = turn_ids[:seq_len]
        if routed is not None:
            routed = routed[: seq_len * 2 * int(dims[0]) * int(dims[1])]

    assert (
        len(input_ids)
        == len(advantages)
        == len(loss_mask)
        == len(position_ids)
        == len(inference_logprobs)
        == len(temperatures)
    ), (
        f"input_ids: {len(input_ids)}, advantages: {len(advantages)}, loss_mask: {len(loss_mask)}, "
        f"position_ids: {len(position_ids)}, inference_logprobs: {len(inference_logprobs)}, temperatures: {len(temperatures)}"
    )
    if teacher_logprobs is not None:
        assert len(teacher_logprobs) == len(input_ids), f"teacher_logprobs: {len(teacher_logprobs)}"
    if turn_ids is not None:
        assert len(turn_ids) == len(input_ids), f"turn_ids: {len(turn_ids)}"

    return MicroBatch(
        input_ids=input_ids,
        advantages=advantages,
        loss_mask=loss_mask,
        position_ids=position_ids,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=teacher_logprobs,
        temperatures=temperatures,
        turn_ids=turn_ids,
        routed_experts=routed,
        routed_experts_dims=dims,
    )


def _extend_routing(bin_content: MicroBatch, bin_len_before: int, sample: MicroBatch) -> None:
    """Append a sample's routing rows to a bin; the side without rows gets -1 rows so they stay aligned.

    A bin whose samples disagree on [layers, top_k] runs without replay; it is marked by empty dims so later
    samples do not re-enable it with misaligned rows.
    """
    if bin_content.routed_experts_dims == []:
        return
    if bin_content.routed_experts is None and sample.routed_experts is None:
        return
    dims = bin_content.routed_experts_dims or sample.routed_experts_dims
    if (bin_content.routed_experts_dims is not None and sample.routed_experts_dims is not None
            and list(bin_content.routed_experts_dims) != list(sample.routed_experts_dims)):
        logger.warning(f"routing replay dims differ inside a micro-batch ({bin_content.routed_experts_dims} vs "
                       f"{sample.routed_experts_dims}); this micro-batch runs without replay")
        bin_content.routed_experts, bin_content.routed_experts_dims = None, []
        return
    head = bin_content.routed_experts if bin_content.routed_experts is not None else no_route_bytes(bin_len_before, dims)
    tail = sample.routed_experts if sample.routed_experts is not None else no_route_bytes(len(sample.input_ids), dims)
    bin_content.routed_experts = head + tail
    bin_content.routed_experts_dims = list(dims)


def packed_samples_into_micro_bs(
    samples: list[tuple[int, MicroBatch]],
    max_seq_len: int,
    num_loras: int,
    single_sample_bins: bool = False,
) -> list[MicroBatch]:
    """Pack samples into micro-batches using First Fit Decreasing.

    Minimizes padding while never truncating. Multimodal samples are
    NOT packed together (variable-sized vision data).

    `single_sample_bins` disables packing entirely — one sample per
    micro-batch. Required for chunked-sequence GRPO: packed-document
    isolation does not exist in the chunked TRAINING attention path
    (kvprefix doc metadata is eval-only), so multi-sample rows would
    attend across sample boundaries (measured 2026-07-27: B-grad cosine
    0.5 vs unchunked). Tail padding chunks are skipped natively, so the
    cost is bounded by pad_to_multiple_of, not seq_len.
    """
    samples.sort(key=lambda x: (x[0], -len(x[1].input_ids)))

    micro_batches: list[MicroBatch] = []

    for idx, sample in samples:
        bins = [] if single_sample_bins else micro_batches
        for bin_content in bins:
            if len(bin_content.input_ids) + len(sample.input_ids) <= max_seq_len:
                bin_len_before = len(bin_content.input_ids)
                bin_content.input_ids.extend(sample.input_ids)
                bin_content.loss_mask.extend(sample.loss_mask)
                bin_content.advantages.extend(sample.advantages)
                bin_content.inference_logprobs.extend(sample.inference_logprobs)
                bin_content.temperatures.extend(sample.temperatures)
                if sample.teacher_logprobs is not None:
                    if bin_content.teacher_logprobs is None:
                        bin_content.teacher_logprobs = []
                    bin_content.teacher_logprobs.extend(sample.teacher_logprobs)
                if bin_content.turn_ids is not None or sample.turn_ids is not None:
                    # Packing mixes single- and multi-turn samples; whichever side
                    # lacks turn ids gets its span backfilled with -1 so turn_ids
                    # stays index-aligned with input_ids.
                    if bin_content.turn_ids is None:
                        bin_content.turn_ids = [-1] * bin_len_before
                    bin_content.turn_ids.extend(sample.turn_ids or [-1] * len(sample.input_ids))
                _extend_routing(bin_content, bin_len_before, sample)
                bin_content.position_ids.extend(sample.position_ids)
                bin_content.lora_num_tokens[idx] += len(sample.input_ids)
                break
        else:
            sample.lora_num_tokens = [0] * num_loras
            sample.lora_num_tokens[idx] = len(sample.input_ids)
            micro_batches.append(sample)

    return micro_batches


def pad_micro_batch(micro_batch: MicroBatch, pad_to_multiple_of: int) -> MicroBatch:
    """Pad a micro-batch so its length is a multiple of pad_to_multiple_of."""
    padding_size = (pad_to_multiple_of - (len(micro_batch.input_ids) % pad_to_multiple_of)) % pad_to_multiple_of

    if not (pad_to_multiple_of > 1 and padding_size > 0):
        return micro_batch

    micro_batch.input_ids.extend([1] * padding_size)
    micro_batch.advantages.extend([0.0] * padding_size)
    micro_batch.loss_mask.extend([False] * padding_size)
    micro_batch.position_ids.extend(list(range(padding_size)))
    micro_batch.inference_logprobs.extend([0.0] * padding_size)
    micro_batch.temperatures.extend([1.0] * padding_size)
    if micro_batch.teacher_logprobs is not None:
        micro_batch.teacher_logprobs.extend([0.0] * padding_size)
    if micro_batch.turn_ids is not None:
        micro_batch.turn_ids.extend([-1] * padding_size)
    if micro_batch.routed_experts is not None and micro_batch.routed_experts_dims:
        micro_batch.routed_experts += no_route_bytes(padding_size, micro_batch.routed_experts_dims)
    # Send padding to the last lora so tokens have ascending lora idx
    micro_batch.lora_num_tokens[-1] += padding_size

    return micro_batch


def prepare_batch(
    rollouts: list[TrainingSample],
    seq_len: int,
    num_train_workers: int,
    idxs: list[int],
    num_loras: int,
    pad_to_multiple_of: int = 1,
    single_sample_bins: bool = False,
) -> list[list[MicroBatch]]:
    """Prepare a batch of samples for each data-parallel worker.

    Each worker gets a list of micro-batches (shape [1, seq_len] each).
    The number of samples per micro-batch is variable (sample packing).
    """
    all_samples = [(idx, prepare_sample(rollout, seq_len)) for idx, rollout in zip(idxs, rollouts)]

    micro_batches = packed_samples_into_micro_bs(
        all_samples, seq_len, num_loras, single_sample_bins=single_sample_bins)
    micro_batches = [pad_micro_batch(micro_batch, pad_to_multiple_of) for micro_batch in micro_batches]

    num_padding_batch = -len(micro_batches) % num_train_workers

    # Ensure each data rank has the same number of micro-batches (prevents hangs with FSDP).
    # Padding batches use zero advantages and False loss_mask so they don't contribute to loss.
    if num_train_workers > 1 and num_padding_batch > 0:
        padded_batch = copy.deepcopy(micro_batches[0])
        padded_batch.advantages = [0.0] * len(padded_batch.input_ids)
        padded_batch.loss_mask = [False] * len(padded_batch.input_ids)
        micro_batches.extend([padded_batch for _ in range(num_padding_batch)])

    assert len(micro_batches) % num_train_workers == 0, (
        "Number of micro batches is not divisible by number of data ranks"
    )

    per_gpu_micro_batches = len(micro_batches) // num_train_workers
    batches_per_gpu = []
    for _ in range(num_train_workers):
        batches = []
        for _ in range(per_gpu_micro_batches):
            batches.append(micro_batches.pop(0))
        batches_per_gpu.append(batches)

    return batches_per_gpu
