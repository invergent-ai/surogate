import verifiers as vf

from surogate.grpo.routing_replay import SampleRouting
from surogate.grpo.transport import TrainingSample
from surogate.grpo.utils.logger import get_logger

# We use list() instead of deepcopy() for flat lists (token IDs, logprobs) - safe because
# primitives are immutable.


def interleave_rollout(output: vf.RolloutOutput, routing_replay: bool = False) -> list[TrainingSample] | None:
    """
    Convert vf.RolloutOutput to trainable rollouts by interleaving trajectory steps
    where the extension property holds.

    When consecutive steps share token prefixes (extension property), they are
    merged into a single sample. When extension breaks (e.g., due to context
    compaction or a change in control-flow), a new sample is started.

    Supports multi-prefix matching to handle interleaved agents. For example,
    [agent1-step1, agent1-step2, agent2-step1, agent1-step3] produces two samples:
    agent1 steps merged together, agent2 step separate.

    Returns a list of samples - could be 1 (extension always held) or up to T
    (extension never held).

    Args:
        output: vf.RolloutOutput containing trajectory data
        routing_replay: also stitch each step's routed-experts payload (``tokens["routed_experts"]``) into
            per-sample routing rows (``TrainingSample.routed_experts``), aligned with prompt_ids + completion_ids
    """
    logger = get_logger()

    trajectory = output["trajectory"]
    if len(trajectory) == 0:
        error = output.get("error")
        stop = output.get("stop_condition")
        logger.warning(
            f"No trajectory steps for example {output['example_id']} (error={error}, stop={stop}). Skipping rollout."
        )
        return None

    has_error = output["error"] is not None
    # this field should be guaranteed because we set temperature in get_sampling_args
    temperature = output["sampling_args"]["temperature"]

    def make_sample(step: vf.TrajectoryStep, step_idx: int) -> TrainingSample:
        """Create a new TrainingSample from a trajectory step."""
        tokens = step["tokens"]
        assert tokens is not None
        if has_error:
            completion_mask = [False] * len(tokens["completion_mask"])
        else:
            completion_mask = [bool(i) for i in tokens["completion_mask"]]
        completion_ids = list(tokens["completion_ids"])

        return TrainingSample(
            prompt_ids=list(tokens["prompt_ids"]),
            prompt_mask=[bool(i) for i in tokens["prompt_mask"]],
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            completion_logprobs=list(tokens["completion_logprobs"]),
            completion_temperatures=[temperature] * len(completion_ids),
            teacher_logprobs=None,
            advantage=None,
            completion_turn_ids=[step_idx] * len(completion_ids),
        )

    def extend_sample(sample: TrainingSample, step: vf.TrajectoryStep, prefix_len: int, step_idx: int) -> None:
        """Extend an existing sample with a new trajectory step (extension property holds)."""
        tokens = step["tokens"]
        assert tokens is not None

        # Extend with new prompt tokens (mask=False, no gradient). These are the
        # env observation for this turn, so they carry this turn's index.
        new_prompt_ids = tokens["prompt_ids"][prefix_len:]
        sample.completion_ids.extend(new_prompt_ids)
        sample.completion_mask.extend([False] * len(new_prompt_ids))
        sample.completion_logprobs.extend([0.0] * len(new_prompt_ids))
        sample.completion_temperatures.extend([temperature] * len(new_prompt_ids))
        assert sample.completion_turn_ids is not None
        sample.completion_turn_ids.extend([step_idx] * len(new_prompt_ids))

        # Extend with new completion tokens
        completion_ids = tokens["completion_ids"]
        sample.completion_ids.extend(completion_ids)
        sample.completion_turn_ids.extend([step_idx] * len(completion_ids))
        if has_error:
            sample.completion_mask.extend([False] * len(tokens["completion_mask"]))
        else:
            sample.completion_mask.extend(bool(i) for i in tokens["completion_mask"])
        sample.completion_logprobs.extend(tokens["completion_logprobs"])
        sample.completion_temperatures.extend([temperature] * len(completion_ids))

    def new_routing(step: vf.TrajectoryStep) -> SampleRouting | None:
        if not routing_replay:
            return None
        routing = SampleRouting()
        add_routing(routing, step)
        return routing

    def add_routing(routing: SampleRouting | None, step: vf.TrajectoryStep) -> None:
        # The step's prompt+completion positions are the sample's positions (extension property), so the
        # payload rows land on the sample directly; see SampleRouting for which request owns a position.
        if routing is not None:
            tokens = step["tokens"]
            routing.add_step(len(tokens["prompt_ids"]), len(tokens["completion_ids"]), tokens.get("routed_experts"))

    # Track multiple active (prefix, sample) pairs to handle interleaved agents
    # Each entry is [prefix_tokens, sample, routing] where prefix_tokens is the accumulated token sequence
    active_samples: list[list] = []

    first_tokens = trajectory[0]["tokens"]
    first_prefix = first_tokens["prompt_ids"] + first_tokens["completion_ids"]
    active_samples.append([first_prefix, make_sample(trajectory[0], step_idx=0), new_routing(trajectory[0])])

    for step_idx, step in enumerate(trajectory[1:], start=1):
        tokens = step["tokens"]
        step_prompt_ids = tokens["prompt_ids"]

        # Check if this step extends ANY active prefix
        matched_idx = None
        for idx, (prefix_tokens, _, _) in enumerate(active_samples):
            if step_prompt_ids[: len(prefix_tokens)] == prefix_tokens:
                matched_idx = idx
                break

        if matched_idx is not None:
            # Extension holds - merge into matched sample
            prefix_tokens, sample, routing = active_samples[matched_idx]
            extend_sample(sample, step, len(prefix_tokens), step_idx=step_idx)
            add_routing(routing, step)
            # Update prefix for this sample
            active_samples[matched_idx][0] = tokens["prompt_ids"] + tokens["completion_ids"]
        else:
            # No prefix matches - start a new sample
            logger.debug(
                f"Extension property broke at step {step_idx + 1} for example {output['example_id']}. "
                f"Starting new sample (active_prefixes={len(active_samples)}, step_prompt_len={len(step_prompt_ids)})."
            )
            new_prefix = tokens["prompt_ids"] + tokens["completion_ids"]
            active_samples.append([new_prefix, make_sample(step, step_idx=step_idx), new_routing(step)])

    for _, sample, routing in active_samples:
        packed = routing.result() if routing is not None else None
        if packed is not None:
            sample.routed_experts, sample.routed_experts_dims = packed

    samples = [sample for _, sample, _ in active_samples]

    # Turn ids stay the trajectory step index and are deliberately NOT re-based
    # per sample. A trajectory frequently splits into several samples — chat
    # templates that re-render earlier turns (e.g. Qwen strips <think> blocks
    # from all but the latest assistant message) break the extension property,
    # so each turn can land in its own sample. Re-basing would relabel every
    # such sample as turn 0 and erase the depth signal the turn diagnostics
    # exist to measure.
    for sample in samples:
        assert sample.completion_turn_ids is not None
        sample.num_turns = len(trajectory)

    return samples
