import msgspec


# Orchestrator -> Packer
class TrainingSample(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A single training example."""

    prompt_ids: list[int]
    prompt_mask: list[bool]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    completion_temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    advantage: float | None = None
    reward: float | None = None
    # 0-based model-turn index per completion token; None for single-turn envs.
    # Env-observation tokens injected between turns carry the index of the turn
    # they precede, so a turn spans [its observation .. its completion].
    completion_turn_ids: list[int] | None = None
    # Total model turns in the source trajectory (before any truncation).
    num_turns: int | None = None
    # Rollout Routing Replay: the rollout engine's MoE expert ids per token, int16 little-endian, C order
    # [len(prompt_ids) + len(completion_ids), layers, top_k]; -1 rows = no replay. None when not collected.
    routed_experts: bytes | None = None
    routed_experts_dims: list[int] | None = None  # [layers, top_k]
    # The trajectory (trial) this sample was cut from, unique per trial within the step's batch: a trial whose turns
    # do not merge into one sequence becomes several samples with the same index. The producer divides each sample's
    # advantage by its trial's sample count, and loss.variant=gspo divides the step by the number of distinct trials,
    # so each trial counts once. None = a trial of its own.
    trial_idx: int | None = None
    # Per completion token: the policy version (the weight broadcast, one per trainer step) whose adapter sampled it;
    # -1 for tokens the model did not sample (completion_mask False). With max_policy_lag set, the trainer drops a
    # sample whose sampled tokens carry several versions, or one older than the lag allows. None = not collected.
    completion_versions: list[int] | None = None


class TrainingBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A batch of training examples with metadata for transport."""

    examples: list[TrainingSample]
    step: int
    run_idx: int | None = None


# Packer -> Trainer
class MicroBatch(msgspec.Struct, array_like=True, gc=False, omit_defaults=True):
    """A micro batch of data for training."""

    input_ids: list[int]
    loss_mask: list[bool]
    advantages: list[float]
    inference_logprobs: list[float]
    position_ids: list[int]
    temperatures: list[float]  # Per-token temperatures used during generation
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    # Per-token 0-based turn index; -1 for prompt/padding tokens. None when the
    # source env is single-turn (turn structure carries no information there).
    turn_ids: list[int] | None = None
    # Rollout Routing Replay rows aligned with input_ids: int16 little-endian, C order
    # [len(input_ids), layers, top_k], -1 = no replay (padding, cache-served prompt tokens).
    routed_experts: bytes | None = None
    routed_experts_dims: list[int] | None = None  # [layers, top_k]
    # Per-token TrainingSample.trial_idx; -1 for padding and for samples without one. None when no sample has one.
    trial_idx: list[int] | None = None
    # Per-token TrainingSample.completion_versions; -1 for prompt tokens, padding and samples without versions. None
    # when no sample has versions.
    policy_version: list[int] | None = None
