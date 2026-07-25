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
    hindsight_logprobs: list[float] | None = None
    hindsight_mask: list[bool] | None = None
    replay_mask: list[bool] | None = None
    opd_reference_logprobs: list[float] | None = None
    replay_weights: list[float] | None = None
    advantage_mask: list[bool] | None = None


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
    opd_reference_logprobs: list[float]
    hindsight_logprobs: list[float]
    hindsight_mask: list[bool]
    replay_mask: list[bool]
    teacher_logprobs: list[float] | None = None
    lora_num_tokens: list[int] | None = None
    replay_weights: list[float] | None = None
