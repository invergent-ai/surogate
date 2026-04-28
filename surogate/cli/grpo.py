"""CLI entry point for split-GPU GRPO: `surogate grpo --train t.yaml --infer i.yaml --orch o.yaml \\
       --vllm-gpus 0,1,2,3 --trainer-gpus 4,5,6,7`

vLLM and the trainer occupy disjoint sets of GPUs. We set ``CUDA_VISIBLE_DEVICES`` for
the trainer side BEFORE any torch import (in this module's ``__main__`` block) and
launch vLLM as a subprocess with its own ``CUDA_VISIBLE_DEVICES``.

For shared-GPU mode, see `surogate grpo-colocate`.
"""

import argparse
import sys

from surogate.utils.logger import get_logger

logger = get_logger()


def _gpu_list(value: str) -> list[int]:
    """Parse '0,1,2,3' into [0, 1, 2, 3]. Rejects empty or duplicate ids."""
    if not value:
        raise argparse.ArgumentTypeError("expected a comma-separated list of GPU ids")
    try:
        ids = [int(x) for x in value.split(",") if x.strip() != ""]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid GPU id in '{value}': {e}") from e
    if not ids:
        raise argparse.ArgumentTypeError("expected at least one GPU id")
    if any(g < 0 for g in ids):
        raise argparse.ArgumentTypeError(f"GPU ids must be non-negative: {ids}")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"duplicate GPU id in '{value}'")
    return ids


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, required=True, help="Path to GRPO training config YAML file")
    parser.add_argument("--infer", type=str, required=True, help="Path to GRPO inference config YAML file")
    parser.add_argument("--orch", type=str, required=True, help="Path to GRPO orchestrator config YAML file")
    parser.add_argument(
        "--vllm-gpus",
        type=_gpu_list,
        required=True,
        help="Comma-separated GPU ids for vLLM (e.g. '0,1,2,3'). Count must equal infer.dp * infer.tp.",
    )
    parser.add_argument(
        "--trainer-gpus",
        type=_gpu_list,
        required=True,
        help="Comma-separated GPU ids for the trainer (e.g. '4,5,6,7'). Count must equal train.gpus.",
    )
    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])

    overlap = sorted(set(args.vllm_gpus) & set(args.trainer_gpus))
    if overlap:
        logger.error(f"--vllm-gpus and --trainer-gpus overlap on {overlap}")
        sys.exit(1)

    # Mask the parent process to the trainer GPUs BEFORE any torch import below.
    # vLLM runs in a spawned subprocess that overrides CUDA_VISIBLE_DEVICES on its
    # own; values are interpreted as driver-level GPU indices in both processes,
    # so the parent's mask does not propagate to the child.
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.trainer_gpus)

    from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
    from surogate.core.config.grpo_orch_config import FileSystemWeightBroadcastConfig, GRPOOrchestratorConfig
    from surogate.core.config.loader import load_config
    from surogate.grpo.config import GRPOTrainConfig
    from surogate.grpo.split import grpo_split

    train_config = load_config(GRPOTrainConfig, args.train)
    infer_config = load_config(GRPOInferenceConfig, args.infer)
    orch_config = load_config(GRPOOrchestratorConfig, args.orch)

    # The CLI is the source of truth for the trainer GPU count; the YAML field
    # becomes optional in split mode.
    train_config.gpus = len(args.trainer_gpus)
    if train_config.model_info.is_moe_model:
        train_config.ep_size = train_config.gpus
        train_config._validate_ep_config()

    train_config.weight_broadcast_type = "filesystem"
    infer_config.weight_broadcast_type = "filesystem"
    orch_config.weight_broadcast = FileSystemWeightBroadcastConfig({"type": "filesystem"})

    grpo_split(
        train_config,
        infer_config,
        orch_config,
        vllm_gpu_ids=args.vllm_gpus,
        trainer_gpu_ids=args.trainer_gpus,
    )
