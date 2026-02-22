"""CLI entry point for unified GRPO: `surogate grpo --train t.yaml --infer i.yaml --orch o.yaml`"""

import sys
import argparse

from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, required=True, help="Path to GRPO training config YAML file")
    parser.add_argument("--infer", type=str, required=True, help="Path to GRPO inference config YAML file")
    parser.add_argument("--orch", type=str, required=True, help="Path to GRPO orchestrator config YAML file")
    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.grpo.config import GRPOTrainConfig
    from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
    from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
    from surogate.grpo.unified import grpo_unified
    from surogate.core.config.grpo_orch_config import ColocateWeightBroadcastConfig

    train_config = load_config(GRPOTrainConfig, args.train)
    infer_config = load_config(GRPOInferenceConfig, args.infer)
    infer_config.__post_init__()
    orch_config = load_config(GRPOOrchestratorConfig, args.orch)
    
    train_config.weight_broadcast_type = "colocate"
    infer_config.weight_broadcast_type = "colocate"
    orch_config.weight_broadcast = ColocateWeightBroadcastConfig({ type: "colocate" })
    
    grpo_unified(train_config, infer_config, orch_config)
