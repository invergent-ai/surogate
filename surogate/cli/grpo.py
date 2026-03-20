"""CLI entry point for unified GRPO: `surogate grpo --train t.yaml --infer i.yaml --orch o.yaml`"""

import sys
import argparse

from surogate.cli.config_overrides import parse_cli_overrides
from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, required=False, default=None, help="Optional path to GRPO training config YAML file")
    parser.add_argument("--infer", type=str, required=False, default=None, help="Optional path to GRPO inference config YAML file")
    parser.add_argument("--orch", type=str, required=False, default=None, help="Optional path to GRPO orchestrator config YAML file")
    return parser


if __name__ == "__main__":
    args, unknown = prepare_command_parser().parse_known_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.grpo.config import GRPOTrainConfig
    from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
    from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
    from surogate.grpo.unified import grpo_unified
    from surogate.core.config.grpo_orch_config import ColocateWeightBroadcastConfig
    from surogate.utils.dict import DictDefault

    overrides = parse_cli_overrides(unknown)
    top_keys = set(overrides.keys())
    allowed = {"train", "infer", "orch"}
    unknown_scopes = sorted(top_keys - allowed)
    if unknown_scopes:
        raise ValueError(
            "Unified GRPO CLI overrides must be scoped as "
            "--train.*, --infer.*, or --orch.*. "
            f"Unexpected top-level override keys: {unknown_scopes}"
        )

    train_overrides = overrides.get("train", {})
    infer_overrides = overrides.get("infer", {})
    orch_overrides = overrides.get("orch", {})

    train_config = load_config(GRPOTrainConfig, args.train, overrides=train_overrides)
    infer_config = load_config(GRPOInferenceConfig, args.infer, overrides=infer_overrides)
    infer_config.__post_init__()
    orch_config = load_config(GRPOOrchestratorConfig, args.orch, overrides=orch_overrides)
    
    train_config.weight_broadcast_type = "colocate"
    infer_config.weight_broadcast_type = "colocate"
    orch_config.weight_broadcast = ColocateWeightBroadcastConfig(
        DictDefault({"type": "colocate"})
    )
    
    grpo_unified(train_config, infer_config, orch_config)
