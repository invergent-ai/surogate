"""CLI entry point for GRPO RL Inference: `surogate grpo-infer config.yaml`"""

import sys
import argparse

from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Path to GRPO inference config YAML file")
    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.core.config.grpo_inference_config import GRPOInferenceConfig
    from surogate.grpo.inference.grpo_infer import grpo_infer

    config = load_config(GRPOInferenceConfig, args.config)
    config.__post_init__()
    
    grpo_infer(config)