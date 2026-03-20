"""CLI entry point for native GRPO training: `surogate grpo-native config.yaml`"""

import sys
import argparse

from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Path to native GRPO config YAML file")
    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.grpo.native_config import NativeGRPOConfig
    from surogate.grpo.native_trainer import native_grpo_train

    config = load_config(NativeGRPOConfig, args.config)
    native_grpo_train(config)
