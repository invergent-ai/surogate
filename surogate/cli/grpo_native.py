"""CLI entry point for native GRPO training: `surogate grpo-native config.yaml`"""

import sys
import argparse

from surogate.cli.config_overrides import parse_cli_overrides
from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default=None,
        help="Optional path to native GRPO config YAML file",
    )
    return parser


if __name__ == "__main__":
    args, unknown = prepare_command_parser().parse_known_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.grpo.native_config import NativeGRPOConfig
    from surogate.grpo.native_trainer import native_grpo_train

    overrides = parse_cli_overrides(unknown)
    config = load_config(NativeGRPOConfig, args.config, overrides=overrides)
    native_grpo_train(config)
