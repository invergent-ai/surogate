"""CLI entry point for GRPO RL training: `surogate grpo-train config.yaml`"""

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
        help="Optional path to GRPO config YAML file",
    )
    return parser


if __name__ == "__main__":
    args, unknown = prepare_command_parser().parse_known_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.grpo.config import GRPOTrainConfig
    from surogate.grpo.trainer import grpo_train

    overrides = parse_cli_overrides(unknown)
    config = load_config(GRPOTrainConfig, args.config, overrides=overrides)
    grpo_train(config)
