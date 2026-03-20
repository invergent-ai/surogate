"""CLI entry point for native OpenAI-compatible inference serving."""

import argparse
import sys

from surogate.cli.config_overrides import parse_cli_overrides


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default=None,
        help="Optional path to serving config YAML file",
    )
    return parser


if __name__ == "__main__":
    args, unknown = prepare_command_parser().parse_known_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.core.config.serve_config import ServeConfig
    from surogate.serve.server import serve

    overrides = parse_cli_overrides(unknown)
    config = load_config(ServeConfig, args.config, overrides=overrides)
    serve(config)
