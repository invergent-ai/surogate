"""CLI entry point for GRPO RL Orchestrator: `surogate grpo-orch config.yaml`"""

import sys
import argparse

from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Path to GRPO Orchestrator config YAML file")
    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])

    from surogate.core.config.loader import load_config
    from surogate.core.config.grpo_orch_config import GRPOOrchestratorConfig
    from surogate.grpo.orchestrator.grpo_orch import grpo_orchestrator

    config = load_config(GRPOOrchestratorConfig, args.config)    
    grpo_orchestrator(config)