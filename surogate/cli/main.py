import argparse
import os
import runpy
import sys
from typing import Dict

from surogate.utils.logger import get_logger

logger = get_logger()

COMMAND_MAPPING: Dict[str, str] = {
    'server': 'surogate.cli.server',
    'migrate': 'surogate.cli.migrate',
}

def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("surogate")
    except Exception:
        try:
            from surogate._version import __version__
            return __version__
        except Exception:
            return "unknown"


def parse_args():
    logger.banner(f"Surogate LLMOps CLI v{_get_version()}")

    parser = argparse.ArgumentParser(description="Surogate LLMOps Framework")
    parser.set_defaults(func=lambda _args, p=parser: p.print_help())
    subparsers = parser.add_subparsers(dest='command', metavar='<command>')

    # server command
    from surogate.cli.server import prepare_command_parser as serve_prepare_command_parser
    serve_prepare_command_parser(subparsers.add_parser('server', help="Start the Surogate HTTP server"))

    # migrate command
    from surogate.cli.migrate import prepare_command_parser as migrate_prepare_command_parser
    migrate_prepare_command_parser(subparsers.add_parser('migrate', help="Database schema migrations"))
    
    args = parser.parse_args(sys.argv[1:])
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands_with_config = ['sft', 'pt', 'grpo_train', 'grpo_infer', 'grpo_orch', 'tokenize']
    if args.command in commands_with_config and not getattr(args, 'config', None):
        parser.print_help()
        sys.exit(1)

    return args

def cli_main():
    """Main CLI entry point for installed 'surogate' command."""
    args = parse_args()

    # Run the command module in-process (avoids a second Python startup).
    # Rewrite sys.argv so the module's __main__ block sees only its own args.
    module_name = COMMAND_MAPPING[args.command]
    sys.argv = [module_name] + sys.argv[2:]
    runpy.run_module(module_name, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    cli_main()
