import argparse
import gc
import importlib.util
import os
import subprocess
import sys
from typing import Dict
import signal

from surogate.utils.logger import get_logger
from surogate.utils.system_info import print_system_diagnostics, get_system_info

logger = get_logger()

COMMAND_MAPPING: Dict[str, str] = {
    'sft': 'surogate.cli.sft',
    'pt': 'surogate.cli.pt',
    'tokenize': 'surogate.cli.tokenize_cmd',
    'serve': 'surogate.cli.serve',
    'eval': 'surogate.cli.eval',
    'ptq': 'surogate.cli.ptq',
}

def parse_args():
    logger.banner("Surogate LLMOps CLI")

    parser = argparse.ArgumentParser(description="Surogate LLMOps Framework")
    parser.set_defaults(func=lambda _args, p=parser: p.print_help())
    subparsers = parser.add_subparsers(dest='command', metavar='<command>')

    # sft command
    from surogate.cli.sft import prepare_command_parser as sft_prepare_command_parser
    sft_prepare_command_parser(subparsers.add_parser('sft', help="Supervised Fine-Tuning"))
    
    # pretrain command
    from surogate.cli.pt import prepare_command_parser as pt_prepare_command_parser
    pt_prepare_command_parser(subparsers.add_parser('pt', help="Pretraining"))

    # tokenize command
    from surogate.cli.tokenize_cmd import prepare_command_parser as tokenize_prepare_command_parser
    tokenize_prepare_command_parser(subparsers.add_parser('tokenize', help="Tokenize datasets for training"))

    args = parser.parse_args(sys.argv[1:])
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    commands_with_config = ['serve', 'pretrain', 'sft', 'pt','tokenize']
    if args.command in commands_with_config and not getattr(args, 'config', None):
        parser.print_help()
        sys.exit(1)

    return args

def cli_main():
    """Main CLI entry point for installed 'surogate' command."""
    args = parse_args()
    file_path = importlib.util.find_spec(COMMAND_MAPPING[args.command]).origin
    python_cmd = sys.executable
    command_args = sys.argv[2:]

    system_info = get_system_info()
    print_system_diagnostics(system_info)

    process = None
    try:
        cmd_args = [python_cmd, file_path, *command_args]
        process = subprocess.Popen(cmd_args, preexec_fn=os.setsid)
        return process.wait()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 0
    finally:
        logger.info("Cleaning up...")
        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't respond
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

        gc.collect()

if __name__ == '__main__':
    exit(cli_main())