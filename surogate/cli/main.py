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
    'grpo': 'surogate.cli.grpo',
    'tokenize': 'surogate.cli.tokenize_cmd',
    'serve': 'surogate.cli.serve',
    'eval': 'surogate.cli.eval',
    'ptq': 'surogate.cli.ptq',
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

    # sft command
    from surogate.cli.sft import prepare_command_parser as sft_prepare_command_parser
    sft_prepare_command_parser(subparsers.add_parser('sft', help="Supervised Fine-Tuning"))
    
    # pretrain command
    from surogate.cli.pt import prepare_command_parser as pt_prepare_command_parser
    pt_prepare_command_parser(subparsers.add_parser('pt', help="Pretraining"))

    # grpo command
    from surogate.cli.grpo import prepare_command_parser as grpo_prepare_command_parser
    grpo_prepare_command_parser(subparsers.add_parser('grpo', help="GRPO Reinforcement Learning from Rule-Based Rewards"))

    # tokenize command
    from surogate.cli.tokenize_cmd import prepare_command_parser as tokenize_prepare_command_parser
    tokenize_prepare_command_parser(subparsers.add_parser('tokenize', help="Tokenize datasets for training"))

    args = parser.parse_args(sys.argv[1:])
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    commands_with_config = ['serve', 'pretrain', 'sft', 'pt', 'grpo', 'tokenize']
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
        exit_code = process.wait()
        # If process died from a signal (negative exit code or 128+signal),
        # give the crash handler time to finish printing
        if exit_code < 0 or exit_code >= 128:
            import time
            time.sleep(0.1)  # Allow crash handler output to complete
        return exit_code
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 0
    finally:
        if process and process.poll() is not None and (process.returncode < 0 or process.returncode >= 128):
            # Process crashed - don't print "Cleaning up" to avoid interleaving with crash output
            pass
        else:
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