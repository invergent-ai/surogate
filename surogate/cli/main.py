# surogate/cli/main.py

import argparse
import gc
import importlib.util
import os
import subprocess
import sys
from typing import Optional, List, Dict
import signal
import torch

from surogate.utils.logger import get_logger
from surogate.utils.system_info import print_system_diagnostics, get_system_info

logger = get_logger()

def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True

def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def parse_args():
    logger.banner("Surogate LLMOps Toolkit CLI")

    parser = argparse.ArgumentParser(description="Surogate LLMOps Framework")
    parser.set_defaults(func=lambda _args, p=parser: p.print_help())
    subparsers = parser.add_subparsers(dest='command', metavar='<command>')

    # Serve command
    from .serve import prepare_command_parser as serve_prepare_command_parser
    serve_prepare_command_parser(subparsers.add_parser('serve', help="Serve a model via API"))

    # Eval command with multiple operation modes
    from .eval import prepare_command_parser as eval_prepare_command_parser
    eval_parser = eval_prepare_command_parser(subparsers.add_parser('eval', help="Evaluate models"))

    # PTQ command
    from .ptq import prepare_command_parser as ptq_prepare_command_parser
    ptq_prepare_command_parser(subparsers.add_parser('ptq', help="Post-training quantization"))

    # SFT command
    from .sft import prepare_command_parser as sft_prepare_command_parser
    sft_prepare_command_parser(subparsers.add_parser('sft', help="Supervised fine-tuning"))

    args = parser.parse_args(sys.argv[1:])

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands_with_config = ['serve', 'pretrain', 'ptq', 'sft']
    if args.command in commands_with_config and not getattr(args, 'config', None):
        parser.print_help()
        sys.exit(1)

    # Validate eval command arguments
    if args.command == 'eval':
        # Check which mode is being used
        if args.list or args.view or args.compare:
            # Viewing results mode - config not needed
            pass
        else:
            # Running evaluation mode - config is required
            if not args.config:
                logger.error("--config is required when running evaluation")
                eval_parser.print_help()
                sys.exit(1)

    return args


COMMAND_MAPPING: Dict[str, str] = {
    'sft': 'surogate.cli.sft',
    'serve': 'surogate.cli.serve',
    'eval': 'surogate.cli.eval',
    'ptq': 'surogate.cli.ptq',
}

def cli_main():
    """Main CLI entry point for installed command."""
    args = parse_args()
    file_path = importlib.util.find_spec(COMMAND_MAPPING[args.command]).origin
    torchrun_args = get_torchrun_args()
    python_cmd = sys.executable
    command_args = sys.argv[2:]

    system_info = get_system_info()
    print_system_diagnostics(system_info)

    torch.cuda.empty_cache()

    if torchrun_args is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // torch.cuda.device_count())
            logger.info(f"Multiple GPUs detected, running on {torch.cuda.device_count()} GPUs")
            cmd_args = [python_cmd, '-m', 'torch.distributed.run', f"--nproc-per-node={torch.cuda.device_count()}", "--standalone", "--nnodes=1", "--node-rank=0", file_path, *command_args]
        else:
            cmd_args = [python_cmd, file_path, *command_args]
    else:
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() // torch.cuda.device_count())
        cmd_args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *command_args]

    process = None
    try:
        process = subprocess.Popen(cmd_args, preexec_fn=os.setsid)
        return process.wait()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        return 0
    except Exception as e:
        logger.error("An error occurred:")
        logger.error(f"Error Type: {{type(e).__name__}}")
        logger.error(f"Error Message: {{e}}")
        import traceback
        traceback.print_exc()
        return 1
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

if __name__ == '__main__':
    exit(cli_main())