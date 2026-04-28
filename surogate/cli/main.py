import argparse
import runpy
import sys

from surogate.utils.logger import get_logger
from surogate.utils.system_info import get_system_info, print_system_diagnostics

logger = get_logger()

COMMAND_MAPPING: dict[str, str] = {
    "sft": "surogate.cli.sft",
    "pt": "surogate.cli.pt",
    "grpo": "surogate.cli.grpo",
    "grpo-colocate": "surogate.cli.grpo_colocate",
    "grpo-train": "surogate.cli.grpo_train",
    "grpo-infer": "surogate.cli.grpo_infer",
    "grpo-orch": "surogate.cli.grpo_orch",
    "vf-init": "surogate.cli.vf_init",
    "vf-eval": "surogate.cli.vf_eval",
    "tokenize": "surogate.cli.tokenize_cmd",
    "merge": "surogate.cli.merge",
    "debug": "surogate.cli.debug",
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
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # sft command
    from surogate.cli.sft import prepare_command_parser as sft_prepare_command_parser

    sft_prepare_command_parser(subparsers.add_parser("sft", help="Supervised Fine-Tuning"))

    # pretrain command
    from surogate.cli.pt import prepare_command_parser as pt_prepare_command_parser

    pt_prepare_command_parser(subparsers.add_parser("pt", help="Pretraining"))

    # grpo command (split-GPU mode)
    from surogate.cli.grpo import prepare_command_parser as grpo_prepare_command_parser

    grpo_prepare_command_parser(subparsers.add_parser("grpo", help="GRPO RL (vLLM and trainer on disjoint GPU sets)"))

    # grpo-colocate command (shared-GPU mode)
    from surogate.cli.grpo_colocate import prepare_command_parser as grpo_colocate_prepare_command_parser

    grpo_colocate_prepare_command_parser(
        subparsers.add_parser("grpo-colocate", help="GRPO RL (vLLM and trainer share GPUs via CUDA IPC)")
    )

    # grpo-infer command
    from surogate.cli.grpo_infer import prepare_command_parser as grpo_infer_prepare_command_parser

    grpo_infer_prepare_command_parser(subparsers.add_parser("grpo-infer", help="GRPO RL Inference"))

    # grpo-train command
    from surogate.cli.grpo_train import prepare_command_parser as grpo_train_prepare_command_parser

    grpo_train_prepare_command_parser(subparsers.add_parser("grpo-train", help="GRPO RL Training"))

    # grpo-orch command
    from surogate.cli.grpo_orch import prepare_command_parser as grpo_orch_prepare_command_parser

    grpo_orch_prepare_command_parser(subparsers.add_parser("grpo-orch", help="GRPO RL Orchestrator"))

    # vf-init command
    from surogate.cli.vf_init import prepare_command_parser as vf_init_prepare_command_parser

    vf_init_prepare_command_parser(subparsers.add_parser("vf-init", help="RL Environment Initialization"))

    # vf-eval command
    from surogate.cli.vf_eval import prepare_command_parser as vf_eval_prepare_command_parser

    vf_eval_prepare_command_parser(subparsers.add_parser("vf-eval", help="RL Environment Evaluation"))

    # tokenize command
    from surogate.cli.tokenize_cmd import prepare_command_parser as tokenize_prepare_command_parser

    tokenize_prepare_command_parser(subparsers.add_parser("tokenize", help="Tokenize datasets for training"))

    # merge command
    from surogate.cli.merge import prepare_command_parser as merge_prepare_command_parser

    merge_prepare_command_parser(subparsers.add_parser("merge", help="Merge a LoRA checkpoint into the base model"))

    # debug command
    from surogate.cli.debug import prepare_command_parser as debug_prepare_command_parser

    debug_prepare_command_parser(subparsers.add_parser("debug", help="Introspection and diagnostics for DSL models"))

    args = parser.parse_args(sys.argv[1:])
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands_with_config = ["sft", "pt", "grpo_train", "grpo_infer", "grpo_orch", "tokenize"]
    if args.command in commands_with_config and not getattr(args, "config", None):
        parser.print_help()
        sys.exit(1)

    return args


def cli_main():
    """Main CLI entry point for installed 'surogate' command."""
    args = parse_args()

    system_info = get_system_info()
    print_system_diagnostics(system_info)

    # Run the command module in-process (avoids a second Python startup).
    # Rewrite sys.argv so the module's __main__ block sees only its own args.
    module_name = COMMAND_MAPPING[args.command]
    sys.argv = [module_name] + sys.argv[2:]
    runpy.run_module(module_name, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    cli_main()
