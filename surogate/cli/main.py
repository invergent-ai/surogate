# surogate/cli/main.py

import argparse
import sys

from surogate.utils.logger import get_logger

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Surogate LLMOps Framework")
    parser.set_defaults(func=lambda _args, p=parser: p.print_help())
    subparsers = parser.add_subparsers(dest='command', metavar='<command>')

    # Serve command
    serve = subparsers.add_parser('serve', help="Serve a model via API")
    serve.add_argument('--host', type=str, help='Host address to bind the server', default='127.0.0.1')
    serve.add_argument('--port', type=int, help='Port number to bind the server', default=8000)
    serve.add_argument('--config', type=str, required=True, help='Path to config file')
    serve.add_argument('--hub_token', type=str, help='Hugging Face/Modelscope token for private model access',
                       default=None)

    sft = subparsers.add_parser('serve', help="Supervised Fine-Tuning (SFT)")
    serve.add_argument('--config', type=str, required=True, help='Path to config file')
    serve.add_argument('--hub_token', type=str, help='Hugging Face/Modelscope token for private model access',
                       default=None)

    # Eval command with multiple operation modes
    eval_parser = subparsers.add_parser('eval', help="Evaluate a model using surogate eval module")
    eval_parser.add_argument('--config', type=str, help='Path to config file (required for run mode)')
    eval_parser.add_argument('--list', action='store_true', help='List all evaluation results')
    eval_parser.add_argument('--view', type=str, metavar='FILENAME', help='View specific evaluation result')
    eval_parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'), help='Compare two evaluation results')
    eval_parser.add_argument('--results-dir', type=str, default='eval_results',
                             help='Results directory (default: eval_results)')

    # PTQ command
    ptq = subparsers.add_parser('ptq', help="Post-training quantization")
    ptq.add_argument('--config', type=str, required=True, help='Path to config file')
    ptq.add_argument('--hub_token', type=str, help='Hugging Face token for private model access',
                     default=None)

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


def cli_main():
    args = parse_args()

    # Lazy imports - only import what's needed for the command
    if args.command == 'serve':
        from surogate.serve.serve import SurogateServe
        logger.info(f"Starting to serve with config {args.config} on {args.host}:{args.port}")
        SurogateServe(**args.__dict__).run()

    elif args.command == 'eval':
        # Determine which eval operation to perform
        if args.list:
            # List all results
            from surogate.eval.results import list_results, display_results_list
            results = list_results(args.results_dir)
            display_results_list(results, args.results_dir)

        elif args.view:
            # View specific result
            from surogate.eval.results import display_results
            from pathlib import Path

            filepath = Path(args.results_dir) / args.view
            if not filepath.exists():
                logger.error(f"Result file not found: {filepath}")
                sys.exit(1)

            display_results(str(filepath))

        elif args.compare:
            # Compare two results
            from surogate.eval.results import compare_results
            from pathlib import Path

            file1, file2 = args.compare
            filepath1 = Path(args.results_dir) / file1
            filepath2 = Path(args.results_dir) / file2

            if not filepath1.exists():
                logger.error(f"First result file not found: {filepath1}")
                sys.exit(1)
            if not filepath2.exists():
                logger.error(f"Second result file not found: {filepath2}")
                sys.exit(1)

            compare_results(str(filepath1), str(filepath2))

        else:
            # Default: Run evaluation
            from surogate.eval.eval import SurogateEval
            logger.info(f"Running evaluation with config {args.config}")
            SurogateEval(**args.__dict__).run()

    elif args.command == 'ptq':
        from surogate.ptq.ptq import SurogatePtq
        logger.info(f"Running PTQ with config {args.config}")
        SurogatePtq(**args.__dict__).run()

    elif args.command == 'sft':
        from surogate.sft.sft import SurogateSFT
        logger.info(f"Starting SFT with config {args.config}")
        SurogateSFT(**args.__dict__).run()


if __name__ == '__main__':
    cli_main()