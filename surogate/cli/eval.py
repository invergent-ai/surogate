import argparse
import sys

from surogate.utils.logger import get_logger
logger = get_logger()

def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to config file (required for run mode)')
    parser.add_argument('--list', action='store_true', help='List all evaluation results')
    parser.add_argument('--view', type=str, metavar='FILENAME', help='View specific evaluation result')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'), help='Compare two evaluation results')
    parser.add_argument('--results-dir', type=str, default='eval_results',
                             help='Results directory (default: eval_results)')
    return parser


if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])

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

