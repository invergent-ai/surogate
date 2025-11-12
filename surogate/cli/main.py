import argparse
import sys

from surogate.utils.logger import get_logger

logger = get_logger()


def parse_args():
    parser = argparse.ArgumentParser(description="Surogate LLMOps Framework")
    parser.set_defaults(func=lambda _args, p=parser: p.print_help())
    subparsers = parser.add_subparsers(dest='command', metavar='<command>')

    serve = subparsers.add_parser('serve', help="Serve a model via API")
    serve.add_argument('--host', type=str, help='Host address to bind the server', default='127.0.0.1')
    serve.add_argument('--port', type=int, help='Port number to bind the server', default=8000)
    serve.add_argument('--config', type=str, required=True, help='Path to config file')
    serve.add_argument('--hub_token', type=str, help='Hugging Face/Modelscope token for private model access', default=None)

    eval_parser = subparsers.add_parser('eval', help="Evaluate a model using EvalScope")
    eval_parser.add_argument('--config', type=str, required=True, help='Path to config file')

    deepeval = subparsers.add_parser('deepeval', help="Evaluate a model using DeepEval framework")
    deepeval.add_argument('--config', type=str, required=True, help='Path to config file')
    deepeval.add_argument('--metric', type=str, help='Evaluation metric or test suite name', default=None)
    deepeval.add_argument('--dataset', type=str, help='Path to dataset or evaluation cases', default=None)
    deepeval.add_argument('--model', type=str, help='Model name or endpoint', default=None)
    deepeval.add_argument('--output', type=str, help='Path to store evaluation results', default='results/deepeval.json')

    ptq = subparsers.add_parser('ptq', help="Post-training quantization")
    ptq.add_argument('--config', type=str, required=True, help='Path to config file')
    ptq.add_argument('--hub_token', type=str, help='Hugging Face/Modelscope token for private model access', default=None)

    args = parser.parse_args(sys.argv[1:])
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands_with_config = ['serve', 'eval', 'deepeval', 'ptq']
    if args.command in commands_with_config and not getattr(args, 'config', None):
        parser.print_help()
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
        from surogate.eval.eval import SurogateEval
        logger.info(f"Running EvalScope evaluation with config {args.config}")
        SurogateEval(**args.__dict__).run()

    elif args.command == 'deepeval':
        from surogate.eval.deepeval_runner import SurogateDeepEval
        logger.info(f"Running DeepEval evaluation with config {args.config}")
        SurogateDeepEval(**args.__dict__).run()

    elif args.command == 'ptq':
        from surogate.ptq.ptq import SurogatePtq
        logger.info(f"Running PTQ with config {args.config}")
        SurogatePtq(**args.__dict__).run()


if __name__ == '__main__':
    cli_main()