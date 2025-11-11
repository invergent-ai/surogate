import argparse
import sys
from typing import Dict

from swift.utils import get_logger

from surogate.eval.eval import SurogateEval
from surogate.ptq.ptq import SurogatePtq
from surogate.serve.serve import SurogateServe

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

    eval = subparsers.add_parser('eval', help="Evaluate a model")
    eval.add_argument('--config', type=str, required=True, help='Path to config file')

    ptq = subparsers.add_parser('ptq', help="Post-training quantization")
    ptq.add_argument('--config', type=str, required=True, help='Path to config file')
    ptq.add_argument('--hub_token', type=str, help='Hugging Face/Modelscope token for private model access', default=None)

    args = parser.parse_args(sys.argv[1:])
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands_with_config = ['serve', 'eval', 'ptq']
    if args.command in commands_with_config and not getattr(args, 'config', None):
        parser.print_help()
        sys.exit(1)

    return args


def cli_main():
    args = parse_args()

    if args.command == 'serve':
        logger.info(f"Starting to serve with config {args.config} on {args.host}:{args.port}")
        SurogateServe(**args.__dict__).run()
    elif args.command == 'eval':
        SurogateEval(**args.__dict__).run()
    elif args.command == 'ptq':
        SurogatePtq(**args.__dict__).run()

if __name__ == '__main__':
    cli_main()