import argparse
import sys

from surogate.utils.logger import get_logger
logger = get_logger()

from surogate.serve.serve import SurogateServe


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, help='Host address to bind the server', default='127.0.0.1')
    parser.add_argument('--port', type=int, help='Port number to bind the server', default=8000)
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--hub_token', type=str, help='Hugging Face/Modelscope token for private model access',
                       default=None)
    return parser


if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])
    SurogateServe(**args.__dict__).run()
