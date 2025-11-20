import argparse
import sys

from surogate.ptq.ptq import SurogatePTQ


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--hub_token', type=str, help='Hugging Face token for private model access',
                     default=None)
    return parser

if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])

    SurogatePTQ(**args.__dict__).run()
