import sys
import argparse
from swift.ray import try_init_ray

from surogate.sft.sft import SurogateSFT

def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--hub_token', type=str, help='Hugging Face token for private model access',
                     default=None)
    return parser

if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])

    try_init_ray()

    SurogateSFT(**args.__dict__).run()
