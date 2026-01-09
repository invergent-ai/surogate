import sys
import argparse

from surogate.core.config.loader import load_config
from surogate.core.config.sft_config import SFTConfig
from surogate.utils.logger import get_logger
logger = get_logger()

from surogate.utils.dict import DictDefault

def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str, help='Path or HTTP(s) URL to config file')
    parser.add_argument('--debug', action='store_true', help='Print tokens along with their labels so you can confirm the correct items are being ignored.')
    parser.add_argument('--hub_token', type=str, help='Hugging Face token for private model access', default=None)

    return parser


if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])
    config = load_config(SFTConfig, args.config)

    from surogate.train.tokenize import tokenize_main
    tokenize_main(config, DictDefault(**args.__dict__))
