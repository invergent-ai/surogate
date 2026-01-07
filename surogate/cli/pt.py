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

    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--hub_token', type=str, help='Hugging Face token for private model access', default=None)

    return parser


if __name__ == '__main__':
    args = prepare_command_parser().parse_args(sys.argv[1:])
    config = load_config(SFTConfig, args.config)

    from surogate.train.pt import pt_main
    pt_main(config, DictDefault(**args.__dict__))