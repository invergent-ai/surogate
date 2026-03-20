import sys
import argparse

from surogate.core.config.loader import load_config
from surogate.core.config.sft_config import SFTConfig
from surogate.cli.config_overrides import parse_cli_overrides
from surogate.utils.logger import get_logger
logger = get_logger()

from surogate.utils.dict import DictDefault

def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default=None,
        help='Optional path or HTTP(s) URL to config file',
    )
    parser.add_argument('--hub_token', type=str, help='Hugging Face token for private model access', default=None)

    return parser


if __name__ == '__main__':
    args, unknown = prepare_command_parser().parse_known_args(sys.argv[1:])
    overrides = parse_cli_overrides(unknown)
    config = load_config(SFTConfig, args.config, overrides=overrides)

    from surogate.train.pt import pt_main
    pt_main(config, DictDefault(**args.__dict__))
