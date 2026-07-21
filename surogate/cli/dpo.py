import argparse
import sys

from surogate.core.config.loader import load_config
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Path or HTTP(s) URL to config file")
    parser.add_argument("--hub_token", type=str, help="Hugging Face token for private model access", default=None)

    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])

    from surogate.dpo.config import DPOTrainConfig
    from surogate.dpo.trainer import dpo_main

    config = load_config(DPOTrainConfig, args.config)
    dpo_main(config, DictDefault(**args.__dict__))
