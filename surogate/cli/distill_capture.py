import argparse
import sys

from surogate.core.config.loader import load_config
from surogate.core.config.sft_config import SFTConfig
from surogate.utils.logger import get_logger

logger = get_logger()

from surogate.utils.dict import DictDefault


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("config", type=str, help="Path or HTTP(s) URL to config file")
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible base URL of a served teacher, e.g. http://localhost:8000/v1 "
        "(overrides distillation.teacher_api_base). Requires a vLLM-compatible server "
        "started with --max-logprobs >= distillation.top_k.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the local teacher model on (default cuda:0). "
        "Ignored with a warning in API mode (--api-base).",
    )
    parser.add_argument(
        "--allow-cross-doc-attention",
        action="store_true",
        help="Allow sdpa fallback when flash-attention-2 is unavailable; packed documents "
        "will attend across document boundaries during capture. "
        "Ignored with a warning in API mode (--api-base).",
    )
    parser.add_argument("--hub_token", type=str, help="Hugging Face token for private model access", default=None)

    return parser


if __name__ == "__main__":
    args = prepare_command_parser().parse_args(sys.argv[1:])
    config = load_config(SFTConfig, args.config)

    from surogate.distill.capture import distill_capture_main

    distill_capture_main(config, DictDefault(**args.__dict__))
