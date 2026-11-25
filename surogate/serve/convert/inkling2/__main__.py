"""``python -m surogate.serve.convert.inkling2 MODEL_DIR OUT.sinfer [--attention ...]``"""

from __future__ import annotations

import argparse
import sys

import torch

from .convert import convert
from .inventory import GATE_UP_ORDERS, parse_attention


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate.serve.convert.inkling2",
                                     description="Inkling-2 checkpoint -> .sinfer (ARTIFACT.md)")
    parser.add_argument("model_dir")
    parser.add_argument("out")
    parser.add_argument("--attention", default="bf16",
                        help="bf16[,fp8-tensor,fp8-channel,fp8-g1x128,fp8-b128] (bf16 is always carried)")
    parser.add_argument("--gate-up-order", choices=GATE_UP_ORDERS, default="interleaved")
    parser.add_argument("--no-mtp", action="store_true", help="leave out the DSpark drafter")
    parser.add_argument("--no-vision", action="store_true", help="leave out the vision tower (text-only artifact); by default it is carried when the checkpoint has one")
    parser.add_argument("--source-sha256", default=None,
                        help="sha256sum output for the checkpoint files (else hashed here)")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    result = convert(args.model_dir, args.out, fp8_schemes=parse_attention(args.attention),
                     gate_up_order=args.gate_up_order, mtp=not args.no_mtp, vision=False if args.no_vision else None,
                     sha256_file=args.source_sha256, log=lambda m: print(m, flush=True))
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
