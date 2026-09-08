"""Convert LFM2-VL safetensors, including its vision encoder and projector."""

import argparse
from pathlib import Path

from surogate.serve.convert.lfm2.convert import convert as convert_lfm2
from . import inventory, recipe

validate_config = inventory.geometry_from_config


def convert(model_dir, out_path, *, device="cuda"):
    return convert_lfm2(model_dir, out_path, device=device, _inventory=inventory, _recipe=recipe)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
