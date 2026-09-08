"""Convert an LFM2-MoE safetensors checkpoint into a serving artifact."""

import argparse
from pathlib import Path

from surogate.serve.convert.lfm2.convert import convert as convert_lfm2
from . import inventory, recipe

validate_config = inventory.geometry_from_config


def convert(model_dir, out_path, *, device="cuda", gguf_repack=None):
    return convert_lfm2(model_dir, out_path, device=device, gguf_repack=gguf_repack,
                        _inventory=inventory, _recipe=recipe)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gguf-repack", type=Path)
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device, gguf_repack=args.gguf_repack)


if __name__ == "__main__":
    main()
