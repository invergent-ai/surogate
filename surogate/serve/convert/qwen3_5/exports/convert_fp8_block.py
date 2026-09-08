"""Checkpoint-driven FP8 conversion with block or row scales."""

from ..convert import _load_config, profile_for_checkpoint
from pathlib import Path
from . import quantized


def convert(model_dir, out_path, *, device="cuda", mtp=True, vision=True):
    return quantized.convert(model_dir, out_path,
                             profile=profile_for_checkpoint(_load_config(Path(model_dir))), device=device, mtp=mtp, vision=vision)


def main(argv=None):
    quantized.main(argv)


if __name__ == "__main__":
    main()
