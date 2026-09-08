"""Checkpoint-driven nvfp4-mlp-only conversion."""

from .. import inventory
from . import quantized


def convert(model_dir, quantized_or_out, out_path=None, *, device="cuda", resources_from=None, mtp=True, vision=True):
    """Use one checkpoint, or a quantized checkpoint with a complete fallback checkpoint."""
    return quantized.convert(
        model_dir, quantized_or_out if out_path is None else out_path,
        profile=inventory.NVFP4_MLP_ONLY,
        quantized_model_dir=None if out_path is None else quantized_or_out,
        device=device, resources_from=resources_from, mtp=mtp, vision=vision,
    )


def main(argv=None):
    quantized.main(argv, profile=inventory.NVFP4_MLP_ONLY)


if __name__ == "__main__":
    main()
