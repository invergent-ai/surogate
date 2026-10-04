import argparse
import os
import sys

from surogate.utils.logger import get_logger

logger = get_logger()


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("--base-model", required=True, help="Path to base model directory or HuggingFace model ID")
    parser.add_argument(
        "--checkpoint-dir", required=True, help="Path to a LoRA checkpoint directory (e.g. output/step_00000050)"
    )
    parser.add_argument("--output", required=True, help="Output directory for the merged model")

    return parser


def main(argv: list[str]) -> int:
    """Run `surogate merge`; returns the process exit code."""
    args = prepare_command_parser().parse_args(argv)

    from surogate.utils.adapter_merge import AdapterMergeError, merge_adapter

    # Resolve base model path (handle HuggingFace model IDs)
    base_model_path = args.base_model
    if not os.path.isdir(base_model_path):
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading base model from HuggingFace: {base_model_path}")
            base_model_path = snapshot_download(base_model_path)
        except Exception as e:
            logger.error(
                f"Base model path '{args.base_model}' is not a local directory "
                f"and could not be downloaded from HuggingFace: {e}"
            )
            return 1

    # Validate checkpoint
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")):
        logger.error(f"No adapter_model.safetensors found in {checkpoint_dir}. Is this a LoRA checkpoint?")
        return 1
    if not os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")):
        logger.error(f"No adapter_config.json found in {checkpoint_dir}.")
        return 1

    logger.info(f"Base model:  {base_model_path}")
    logger.info(f"Checkpoint:  {checkpoint_dir}")
    logger.info(f"Output:      {args.output}")

    try:
        merge_adapter(
            base_model_path=base_model_path,
            adapter_path=checkpoint_dir,
            output_path=args.output,
        )
    except AdapterMergeError as e:
        logger.error(f"Merge failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
