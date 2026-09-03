"""Turn a checkpoint we trained into a GGUF the engine can serve.

A model someone downloads already exists as a GGUF and is served where it lies; nothing here
touches it. A model trained here has no published GGUF, and this is the step that produces one:

    surogate sft ...                                   # a LoRA checkpoint
    surogate merge --base-model B --checkpoint-dir C --output merged
    surogate quantize --model merged --output model.gguf --type q4_k_m
    surogate serve model.gguf

Two facts shape the implementation. The quantisation arithmetic is llama.cpp's, because every
published GGUF was made with it and because the reference K-quant encoders exist nowhere else:
the `gguf` Python package writes Q8_0 but raises `NotImplementedError` for Q4_K, Q5_K and Q6_K.
And a K-quant is produced in two passes, not one: llama.cpp's converter reads the Hugging Face
checkpoint and writes a BF16 GGUF (it holds the per-architecture tensor mapping and the
tokenizer), then `llama-quantize` reads that and applies the type mixture. So this command
drives those two programs and does no arithmetic of its own, which is also the shape unsloth
ships (`study/unsloth-zoo/unsloth_zoo/llama_cpp.py`).

The intermediate is two bytes a parameter, so a 35 B model wants ~70 GB of scratch beside the
output; the space is checked before the first pass rather than discovered during it.
"""

import argparse
import os
import shutil
import subprocess
import sys

from surogate.utils.logger import get_logger

logger = get_logger()

# `llama-quantize --help` lists these; the K-quants are the product and the rest are here
# because a caller may legitimately want them. Anything llama.cpp accepts still works: the
# value is passed through, and llama.cpp rejects what it does not know.
COMMON_TYPES = (
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_k_s", "q4_k_m",
    "q5_k_s", "q5_k_m", "q6_k", "q8_0", "bf16", "f16", "f32",
)

# The vendored tree, used when nothing else says otherwise. It is a checkout rather than part of
# the package, so the environment variable exists for an installed llama.cpp elsewhere.
VENDORED_LLAMA_CPP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "study", "llama.cpp-master",
)


def prepare_command_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Quantize a trained checkpoint into a GGUF the engine serves"
        )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face checkpoint directory (e.g. the output of `surogate merge`), "
        "or an unquantised .gguf to quantize directly",
    )
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument(
        "--type",
        default="q4_k_m",
        help=f"Quantization type, default q4_k_m. Common: {', '.join(COMMON_TYPES)}",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="Quantizer threads, default the CPU count"
    )
    parser.add_argument(
        "--imatrix",
        default=None,
        help="Importance matrix file, passed to llama-quantize; required by the IQ types",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the BF16 GGUF the first pass writes (deleted by default)",
    )
    parser.add_argument(
        "--llama-cpp",
        default=None,
        help="llama.cpp directory holding the converter and the built quantizer "
        "(default: $SUROGATE_LLAMA_CPP, else the vendored checkout)",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build llama-quantize if it is missing instead of failing with the command",
    )
    return parser


def _llama_cpp_dir(explicit):
    """The llama.cpp checkout to drive: the flag, then the environment, then the vendored tree."""
    for candidate in (explicit, os.environ.get("SUROGATE_LLAMA_CPP"), VENDORED_LLAMA_CPP):
        if candidate:
            return os.path.abspath(candidate)
    return None


def _find_quantizer(llama_cpp, build):
    """The `llama-quantize` binary, built on request when it is the only thing missing."""
    candidates = (
        os.path.join(llama_cpp, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp, "build", "bin", "Release", "llama-quantize.exe"),
        os.path.join(llama_cpp, "llama-quantize"),
    )
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    command = ["cmake", "--build", "build", "--target", "llama-quantize", "-j", "16"]
    if not build:
        logger.error(
            f"llama-quantize is not built in {llama_cpp}. Build it with:\n"
            f"    cd {llama_cpp} && {' '.join(command)}\n"
            f"or pass --build to have this command do it."
        )
        return None
    logger.info(f"Building llama-quantize in {llama_cpp}")
    if subprocess.run(command, cwd=llama_cpp).returncode != 0:
        logger.error("The llama-quantize build failed; its output is above.")
        return None
    return _find_quantizer(llama_cpp, build=False)


def _find_converter(llama_cpp):
    """llama.cpp's Hugging-Face-to-GGUF script, whichever name this revision uses."""
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        path = os.path.join(llama_cpp, name)
        if os.path.isfile(path):
            return path
    logger.error(f"No convert_hf_to_gguf.py in {llama_cpp}; is that a llama.cpp checkout?")
    return None


def _checkpoint_bytes(model_dir):
    """Bytes of weight files in a checkpoint, for the scratch-space estimate."""
    total = 0
    for entry in os.scandir(model_dir):
        if entry.is_file() and entry.name.endswith((".safetensors", ".bin", ".pt")):
            total += entry.stat().st_size
    return total


def _run(command, what):
    logger.info(f"{what}: {' '.join(str(part) for part in command)}")
    if subprocess.run(command).returncode != 0:
        logger.error(f"{what} failed; its output is above.")
        return False
    return True


def main(args):
    llama_cpp = _llama_cpp_dir(args.llama_cpp)
    if llama_cpp is None or not os.path.isdir(llama_cpp):
        logger.error(
            "No llama.cpp checkout found. Pass --llama-cpp DIR or set SUROGATE_LLAMA_CPP."
        )
        return 1
    if not args.output.endswith(".gguf"):
        logger.error(f"--output must end in .gguf, got {args.output}")
        return 1
    if os.path.abspath(args.model) == os.path.abspath(args.output):
        logger.error("--model and --output are the same file")
        return 1
    if args.imatrix is not None and not os.path.isfile(args.imatrix):
        logger.error(f"imatrix file does not exist: {args.imatrix}")
        return 1

    quantizer = _find_quantizer(llama_cpp, args.build)
    if quantizer is None:
        return 1

    output_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(output_dir, exist_ok=True)

    # A .gguf input is already the quantizer's operand; a directory needs the first pass. The
    # intermediate lands beside the output, where the space was checked.
    intermediate = None
    if os.path.isdir(args.model):
        converter = _find_converter(llama_cpp)
        if converter is None:
            return 1
        weights = _checkpoint_bytes(args.model)
        if weights == 0:
            logger.error(f"No .safetensors or .bin weight files in {args.model}")
            return 1
        free = shutil.disk_usage(output_dir).free
        # The BF16 intermediate is ~2 bytes a parameter; a BF16 checkpoint is already that, so
        # its own size is the estimate, plus the output itself at less than the same again.
        needed = int(weights * 1.1)
        if free < needed:
            logger.error(
                f"{output_dir} has {free / 1e9:.1f} GB free; the BF16 GGUF this writes needs "
                f"about {needed / 1e9:.1f} GB. Point --output somewhere larger."
            )
            return 1
        base = os.path.basename(args.output)[: -len(".gguf")]
        intermediate = os.path.join(output_dir, f"{base}.bf16.gguf")
        if not _run(
            [sys.executable, converter, os.path.abspath(args.model),
             "--outtype", "bf16", "--outfile", intermediate],
            "Converting the checkpoint to a BF16 GGUF",
        ):
            return 1
        source = intermediate
    elif os.path.isfile(args.model):
        source = os.path.abspath(args.model)
    else:
        logger.error(f"--model is neither a directory nor a file: {args.model}")
        return 1

    threads = args.threads if args.threads is not None else (os.cpu_count() or 1)
    command = [quantizer]
    if args.imatrix is not None:
        command += ["--imatrix", os.path.abspath(args.imatrix)]
    command += [source, os.path.abspath(args.output), args.type, str(threads)]
    quantized = _run(command, f"Quantizing to {args.type}")

    # The intermediate is large and single-purpose, so it goes unless asked for -- but never
    # when the pass that reads it failed, since a retry would otherwise redo the conversion.
    if intermediate is not None and os.path.isfile(intermediate):
        if quantized and not args.keep_intermediate:
            os.remove(intermediate)
        else:
            logger.info(f"BF16 GGUF kept at {intermediate}")
    if not quantized:
        return 1

    size = os.path.getsize(args.output)
    logger.info(f"Wrote {args.output} ({size / 1e9:.2f} GB). Serve it with:")
    logger.info(f"    surogate serve {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(prepare_command_parser().parse_args(sys.argv[1:])))
