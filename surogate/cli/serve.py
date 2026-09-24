# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# `surogate serve ...` — the native serving engine (design/serve-engine-plan.md).
#
# The engine is the C++ server at csrc/src/serve, built by
# `make serve-build`. This wrapper resolves the binary and os.execv's it, so no
# Python (and no Python CUDA context) stays in the serving process. It must run
# BEFORE any CUDA-touching import in surogate.cli.main, mirroring jackalope.

import os
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

_USAGE = """\
usage: surogate serve <model> [engine options...]
       surogate serve --generate <model> --prompt "..." [options...]
       surogate serve --embed <model> [--frontend DIR] [options...]
       surogate serve --stt <model> [--lm PATH] [--device N|cpu] [options...]
       surogate serve --tts <model> [--voice NAME] [--device cpu|N] [options...]

<model> is a Hugging Face repo id, a local safetensors model directory, or a
GGUF file. First use converts transparently into a local cache; after that,
loads are instant.

  surogate serve Qwen/Qwen3.6-27B
  surogate serve ~/models/qwen3.6-27b-hf/
  surogate serve ~/models/qwen3.6-27b-Q4_K_M.gguf

Serves OpenAI-/Anthropic-compatible HTTP (default), runs a one-shot generation
with --generate (streams the answer to stdout), or serves /v1/embeddings for an
encoder model with --embed.

Common server options (full list: surogate serve --engine-help):
  --host 0.0.0.0 --port 8080     bind address
  --max-model-len N              per-sequence context ceiling
  --kv-capacity N|auto           KV pool size ('auto' = free VRAM minus 1 GiB)
  --max-num-seqs N               simultaneous requests (default 1)
  --kv-cache-dtype auto|fp8|bf16|int8
                                KV cache precision (default auto: bf16 for a pure-attention
                                 model, fp8 where linear-attention layers carry the stack)
  --no-cache                     rebuild the conversion cache instead of reusing it
  --spec mtp --draft-tokens 3    speculative decoding
  --dflash-model PATH           separate Muse-Glimmer DFlash GGUF checkpoint
  --mmproj PATH                 matching vision projector for a supported vision GGUF
  --offload-vision               store image encoder/projector weights in system RAM
  --offload-embeddings           store token embeddings in system RAM
  --offload-output-head          store output-head weights in system RAM
  --decision-temperature T       calibrate the decisions endpoint: answers are read from
                                 softmax(option logits / T); default 1 (unchanged)

--generate runs one shot and has its own spellings for a few options
(--max-context, --kv-dtype, --max-new): surogate serve --generate --engine-help.

--embed serves an encoder model: --device N|cpu chooses the backend, and
--frontend DIR optionally overrides the tokenizer embedded in the GGUF. CPU serving wants
OMP_WAIT_POLICY=ACTIVE and OMP_NUM_THREADS set to the physical cores of one NUMA
node.

--stt serves surogate speech recognition models on GPU or CPU. Use --lm PATH to add
a language model. See docs/inference/speech.md for file uploads and live audio.

--tts serves surogate speech generation models with named voices, on CPU or a GPU.
See docs/inference/tts.md.

From a source checkout, build the engine first with: make serve-build
"""


def _repo_root() -> Path | None:
    # surogate/cli/serve.py -> surogate/cli -> surogate -> repo root
    root = Path(__file__).resolve().parent.parent.parent
    return root if (root / "csrc" / "src" / "serve").is_dir() else None


_MODES = {
    # mode -> (binary name, env override)
    "server": ("surogate-engine", "SUROGATE_SERVE_BIN"),
    "generate": ("surogate-engine-cli", "SUROGATE_ENGINE_CLI_BIN"),
    "embed": ("surogate-embed", "SUROGATE_EMBED_BIN"),
    "stt": ("surogate-stt", "SUROGATE_STT_BIN"),
    "tts": ("surogate-tts", "SUROGATE_TTS_BIN"),
}

# Arity is needed before preparing a model: an option value may itself start with
# "--", and options may precede the positional model. Native parsers still own
# value validation. Tests check this inventory against every native parser.
_COMMON_VALUES = frozenset("""
    --device --host-moe-layers --gpu-layers -ngl --n-gpu-layers --expert-slots
    --host-expert-bank --cpu-moe-share --cpu-moe-prefill-share --cpu-moe-min-tokens
    --devices --kv-capacity --spec --draft-tokens --spec-max-lanes
    --temperature --top-p --top-k --min-p --presence-penalty --frequency-penalty --seed
""".split())
_COMMON_SWITCHES = frozenset("""
    --vision --lm-head-draft --no-thinking --greedy --spec-adaptive
    --offload-vision --offload-embeddings --offload-output-head
""".split())
_VALUE_OPTIONS = {
    "server": _COMMON_VALUES | frozenset("""
        --host --port --api-key --api-key-file --served-model-name --max-model-len --max-num-seqs
        --max-pending-requests --pending-timeout-ms --max-num-batched-tokens
        --log-stats-interval-ms --max-request-mib --media-cache-mib --media-live-mib
        --media-preprocess-threads --request-log-jsonl --kv-cache-dtype --kv-cache-dtype-skip-layers
        --default-max-tokens --reasoning-parser --tool-call-parser --chat-template
        --model-priority --model --lora-modules --max-loras --max-lora-rank
        --decision-temperature --gpu-memory-limit-mib
    """.split()),
    "generate": _COMMON_VALUES | frozenset("""
        --prompt --messages --max-new --max-context --prefill-chunk --kv-dtype
        --reasoning-effort --stop-token-id --stop --reasoning-stop
    """.split()),
    "embed": frozenset("--host --port --device --served-model-name".split()),
    "stt": frozenset("--host --port --device --served-model-name --api-key --api-key-file --max-num-seqs --threads --cpu-kernels".split()),
    "tts": frozenset("--host --port --device --served-model-name --api-key --api-key-file --voice --max-pending-requests --max-num-seqs --request-timeout --max-input-characters --threads --codec-threads --cpu-kernels".split()),
}
_SWITCH_OPTIONS = {
    "server": _COMMON_SWITCHES | frozenset("""
        --elastic-kv --no-elastic-kv
        --elastic-kv-overcommit --enforce-eager --no-prefix-reuse
        --enable-prefix-caching --no-enable-prefix-caching --enable-auto-tool-choice
        --enable-sleep-mode --enable-lora --preserve-thinking --cors
    """.split()),
    "generate": _COMMON_SWITCHES | frozenset("""
        --raw-output --print-token-ids --prefill-warmup --no-cuda-graph
    """.split()),
    "embed": frozenset(),
    "stt": frozenset(),
    "tts": frozenset(),
}


def _parse_invocation(args: list[str]) -> tuple[str, str | None, list[str], bool, str | None]:
    """Return mode, model, native options, cache policy and preparation resource."""
    values = frozenset().union(*_VALUE_OPTIONS.values(), {"--frontend", "--mmproj", "--dflash-model", "--lm"})
    switches = frozenset().union(*_SWITCH_OPTIONS.values(), {
        "--generate", "--embed", "--stt", "--tts", "--no-cache", "--engine-help", "--help", "-h",
    })
    parsed: list[tuple[str, str | None]] = []
    models: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        i += 1
        if token == "--":
            models.extend(args[i:])
            break
        if not token.startswith("-"):
            models.append(token)
            continue
        flag, equals, inline = token.partition("=")
        if flag in values:
            if equals:
                value = inline
            elif i < len(args):
                value = args[i]
                i += 1
            else:
                raise ValueError(f"{flag} needs a value")
            parsed.append((flag, value))
        elif flag in switches and not equals:
            parsed.append((flag, None))
        else:
            raise ValueError(f"unknown option: {flag}")

    flags = {flag for flag, _ in parsed}
    if len(flags & {"--generate", "--embed", "--stt", "--tts"}) > 1:
        raise ValueError("--embed, --generate, --stt and --tts are different modes")
    mode = next((name for name in ("tts", "stt", "embed", "generate") if f"--{name}" in flags), "server")
    if flags & {"--engine-help", "--help", "-h"}:
        return mode, None, ["--help"], True, None
    if len(models) != 1 or not models[0]:
        raise ValueError("exactly one model is required")
    native: list[str] = []
    frontend = None
    for flag, value in parsed:
        if flag in {"--generate", "--embed", "--stt", "--tts", "--no-cache"}:
            continue
        if flag == "--lm" and mode == "stt":
            frontend = value
            continue
        if flag == "--frontend" and mode == "embed":
            if not value:
                raise ValueError("--frontend needs a directory")
            frontend = value
            continue
        if flag == "--mmproj" and mode in ("server", "generate"):
            if not value:
                raise ValueError("--mmproj needs a GGUF file")
            frontend = value
            continue
        if flag == "--dflash-model" and mode in ("server", "generate"):
            if not value:
                raise ValueError("--dflash-model needs a GGUF file")
            native.extend([flag, value])
            continue
        if flag not in _VALUE_OPTIONS[mode] | _SWITCH_OPTIONS[mode]:
            raise ValueError(f"{flag} is not supported in {mode} mode")
        native.append(flag)
        if value is not None:
            native.append(value)
    return mode, models[0], native, "--no-cache" not in flags, frontend


# Where the wheel puts the product binaries (csrc/CMakeLists.txt, install component
# `serve`). An installed package has no csrc/ to look in, and the engine is not a
# command of its own, so it is not on PATH either -- this is the only place it is.
_INSTALLED_BIN_DIR = Path(__file__).resolve().parent.parent / "serve" / "_bin"


def _resolve_binary(mode: str) -> str | None:
    name, env_var = _MODES[mode]
    env = os.environ.get(env_var)
    if env and Path(env).is_file():
        return env
    root = _repo_root()
    if root is not None:
        cand = root / "csrc" / "build-serve" / name
        if cand.is_file():
            return str(cand)
        if mode in ("stt", "tts"):
            cand = root / "csrc" / f"build-{mode}" / name
            if cand.is_file():
                return str(cand)
    cand = _INSTALLED_BIN_DIR / name
    if cand.is_file():
        return str(cand)
    return shutil.which(name)


def maybe_exec_serve() -> None:
    """Prepare assets, then replace Python with the selected native engine."""
    argv = sys.argv
    if len(argv) < 2 or argv[1] != "serve":
        return

    rest = argv[2:]
    if not rest or rest[0] in ("-h", "--help"):
        sys.stderr.write(_USAGE)
        sys.exit(0 if rest else 1)

    try:
        mode, model, rest, reuse_cache, frontend = _parse_invocation(rest)
    except ValueError as error:
        sys.stderr.write(f"surogate serve: {error}\n")
        sys.exit(2)

    binary = _resolve_binary(mode)
    if binary is None:
        name = _MODES[mode][0]
        sys.stderr.write(
            f"surogate serve: the serving engine is not built ({name} not found).\n"
            f"Build it first:  {'make serve-tts-build' if mode == 'tts' else 'make serve-build'}\n"
            "An installed wheel ships it; a source tree builds it into csrc/build-serve.\n"
        )
        sys.exit(127)

    if mode == "tts":
        if model is not None:
            from surogate.serve.tts.assets import prepare_bundle

            # The server takes the last --device given, and so does the check here.
            devices = [rest[i + 1] for i, arg in enumerate(rest[:-1]) if arg == "--device"]
            try:
                bundle = prepare_bundle(model, reuse_cache=reuse_cache,
                                        echo=lambda m: print(m, file=sys.stderr),
                                        device=devices[-1] if devices else "cpu")
            except ValueError as error:
                sys.stderr.write(f"surogate serve: {error}\n")
                sys.exit(2)
            if "--served-model-name" not in rest[::2]:
                rest.extend(["--served-model-name", model])
            rest = [str(bundle.root), *rest]
        os.execv(binary, [binary, *rest])
        return

    if model is not None:
        from surogate.serve.ingest import ensure_encoder_weights, ensure_engine_weights

        dflash_model = None
        speculative = None
        native = []
        i = 0
        while i < len(rest):
            flag = rest[i]
            takes_value = flag in _VALUE_OPTIONS[mode] or flag == "--dflash-model"
            width = 2 if takes_value else 1
            if flag == "--dflash-model":
                if dflash_model is not None:
                    raise SystemExit("surogate serve: --dflash-model may be specified only once")
                dflash_model = rest[i + 1]
            else:
                native.extend(rest[i:i + width])
                if flag == "--spec":
                    speculative = rest[i + 1]
            i += width
        rest = native
        if dflash_model is None and speculative == "dflash":
            dflash_model = "auto"
        # Preparation follows the same logical CUDA device as the native runtime.
        # An explicit conversion override remains useful when CPU RAM is preferable.
        selected = {}
        i = 0
        while i < len(rest):
            flag = rest[i]
            if flag in _VALUE_OPTIONS[mode]:
                if flag in ("--device", "--devices", "--served-model-name"):
                    selected[flag] = rest[i + 1]
                i += 2
            else:
                i += 1
        device = selected.get("--devices", selected.get("--device", "0")).split(",")[0]
        conversion_device = "cpu" if device == "cpu" else f"cuda:{device}"
        previous_conversion_device = os.environ.get("SUROGATE_CONVERT_DEVICE")
        os.environ.setdefault("SUROGATE_CONVERT_DEVICE", conversion_device)
        kwargs = dict(reuse_cache=reuse_cache, echo=lambda m: print(m, file=sys.stderr))
        try:
            with redirect_stdout(sys.stderr):
                if mode == "stt":
                    from surogate.serve.speech import ensure_speech_weights
                    resolved = ensure_speech_weights(model, lm=frontend, **kwargs)
                elif mode == "embed":
                    resolved = ensure_encoder_weights(model, frontend=frontend, **kwargs)
                else:
                    if frontend is not None:
                        kwargs["mmproj"] = frontend
                    if dflash_model is not None:
                        kwargs["dflash_model"] = dflash_model
                    resolved = ensure_engine_weights(model, **kwargs)
        finally:
            if previous_conversion_device is None:
                os.environ.pop("SUROGATE_CONVERT_DEVICE", None)
        # Keep the public identity from the user's original argument, before the
        # checkpoint is replaced by its prepared cache path (as vLLM does before
        # resolving/redirecting a model). An explicit deployment alias wins.
        if mode in ("server", "embed", "stt") and "--served-model-name" not in selected:
            rest.extend(["--served-model-name", model])
        rest = [str(resolved), *rest]

    os.execv(binary, [binary] + rest)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "serve"] + sys.argv[1:]
    maybe_exec_serve()
