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
from pathlib import Path

_USAGE = """\
usage: surogate serve <model> [engine options...]
       surogate serve --generate <model> --prompt "..." [options...]
       surogate serve --embed <model> [--frontend DIR] [options...]

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
  --max-num-seqs N               concurrent lanes (default 1)
  --kv-cache-dtype auto|fp8|bf16 KV cache precision (default auto: bf16 for a pure-attention
                                 model, fp8 where linear-attention layers carry the stack)
  --no-cache                     rebuild the conversion cache instead of reusing it
  --spec mtp --draft-tokens 3    speculative decoding

--generate runs one shot and has its own spellings for a few options
(--max-context, --kv-dtype, --max-new): surogate serve --generate --engine-help.

--embed serves an encoder model: --device N|cpu chooses the backend, and
--frontend DIR supplies the tokenizer when converting a .gguf. CPU serving wants
OMP_WAIT_POLICY=ACTIVE and OMP_NUM_THREADS set to the physical cores of one NUMA
node.

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
}


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
    return shutil.which(name)


def maybe_exec_serve() -> None:
    """If invoked as `surogate serve ...`, exec the engine binary (never returns)."""
    argv = sys.argv
    if len(argv) < 2 or argv[1] != "serve":
        return

    rest = argv[2:]
    if not rest or rest[0] in ("-h", "--help"):
        sys.stderr.write(_USAGE)
        sys.exit(0 if rest else 1)

    # --generate switches to the one-shot CLI binary, --embed to the encoder
    # server; --engine-help passes --help through so the binary's own option
    # surface stays canonical.
    mode = "server"
    if "--generate" in rest:
        rest = [a for a in rest if a != "--generate"]
        mode = "generate"
    if "--embed" in rest:
        if mode == "generate":
            sys.stderr.write("surogate serve: --embed and --generate are different modes.\n")
            sys.exit(2)
        rest = [a for a in rest if a != "--embed"]
        mode = "embed"
    if "--engine-help" in rest:
        rest = ["--help" if a == "--engine-help" else a for a in rest]

    binary = _resolve_binary(mode)
    if binary is None:
        name = _MODES[mode][0]
        sys.stderr.write(
            f"surogate serve: the serving engine is not built ({name} not found).\n"
            "Build it first:  make serve-build   (from the surogate repo root)\n"
        )
        sys.exit(127)

    # `--no-cache` rebuilds the index instead of reusing one that is already there. It is
    # for a converter change: the cache is keyed on the *checkpoint*, so editing a recipe
    # leaves the stale entry looking valid. It is consumed here, not passed to the engine.
    reuse_cache = "--no-cache" not in rest
    rest = [a for a in rest if a != "--no-cache"]

    # Resolve the model spec (first non-flag argument) through the ingest
    # layer: safetensors dirs / HF repo ids convert transparently into the
    # internal cache; GGUF and unsupported models get clear messages.
    model_index = next(
        (i for i, a in enumerate(rest) if not a.startswith("-")
         and (i == 0 or not rest[i - 1].startswith("--") or "=" in rest[i - 1]
              or rest[i - 1] in ("--vision", "--greedy", "--no-cuda-graph",
                                 "--no-prefix-reuse", "--lm-head-draft",
                                 "--no-thinking", "--preserve-thinking", "--cors",
                                 "--raw-output", "--print-token-ids"))),
        None,
    )
    if mode == "embed":
        # The encoder server takes its model positionally, like the generative
        # engine does; --frontend is a conversion input, consumed here.
        frontend = None
        if "--frontend" in rest:
            i = rest.index("--frontend")
            if i + 1 >= len(rest):
                sys.stderr.write("surogate serve: --frontend needs a directory\n")
                sys.exit(2)
            frontend = rest[i + 1]
            del rest[i:i + 2]
            if model_index is not None and model_index > i:
                model_index -= 2
        if model_index is None:
            sys.stderr.write("surogate serve --embed: a model is required\n")
            sys.exit(2)
        from surogate.serve.ingest import ensure_encoder_weights

        resolved = ensure_encoder_weights(rest[model_index], frontend=frontend,
                                          reuse_cache=reuse_cache,
                                          echo=lambda m: print(m, file=sys.stderr))
        rest = [*rest[:model_index], str(resolved), *rest[model_index + 1:]]
    elif model_index is not None:
        from surogate.serve.ingest import ensure_engine_weights

        resolved = ensure_engine_weights(rest[model_index], reuse_cache=reuse_cache,
                                         echo=lambda m: print(m, file=sys.stderr))
        rest = [*rest[:model_index], str(resolved), *rest[model_index + 1:]]

    os.execv(binary, [binary] + rest)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "serve"] + sys.argv[1:]
    maybe_exec_serve()
