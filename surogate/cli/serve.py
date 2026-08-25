# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# `surogate serve ...` — the native serving engine (design/serve-engine-plan.md).
#
# The engine is the C++ server at csrc/src/serve (NInfer-derived), built by
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

<model> is a Hugging Face repo id, a local safetensors model directory, or a
GGUF file. First use converts transparently into a local cache; after that,
loads are instant.

  surogate serve Qwen/Qwen3.6-27B
  surogate serve ~/models/qwen3.6-27b-hf/
  surogate serve ~/models/qwen3.6-27b-Q4_K_M.gguf

Serves OpenAI-/Anthropic-compatible HTTP (default), or runs a one-shot
generation with --generate (streams the answer to stdout).

Common engine options (full list: surogate serve --engine-help):
  --host 0.0.0.0 --port 8080     bind address (server mode)
  --max-context N                per-sequence context ceiling
  --kv-capacity N|auto           KV pool size ('auto' = free VRAM minus 1 GiB)
  --max-concurrency N            concurrent requests (server mode, 1-8)
  --spec mtp --draft-tokens 3    speculative decoding
  --kv-dtype bf16|int8           KV cache precision

The engine binary is resolved from, in order:
  1. $SUROGATE_SERVE_BIN / $SUROGATE_ENGINE_CLI_BIN (explicit paths)
  2. the repo build tree (csrc/build-serve/) when running from a checkout
  3. $PATH (surogate-engine / surogate-engine-cli)
Build it from a checkout with: make serve-build
"""


def _repo_root() -> Path | None:
    # surogate/cli/serve.py -> surogate/cli -> surogate -> repo root
    root = Path(__file__).resolve().parent.parent.parent
    return root if (root / "csrc" / "src" / "serve").is_dir() else None


def _resolve_binary(server_mode: bool) -> str | None:
    name = "surogate-engine" if server_mode else "surogate-engine-cli"
    env = os.environ.get("SUROGATE_SERVE_BIN" if server_mode else "SUROGATE_ENGINE_CLI_BIN")
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

    # --generate switches to the one-shot CLI binary; --engine-help passes
    # --help through to the engine so its full option surface stays canonical.
    server_mode = True
    if "--generate" in rest:
        rest = [a for a in rest if a != "--generate"]
        server_mode = False
    if "--engine-help" in rest:
        rest = ["--help" if a == "--engine-help" else a for a in rest]

    binary = _resolve_binary(server_mode)
    if binary is None:
        name = "surogate-engine" if server_mode else "surogate-engine-cli"
        sys.stderr.write(
            f"surogate serve: engine binary '{name}' not found.\n"
            "Build it first:  make serve-build   (from the surogate repo root)\n"
            "or point SUROGATE_SERVE_BIN at an existing binary.\n"
        )
        sys.exit(127)

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
    if model_index is not None:
        from surogate.serve.ingest import ensure_engine_weights

        resolved = ensure_engine_weights(rest[model_index],
                                         echo=lambda m: print(m, file=sys.stderr))
        rest = [*rest[:model_index], str(resolved), *rest[model_index + 1:]]

    os.execv(binary, [binary] + rest)


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "serve"] + sys.argv[1:]
    maybe_exec_serve()
