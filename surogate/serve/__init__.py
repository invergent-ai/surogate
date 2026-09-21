# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# surogate.serve — the serving-engine integration layer (design/serve-engine-plan.md).
#
#   ingest.py   input classification (safetensors dir / HF repo id / GGUF),
#               transparent conversion into the internal cache, atomic publish.
#   gguf/       GGUF support: bridge.py (container read, dequant, HF-dir
#               synthesis) + one module per model family that needs
#               llama.cpp export-transform inversions (qwen35.py, ...).
#
# The C++ engine itself is vendored at csrc/src/serve/sinfer and built by
# `make serve-build`; surogate/cli/serve.py is the thin exec wrapper.

# `gguf` is llama.cpp's Python library, and the conversion paths below import it. It has to be
# the copy that matches the converter this build pinned: the pip release lags it, and a
# checkpoint whose architecture the pinned converter knows can fail against an older installed
# `gguf`. The build installs the pair into `_llama_cpp/`, so that copy goes on `sys.path` here,
# ahead of anything in site-packages, where every import under `surogate.serve` passes through.
#
# Without this an installed wheel serves no GGUF at all: `surogate serve model.gguf` died with
# `ModuleNotFoundError: No module named 'gguf'`, which a development checkout never sees
# because a pip `gguf` happens to be in its virtualenv. `SUROGATE_LLAMA_CPP` still points the
# pair at a checkout of your own, and a source tree that has configured cmake but installed
# nothing falls back to what the build fetched.
def _vendored_gguf_py() -> str | None:
    import glob
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    roots = []
    explicit = os.environ.get("SUROGATE_LLAMA_CPP")
    if explicit:
        roots.append(os.path.abspath(explicit))
    roots.append(os.path.join(here, "_llama_cpp"))
    repo_root = os.path.dirname(os.path.dirname(here))
    roots.extend(sorted(glob.glob(os.path.join(repo_root, "build", "*", "_deps", "llama_cpp-src"))))
    for root in roots:
        candidate = os.path.join(root, "gguf-py")
        if os.path.isdir(os.path.join(candidate, "gguf")):
            return candidate
    return None


def _install_vendored_gguf() -> None:
    import sys

    path = _vendored_gguf_py()
    if path and path not in sys.path:
        sys.path.insert(0, path)


_install_vendored_gguf()
