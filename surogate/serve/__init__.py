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
