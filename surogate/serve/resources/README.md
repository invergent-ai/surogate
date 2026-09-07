# Embedded static resources for registered serving targets

Checkpoint-invariant config files (`config.json`, `generation_config.json`,
`preprocessor_config.json`, `video_preprocessor_config.json`) for each
registered converter target, vendored from the canonical Apache-2.0 model
repositories so that GGUF-sourced conversions run fully offline. The
tokenizer and chat template are NOT stored here — they are reconstructed
from the GGUF itself (surogate/serve/gguf/frontend.py).

Provenance: Qwen/Qwen3.6-27B, Qwen/Qwen3.8-27B, Qwen/Qwen3.6-35B-A3B
(Apache-2.0), fetched 2026-08-24. The vendored converter verifies the
pinned files by SHA-256 at conversion time.
