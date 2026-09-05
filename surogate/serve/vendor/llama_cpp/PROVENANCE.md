# llama.cpp's HF converter, vendored

A K-quant takes two passes. This is the first: `convert_hf_to_gguf.py` reads a Hugging Face
checkpoint and writes a BF16 GGUF, because it holds the per-architecture tensor mapping and
the tokenizer. `llama-quantize` — vendored at `csrc/src/third_party/llama.cpp` — then reads
that and applies the type mixture.

`gguf-py/` comes with it and is not optional. The converter puts it on `sys.path` ahead of
anything installed, and it has to: the pip `gguf` package is a release of its own and lags the
converter, so the pair went out of step immediately — a checkpoint whose architecture the
pinned converter knows failed with `MODEL_ARCH has no attribute 'DFLASH'` against the
installed one. Converter and library are one revision or they are broken.

Taken from the same revision as the C++ half; see that tree's `PROVENANCE.md` for the commit,
the reason the dependency is vendored rather than pointed at, and how to update.

MIT, upstream's `LICENSE` beside the C++ half.
