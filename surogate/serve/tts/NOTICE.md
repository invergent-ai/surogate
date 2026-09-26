# Native TTS package provenance

Asset preparation pins `surogate/amami-357m-ro` at revision
`5393e9bf69ec08ef2c89bf387bf3791c446fb26a` and verifies its native CPU package
(`cpu/`), or for a GPU its GPU variant (`gpu/`) from the same revision.
The package supplies the GGML model runtime and includes its runtime and GGML
licenses, build provenance and patch sources. Surogate supplies the configurable
worker, adapter and optional optimized CPU backend.

The native server and Romanian frontend are documented in
[csrc/src/serve/tts/NOTICE.md](../../../csrc/src/serve/tts/NOTICE.md).
Model and voice licensing is separate from the serving implementation; see the
[model card](https://huggingface.co/surogate/amami-357m-ro).
