# Native TTS package provenance

Asset preparation pins `surogate/surogate-ro-tts` at revision
`2bf175b4edc7b3ca7261d80e4d4ad85117c4f0a4` and verifies its native package.
The package supplies the GGML model runtime and includes its runtime and GGML
licenses, build provenance and patch sources. Surogate supplies the configurable
worker, adapter and optional optimized CPU backend.

The native server and Romanian frontend are documented in
[csrc/src/serve/tts/NOTICE.md](../../../csrc/src/serve/tts/NOTICE.md).
Model and voice licensing is separate from the serving implementation; see the
[model card](https://huggingface.co/surogate/surogate-ro-tts).
