# TTS implementation provenance

The C++ Romanian normalization is adapted from `ro_tts/frontend.py`, version
2.2.2, in [invergent-ai/training_tts](https://github.com/invergent-ai/training_tts/tree/3df0c664a440527b6c4f339e275f43f5c4da1736).
It uses the validated defaults with numeric-list boundaries enabled. Optional
research respelling and legal-reference switches are not enabled in serving.
Number spelling is implemented in C++; num2words is not a runtime dependency.

The Romanian sentence-splitting logic is adapted from NVIDIA NeMo, copyright
2023 NVIDIA CORPORATION & AFFILIATES, under Apache-2.0. The original license is
available at <https://www.apache.org/licenses/LICENSE-2.0>.

The model package supplies the native synthesis libraries, based on
[NVIDIA NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp), revision
`07003daa7eefea542076310722ccaa89709ee3c3`, with the release's long-form history
fix. The package includes runtime and GGML licenses, build provenance and patch
sources. Surogate's persistent worker loads those libraries through the
`native_bridge.cpp` adapter. `vendor/magpie_runtime.h` preserves the Apache-2.0
header from that pinned runtime. This adapter uses its private C++ ABI, so the
worker restricts loading to the supported runtime and GGML-base fingerprints.
The HTTP server communicates with the worker through pipes.

The optional optimized CPU backend uses GGML revision
`c03b4e2bcece5134827881af90242086daf75be5`, under the MIT license reproduced in
`vendor/GGML-LICENSE`. `patches/ggml-cpu.patch` preserves the NVIDIA runtime's
GGML ABI additions and adds Surogate's AVX-512 FP32 projection and FP16 codec
matrix kernels. The model package retains its own runtime and model licenses.

The native frontend matches the frozen token sequences for all 301 Romanian
stress-benchmark inputs. The 30 foreign-language rows remain in the fixture for
provenance and are outside this Romanian endpoint's tokenizer contract.
