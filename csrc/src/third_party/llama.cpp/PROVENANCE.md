# llama.cpp, vendored

`surogate quantize` produces a GGUF from a checkpoint we trained, and the quantisation
arithmetic is llama.cpp's rather than ours — deliberately, because every published GGUF was
made with these encoders and an encoder of our own would have to match `make_qkx2_quants` and
the per-tensor mixture in `llama-quant.cpp` closely enough not to be quietly worse than the
file a user could have downloaded instead.

Until 2026-09-05 that dependency was a path: the command looked for `study/llama.cpp-master`,
a 459 MB clone that git did not track, was not pinned, and did not exist on any machine but
this one. A user who installed the wheel got a command that failed on first use, and two
machines could produce different weights from the same checkpoint with nothing to say why.

## What is here

| upstream | why |
|---|---|
| `ggml/` (CPU backend only, plus the loose sources and headers) | the tensor library |
| `src/` | libllama: the model loader and `llama_model_quantize` |
| `common/` | `llama-common`, which `llama-quantize-impl` links |
| `vendor/` | upstream's own third party, which `common` links (`nlohmann`, `sheredom`, `cpp-httplib`) |
| `tools/quantize/` | the program itself |
| `cmake/`, `ggml/cmake/` | the helpers the subdirectories call (`llama_add_compile_flags`, `license_add_file`, build-info) |
| `include/`, `ggml/include/` | the public headers |

**Not here**, and the reason our own `CMakeLists.txt` exists rather than upstream's root: the
server, the UI, the app, the tests, the examples, the pocs, and seventeen of the eighteen
tools. Every accelerator backend is excluded too — CUDA, Metal, Vulkan, SYCL, OpenCL, BLAS,
RPC, Hexagon and the rest. `llama-quantize` reads a GGUF and writes a GGUF; it is CPU-only by
nature, which is also why upstream can ship prebuilt CPU archives of it.

## The revision

    commit  163a40796f0ebaae246325f8d2e15028b413fa9d
    describe b10797-10-g163a40796
    date    2026-09-04

`CMakeLists.txt` sets `LLAMA_BUILD_NUMBER` and `LLAMA_BUILD_COMMIT` to match, because upstream
reads those out of git and a vendored copy has none — without them `llama-quantize --version`
cannot say which arithmetic produced a file, which is the whole point of pinning.

## Updating

Re-copy the directories in the table from a newer checkout, refresh the three revision facts
above and the two CMake variables, then rebuild and re-run the gate: quantise Qwen3.5-0.8B to
`q4_k_m` and confirm the served perplexity is unchanged (2026-09-05: **14.9559** against
llama-perplexity's 14.9713 on the same file). A quantiser that changes its arithmetic changes
every artifact made after it, and that number is what would notice.

## Licence

MIT, upstream's `LICENSE` beside this file. Permission to take llama.cpp code was given by the
owner on 2026-09-02.
