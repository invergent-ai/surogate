# Serving engine audit — 2026-09-12

Audited revision: `7f123876`.

This is an engineering backlog and validation record. It covers the launcher and
ingestion, generation and embedding HTTP endpoints, tool/JSON constraints, adapters,
request scheduling, prefix/KV storage, pipeline placement, host offload, vision,
speculation, and their test infrastructure. Shared execution paths and family guards
were reviewed; this is not a claim that every kernel or checkpoint combination has
been independently validated.

No serving implementation was changed during this audit. Broad throughput benchmarks
remain deferred. Audio and server-side Responses storage are intentionally outside
the requested scope.

## Follow-up status — 2026-09-13

A1 and A2 are resolved: embedding token IDs and sequence lengths are validated before
execution on both backends; dimensions and base64 output are honored, and unknown model
names are rejected. Embedding requests may omit `model` to use the running encoder.
Served names for generation and embeddings now preserve the original startup model argument,
matching vLLM's default naming behavior; explicit deployment aliases still override it.

Validation: the embedding request and server option C++ tests pass; 8 HTTP regression cases
pass across CPU and GPU, with the GPU server observed under Compute Sanitizer; 52 relevant
launcher cases pass. The two pre-existing A3 inventory failures were excluded from that
scoped launcher run. Live text and embedding launches both reported the supplied GGUF path
in `/v1/models` and successful responses. The findings below retain the original audit
reproductions. A3 is also resolved: the shared launcher switch inventory now forwards
`--spec-adaptive` for server and one-shot generation, before or after the model argument.
All 54 launcher tests pass, including the previously failing native inventories and native
help checks. A4–A5 and the subsequent backlog remain open.

## Fix first

### A1 — resolved: invalid embedding token IDs can terminate GPU serving

**Reproduced with the local EmbeddingGemma Q8_0 checkpoint on physical GPU 2.**
`POST /v1/embeddings` accepts an integer input array without checking its IDs against
the vocabulary. The GPU encoder checks sequence lengths, then sends those IDs to a
W8 gather that uses them directly as row offsets.

```json
{"model":"embeddinggemma","input":[2147483647]}
```

Under Compute Sanitizer, this produced out-of-bounds global reads in
`embed_gather_w8_kernel`, followed by `cudaErrorIllegalAddress`; the HTTP connection
closed without a response. A valid request in the same process succeeded first.
A fresh device test passed afterward; the failure was confined to the test server process.
The CPU gather checks IDs, but its exception is currently rendered as HTTP 500.

Validate token types, integer range, vocabulary bounds, and sequence lengths before
dispatch. Invalid inputs should produce a request error and leave the encoder usable.
Cover both backends and verify a valid request still succeeds after each rejection.

Evidence: [HTTP parsing](../../serve/encoder/embedding_server.cpp#L51),
[GPU encoder validation](../../serve/encoder/gemma_embedding.cpp#L218),
[unchecked W8 gather](../../serve/ops/kernel/embed_gather.cuh#L142),
[CPU bounds check](../../serve/encoder/cpu/cpu_ops.cpp#L696).

### A2 — resolved: embedding response options are silently ignored

**Reproduced over HTTP in the same isolated GPU process.**

| Request | Actual response |
|---|---|
| `dimensions: 128` | HTTP 200, 768-dimensional vector |
| `encoding_format: "base64"` | HTTP 200, JSON float array |
| `model: "not-the-loaded-model"` | HTTP 200, actual model's vector labeled with the requested name |

The handler reads only `model` and `input`. Implement supported options or reject
them explicitly; establish and validate the served model identity. Length errors
also need HTTP 400 rather than the inference catch-all's HTTP 500.

Evidence: [embedding handler](../../serve/encoder/embedding_server.cpp#L173).

### A3 — resolved: the launcher rejects documented adaptive DFlash

**Reproduced without loading a model.** Both launcher modes reject
`--spec-adaptive` with `ValueError: unknown option: --spec-adaptive`.
The native parsers support it, but the Python option inventory omits it.
`test_native_option_inventory[server]` and `[generate]` fail for this reason.

The switch is now in the shared launcher inventory. The complete launcher suite, including
both native option inventories and help checks, passes.

Evidence: [launcher switches](../../../../surogate/cli/serve.py#L84),
[native server parser](../../serve/serve/serve_options.cpp#L367),
[inventory test](../../../../tests/serve/test_cli.py#L14).

### A4 — P2: model wake waits bypass the request deadline and cancellation

**Confirmed by control-flow review; not exercised with a three-minute wake timeout.**
HTTP handlers call `ensure_awake()` before `GenerationService::prepare()` acquires
the request lifetime. Wake admission has its own 180-second timeout and accepts no
request deadline or disconnect callback. Consequently, `--pending-timeout-ms` does
not cover this wait, and disconnected callers can continue occupying HTTP workers
while waiting for another model to release memory. Wake transitions also run under
the scheduler's common mutex.

Carry one request lifetime through model selection, wake admission, preparation,
and engine admission. Make the wait cancellable and return an attributed timeout.

Evidence: [HTTP ordering](../../serve/serve/http_server.cpp#L754),
[wake loop](../../serve/serve/model_scheduler.cpp#L49),
[180-second default](../../serve/serve/model_scheduler.h#L75),
[request lifetime](../../serve/serve/generation_service.cpp#L372).

### A5 — P2: resolve the remaining numerical and parser test disagreements

All three C++ failures reproduced individually on otherwise idle devices/processes.

| Test | Finding | Follow-up |
|---|---|---|
| `sinfer_head_linear_test` | 14 comparisons fail. Its shared oracle now rounds W8 weights to BF16, while this operation explicitly specifies FP32 dequantization. A temporary diagnostic using its documented FP32 weight contract passes against the unchanged library. | Give the head operation its own correct reference; keep the generic W8 reference for kernels that materialize BF16 weights. |
| `sinfer_kimi_delta_net_test` | Uniform-gate KDA/GDN comparison fails: example output `0.029541` versus `0.0296631`. Its other checks complete; the cause of the cross-operation disagreement remains unresolved. | Compare both implementations against the same independent recurrence, including state storage boundaries; determine whether the kernel or comparison contract needs correction. |
| `sinfer_output_parsers_test` | The Spark test calls text after a complete tool call malformed. The shared parser now preserves trailing answer text, and the Qwen parser test explicitly requires that behavior. | Reconcile the Spark expectation with the shared behavior and cover streaming suffix handling. |

Evidence: [head contract](../../serve/api/ops/head_linear.h#L28),
[head reference](ops/test_head_linear.cpp#L61),
[head kernel](../../serve/ops/kernel/head_linear.cuh#L67),
[KDA comparison](ops/test_kimi_delta_net.cpp#L220),
[Spark expectation](test_output_parsers.cpp#L167),
[shared suffix parsing](../../serve/serve/tool_call_parser.cpp#L235),
[suffix contract test](test_tool_call_parser.cpp#L80).

## Memory and throughput backlog

These are confirmed architectural limits or optimization opportunities. No new
speedup or memory saving was measured in this audit.

| ID | Open item and impact | Evidence / next step |
|---|---|---|
| M1 | **Windowed attention retains full-history KV.** Window masks reduce attention work, but owning attention layers share the same logical page capacity. Old local-attention K/V remains allocated as context grows. | [Cache planning](../../serve/family/impl/state/decoder_state.cpp#L18), [common page allocation](../../serve/core/paged_kv_cache.cpp#L61). Add separate bounded storage/page retirement for windowed layers while preserving positions, prefix reuse, and speculation. Measure actual engine memory, including granularity. |
| M2 | **Common prefixes are not shared across active request lanes.** Admission looks for reuse in idle lanes; the page allocator gives requests exclusive page ownership. Simultaneous branches of the same prefix repeat storage/work. Completed-turn reuse already works. | [Idle-lane selection](../../serve/runtime/engine/concurrent_executor.h#L820), [page allocator](../../serve/core/paged_kv_cache.h#L128). Explore immutable shared prefix pages with reference counts and copy-on-write tails; recurrent models also need compatible state checkpoints. |
| M3 | **Multi-GPU execution uses pipeline stages and host-staged transfers.** Stage boundaries travel device → pinned host memory → device; there is no direct peer transfer path or tensor/expert parallel serving backend. Weight-balanced partitioning does not model measured link bandwidth or stage compute time. Single-request latency still traverses every stage. | [Pipeline transport](../../serve/runtime/engine/pipeline_instance.h#L1), [partition inputs](../../serve/targets/registry.cpp#L480), [partition algorithm](../../serve/runtime/engine/pipeline_partition.h#L12). Probe peer access before proposing P2P, retain the host fallback, and use the machine's NUMA/x8/x16 topology. Evaluate tensor parallelism separately. |
| M4 | **High request concurrency still creates many host threads and admission plans.** The HTTP pool is sized to active plus pending capacity across all models. Each waiting request can keep per-lane plan arrays; lane selection scans the configured lanes. | [HTTP pool](../../serve/serve/http_server.cpp#L149), [lane plans](../../serve/runtime/engine/concurrent_executor.h#L820). Measure thousands of clients, slow consumers, host memory, and admission CPU time; consider bounded asynchronous HTTP handling. This is not the old 128-active-request limit. |
| M5 | **Small expert caches disable wider CPU sharing.** A cache smaller than one layer's experts forces CPU prefill share to zero and narrows the decode band. Wider rounds compute experts on GPU, potentially requiring repeated transfers. | [Expert-cache restrictions](../../serve/family/impl/moe/expert_cache.cpp#L1420). Extend streamed expert batches with CPU/GPU split scheduling, then measure against the existing GPU fallback. |
| M6 | **Some GGUF MLP paths still materialize avoidable intermediates.** K-quant SwiGLU prefill fusion requires equal gate/up formats. Q8_0/IQ4_NL pairs are covered; mixed K-quant pairs use the fallback. The existing combined SwiGLU/down path admits only wide NVFP4. | [Prefill admission](../../serve/ops/linear/ggml/ggml_swiglu.cu#L22), [combined MLP admission](../../serve/ops/wrapper/linear_swiglu_down_add.cpp#L28). Inventory actual checkpoint formats and measure candidate fusions before changing dispatch. |
| M7 | **Embedding requests are serialized across HTTP calls.** GPU batching combines sequences within one request; the dedicated runner executes one request at a time. It has no cross-request batcher or request deadline/cancellation integration. The CPU backend loops over sequences. | [Embedding runner](../../serve/encoder/embedding_server.cpp#L120), [CPU batch loop](../../serve/encoder/cpu/cpu_gemma_embedding.cpp#L339). First fix A1/A2, then add bounded queueing and measure dynamic batching. |
| M8 | **Speculation still has bounded round widths.** DFlash executes at most eight lanes per GPU round and proposes at most 15 tokens; MTP proposes at most five. More requests use additional rounds. Adaptive graph/profile preparation also consumes startup time and memory. | [DFlash lane limit](../../serve/api/types.h#L97), [draft validation](../../serve/family/impl/runtime/layouts_impl.h#L880). Treat larger limits as measured throughput work, not automatic improvements. |

## Coverage and API limits

| Area | Remaining scope |
|---|---|
| JSON and tool schemas | JSON output still rejects unsupported intersections, overlapping `oneOf`, conditionals/general negation, uniqueness, and remote references. Tool schemas are narrower: patterns, string lengths/formats, `multipleOf`, `allOf`, `oneOf`, and assertion siblings beside `$ref`/`anyOf` are rejected. Tool schema validation has a 64-level nesting limit. See [tool validator](../../serve/serve/tool_constraints.cpp#L11), [JSON compiler](../../serve/runtime/contract/constraint.cpp#L159), and the existing [user-facing restrictions](../../../../docs/inference/api.md#structured-output). Constrained speculative sampling is already implemented. |
| Completion API | `n > 1`, completion `best_of > 1`, `echo`, `suffix`, and pre-tokenized `/v1/completions` prompts remain refused. Generation has a separate token-input endpoint. [Parsing](../../serve/serve/openai_schema.cpp#L486), [completion validation](../../serve/serve/openai_schema.cpp#L612). These are compatibility additions, not silent wrong answers. |
| Adapter coverage | Full saved matrices outside embeddings/output heads still require merging. Loading recognizes LoRA/DoRA rather than arbitrary PEFT methods, and requires one `adapter_model.safetensors` file. Spatial LoRA B kernels must be unit-sized. Unsupported tensors are rejected explicitly. [Registry](../../serve/serve/lora_registry.cpp#L113). Existing expert prefill, CPU expert adapters, vision adapters, biases, and rank/alpha overrides are implemented. |
| Additional-model ingestion | The positional model is prepared automatically; additional `--model` entries still require prepared `.sinfer` paths. [Launcher preparation](../../../../surogate/cli/serve.py#L230), [registry guard](../../serve/targets/registry.cpp#L43). Prepare all entries through the same ingestion API. |
| LFM2 quantized memory | LFM2 conversion repacks compatible weights to W8 and dequantizes/re-encodes other GGUF formats into W8. Low-bit source size therefore does not describe loaded memory or prepared artifact size. [Converter](../../../../surogate/serve/convert/lfm2/convert.py#L1). Native GGUF execution here remains useful memory work. |
| GLM sparse context / vision | The native GLM converter still omits the sparse indexer and caps context at `min(max_context, index_topk + index_pool - 1)`. Vision-bearing GLM and Qwen4exp GGUFs are refused by their text recipes. [GLM bound](../../../../surogate/serve/convert/glm5_next/inventory.py#L104), [GLM omissions](../../../../surogate/serve/convert/glm5_next/convert.py#L50), [Qwen4exp](../../../../surogate/serve/convert/qwen4exp/convert.py#L204). GLM already uses a latent KV geometry; the old expanded-KV limitation should not be repeated. |
| Tokenizer coverage | Added tokens with `single_word`, `lstrip`, `rstrip`, or nontrivial normalized matching are rejected. These can prevent a checkpoint from loading even when its tensor architecture is supported. [Validation](../../serve/family/impl/frontend/tokenizer.cpp#L185). |
| Hardware / encoder families | The common generation planner requires SM120. CPU execution is available for embeddings, not full generative inference. Embedding ingestion registers GemmaEmbedding, whose current attention backends require one KV head; arbitrary embedding architectures are not supported. [Generation guard](../../serve/family/impl/runtime/layouts_impl.h#L910), [encoder registry](../../../../surogate/serve/ingest.py#L664), [encoder geometry](../../serve/encoder/gemma_embedding.h#L55). |

## Test coverage and maintenance

The aggregate suite can pass important areas without exercising their full data paths.
This audit had **12 C++ skips and 145 Python skips**. The physical multi-device,
high-concurrency, real-prefix, DFlash, and vision integration cases depend on explicitly
provided artifacts or environment settings. Earlier physical validations are useful
evidence, but this audit does not establish a fresh checkpoint × quantization ×
adapter × offload × speculation × vision × GPU-count matrix.

The GGML operator executable returns success when its external fixtures are missing;
only its built-in zero-weight cases run. That fixture directory was absent in this
audit run. It also permits the shared-only mode to
finish successfully with zero cases. The default suite should generate/require its
fixtures or report the missing coverage as a skip. See
[fixture loop](ops/linear/ggml/test_ggml_k.cpp#L903) and
[success exit](ops/linear/ggml/test_ggml_k.cpp#L987).

Other maintenance items: generated float literals still use six decimal places in
scientific notation, which does not guarantee FP32 round trips
([emitter](../../../../surogate/serve/tools/generate/emit_config.py#L15)); `TODO.md` and
`TODOv2.md` contain stale architecture counts, frontend expectations, and completed
refactoring prerequisites. They should be reconciled with current code rather than
used as the source of truth for remaining features.

The full Qwen3-VL 235B checkpoint remains unvalidated, as the guide already states.
Passing its operator geometry test is not full-checkpoint validation.

## Existing capabilities verified in the review

The old restrictions on 128 active requests, multi-GPU model/sleep combinations,
sequential vision blocking all active text generation, missing completed-turn reuse,
20-candidate filtered sampling, and lack of constrained speculative sampling are
not current open items. The relevant implementations and tests are present.
The W8 GDN snapshot test fixed in `7f123876` also passed in this aggregate run.

Server-side Responses state was deliberately removed. Native audio was explicitly
deferred. Neither is proposed for reintroduction by this audit. Training-only GLM
restrictions and the previously excluded training matmul failures were not tested.

## Validation record

- Built `serve-tests`, `surogate-engine`, and `surogate-engine-cli` in `csrc/build-serve`.
- Python: `pytest -q tests/serve tests/test_serve_contract.py`: **487 passed,
  2 failed, 145 skipped**. Both failures are A3.
- C++: all **146 registered tests**, split by CTest index across physical GPUs
  **0, 1, 4, 6**, two tests per GPU process group: **131 passed, 3 failed,
  12 skipped**. The failures are A5.
- Isolated reruns reproduced each C++ failure. The head projection diagnostic with
  FP32-dequantized reference weights passed without changing the library.
- Embedding HTTP probes used the original local `embeddinggemma-300M-Q8_0.gguf`
  through a temporary prepared artifact and Compute Sanitizer on physical GPU 2.
  They reproduced A1/A2; no external server or network model download was involved.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` was set for GPU jobs. This was validation, not a
  performance comparison; no results were added to `BENCHMARKS.md`.

CTest reproduction, after building all targets:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 SUROGATE_TEST_ORACLE_THREADS=4 \
  ctest --test-dir csrc/build-serve --output-on-failure -j 2 -I 1,,4
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 SUROGATE_TEST_ORACLE_THREADS=4 \
  ctest --test-dir csrc/build-serve --output-on-failure -j 2 -I 2,,4
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 SUROGATE_TEST_ORACLE_THREADS=4 \
  ctest --test-dir csrc/build-serve --output-on-failure -j 2 -I 3,,4
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 SUROGATE_TEST_ORACLE_THREADS=4 \
  ctest --test-dir csrc/build-serve --output-on-failure -j 2 -I 4,,4
```

Small local evidence files are retained under `/tmp/surogate-serving-audit-*`,
including the result JSON, CTest XML/logs, isolated rerun logs, sanitizer output,
and probe sources. Temporary prepared model and pytest data are removed after
the audit; original model files are preserved. Cleanup reclaimed 673,087,488 allocated
bytes (about 642 MiB).

## Suggested execution order

1. Fix embedding request validation and response semantics (A1/A2).
2. Expose adaptive speculation through the launcher (A3).
3. Resolve the three suite disagreements, particularly KDA/GDN arithmetic (A5).
4. Carry cancellation/deadlines through model wake admission (A4).
5. Measure windowed KV storage savings, then active-prefix sharing and pipeline
   transfer/placement improvements (M1–M3).
6. Prioritize schema and model/adapter coverage additions by the workloads that
   currently hit their explicit rejection paths.
