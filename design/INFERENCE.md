# Inference engine — progress log

Running record of the serving-engine work so it can be picked up cold. Newest entries at the
bottom of each section; the plan itself is `design/serve-engine-plan.md`, the Flash-Next design
`design/serve-engine-flash-next.md`, parked items `design/serve-engine-backlog.md`, the board of
record `surogate/serve/BENCHMARKS.md`.

## Objective (2026-08-28)

Serve `models/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` (111 GB, GGUF arch
`qwen4exp`) on one RTX 5090 with CPU offloading (FreeToken/llama.cpp style) and on 8 × 5090
with pipeline and expert parallelism (vLLM style, no P2P/NVLink on consumer cards). The core
stays model- and SM-agnostic: every new architecture is a target on top of shared contracts.

Phase 1 = onboard the model on the qwen3_6 family runtime with simple expert streaming;
phase 2 = FreeToken hybrid (GPU slot cache + CPU expert compute + bandwidth-matched split);
phase 3 = PP across 8 GPUs; phase 4 = EP measured against PP.

## Status

| step | state | evidence |
|---|---|---|
| Model contract (hc, PLE, GDN sigmoid gate, MoE, indexer) | done | `design/serve-engine-flash-next.md` §5; llama.cpp `study/llama.cpp-master/src/models/qwen4exp.cpp` is the oracle |
| GGUF-native converter `tools/convert/qwen4exp/` | done | commit 81d818d9; W8 experts, un-tiled GDN V heads, q/k norms stored HF-style (γ−1) |
| Artifact `/home/densemax2/work/models/ninfer/qwen3_8_flash_next.ninfer` | done, verified | 163.1 GB in 1046 s; scratchpad `verify_flash_artifact.py`: W8 repacks and the 28.8 GB PLE table bit-exact vs the GGUF, requantised gate/up ≤ 5.4e-3 rel-L2, VERIFY_DONE bad=0 |
| llama.cpp baseline | done | BENCHMARKS.md: 8×5090 39.3 tok/s @1 / 28.8 @32; 1×5090 CPU-MoE 7.1 / 16.3 |
| Sigmoid-gated RMSNorm, W8 dispatch arms (13312/16384 × 2560, 2560 × 6144) | done | commit d2174926 |
| Sparse-MoE geometry from the weights (stage A) | done | commit 13a08634 |
| Sparse-MoE kernels instantiated per geometry, 512/10/640/2560 compiled (stage B) | done, 35B probe clean | see 2026-08-28 entry below |
| BF16 cuBLASLt GEMM route (`ops/linear/bf16/bf16_cublaslt.*`) | written, not yet built | for hc down/up/inject, PLE key/value, GDN a_b at 96×2560 |
| Hyper-connection op (`api/ops/hyper_connection.h`, `ops/hyper_connection/`) | written, not yet built | mix / combine / broadcast_streams |
| PLE op (`api/ops/ngram_ple.h`, `ops/ngram_ple/`) | in progress | device-side hash, IQ4_NL row gather from pinned host, group norms, gate, dilated conv with per-slot state |
| qwen3_6 family: residual-width trait + norm/embed/final/prologue hooks | not started | three layer loops: `run_layers`, `mixed_chunk_multi`, `mixed_graph_window` |
| PLE state pool (per-slot conv history [10240,9] + token history [2]) | not started | mirror `core/linear_attention_state.*`; 7 slot-lifecycle sites in `program_impl.h` / `text_context_impl.h` |
| Target `targets/qwen4exp/` (package, bindings, variant, registry) | not started | model_id `qwen3.8-flash-next`, weights_id `w8-hc-v1`, target key `qwen4exp` |
| Expert streaming v0 (pinned host bank, per-layer coalesced H2D of touched experts) | not started | bind experts + PLE table ValidateOnly, `Reader::read_direct` into pinned memory |
| Parity vs llama.cpp, first throughput row | not started | |

## Decisions that shape the code

- Experts stay W8G32 in the artifact: regrouping K-quants into the kernels' group-64 formats
  measures 11–12 % rel-L2 error; W8 0.55 %; Q8_0/Q4_0/Q5_0/IQ4_NL repack bit-exactly.
- Hyper-connections live inside the target Variant. The family's `Variant` already owns every
  residual read/write (attention/GDN projections, `gdn_norm_control_projection`,
  `post_mixer`) and the arena scopes per mixer/MLP, so the family only needs a residual-width
  trait (`TextConfig::residual`, default `hidden`) at the residual planes plus hooks with
  bit-identical defaults for the five existing targets: attention-side norm, post-mixer norm,
  embed→residual, final residual→hidden, per-layer prologue (PLE).
- Flash-Next projections compose plain W8 `ops::linear` + `extract_bf16_columns` +
  `causal_conv1d_silu_snapshot` instead of extending the geometry-specialised fused wrappers
  (`attn_input_proj`, `gdn_input_proj`, `linear_add`, the BF16 GDN gating family).
  `gdn_input_projection_record` is speculative-only (no MTP/DFlash for this model) and is
  refused.
- BF16 dense problems (hc, PLE, a_b) run through cuBLASLt with cached plans; prewarm and
  prepare every shape before stream capture.
- Norm convention: GGUF gammas are folded (1+w). Only the attention q/k norms are stored as
  HF-style w because the family applies them with `unit_offset=true`; everything else is
  consumed with the folded gamma (`unit_offset=false`, or the op's own FP32 gamma).
- Sparse MoE: kernel bodies are `*_body.inc` files included once per geometry namespace
  (`geometry_qwen36`, `geometry_flash_next`); the wrapper derives the geometry from the router
  and shared-down shapes plus `SparseMoeWeights::experts_per_token` and refuses anything
  unregistered. Q5/Q6 routed-down paths need intermediate 512 (eight 64-wide groups per row)
  and throw for other geometries; Flash-Next uses W8+W8.
- PLE hash runs on device (ids are device tensors inside captured graphs); per-column metadata
  (segment begin, state slot, segment-last flag) is staged like the family's other round
  inputs. The table is read zero-copy from pinned host memory (52 GB/s measured; 1.4 KB per
  token).
- Expert bank and PLE table are bound ValidateOnly and read with `Reader::read_direct` into
  pinned host memory at load: the `Reader` dies after construction.

## How to run / verify

- Convert: `python -m surogate.serve.tools.convert.qwen4exp.convert --gguf <shard1> --frontend
  models/Qwen3.8-Flash-Next-frontend --out <path>.ninfer --device cuda` (GPU 7 was used).
- Verify an artifact against the GGUF: scratchpad `verify_flash_artifact.py` (decode objects,
  compare with gguf-py dequantisation using the converter's algebra; PLE table byte-compare).
- Build the engine: `cmake --build csrc/build-serve --parallel 32 --target surogate-engine`
  (never while an engine process is live: mmap SIGBUS).
- 35B regression probe: scratchpad `probe_35b.sh` (coherence + loadgen, GPU 1, port 8898).
- llama.cpp oracle: `study/llama.cpp-master/build/bin/llama-server` (readiness = `/health`).

## Log

### 2026-08-28

- Converter finished and verified; q/k norms rewritten in place to the HF convention
  (`patch_qk_norms.py`, 24 objects, max error 0) and the convention baked into `convert.py`.
- Disk: purged pip/uv/vLLM/cpptools caches and the launcher's artifact cache, removed the
  superseded HF checkpoints (27B-FP8, both 35B NVFP4 mirrors, five third-party 4B quants);
  242 GB free. Candidates left for the owner: HF `Qwen3.6-35B-A3B` BF16 (67 GB, only needed
  to reconvert the parked 35B), `Qwen3.5-9B` (19 GB), `ro-models/.local/artifacts/ro2/train/tmp_*`
  (≈86 GB of `tmp_` training outputs), `surogate-ro-model/.git` (28 GB), four sibling `.venv`s
  (43 GB), `actions-runner/_work` (13 GB), `LTX-2` weights (52 GB), modelscope datasets (13 GB).
- Sparse MoE stage A (geometry from weights) and stage B (per-geometry instantiation with the
  literal fixes: router dot loop, 9-warp block sizes, `expert * 2*intermediate` rows, top-k
  template, small-T S1 warps and S2 shared-memory batching, select/scan/gather/reduce launch
  shapes, W8 scale staging at non-multiple-of-8 group counts, SM count from the device) both
  build; 35B probe on the new kernels: coherent, 0 fatals, 1,147 tok/s at 32 users /
  `--max-num-seqs 32` (board row is 100 users / 128 lanes / chunk 4096: 1,942).
- Written but unbuilt: BF16 cuBLASLt route, hyper-connection op.
