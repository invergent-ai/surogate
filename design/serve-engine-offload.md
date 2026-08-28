# CPU offload for the serving engine — what to take from FreeToken

Read of `study/FreeToken` (2026-08-28), against an engine that already has a
paged KV cache, CUDA-graph-captured decode rounds and NVFP4/FP8 resident
weights. The question was which of its offloading machinery is worth reusing.

## What FreeToken actually offloads

**Routed MoE expert weights, and nothing else.** Not KV, not activations, not
the router. `engine/engine.py:460` loads everything except `experts` resident;
`models/deepseek_v4/moe.py:128` keeps the gate, the shared experts and the
hash table on device. There is no host KV tier anywhere in `kvcache/` — so if
we ever want KV offload we are building it, not porting it.

The unit is one **(layer, expert)** row, decomposed into *banks* — one parallel
tensor per component of the quantised expert (`moe/offload_cache.py:36`,
`_BANK_SCHEMAS`: 2 banks for bf16, 4 for fp8-block, 6 for native NVFP4). Host
side is one allocation per layer; GPU side is one flat slot pool shared by all
layers, keyed `layer_id * num_experts + expert`.

## The one idea that is genuinely load-bearing

**A device-side gather kernel that dereferences pinned host pointers over PCIe
from inside a CUDA kernel** — `kernel/csrc/jit/fast_index_copy.cuh:487`,
`fast_index_copy_multi`. Not `cudaMemcpyAsync`. The row count comes from a
device `int64[1]` (`p.valid_length[0]`), the indices are device tensors, and
the host pointers are UVA aliases of pinned memory.

That is what makes the whole decode step **graph-capturable**: a
`cudaMemcpyAsync`'s byte count is a host argument fixed at capture time, so a
variable-count fetch cannot live inside a captured graph. Ours are captured
too, so this is the primitive we would need, and it generalises past experts to
"gather K variable rows from host into a GPU pool inside a captured graph".

Measured cost in-tree: **~31 GB/s over PCIe against ~3 TB/s HBM**
(`kernel/fast_index_copy.py:166`). A miss is ~100x the per-byte cost of a hit,
which is the entire cost model.

## Worth taking, in order

1. **`fast_index_copy_multi`** — ~100 lines, self-contained. Its single-bank
   sibling uses `ld.global.L1::no_allocate` / `st.global.wt` so streamed bytes
   do not pollute L1/L2 (`fast_index_copy.cuh:36`); the multi-bank one does
   not. Free win if we port it with those hints.
2. **Pin-after-fill** (`moe/host_banks.py:108`). Allocate a lazy anonymous
   `mmap`, fill it, *then* `cudaHostRegister`. Registering first faults and
   zero-fills every page and that work is immediately overwritten — they
   measured ~47 s wasted on 137 GiB. Also: `cudaHostRegister` is
   driver-serialised, so drain it from one background thread and load time
   becomes `max(read, settle)` (`PinPipeline`, line 284).
3. **The graph-safe copy descriptor** (`offload_cache.py:354`,
   `_build_copy_plan`): per-bank `(dst_base, src_base_per_layer, row_bytes)`
   precomputed once into device int64 tensors at fixed addresses, so layer
   selection is a static index per graph node. This maps onto our paged-cache
   descriptor style directly.
4. **The flat-id slot pool** — `id = layer * num_experts + expert` collapses
   the pair to one integer, so eviction needs no decode and "does this slot
   belong to layer L" is a range check.
5. **`decode_routing_stats`** (`offload_cache.py:948`) computes an
   `oracle_hit_at_slots` upper bound from the observed routing distribution.
   Before building any policy past LRU, this says whether one is worth it.

## Two hazards they paid for and wrote down

- **`cudaMemcpyBatchAsync` degrades to a synchronous copy** when a batch mixes
  large entries with sub-256 KB ones on registered host memory (CUDA 13.0,
  H100, bisected). One 5-22 KB entry beside a large one blocks the calling
  thread for the whole transfer. Cost when they hit it: **−22 % end to end** on
  gpt-oss at 2048 tokens (`offload_cache.py:17`). Keep every entry >= 256 KB.
- **A spin-wait kernel for a GPU-CPU handshake pins reported utilisation at
  99 %**, and power governors respond by clamping CPU frequency — a net decode
  regression. Use `cuStreamWriteValue64` / `cuStreamWaitValue64` front-end
  memops instead (`moe/cpu_executor.py:41`). Their host-func alternative costs
  ~30-50 us per call, twice per MoE layer per step, ~6 ms/step at 75 layers.

## What not to take

- **The CPU-compute path** (`csrc/cpu_moe/cpu_moe_ext.cpp`, 107 KB of AVX-512
  GEMV plus coordinator, flag protocol, watchdog, core pinning). It only pays
  when host RAM bandwidth exceeds ~2x PCIe — consumer boxes with narrow PCIe.
  Extract the handshake, not the kernels.
- **The LRU itself** is an external pinned dependency (`flashlib==0.3.0`), not
  vendored. The in-repo hybrid Triton mirror (`moe/offload_kernels.py:291`) is
  a readable single-block reconstruction: argmin over a `usage` timestamp,
  active slots protected by masking to `INT64_MAX`, three phases. Budget for
  writing it.
- **Their serialised decode gather is the design's ceiling, not its
  recommendation**: `ensure_experts -> copy_missing -> GEMM` on one stream, so
  every miss is a full PCIe stall in front of the layer's GEMM. There is no
  cross-layer prefetch and no routing prediction anywhere — even for DSV4's
  hash router, where the routing is token-id-determined and therefore knowable
  arbitrarily far ahead. Multi-stream capture with event edges would let layer
  L's GEMM overlap layer L+1's speculative gather. They did not try it.

## Where this leaves us

Nothing here is needed for the models on the board. The 35B fits one 5090 with
8.4 GB to spare, and the 27B and below are resident. Offload is for the 122B
and 397B tiers, and when we get there the shape of the work is: the gather
kernel (item 1), the pinned bank lifecycle (item 2), the graph-safe descriptor
(item 3), and an LRU we write ourselves. The bandwidth-matched split
(`offload_kernels.py:345`, Q16 fixed point computed in-kernel because the miss
count only exists device-side under capture) is the piece to copy if we ever
want two execution resources sharing one step.
