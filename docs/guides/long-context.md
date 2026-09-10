# Long Context Training

Use `long_context: true` to reduce MLP activation memory during training
and policy scoring. It processes tokens in tiles through the up/gate projection,
SwiGLU and down projection. Backward recomputes each tile and immediately consumes
it, accumulating weight and LoRA gradients across tiles.

```yaml
sequence_len: 32768
long_context: true
recompute: true
per_device_train_batch_size: 1
lora_dropout: 0
```

Choose a sequence length within the model's context limit and available memory.
The same training options apply to SFT and GRPO. See the
[local GLM example](../../examples/sft/glm/dummy-long-context.yaml) for a small
checkpoint that can exercise this path on one GPU.

## Dense MLP memory

An MLP processes each token independently. Tiling reduces its intermediate
storage from `O(B*T*intermediate_size)` to
`O(min(B*T, hidden_size)*intermediate_size)`. Inputs, outputs and residual streams
still cover the full sequence. The compiler excludes tile intermediates from
full-sequence activation arenas and saved tensors.

For example, one BF16 intermediate buffer with width 11,008, hidden size 4,096
and batch size one uses:

| Tokens | Ordinary buffer | Tiled buffer |
|---:|---:|---:|
| 2,048 | 43 MiB | 43 MiB |
| 32,768 | 688 MiB | 86 MiB |
| 131,072 | 2,752 MiB | 86 MiB |

The fused up/gate projection is twice this size; SwiGLU has the size shown.
These are individual buffer sizes, not total training memory. Backward also
needs tile gradients and, for GLM, unclamped projection values.

Tiling trades additional launches and backward recomputation for memory.
The cost depends on the model, tile count and hardware. Short inputs use one
tile and therefore gain little memory.

## Supported paths

The compiler recognizes the dense, bias-free up/gate → SwiGLU → down pattern
used by Llama, Qwen3, dense Qwen3.5 blocks, Qwen3-VL and GLM's leading dense
layers. It preserves GLM's asymmetric activation clamps and applies LoRA in
both forward execution and backward recomputation.

GLM also tiles routed and shared expert MLPs in resident BF16 training with
`ep_size: 1`. Routed tiles operate on the permuted tokens, preserving routing,
asymmetric clamps and grouped LoRA arithmetic. Wide expert intermediates shrink
from `O(B*T*top_k*moe_intermediate_size)` to
`O(min(B*T*top_k, hidden_size)*moe_intermediate_size)`. The permutation, routing
and hidden-width input/output buffers still span the full sequence.

Expert tiling with CPU weight streaming or expert parallelism is not covered.
Other models' MoE branches and activations such as NemotronH's ReLU-squared MLP
retain their existing execution paths.

BF16 full training and LoRA are covered by native GPU regression tests, including
packed documents, gradient accumulation and recomputation. Tiling changes GEMM
shapes and gradient reduction order, so BF16 gradients need not be bit-identical.
The tests compare numerical tolerances and check scoring with identical weights.

Use `lora_dropout: 0`: nonzero adapter dropout is rejected because tile
recomputation does not preserve the ordinary path's dropout mask. The tiled
GEMMs do not use the ordinary FP8/FP4 recipe dispatch; those recipes are not
validated by the BF16 checks described here.

`use_cuda_graphs: true` can remain enabled. The runtime uses split execution:
tiled MLP groups run eagerly, and eligible surrounding segments use CUDA graphs.
Capture-unsafe operations can still force eager execution.

## GLM sparse indexer memory

GLM's frozen DSA indexer automatically scores **128 queries at a time** in both
training and prefill. This is independent of `long_context`. Keys are pooled
once; each query tile is scored, selected and expanded before its scratch
buffer is reused. Packed-document boundaries and deterministic tie ordering
are preserved.

The FP32 score buffer shrinks from `B*T*ceil(T/index_kpool)` elements to
`B*min(T,128)*ceil(T/index_kpool)`. At 131,072 tokens, batch size one and pool
size 16, that is **4 GiB → 4 MiB** for scores alone. Key buffers and selected
token indices remain linear in sequence length; the latter also scale with
`index_topk` and the optional tail width. Tiling bounds score storage, but each
query still scans the pools and top-k still uses sorting.

GLM native-colocate stores normalized MLA latents and reconstructs only the
selected K/V projections during decode. Its MLA and indexer histories grow in
128-token pages, returned to a shared pool when a request finishes. Reconstruction applies the original BF16 projection and LoRA
operations to preserve rollout/scoring parity; it trades additional projection
work for lower persistent cache memory. Use `options.glm_rollout_parity = True`
for consistent GLM rollout/scoring arithmetic. Native co-locate enables it automatically.

GLM workspace sizing keeps the shape-based heuristic and graph bounds, without
the legacy multi-GiB minimum and MoE allowance. Graphs containing no standard
FlashAttention operations do not reserve its 1 GiB attention workspace. Saved
MoE buffers are allocated as needed before capture.

On GPU 7 (RTX 5090), the 4,096-token dummy GLM with BF16 LoRA rank 8 and
`recompute: true` used the following device allocations after two updates:

| Configuration | Allocated memory |
|---|---:|
| Before expert tiling and workspace changes | 7,306 MiB |
| Current runtime, `long_context: false` | 4,962 MiB |
| Current runtime, `long_context: true` | 4,852 MiB |

The combined reduction is about 34%. These measurements include library caches
and describe this small checkpoint; they are not transient peak measurements
or full-model capacity estimates. Reproduce the current comparison in separate
processes on an otherwise idle GPU:

```bash
CUDA_VISIBLE_DEVICES=7 python benchmarks/bench_glm_memory.py
CUDA_VISIBLE_DEVICES=7 python benchmarks/bench_glm_memory.py --tiled
```

Model/optimizer state, KDA chunk states, residual streams, token permutations and
sparse attention indices can still be substantial at long context. See
[GLM limitations](../../examples/sft/glm/README.md) and [native-colocate](rl-colocate.md).

## See also

- [Memory](memory.md)
- [Offloading](offloading.md)
- [Config reference](../reference/config.md)
