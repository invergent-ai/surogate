# Tiny GLM-5.3-Flash

Create a deterministic, randomly initialized 2.34M-parameter checkpoint locally:

```bash
python examples/sft/glm/create_dummy.py --output models/dummy-glm-5.3-flash --index-topk 32
surogate sft examples/sft/glm/dummy-full.yaml
surogate sft examples/sft/glm/dummy-lora.yaml
```

Run from the repository root. The generator needs Transformers with `glm5_next`
support, PyTorch, tokenizers and safetensors. It downloads no model or tokenizer
and refuses to overwrite a nonempty directory. Training uses one NVIDIA GPU;
select a free device with `CUDA_VISIBLE_DEVICES`.

The model keeps the released architecture's 3-KDA/1-MLA cycle across eight layers,
three leading dense MLPs, four routed experts with top-2 selection, a shared expert,
four residual streams and 20 Sinkhorn iterations. Hidden width is 128, KDA has two
32-wide heads, and MLA has two 64-wide heads. A byte-level tokenizer and tiny vision
tower make the checkpoint loadable with Transformers. The examples train text.
Random weights and the bundled dataset are for implementation checks, not model
quality. This fixture does not establish memory requirements for the 320B model.

The byte tokenizer includes GLM's reasoning/tool control tokens and the released
[chat template](chat_template.jinja), copied from `zai-org/GLM-5.3-Flash-BF16`
revision `a5b45eb41df6402735dedc900be14a42e8d5e538` under its
[MIT license](TEMPLATE_LICENSE). It renders actual tool schemas, calls and
observations. Tool prompts need a larger context than the short SFT examples.

To test a tool call, execution, token-preserving second turn, reward, native
policy scoring and GRPO adapter update:

```bash
CUDA_VISIBLE_DEVICES=7 pytest tests/grpo/test_tool_protocol_gpu.py --slow -q -k glm
```

This test creates its own 2,048-token dummy fixture. Sampling is scripted to
exercise tool use with random weights; logits, log probabilities, loss masks,
gradients and the optimizer update are real. It checks integration, not learned
agent quality. Function-tool support is available across the supported GRPO
families; see [agentic rollouts](../../../docs/guides/rl-colocate.md#agentic-tool-rollouts).

The checkpoint uses the released tensor names, separate Q/K/V convolution weights
and separate expert projections, exercising the real importer. Native tests compare
forward outputs to Transformers and test CUDA derivatives against autograd:

```bash
pytest tests/test_glm5_next_dsl.py -q
CUDA_VISIBLE_DEVICES=0 pytest tests/train/test_glm5_training.py tests/train/test_kda_triton.py tests/train/test_glm_dsa.py -q
```

The shared-policy GRPO check generates tokens, scores them, updates the adapter,
and verifies that serving keeps the same base allocation. It uses the local dummy
checkpoint and needs no reward environment or external inference service:

```bash
CUDA_VISIBLE_DEVICES=0 SUROGATE_SHARED_MODEL="$PWD/models/dummy-glm-5.3-flash" \
  SUROGATE_SHARED_GRAPHS=1 SUROGATE_SHARED_TOKENS=120 \
  pytest tests/grpo/test_shared_model_gpu.py --slow -q
```

The same test file also creates its own random sparse fixture and checks 248-token
rollouts against packed training scores before and after real updates, with one
or two training rows, greedy and seeded sampled generation, CUDA graphs, and
non-default LoRA scaling. These cases need
no `SUROGATE_SHARED_MODEL` setting; select them with `-k glm_long_rollout`.

For the 2,040-token diagnostic, create a separate checkpoint with a larger context:

```bash
python examples/sft/glm/create_dummy.py --output models/dummy-glm-long \
  --index-topk 32 --max-sequence-length 2048
CUDA_VISIBLE_DEVICES=0 SUROGATE_SHARED_MODEL="$PWD/models/dummy-glm-long" \
  SUROGATE_SHARED_GRAPHS=1 SUROGATE_SHARED_BATCH=2 \
  SUROGATE_SHARED_SEQ_LEN=2048 SUROGATE_SHARED_TOKENS=2040 \
  pytest tests/grpo/test_shared_model_gpu.py --slow -q -k generation_scoring
```

On GPU 7 (RTX 5090), the 2,040-token diagnostic matched training log probabilities
within `3.9e-7` before and after adapter updates. This validates the random tiny
checkpoint at that context length; it is not a full-checkpoint throughput or
memory measurement.

Native-colocate uses the trainer's model. GLM prefills each prompt once and keeps
FP32 KDA state, convolution history, pooled indexer keys and MLA KV history across
decode calls. It processes one request at a time, resetting state between requests
and after adapter or base-weight changes. Training and serving share the base and
adapter allocations, including when training uses CUDA graphs.
The frozen reference policy excludes LoRA updates. Use BF16 adapters with the MoE
layers. This fixture fits one GPU; the full checkpoint's base weights must still fit
the selected training topology.

Text training supports sequences longer than `index_topk` (32 with the command
above, 256 by default in the generator, 2048 in the
released checkpoint). Eight native Triton DSA kernels normalize and pool keys,
score and select pools, expand selected token indices, and run sparse attention
forward/backward. They live in
[`surogate/kernels/triton/glm_dsa.py`](../../../surogate/kernels/triton/glm_dsa.py)
and are registered through the JIT manifest compiler. The indexer is frozen, as
in the Transformers reference. Packed samples reset convolution, recurrence and
pooling; sparse indices cannot cross document boundaries. Tests lower top-k to
eight so the small fixture exercises sparse selection in full and LoRA training.

Current limits:

- BF16 model activations, normalized ungrouped routing, and a full indexer on each
  MLA layer. Pool size must be a power of two and divide `index_topk`. Both tail
  selection settings are supported. Context is bounded by the configured maximum
  sequence length and available memory.
- Attention visits selected keys, but indexer scores still need
  `O(B * T * ceil(T / pool_size))` FP32 workspace during training/prefill. This is
  not yet a memory-efficient long-context indexer. Top-k uses sorting rather than
  a tuned radix-selection kernel.
- Native-colocate requires one GPU with resident unquantized base weights and
  BF16 LoRA adapters. Its MLA cache stores expanded per-head K/V; latent compression,
  paged caches, continuous batching and captured decode graphs are not implemented.
  Prefill uses the recurrent FLA kernel, without the training chunk kernel's
  parallelism across tokens.
- Native-colocate automatically uses the same recurrent KDA forward for rollout
  and training/scoring, plus fixed-reduction GEMMs and deterministic forward
  additions. This resolves the previous long-rollout mismatch from BF16 rounding
  changing hard expert/pool selection. It costs training-forward parallelism;
  backward still uses FP32 FLA chunk intermediates and derivatives. Ordinary SFT
  keeps the parallel chunk forward. Direct Python callers must set
  `options.glm_rollout_parity = True` before constructing the trainer, with
  `doc_masking: true`. The GLM rollout/scoring regression tolerance is `1e-5`.
- Indexer auxiliary training, vision training, MTP training and sequence-chunked
  pipeline state carry remain unsupported. Full expert-weight training requires
  `ep_size: 1`. Native exports contain the text-training weights, not a complete
  multimodal checkpoint.

BF16 training now uses the vendored FLA 0.5.2 chunked KDA kernels in
[`surogate/kernels/triton/fla_kda`](../../../surogate/kernels/triton/fla_kda/README.md).
The trainer compiles and caches 19 KDA kernel manifests, and the C++ runtime launches
them directly. No installed `flash-linear-attention` package is required.
The upstream MIT license and pinned revision ship with the package.

The integrated path supports forward/backward, Q/K normalization, GLM's bounded
per-channel decay, packed documents, CUDA graph replay, recomputation, full
training, LoRA and native-colocate. Forward intermediates are recomputed in
backward and scratch storage is reused between layers. Head dimensions 8, 16,
32, 64 and 128 are covered by GPU tests; the dummy model uses 32.
`doc_masking: false` uses the CUDA reference path for training KDA. Isolated FP32
KDA tests also use that reference; the complete DSA model requires BF16 activations.
Fragmented packing increases
temporary state memory because each document needs its own chunk states.

To reproduce the numerical and native kernel timing comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_glm_kda.py --length 256 --dim 128
```

On an RTX 5090, `[B,T,H,D]=[1,256,2,128]` gave a maximum BF16 output difference
of `0.000031` and gradient relative RMS differences below `5e-6` against the CUDA
reference. CUDA graph replay with FP32 chunk intermediates measured:

| KDA operation | CUDA reference | FLA chunk forward | FLA rollout-parity forward |
|---|---:|---:|---:|
| Forward | 4.10 ms | 0.181 ms | 0.258 ms |
| Forward + backward | 174.0 ms | 1.247 ms | 1.324 ms |

These timings include GPU document metadata preparation and backward recomputation,
exclude compilation, and use about 5.67 MiB of temporary KDA storage. They measure
isolated KDA kernels, not full-model training speedups or the additional cost of
fixed-reduction GEMMs. The recurrent forward processes tokens sequentially, so its
cost grows with document length. Both forward modes use the FLA chunk backward.
