# Tiny GLM-5.3-Flash

Create a deterministic, randomly initialized 2.34M-parameter checkpoint locally:

```bash
python examples/sft/glm/create_dummy.py --output models/dummy-glm-5.3-flash
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

The checkpoint uses the released tensor names, separate Q/K/V convolution weights
and separate expert projections, exercising the real importer. Native tests compare
forward outputs to Transformers and test CUDA derivatives against autograd:

```bash
pytest tests/test_glm5_next_dsl.py -q
CUDA_VISIBLE_DEVICES=0 pytest tests/train/test_glm5_training.py tests/train/test_kda_triton.py -q
```

The shared-policy GRPO check generates tokens, scores them, updates the adapter,
and verifies that serving keeps the same base allocation. It uses the local dummy
checkpoint and needs no reward environment or external inference service:

```bash
CUDA_VISIBLE_DEVICES=0 SUROGATE_SHARED_MODEL="$PWD/models/dummy-glm-5.3-flash" \
  SUROGATE_SHARED_GRAPHS=1 pytest tests/grpo/test_shared_model_gpu.py --slow -q
```

Native-colocate uses the trainer's model and recomputes the prefix for each token.
The frozen reference policy excludes LoRA updates. Use BF16 adapters with the MoE
layers. This fixture fits one GPU; the full checkpoint's base weights must still fit
the selected training topology.

Text training currently requires `sequence_len <= index_topk` (256 here, 2048 in
the released checkpoint), normalized ungrouped routing and tail selection enabled.
Sparse DSA selection, indexer auxiliary training, vision training, MTP training and
sequence-chunked pipeline state carry are not implemented. Full expert-weight
training currently requires `ep_size: 1`. Native exports contain the text-training
weights, not a complete multimodal checkpoint. Packed samples reset
both convolution history and KDA state.

BF16 training now uses the vendored FLA 0.5.2 chunked KDA kernels in
[`surogate/kernels/triton/fla_kda`](../../../surogate/kernels/triton/fla_kda/README.md).
The trainer compiles and caches 16 kernel manifests, and the C++ runtime launches
them directly. No installed `flash-linear-attention` package is required.
The upstream MIT license and pinned revision ship with the package.

The integrated path supports forward/backward, Q/K normalization, GLM's bounded
per-channel decay, packed documents, CUDA graph replay, recomputation, full
training, LoRA and native-colocate. Forward intermediates are recomputed in
backward and scratch storage is reused between layers. Head dimensions 8, 16,
32, 64 and 128 are covered by GPU tests; the dummy model uses 32. FP32 activations
or `doc_masking: false` use the CUDA reference path. Fragmented packing increases
temporary state memory because each document needs its own chunk states.
Native-colocate still recomputes prefixes; persistent recurrent decode state is
not implemented by this training-kernel integration.

To reproduce the numerical and native kernel timing comparison:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_glm_kda.py --length 256 --dim 128
```

On an RTX 5090, `[B,T,H,D]=[1,256,2,128]` gave a maximum BF16 output difference
of `0.00037` and gradient relative RMS differences below `0.7%` against the CUDA
reference. CUDA graph replay measured:

| KDA operation | CUDA reference | Vendored FLA | Speedup |
|---|---:|---:|---:|
| Forward | 4.22 ms | 0.096 ms | 44× |
| Forward + backward | 174.6 ms | 0.339 ms | 514× |

These timings include GPU document metadata preparation and backward recomputation,
exclude compilation, and use about 3.92 MiB of temporary KDA storage. They measure
isolated KDA kernels, not full-model training speedups. Both 10-step CLI examples
and the shared-policy GRPO check also pass with this backend.
