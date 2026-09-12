# Qwen3.5 shared activation quantization measurement

Measured on 2026-09-12 using the local `Qwen_Qwen3.5-9B-Q5_K_M.gguf` checkpoint
and engine commit `6d2de05f`. No serving dispatch or workspace planning changed.

## Result

Sharing activation quantization provides a small projection-latency improvement
for this checkpoint. It does not reduce scratch memory.

At 2,048 tokens, repeated-execution times below average the two sweep medians.
Percentages show the reduction in those projection times, not full-model latency.

| Projection | Main layers with these formats | Baseline | Shared | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| GDN: Q8_0 QKV, Q5_K gate | 24 | 2,537.1 µs | 2,517.1 µs | 0.8% |
| Attention: Q6_K Q/K/gate/V | 4 | 3,149.1 µs | 3,077.5 µs | 2.3% |
| Attention: Q5_K Q/gate/V, Q6_K K | 4 | 2,038.8 µs | 1,962.0 µs | 3.8% |
| MTP attention: Q8_0 Q/K/gate/V | Separate draft layer | 2,382.8 µs | 2,307.7 µs | 3.2% |

The two sweeps reproduced the repeated-execution reductions: GDN 0.76–0.81%,
Q6_K attention 2.26–2.29%, mixed attention 3.75–3.78%, and MTP 3.13–3.17%.
At 128 tokens, the first sweep's corresponding reductions were 2.4%, 2.9%,
3.8%, and 3.8%.

Applying the measured median paired savings to the counts of matching main
layers gives **1.06–1.09 ms per 2,048-token chunk** under repeated execution.
The cold runs give 1.23–1.26 ms using paired savings. These are extrapolations
from representative layers, not measured full-model first-token latency. They
exclude convolution, GDN recurrence, attention, MLP, output head, and scheduling.
The MTP layer is excluded from the main-layer estimate.

Cold samples were noisier. In particular, the second Q6_K attention run's
difference between independent medians was 150.6 µs, while its median paired
saving was 77.8 µs, close to the first run's 75.8 µs. The CSV retains both
summaries and minimum paired savings, including negative minima; the larger
independent-median difference is not used for the extrapolation.

## Coverage and method

The checkpoint contains 32 main text layers and one MTP layer. The extractor
groups layers by projection formats and reads actual weights from layers 0, 3,
7, and 32. It preserves native quantized blocks, separates interleaved query
and gate rows, and restores GDN's grouped value-head order using row
permutations. There is no dequantization or requantization of weights. Input
activations use the benchmark's deterministic synthetic BF16 ramp.

All projections have input width 4,096. Attention's four outputs have widths
4,096 / 1,024 / 4,096 / 1,024; GDN's QKV and gate outputs have widths
8,192 / 4,096. The baseline invokes existing `linear_launch` separately for
each projection. The candidate quantizes once, then invokes the existing
`linear_prequantized_launch` for each projection. All buffers and projection
kernels are shared between the two variants.

GPU and timing conditions match the
[previous Q/K/V experiment](ggml_qkv_quantization.md): physical GPU 4,
RTX 5090 32 GiB, PCI `0000:81:00.0`, PCIe x16, NUMA node 1. CPU and memory
allocation bind to node 1. Other GPUs were idle. CUDA events time captured
graphs with all inputs already on the GPU, 31 paired samples, alternating
variant order, and eight warmups of each graph. Clocks are not locked.
Repeated execution captures 20 operations; cold execution captures one and
writes a 256 MiB eviction buffer before each timed sample.

Two complete sweeps each cover 27 cases: seven token counts
(1, 8, 16, 32, 128, 512, 2,048) for both attention format groups and MTP, plus
six counts (8 through 2,048) for GDN. Single-token GDN is excluded because its
specialized projection/convolution decode route already shares quantization.
Both sweeps pass bit-for-bit finite-output checks in eager execution and after
graph replay. Every median improved in both cache conditions. Graph inspection
confirms eight kernels become five for attention, and four become three for GDN.

Both paths use the same **9 MiB** projection scratch allocation at 2,048 tokens.
The baseline already reuses scratch sequentially. Input, output, weight, and
KV-cache allocations are unchanged; engine-wide peak VRAM is not measured.

The [CSV](qwen35_quantization.csv) retains all 108 cache-condition rows from
the two sweeps. The [fixture manifest](qwen35_quantization_fixtures.json) records
source tensors, format groups, dimensions, and SHA-256 hashes of extracted
blocks. Temporary weight files were removed after measurement.

## Reproduce

From the repository root, with serving benchmarks enabled:

```bash
.venv/bin/python csrc/src/testing/serve/bench/experiments/prepare_qwen35_quant_fixtures.py \
  models/Qwen_Qwen3.5-9B-Q5_K_M.gguf /tmp/surogate-qwen35-quant-fixtures
cmake --build csrc/build-serve --target sinfer_ggml_qkv_quant_bench -j 8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  numactl --cpunodebind=1 --membind=1 \
  csrc/build-serve/serve_bench/sinfer_ggml_qkv_quant_bench \
  --fixtures /tmp/surogate-qwen35-quant-fixtures > qwen35-first.csv
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  numactl --cpunodebind=1 --membind=1 \
  csrc/build-serve/serve_bench/sinfer_ggml_qkv_quant_bench \
  --fixtures /tmp/surogate-qwen35-quant-fixtures > qwen35-repeat.csv
```

The fixture output directory must not already exist. Optional benchmark filters
are `--shape` using a manifest case name, `--tokens N`, and `--repeat N`.
The measured `libsinfer.so` SHA-256 is
`b21f733253063e0f5bcf62c954d26fa1b7e6ce6408e4a7bde161d3f0934d3c96`.

These results cover this 9B GGUF checkpoint's projection formats. They do not
establish speedups for other Qwen3.5 sizes or precisions, adapter execution,
multi-GPU serving, or complete MTP rounds.
