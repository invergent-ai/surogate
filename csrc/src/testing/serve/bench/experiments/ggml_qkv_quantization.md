# Shared GGUF Q/K/V activation quantization measurement

Measured on 2026-09-12 against engine commit `74195813`. This is an isolated
operator experiment. The engine's dispatch and workspace planning are unchanged.

## Method

The baseline calls the production `linear_launch` three times, as the GGUF
Q/K/V row-projection path does. Each call quantizes the same BF16 activation and
projects one output. The candidate calls `quantize_q8_1_planes_launch` once and
then calls the existing `linear_prequantized_launch` for each output. Both paths
use the same weights, input, output buffers, scratch allocation, and projection
kernels. Graph inspection confirms six kernels versus four.

The benchmark uses finite, deterministic synthetic GGUF blocks. Shape names
describe projection dimensions; these are not full-model measurements. It covers
Q4_K, Q5_K, Q6_K, Q8_0, and IQ4_NL homogeneous Q/K/V weights, plus Q5_K/Q5_K/Q6_K
and Q8_0/Q5_K/Q6_K mixtures. Input token counts are 1, 8, 16, 32, 128, 512, and
2,048. The mixed Q5_K/Q5_K/Q6_K combination matches the first attention layer's
formats in the local TinyLlama Q5_K_M checkpoint.

Hardware is physical GPU 4, RTX 5090 32 GiB, PCI `0000:81:00.0`, PCIe x16,
NUMA node 1. The process binds CPU and memory allocation to node 1 and sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. All weights and inputs are resident on the GPU
before timing; no PCIe transfer is timed. Other GPUs were idle. GPU clocks were
not locked.

Each sweep has 196 cases. Every case verifies finite, bit-for-bit equal Q/K/V
outputs in eager execution and after graph replay. Each cache condition then
takes 31 paired samples, alternating the order of baseline and candidate.
CUDA events measure graph execution. Repeated execution batches 20 operations
per graph; cold execution uses one operation and writes a 256 MiB eviction buffer
before each sample, outside the timed region. The GPU has 96 MiB L2 cache.
Eight launches of both graphs precede timing. The complete sweep is run twice.

The CSV retains each sweep's median baseline and candidate time, the percentage
reduction between those medians, and the median and minimum paired time saving.
An individual negative paired saving records timing noise; it does not get
discarded. Repeated execution can still exceed cache capacity for large shapes.

## Results

Both sweeps completed, with exact equality in all 196 cases per sweep. Every
case's median improved in both cache conditions. The complete results are in
[`ggml_qkv_quantization.csv`](ggml_qkv_quantization.csv).

Representative 2,048-token results below use repeated execution. Times are the
mean of the two sweep medians; the reduction range shows the individual sweeps.

| Projection shape | Q/K/V formats | Baseline | Shared quantization | Time reduction |
| --- | --- | ---: | ---: | ---: |
| TinyLlama | Q5_K / Q5_K / Q6_K | 289.2 µs | 262.5 µs | 9.0–9.5% |
| Qwen3-0.6B | Q8_0 / Q8_0 / Q8_0 | 261.5 µs | 247.5 µs | 5.3–5.4% |
| Llama-8B | Q8_0 / Q8_0 / Q8_0 | 1,443.3 µs | 1,389.5 µs | 3.7% |
| Llama-70B | Q8_0 / Q8_0 / Q8_0 | 4,739.8 µs | 4,655.2 µs | 1.8% |

The corresponding cold runs improved by 8.5–9.7%, 5.4–5.5%, 3.6–3.7%, and
1.9–2.4%, respectively. The repeat run reproduced the direction and approximate
magnitude despite unlocked clocks. Individual long-projection samples sometimes
regressed, as recorded in the minimum paired savings column; those samples were
retained in the results.

This supports sharing activation quantization as a modest latency optimization.
It saves about 14–85 µs per layer for the representative shapes above, while the
three matrix multiplications remain. An engine-level measurement is still needed
to determine the effect on first-token latency. It offers no operator scratch
memory saving.

## Memory

There is no scratch-memory reduction from sharing this quantization. The current
three projections already reuse one allocation sequentially. At 2,048 tokens,
both paths use:

| Projection shape | Hidden width | Q/K/V rows | Scratch per path |
| --- | ---: | --- | ---: |
| TinyLlama | 2,048 | 2,048 / 256 / 256 | 4.5 MiB |
| Qwen3-0.6B | 1,024 | 2,048 / 1,024 / 1,024 | 2.25 MiB |
| Llama-8B | 4,096 | 4,096 / 1,024 / 1,024 | 9 MiB |
| Llama-70B | 8,192 | 8,192 / 1,024 / 1,024 | 18 MiB |

The input, output, weight, and KV-cache storage are unchanged. This experiment
does not measure engine-wide peak VRAM or propose a change to its reservations.

## Reproduce

From the repository root, using a configured serving build with
`SUROGATE_SERVE_BENCH=ON`:

```bash
cmake --build csrc/build-serve --target sinfer_ggml_qkv_quant_bench -j 8
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  numactl --cpunodebind=1 --membind=1 \
  csrc/build-serve/serve_bench/sinfer_ggml_qkv_quant_bench > qkv-first.csv
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  numactl --cpunodebind=1 --membind=1 \
  csrc/build-serve/serve_bench/sinfer_ggml_qkv_quant_bench > qkv-repeat.csv
```

Optional filters: `--shape tinyllama|qwen3_06b|llama_8b|llama_70b`, `--tokens N`,
and `--repeat N`. No checkpoint conversion, download, or model export is needed.

The measured `libsinfer.so` SHA-256 is
`b21f733253063e0f5bcf62c954d26fa1b7e6ce6408e4a7bde161d3f0934d3c96`.

Full-model first-token latency, adapters, input permutations, and pipeline
integration are outside this measurement. Operator percentages must not be
reported as model-level speedups.
