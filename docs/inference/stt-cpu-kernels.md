# STT CPU kernel measurements

The optimized FP32 path is now the default for CPU serving with MKL-enabled
LibTorch. `--cpu-kernels reference` retains the original implementation.
The earlier oneDNN linear rewrite has been replaced; numerical tolerances
were not relaxed to enable this path.

## Changes

- Combine the query/key/value projections for full attention, and key/value
  projections for streaming. Single-frame GEMV keeps its original reduction.
- Reuse projected position encodings. Full segments and streaming chunks have
  separate caches, each capped at 32 MiB; longer shapes run without caching.
- Use a strided view for the retained relative-attention columns, removing the
  padded temporary and its copy. Frontend ReLU also reuses its output buffer.
- Pack feed-forward output weights through LibTorch's MKL operator when a
  shape recurs. Each matrix retains one packed shape. Occasional different
  lengths use ordinary GEMM; replacement requires three observations. Small
  shapes, noncontiguous inputs and incompatible builds keep the original path.
- Cache language-model backoff expansions inside a transcription request.
  The cache is bounded to four times the beam width. Vocabulary expansion,
  pruning, score arithmetic and tie ordering remain unchanged.

Packed layouts are tied to the input row count; Surogate also keys them by
compute-thread count and immutable weight identity. The operator falls back
to ordinary linear evaluation for an incompatible shape. See the
[PyTorch MKL implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/mkldnn/Linear.cpp#L353).
Small-work dispatch is guarded because packed and unpacked MKL kernels can
use different reductions. These guards are exercised at multiple thread counts.

The caches add memory for combined attention weights, retained packed weights
and position tensors. They do not grow with the number of requests. No model
quantization, activation approximation or weight update is involved.

## Measured latency

September 18, 2026; AMD EPYC 9124; LibTorch 2.13.0+cu130; four compute threads
pinned to physical cores 16–19 on one socket. The host had other activity.
Backend order was reference/optimized, then optimized/reference.

| Workload | Reference | Optimized | Less processing time |
|---|---:|---:|---:|
| Offline, 4.92-second recording | 114.3 ms | 97.8 ms | 14.5% |
| Offline, 20.20-second recording | 397.5 ms | 336.1 ms | 15.4% |
| Offline, varied lengths, mean per clip | 98.6 ms | 94.6 ms | 4.0% |
| Streaming, 4.92-second recording | 678.5 ms | 632.3 ms | 6.8% |
| Streaming, 20.20-second recording | 2258.9 ms | 2030.3 ms | 10.1% |

Repeated offline values are medians of 12 warm runs per backend. The varied
workload uses 21 different Romanian recordings across three voices, twice per
backend, after two warm-up requests. Its smaller gain is an important limit:
repeated-length measurements benefit more from cached work.

Streaming values are medians of six warm runs per backend, feeding 160 ms
packets without real-time arrival delays. They include VAD, partial decoding,
and full-segment finalization, and exclude stream construction. Offline values
include mel extraction, encoder, CTC and language-model decoding. Both exclude
audio-file decoding and HTTP overhead. These are processing benchmarks, not
first-request latency or a guarantee for another CPU.

## Correctness

All checks below compare optimized execution with the original FP32 path:

- **903 recordings:** all 301 Romanian stress sentences in Doina, Tudor and
  Radu. Encoder and CTC tensors are bit-identical; both CTC+LM and TDT
  transcripts match exactly.
- **57 streaming recordings, 263 chunks:** full-segment and chunk tensors are
  bit-identical; every TDT partial and final decoder transcript agrees.
- **240 boundary cases:** silence, low-amplitude audio and noise, from 10 ms to
  2.24 seconds, with both models at 1/2/4/8 threads. All tensors and transcripts
  match exactly, including 156 streaming chunks.
- **1,266 operation cases:** relative-shift aliasing and arithmetic, packed
  matrix evaluation, changing weights/shapes/thread counts, noncontiguous
  inputs, optional bias and small-matrix fallbacks.

Integration checks also pass: 12 real-model HTTP tests with automatic and
reference kernels, 644 Python serving tests (218 skipped), and 168 native
serving tests (23 skipped because their optional fixtures were absent).

The model comparison retains the original per-element limits and relative-RMS
threshold of `1e-5`, reports bit equality separately, and now fails if either
numerical check fails. Transcript equality means scoring those recordings with
either text normalization gives the same WER. This is not a new accuracy
benchmark on independent human speech or a guarantee of bit equality on every
CPU/library version.

## Reproduce

```bash
cmake --build csrc/build-serve --target \
  surogate-stt speech-cpu-bench test_speech_cpu_ops test_speech_cpu_kernels
csrc/build-serve/test_speech_cpu_ops

# MODEL is a prepared directory containing speech.json and acoustic.safetensors.
SUROGATE_CPU_REFERENCE=1 taskset -c 16-19 \
  csrc/build-serve/speech-cpu-bench "$MODEL" 4 7 short.wav long.wav
taskset -c 16-19 \
  csrc/build-serve/speech-cpu-bench "$MODEL" 4 7 short.wav long.wav

# For the streaming model, measure packet processing and finalization:
SUROGATE_CPU_STREAMING=1 taskset -c 16-19 \
  csrc/build-serve/speech-cpu-bench "$STREAMING_MODEL" 4 4 speech.wav

# Each manifest entry contains id, voice and a local audio path.
csrc/build-serve/test_speech_cpu_kernels "$MODEL" cases.json report.jsonl 4
```

Build with `SUROGATE_SERVE_TESTS=ON` to expose the test targets. Choose an
affinity appropriate to the host. `--threads N` remains the public serving
parameter; `--cpu-kernels reference` and `optimized` allow direct comparisons.
