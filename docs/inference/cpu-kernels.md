# CPU speech kernel validation

Validation date: September 18, 2026. Host: AMD EPYC 9124, two sockets, 32
physical cores in total. TTS timings use four physical cores on one socket,
with four generator threads and four codec threads sharing those cores.
The machine is shared; timings include background load and are not a capacity
guarantee for an otherwise idle production host.

## Kernel selection

| Service | Default (`auto`) | Explicit `optimized` |
|---|---|---|
| TTS | AVX-512 backend when available; otherwise the package backend | Requires the compatible AVX-512 backend |
| STT | Original LibTorch CPU path | Experimental oneDNN linear kernels and channels-last frontend convolutions |

TTS reuses FP32 vectors across projection outputs and FP16 conversions across
2×3 codec matrix tiles. It preserves the original accumulation order, including
scalar tails for irregular convolution dimensions. STT keeps FP32 and uses
cached oneDNN weight layouts; it retains the original SiLU activation.

## TTS measurements

The HTTP comparison uses benchmark sentences `001` and `293` with Doina,
Tudor and Radu: six distinct recordings, approximately 4.8–5.6 seconds for
the short sentence and 20 seconds for the long passage. Each backend runs
each case twice, in reference/optimized then optimized/reference order.
Model startup and server warm-up are excluded; complete HTTP request latency
is included. Throughput is total generated audio duration divided by total
request time. Real-time factor (RTF) is its inverse; an RTF above 1 is slower
than real time.

| Backend | Throughput (audio-seconds/second) | RTF |
|---|---:|---:|
| Original package kernels | 0.403 | 2.481 |
| Optimized AVX-512 kernels | **0.461** | **2.168** |

Across 24 HTTP requests, the optimized backend delivered **14.4% more
throughput**, or **12.6% less generation time** for the same audio. A ten-second
clip therefore corresponds to approximately 21.7 seconds of processing at this
measured average. Per-voice optimized throughput was 0.458 for Doina, 0.462 for
Tudor and 0.464 for Radu. All 24 WAVs matched their frozen reference hashes.

The standalone short-clip probe averaged 8.36 seconds to generate 4.92 seconds
of audio (0.59 audio-seconds per second). That is a separate three-run,
single-sentence measurement, not the broader HTTP average.

### Correctness

- 3,520 matrix cases match GGML's original dot-product kernels bit-for-bit.
  These cover irregular reduction lengths, odd matrix edges, padded rows,
  single and paired conditioning inputs, and 1/2/4/8 threads.
- HTTP controls compare complete WAV hashes against the accepted native
  release, including long-form history and switching all three voices.
- The separately exported CV Male and CV Female voices also retain their
  exact reference WAVs through the standalone build and automatic kernel selection.
- All 301 Romanian stress inputs retain their frozen frontend token sequences.
- The model weights, codec weights and decoding settings are unchanged. This
  validation does not establish a new 331-sentence WER or unseen-voice score.

The native model package is pinned to HF revision
`2bf175b4edc7b3ca7261d80e4d4ad85117c4f0a4`. The optimized backend uses GGML
`c03b4e2bcece5134827881af90242086daf75be5` with the model's ABI additions and
Surogate's matrix kernels. The worker checks CPU features and runtime ABI
fingerprints before loading it.

## STT numerical qualification

With four threads and affinity to physical cores on the other socket, warm
stage measurements gave the following median total latency. Each value uses
12 warm repetitions; backend order was reversed between the two rounds.
These timings include mel extraction, encoder, CTC and language-model decoding,
and exclude audio-file decoding and HTTP overhead.

| Audio duration | Original kernels | Experimental oneDNN | Time reduction |
|---|---:|---:|---:|
| 4.92 seconds | 134.6 ms | 107.5 ms | 20.1% |
| 20.20 seconds | 461.3 ms | 379.7 ms | 17.7% |

Both the CTC decoder with language-model rescoring and the greedy TDT decoder
produced identical transcripts for **903/903 recordings**: all 301 Romanian
sentences in each of Doina, Tudor and Radu. This compares Surogate STT with
itself under different CPU kernels; it does not use Whisper.

Intermediate tensors are not identical. In that run:

| Diagnostic | Maximum observed |
|---|---:|
| Encoder absolute difference | 0.0028553 |
| CTC absolute difference | 0.0057707 |
| Encoder relative RMS difference | 0.00593% |
| CTC relative RMS difference | 0.00156% |

Sixteen recordings exceeded the existing per-element tensor tolerances;
thirteen exceeded the additional relative-RMS threshold of `1e-5`.
The numerical qualification therefore **did not pass**. These thresholds
were retained, and the faster STT path remains opt-in. Identical transcripts
on these recordings do not prove identical behavior on all future audio.

The Python serving suite passed 644 tests (218 skipped). The existing native
serving suite completed 187 checks with no failures; the added matrix-kernel
test also passed. Real-model HTTP checks passed for both offline and streaming
STT, with both default and experimental kernels (12 tests, two compute threads).

## Reproduce

Build the native servers and test drivers in a source checkout. TTS-only builds
do not require LibTorch or CUDA:

```bash
make serve-tts-build
cmake --build csrc/build-tts --target test_tts_cpu_kernels
csrc/build-tts/test_tts_cpu_kernels

# NATIVE_PACKAGE is the verified export containing voices.json and the GGUF files.
.venv/bin/python csrc/src/testing/serve/bench/tts_cpu_bench.py \
  --binary csrc/build-tts/surogate-tts --model "$NATIVE_PACKAGE" \
  --cpus 16-19 --threads 4 --codec-threads 4 --rounds 2 \
  --output /tmp/tts-cpu-benchmark.json
```

Choose a CPU list appropriate to the machine. The benchmark's frozen six-case
fixture targets the pinned released voices; custom exports need their own
`--cases` JSON with `id`, `voice`, `input`, `seed` and reference WAV `sha256`.

For STT, `speech-cpu-bench MODEL THREADS REPEATS AUDIO...` reports mel, encoder,
CTC and language-model decoding times. It selects the experimental backend;
set `SUROGATE_CPU_REFERENCE=1` for the original kernels. Repeat zero is the
cold request for each recording. `test_speech_cpu_kernels MODEL CASES REPORT`
performs the numerical and transcript comparison; its cases JSON contains
`id`, `voice` and local `audio` paths, and it returns nonzero on qualification
failures. Model-free CTest runs skip this check rather than substituting fake
weights.
