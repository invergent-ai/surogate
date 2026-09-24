# Romanian text-to-speech

Serve the published Romanian model on CPU with **Doina**, **Tudor** and **Radu** (for a GPU,
see [GPU](#gpu)):

```bash
surogate serve --tts surogate/surogate-ro-tts --device cpu --threads 4 --port 8080
```

From a source checkout, build the native server first:

```bash
# Ubuntu/Debian build dependency:
sudo apt-get install libicu-dev patch
make serve-tts-build
```

The full `make serve-build` and wheel build also include `surogate-tts`.
The TTS-only target requires C++20, CMake, Ninja, `patch` and ICU 70 or newer; it does
not require the CUDA toolkit or LibTorch. The runtime needs the ICU libraries
matching the build. `SUROGATE_TTS_BIN` can select a separately built executable.

Python resolves and verifies assets, then replaces itself with `surogate-tts`.
HTTP, Romanian normalization and tokenization run in C++. The HTTP setup is
shared with native STT through `csrc/src/serve/serve/audio_http.h`. A native
GGML worker (one by default, see `--max-num-seqs`) keeps the generator and codec
loaded between requests. No Python,
Torch, NeMo or training-recipe process remains in the TTS inference path.

The published binary targets Linux x86-64 and was validated on AMD EPYC 9124.
Other CPUs require a compatible native build and validation. With `--device cpu`, the
default, speech generation uses CPU threads and no GPU. Voice creation happens separately on offline
GPUs; serving a voice does not perform cloning or training.

## CPU kernels and threads

`--threads N` sets the generator's compute threads (default 4, range 1–256).
`--codec-threads N` separately controls the audio codec. Its default, or `0`,
inherits `--threads`. Both stages can run concurrently; these values are thread
counts per stage, not a reservation of physical cores. For example:

```bash
surogate serve --tts surogate/surogate-ro-tts --threads 4 --codec-threads 4
```

`--cpu-kernels auto` selects Surogate's AVX-512 GGML backend when the CPU and
build support it. The FP32 projection kernels reuse vectors across outputs;
the FP16 codec matrix kernel shares conversions across a 2×3 output tile,
including irregular convolution dimensions. Accumulation keeps the reference
FP32 arithmetic and reduction order; model precision and sampling stay fixed.

Use `--cpu-kernels reference` to select the model package's original backend for
comparisons. `--cpu-kernels optimized` requires the AVX-512 backend and fails
if it is unavailable. `auto` falls back to the package backend. That fallback
still requires a CPU compatible with the published native package.

The build downloads a hash-pinned GGML source archive and applies the model's
ABI additions plus Surogate's kernels. Set CMake's
`-DSUROGATE_TTS_CPU_KERNELS=OFF` to omit this backend. The worker checks the
model runtime's ABI fingerprints before loading its adapter. Newly exported
voices must use the same supported native runtime; unrelated runtime builds
need a matching adapter.

On a multi-socket host, measure with a consistent CPU affinity on one socket.
Increasing threads does not necessarily improve latency. See the
[CPU kernel validation report](cpu-kernels.md) for measurements and checks.

## GPU

`--device N` runs the generator, the codec and sampling on CUDA device N. N counts among the
devices the server can see, as it does for the LLM server. The default, `--device cpu`, is
unchanged, and `--threads` and `--cpu-kernels` apply to the CPU only.

A GPU needs the package's **GPU variant**:

- Its model, codec, tokenizer and voices are the CPU package's.
- Its `lib/` holds the same Magpie runtime built with CUDA, including `libggml-cuda`, and runs on
  compute capability 12.0 (RTX 50-series) only.
- The CPU package is refused on a GPU with a clear error.

The published model has its GPU variant in the repository's `gpu/` folder. With `--device N`,
`surogate/surogate-ro-tts` downloads and verifies that variant, pinned like the CPU package:

```bash
surogate serve --tts surogate/surogate-ro-tts --device 0 --port 8080
```

A downloaded copy, or a local variant, is served the same way:

```bash
hf download surogate/surogate-ro-tts --revision e6b1372cb1b3db6c205ebfc589fba8d630ed438b \
  --include "gpu/*" --local-dir surogate-ro-tts
surogate serve --tts surogate-ro-tts/gpu --device 0 --port 8080
```

On an RTX 5090 one request runs at about 25 times real time, with about 25 ms to the first
audio. Each server holds about 2.4 GB of GPU memory per worker. The worker needs the NVIDIA driver
and the CUDA 13 runtime (`libcudart`, `libcublas`) on the host, as the LLM server does.
`/health` reports `"device": "cuda:N"`. The variant also serves `--device cpu`, with the same
results as the CPU package, but still needs those libraries. Use the CPU package on hosts without
them.

The worker accepts only the CUDA runtime whose checksums it pins, as it does for the CPU runtime,
so the variant is published as built. Its CUDA-compiled libraries are not bit-for-bit reproducible:
a rebuild in another build directory gives the same `libggml`, `libggml-base` and `libggml-cpu`,
but a different `libnemo_speech_tts` and `libggml-cuda`. This is how that build was made, for
provenance and for qualifying a replacement. It uses `nemo-speech-cpp` at `07003daa` with the package's
`longform_history_fix.patch` and `longform_context_cache.h`, and the CPU package's TTS-only
options plus CUDA. The build maps source paths, links no NCCL, and loads its libraries from
their own directory:

```bash
SRC=$PWD/nemo-speech-cpp
cmake -S $SRC -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DNEMO_SPEECH_BUILD_TTS=ON -DNEMO_SPEECH_GGML_PATCHED=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_C_FLAGS=-ffile-prefix-map=$SRC=nemo-speech-cpp \
  -DCMAKE_CXX_FLAGS=-ffile-prefix-map=$SRC=nemo-speech-cpp \
  -DCMAKE_CUDA_FLAGS=-Xcompiler=-ffile-prefix-map=$SRC=nemo-speech-cpp \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON '-DCMAKE_INSTALL_RPATH=$ORIGIN' \
  -DNEMO_SPEECH_BUILD_ASR=OFF -DNEMO_SPEECH_BUILD_CLI=OFF -DNEMO_SPEECH_BUILD_HTTP=OFF \
  -DNEMO_SPEECH_BUILD_GRPC=OFF -DNEMO_SPEECH_BUILD_TESTS=OFF -DNEMO_SPEECH_BUILD_TOOLS=OFF
cmake --build build-cuda --target nemo_speech_tts ggml-cuda
python -m surogate.serve.tools.tts.gpu_variant CPU_PACKAGE build-cuda/bin GPU_VARIANT
```

The tool refuses a build whose checksums are not the pinned ones. A different build must first be
qualified (quality and speed), then pinned in `native_worker.cpp` and in the tool.

## Model download

The first start downloads only the native CPU package, approximately 1.15 GiB,
from HF revision `2bf175b4edc7b3ca7261d80e4d4ad85117c4f0a4`, or with `--device N`
only its GPU variant, approximately 1.35 GiB, from revision
`e6b1372cb1b3db6c205ebfc589fba8d630ed438b`. It verifies the voice profile and
every file listed in it, then completes a short warm-up before accepting
requests. Subsequent starts verify and reuse the cached package.

Authenticate for the private repository with `hf auth login` or `HF_TOKEN`.
The package is cached under `~/.cache/surogate/serve`; `SUROGATE_SERVE_CACHE`
changes that directory. `--no-cache` refreshes the pinned HF package. A cached
or local package can start without network access.

## Generate speech

```bash
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"surogate/surogate-ro-tts","input":"Bună ziua! Cu ce vă pot ajuta?","voice":"Doina","response_format":"wav"}' \
  --output doina.wav
```

Change `voice` to `Tudor` or `Radu` to switch voices. Voice names are matched
without case sensitivity. Discover the available names with:

```bash
curl http://localhost:8080/v1/audio/voices
curl http://localhost:8080/v1/models
curl http://localhost:8080/health
```

`/health` becomes ready after every worker has loaded the model. `/v1/models` reports the
original startup model argument. Use `--served-model-name my-tts` to set an alias;
requests that provide `model` must use that alias.

### Request fields

| Field | Accepted values |
|---|---|
| `input` | Required Romanian text, 1–4096 characters (see `--max-input-characters`) |
| `model` | Optional served model ID |
| `voice` | A name from `/v1/audio/voices`; defaults to the first voice in the package |
| `response_format` | `wav` (default) or `pcm` |
| `speed` | `1` |
| `seed` | Integer from 0 to 2147483647; defaults to 9 |
| `stream_format` | Omitted (the complete recording), `audio` or `sse`; see [Streaming](#streaming) |

Both formats contain mono, 22,050 Hz, signed 16-bit little-endian audio. PCM has
no WAV header. `X-Audio-Sample-Rate` reports the rate. MP3, style instructions and
speed changes are unavailable. Unsupported fields return a JSON error. Reserved tokenizer symbols, invalid
structured dates/times and oversized expanded text are rejected. A `Range` header is refused
with HTTP 416.

An input of 45 words or more is synthesized sentence by sentence. `--max-input-characters N`
raises the 4096-character limit, up to 16384, so a long reply can go in one request; streamed, it
starts playing as soon as its first audio is made. The limits that scale with N:
- a sentence for every 16 characters allowed (at least 64 sentences);
- the text as spoken, with numbers, dates and abbreviations written out, may be four times the
  input, and at most 16,384 characters in any case: about 24 minutes of speech, within the 96 MiB
  of audio a request may produce;
- at most 4,096 numbers, dates and similar values per input, whatever N.

Raise `--request-timeout` with N: on four CPU cores synthesis takes about twice as long as the
audio lasts.

Every successful response carries `X-Usage-Characters`: the characters of
`input` it was billed for. They are counted as sent, in Unicode code points, as
the input limit counts them and as Python's `len()` does. A precomposed
Romanian letter with a diacritic (ă, â, î, ș, ț) is one character, although it
takes two UTF-8 bytes, so a gateway that counts request bytes would overcharge
Romanian text by about 10%. A decomposed letter (a base letter followed by a
combining mark) counts as two, and whitespace and bracketed spans that are not
spoken count too. An account service that counts characters itself should use
the same rule. Error responses (4xx and 5xx, including queue-full, timeout and
worker failures) carry no count.

### Streaming

Without `stream_format`, the response carries the complete recording once it has been
synthesized. With it, the audio is sent as the runtime produces it, and the first audio reaches
the client after the runtime's time to first audio plus network time. Measured over HTTP on
one machine with benchmark sentences of 3 to 5 seconds of audio: 32 ms on an RTX 5090 (median of
20 requests, p90 38 ms), and 1.3 s on four CPU cores, where the whole recording takes 9.4 s. A
3,524-character input (5 minutes of audio) started playing after 32 ms on the RTX 5090; its
synthesis took 17.5 s. On the CPU, streamed audio is the same, sample for sample, as the whole
recording.

- `stream_format: "audio"` sends the recording's bytes in a chunked body as they are produced.
  With `wav`, a header of unknown length comes first: its size fields are `0xFFFFFFFF`, which
  players read as "until the end". With `pcm`, the raw samples come alone.
- `stream_format: "sse"` sends the same bytes base64-encoded in `speech.audio.delta` events
  (`{"type": "speech.audio.delta", "audio": "…"}`). A `speech.audio.done` event with
  `usage.input_characters` and `usage.audio_seconds` ends a complete stream.

The request is queued like any other before its headers go out. A full queue, a queue timeout or
an invalid request therefore still gets its HTTP error status and carries no
`X-Usage-Characters`.

The client sets the pace of delivery, not of synthesis. The worker is handed to the next request
as soon as the last audio is made; audio the client has not read yet waits in the server (at most
the 96 MiB a request may produce). `--request-timeout` bounds queueing plus synthesis, not
delivery. A client that reads nothing for 5 seconds is disconnected. A request, streamed or whole,
keeps its place in the queue until its audio has been delivered, so `--max-pending-requests` also
bounds how many listeners can be served at once: raise it for many clients that play the audio as
it arrives. Audio not yet read takes memory, up to 96 MiB per request (about 12 GiB with 128
pending requests). An idle keep-alive connection is closed after 1 second, so a connection pool in
front of the server should reuse connections sooner or not keep them.

Once the stream has started, `X-Usage-Characters` has already been sent. A stream that fails
partway was not delivered: a raw stream then ends without its final chunk, which clients report
as an incomplete response, and an SSE stream ends with an `error` event instead of
`speech.audio.done`. Bill a streamed request only when it completes: its status (200) and
`X-Usage-Characters` arrive before that is known.

`speech.audio.done` carries this server's usage, `input_characters` and `audio_seconds`, not
the `input_tokens`, `output_tokens` and `total_tokens` of OpenAI's event.

The endpoint uses the OpenAI speech request shape for the supported fields, with
WAV as its default format. For an OpenAI client, request `response_format="wav"`
explicitly. This service speaks Romanian; it does not reproduce the separate
foreign-language routing used for 30 rows of the research stress benchmark.

## Local and newly cloned voices

Pass the native export directory or its `voices.json`:

```bash
surogate serve --tts /models/my-voice-native --voice "My voice" --port 8080
surogate serve --tts /models/my-voice-native/voices.json --voice "My voice"
```

Use the native package produced by the
[published voice-creation pipeline](https://github.com/invergent-ai/training_tts/blob/main/docs/CREATE-A-VOICE.md).
Keep its model, codec, runtime libraries, tokenizer and profile together.
The server verifies their hashes and uses the names and decoding settings from
the profile. A bare NeMo checkpoint or Python-only voice profile must first be
exported with that pipeline's `ro_tts.export_native` command.

`--voice` chooses the default name; clients can still select any voice in the
loaded package. Adding a new voice requires a new native export and a server
restart. It does not require a separate model process for every name.

## Queue, authentication and shutdown

`--max-num-seqs N` (default 1, at most 16) synthesizes N requests at once. Each runs on its own
worker process with its own copy of the model: about 2.4 GB of GPU memory per worker on a GPU, and
on the CPU about 1.5 GB of memory and `--threads` + `--codec-threads` threads per worker. All
workers load before the server answers, and startup names the worker that failed to load. A
worker that fails is replaced at once without disturbing the others. Requests go to a worker whose
model is loaded first. A worker that dies while idle is restarted in the background. One that dies
before its model loads (for example when another process took the GPU memory) is retried every 10
seconds, and meanwhile takes no request while another worker can serve; when none can, requests get
HTTP 503 at once. `/health` stays 200 while any worker can serve, with `"status": "degraded"` and
`"workers": {"ready": R, "total": N}`, and turns 503 when none can: a worker counts once its model
has loaded, or while it loads in place of one that had.

On one RTX 5090 with `--max-num-seqs 4`, streaming Doina, each request's processing time per
second of audio was (median, and the maximum in brackets; below 1 is faster than real time):

| Concurrent requests | Alone | Next to Rune serving 8 chats at once |
|---|---|---|
| 1 | 0.04 (0.04) | 0.18 (0.44) |
| 2 | 0.09 (0.13) | 0.66 (0.90) |
| 3 | 0.19 (0.28) | 0.90 (1.09) |
| 4 | 0.26 (0.55) | 0.88 (1.43) |

Rune was the Q4_K_M build with `--gpu-memory-limit-mib 20000 --max-num-seqs 8`, beside four TTS
workers (about 2.46 GB each) on the same card. First audio took 30 ms alone and 110 to 180 ms (median) next to Rune.
Next to a busy LLM server, 2 requests at a time each stay faster than real time; a third one does
not always. The workers share the GPU by time slicing and do not batch.

Eight additional requests can wait by default, and they are served in arrival order. Set
`--max-pending-requests N` to change the queue size, or `0` to reject requests while every worker
is busy or a stream is still being delivered. A full queue returns HTTP 429.
`--request-timeout SECONDS` bounds queueing plus synthesis time (default 300);
expiration returns HTTP 504. When a client leaves, its request stops at the worker's next
audio chunk and the worker serves the next request without reloading the model. A worker that
produces no audio for 30 seconds after that is stopped instead. A worker that failed or timed out
is replaced at once, unless it died before its model loaded (then it is retried every 10 seconds).
At shutdown the workers are stopped at once, even in the middle of a request.

The server binds to `127.0.0.1` by default. Use `--host` to choose the interface.
`--api-key-file PATH` enables bearer authentication on every endpoint, with the key read from a
file. Unlike `--api-key KEY`, it keeps the key out of the process's command line, where every
local user can read it:

```bash
surogate serve --tts surogate/surogate-ro-tts --api-key-file /etc/surogate/tts.key
curl -H "Authorization: Bearer $(cat /etc/surogate/tts.key)" http://localhost:8080/v1/audio/voices
```

Normal shutdown stops the native workers and removes temporary audio. Finished
request audio and per-request native statistics are not retained on disk.

`GET /metrics` reports the server in Prometheus text format, like the LLM server:

- `surogate_up`;
- `surogate_requests{state="running"}`: speech requests, counted from arrival, including while
  they wait in the queue, until their audio has been written or their client has gone;
- `surogate_requests_total{endpoint,outcome}`;
- `surogate_characters_total`: input characters billed;
- `surogate_audio_seconds_total`: seconds of audio synthesized.

Wait for `surogate_requests{state="running"}` to reach 0 to drain a server before restarting it.

## Quality

Serving keeps the published model, Romanian normalization, tokenizer and fixed
per-voice decoding settings. The 301 Romanian benchmark inputs produce the same
token sequences as the accepted native release. HTTP control recordings are
compared byte-for-byte with the saved native outputs, including voice switching
and long text. This is an integration check; it does not establish new WER or
the quality of a newly cloned voice. Published scores and licensing remain in
the [model card](https://huggingface.co/surogate/surogate-ro-tts).

## Validation

```bash
make serve-tts-build
.venv/bin/python -m pytest -q tests/serve/test_tts.py tests/serve/test_cli.py
SUROGATE_TTS_TEST_MODEL=surogate/surogate-ro-tts \
  .venv/bin/python -m pytest -q tests/serve/test_tts_http.py
```

The first command builds both the server and a frontend test driver. Unit tests
exercise the native HTTP server using a protocol fixture; the opt-in test loads
the real HF model and compares the three released voices with frozen audio
hashes. The tokenizer regression fixture covers all 301 Romanian inputs.
