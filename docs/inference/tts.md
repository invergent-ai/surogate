# Romanian text-to-speech

Serve the published Romanian model on CPU with **Doina**, **Tudor** and **Radu**:

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
shared with native STT through `csrc/src/serve/serve/audio_http.h`. One native
GGML worker keeps the generator and codec loaded between requests. No Python,
Torch, NeMo or training-recipe process remains in the TTS inference path.

The published binary targets Linux x86-64 and was validated on AMD EPYC 9124.
Other CPUs require a compatible native build and validation. Speech generation
uses CPU threads and no GPU. Voice creation happens separately on offline
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

## Model download

The first start downloads only the native CPU package, approximately 1.15 GiB,
from HF revision `2bf175b4edc7b3ca7261d80e4d4ad85117c4f0a4`. It verifies the
voice profile and every file listed in it, then completes a short warm-up before
accepting requests. Subsequent starts verify and reuse the cached package.

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

`/health` becomes ready after the model is loaded. `/v1/models` reports the
original startup model argument. Use `--served-model-name my-tts` to set an alias;
requests that provide `model` must use that alias.

### Request fields

| Field | Accepted values |
|---|---|
| `input` | Required Romanian text, 1–4096 characters |
| `model` | Optional served model ID |
| `voice` | A name from `/v1/audio/voices`; defaults to the first voice in the package |
| `response_format` | `wav` (default) or `pcm` |
| `speed` | `1` |
| `seed` | Integer from 0 to 2147483647; defaults to 9 |

Both formats contain mono, 22,050 Hz, signed 16-bit little-endian audio. PCM has
no WAV header. `X-Audio-Sample-Rate` reports the rate. Responses contain the
complete recording; incremental audio streaming, MP3, style instructions and
speed changes are unavailable. Unsupported fields return a JSON error. Reserved tokenizer symbols, invalid
structured dates/times and oversized expanded text are rejected. Split long
inputs into separate requests.

Every successful response carries `X-Usage-Characters`: the characters of
`input` it was billed for. They are counted as sent, in Unicode code points, as
the 4096-character limit counts them and as Python's `len()` does. A precomposed
Romanian letter with a diacritic (ă, â, î, ș, ț) is one character, although it
takes two UTF-8 bytes, so a gateway that counts request bytes would overcharge
Romanian text by about 10%. A decomposed letter (a base letter followed by a
combining mark) counts as two, and whitespace and bracketed spans that are not
spoken count too. An account service that counts characters itself should use
the same rule. Error responses (4xx and 5xx, including queue-full, timeout and
worker failures) carry no count.

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

One request runs at a time against the shared model. Eight additional requests
can wait by default. Set `--max-pending-requests N` to change the queue size, or
`0` to reject requests while the worker is busy. A full queue returns HTTP 429.
`--request-timeout SECONDS` bounds queueing plus synthesis time (default 300);
expiration returns HTTP 504. Interrupted inference discards the worker's output
before another request uses it. The next request reloads a stopped worker.

The server binds to `127.0.0.1` by default. Use `--host` to choose the interface.
`--api-key-file PATH` enables bearer authentication on every endpoint, with the key read from a
file. Unlike `--api-key KEY`, it keeps the key out of the process's command line, where every
local user can read it:

```bash
surogate serve --tts surogate/surogate-ro-tts --api-key-file /etc/surogate/tts.key
curl -H "Authorization: Bearer $(cat /etc/surogate/tts.key)" http://localhost:8080/v1/audio/voices
```

Normal shutdown stops the native worker and removes temporary audio. Finished
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
