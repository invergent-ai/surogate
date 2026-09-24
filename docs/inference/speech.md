# Romanian speech recognition

For speech generation with named voices on CPU, see [Romanian text-to-speech](tts.md).

Serve Romanian transcription on a GPU or CPU with either model:

| Model | Use |
|---|---|
| [`surogate/surogate-ro-110m-tdt-ctc`](https://huggingface.co/surogate/surogate-ro-110m-tdt-ctc) | Transcribe complete audio files |
| [`surogate/surogate-ro-110m-streaming`](https://huggingface.co/surogate/surogate-ro-110m-streaming) | File transcription or live audio with partial and final transcripts |

Start the non-streaming model:

```bash
surogate serve --stt surogate/surogate-ro-110m-tdt-ctc --device 0 --port 8080
```

It processes the complete recording before returning text and automatically uses
the repository's Romanian language model. Live-audio requests return an error
with instructions to use file transcription instead.

For live audio, install the streaming dependency and start the streaming model:

```bash
pip install 'silero-vad==6.2.1'
surogate serve --stt surogate/surogate-ro-110m-streaming --device 0 --port 8080
```

Use `--device cpu` to run without a GPU. CPU-only PyTorch is supported. From a
source checkout, build with `make serve-build`, or use `make serve-stt-build`
to build just the speech server without a CUDA toolkit. The latter needs a C++
compiler, FFmpeg development libraries, and PyTorch in the active environment;
set `STT_PYTHON=/path/to/python` to choose that environment.
Installed wheels include the speech server. For a private
Hugging Face repository, authenticate with `hf auth login` or set `HF_TOKEN`.

### CPU compute options

`--threads N` sets CPU compute threads (default 4, range 1–256). The setting
also applies inside HTTP inference workers; it is independent of the number
of HTTP connections or streaming sessions.

```bash
surogate serve --stt surogate/surogate-ro-110m-tdt-ctc --device cpu --threads 4
```

`--cpu-kernels auto` (the default) enables the optimized FP32 path on CPU
with an MKL-enabled LibTorch build. It combines attention projections, caches
bounded position projections, removes a relative-attention padding copy, and
reuses packed feed-forward weights for recurring shapes. Language-model
backoff scores are also cached within each request. Changing clip lengths use
ordinary GEMM until a shape recurs, avoiding repeated packing overhead.

`--cpu-kernels reference` selects the original implementation for comparisons.
`--cpu-kernels optimized` requires CPU and MKL support; `auto` falls back to the
reference path if unavailable. LibTorch remains the runtime dependency. Model
precision, decoding rules and scores are unchanged.

The current path replaces the earlier experimental oneDNN linear rewrite. It
passed exact encoder/CTC tensor comparisons and both decoder transcript checks
on all 903 Romanian control recordings, plus streaming and boundary cases.
See [STT CPU kernel measurements](stt-cpu-kernels.md) for latency, cache behavior
and validation scope.

The first start downloads the selected acoustic checkpoint and its Romanian
language model, then prepares reusable files under `~/.cache/surogate/serve`.
Allow about 5 GB of disk space for the downloaded and prepared files. Later
starts reuse them. `SUROGATE_SERVE_CACHE` changes the preparation cache location.

To use local files:

```bash
surogate serve --stt /models/Ib_final.nemo \
  --lm /models/ro_4gram.arpa --device cpu

surogate serve --stt /models/Is_ctc_final_20260915.nemo \
  --lm /models/ro_4gram.nemo --device 0
```

The matching language model can be a NeMo `.nemo` archive or a token-ID `.arpa`
file. Preparation is automatic and cached. The non-streaming model does not
need the `silero-vad` dependency.

The served model ID is the model argument supplied at startup. `/v1/models`
reports it; `--served-model-name` sets a deployment alias. `--api-key` enables
bearer-token authentication on every endpoint.

## Transcribe a file

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav
```

The response is `{"text":"…","usage":{"type":"duration","seconds":2.63}}`.
WAV, FLAC, MP3, Ogg, M4A, AAC, and WebM audio
are supported; stereo and other sample rates are converted automatically.
Uploads are limited to 64 MiB and ten minutes. The non-streaming model processes
the entire file together; longer files need more memory. The streaming model
divides recordings at pauses, with a maximum segment length of 60 seconds.

Optional form fields:

| Field | Values |
|---|---|
| `model` | The served model ID; omit to use the running model |
| `language` | `ro` |
| `response_format` | `json` (default), `text`, or `verbose_json` |

`verbose_json` also returns duration and language. Translation, word
alignment, subtitle formats, and prompting are unavailable. Unsupported fields
return an error.

Every successful transcription reports the audio length it is billed for, in
seconds, whatever the format:

- the `X-Audio-Duration-Seconds` response header, on every format;
- `usage`, OpenAI's object for audio billed by duration
  (`{"type": "duration", "seconds": …}`), in `json` and `verbose_json`.

The two carry the same number, and so does `verbose_json`'s `duration`. A
gateway can meter a transcription with it, without decoding the file or reading
the body. A failed transcription carries neither. The value is the length of the
decoded audio, counted in 16 kHz samples (62.5 µs each):

- WAV, FLAC and other lossless files are billed for exactly their length.
- MP3 is billed for its audio, without the encoder's padding, when the file has
  a gapless (Xing/LAME) header, as encoders write by default.
- AAC and M4A can include up to one AAC frame of encoder padding at the end:
  1024 samples at the file's own rate, which is 23 ms at 44.1 kHz and 64 ms at
  16 kHz.

The header is a decimal number of seconds that always reads back as the same
value, for example `2.63` or `1.0`. The shortest audio, under a millisecond, is
written in exponent form (`6.25e-05` for one sample), so parse it as a float.

## Stream microphone audio

Use the streaming model for this endpoint. Create a stream:

```bash
curl -X POST http://localhost:8080/v1/audio/streams
```

The response includes an `id`. Send **mono, 16 kHz, signed 16-bit little-endian
PCM**, without a WAV header, to that stream:

```bash
curl -X POST "http://localhost:8080/v1/audio/streams/STREAM_ID" \
  -H 'Content-Type: application/octet-stream' --data-binary @chunk.pcm
```

Each response contains an `events` array. A `partial` event updates the current
utterance. A `final` event replaces its partial text and completes the segment.
Final text can differ from the partial text. Silence of approximately 640 ms
ends an utterance; the model also needs about one second of audio lookahead.

Send chunks in order and wait for each response before submitting the next chunk
for that stream. Continue recording and buffer new audio while waiting. Chunks
may contain any whole number of samples, up to ten seconds; 32–160 ms chunks
work well for live capture. Keep sending captured silence so pauses can be detected.

Drain the last audio and close the stream:

```bash
curl -X POST "http://localhost:8080/v1/audio/streams/STREAM_ID?finish=true"
```

The last request may also include PCM. Use `DELETE /v1/audio/streams/STREAM_ID`
to cancel a stream and discard its unfinished audio. Streams expire after two
minutes without a request. `--max-num-seqs` sets the number of live streams
(default 8); each keeps its own audio and transcript state. Inference requests
share one model and run one at a time.

Speech serving uses its own server mode. It does not add audio inputs to chat
completion requests or provide an OpenAI Realtime WebSocket endpoint.
