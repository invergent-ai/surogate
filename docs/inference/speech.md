# Romanian speech recognition

Serve [`surogate/surogate-ro-110m-streaming`](https://huggingface.co/surogate/surogate-ro-110m-streaming)
for Romanian transcription on a GPU or CPU. File uploads return completed transcripts.
Live audio streams return partial transcripts and corrected final transcripts after a pause.

Install the speech preparation dependency, then start the server:

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

The first start downloads the selected acoustic checkpoint and its Romanian
language model, then prepares reusable files under `~/.cache/surogate/serve`.
Allow about 5 GB of disk space for the downloaded and prepared files. Later
starts reuse them. `SUROGATE_SERVE_CACHE` changes the preparation cache location.

To use local files:

```bash
surogate serve --stt /models/Is_ctc_final_20260915.nemo \
  --lm /models/ro_4gram.nemo --device 0
```

The served model ID is the model argument supplied at startup. `/v1/models`
reports it; `--served-model-name` sets a deployment alias. `--api-key` enables
bearer-token authentication on every endpoint.

## Transcribe a file

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav
```

The response is `{"text":"…"}`. WAV, FLAC, MP3, Ogg, M4A, AAC, and WebM audio
are supported; stereo and other sample rates are converted automatically.
Uploads are limited to 64 MiB and ten minutes. Longer recordings are divided
at pauses, with a maximum segment length of 60 seconds.

Optional form fields:

| Field | Values |
|---|---|
| `model` | The served model ID; omit to use the running model |
| `language` | `ro` |
| `response_format` | `json` (default), `text`, or `verbose_json` |

`verbose_json` also returns duration and language. Translation, word
alignment, subtitle formats, and prompting are unavailable. Unsupported fields
return an error.

## Stream microphone audio

Create a stream:

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
