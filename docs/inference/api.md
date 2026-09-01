# OpenAI-compatible API

`surogate serve` speaks the OpenAI wire format natively — point any OpenAI client at it by
overriding the base URL. The same process also serves the Anthropic Messages API, so both
client ecosystems work against one endpoint without a proxy.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Explain paged attention in two sentences."}],
).choices[0].message.content)
```

`--api-key KEY` turns on bearer-token auth; without it any key is accepted. `--cors` enables
browser cross-origin requests. `--served-model-name ID` overrides the model id the server
reports, which is how you keep a client's hard-coded model string working.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat Completions, streaming or not |
| `POST` | `/v1/responses` | Responses API |
| `GET` | `/v1/responses/{id}` | Fetch a stored response |
| `DELETE` | `/v1/responses/{id}` | Delete a stored response |
| `POST` | `/v1/responses/{id}/cancel` | Cancel an in-flight response |
| `GET` | `/v1/responses/{id}/input_items` | List the input items of a stored response |
| `POST` | `/v1/responses/input_tokens` | Count input tokens without generating |
| `POST` | `/v1/responses/compact` | Compact a stored conversation |
| `GET` | `/v1/models`, `/v1/models/{id}` | Model listing |
| `POST` | `/v1/messages` | **Anthropic** Messages API |
| `POST` | `/v1/messages/count_tokens` | **Anthropic** token counting |
| `POST` | `/sleep` | Release a model's VRAM, state parked in host RAM (`--enable-sleep-mode`; `?model=NAME` on multi-model servers) |
| `POST` | `/wake_up` | Restore a model; sub-second for most models (`?model=NAME`) |
| `GET` | `/is_sleeping` | Sleep state (`?model=NAME`) |
| `POST` | `/v1/load_lora_adapter` | Load a PEFT adapter at runtime (`--enable-lora`; `?model=NAME` targets a specific model on multi-model servers) |
| `POST` | `/v1/unload_lora_adapter` | Unload an adapter by name (`?model=NAME`) |
| `GET` | `/health` | Readiness probe |
| `POST` | `/v1/embeddings` | Embeddings — served by `surogate serve --embed`, a separate process |

## Chat Completions

### Supported request fields

| Field | Notes |
|---|---|
| `model` | Matched against the served model id |
| `messages` | Roles `system`, `user`, `assistant`, `tool` |
| `stream`, `stream_options.include_usage` | SSE; with `include_usage` a dedicated usage chunk precedes `[DONE]` |
| `max_completion_tokens`, `max_tokens` | `max_completion_tokens` wins; default from `--default-max-tokens` (8192) |
| `temperature`, `top_p`, `top_k` | `top_k` is the common non-OpenAI extension; `min_p` is a launch flag only |
| `presence_penalty`, `frequency_penalty` | |
| `seed` | Reproducible sampling |
| `stop` | String or array of strings |
| `logit_bias` | Keys are integer token ids |
| `tools`, `tool_choice` | `tool_choice`: `none`, `auto`, `required`, or a named function object |
| `response_format` | Only `{"type": "text"}` |
| `reasoning_effort` | `low`, `medium`, `xhigh` on models that support it |
| `chat_template_kwargs.enable_thinking` | Per-request thinking toggle |
| `chat_template_kwargs.preserve_thinking` | Retain closed-turn reasoning in later prompts |
| `ignore_eos` | Benchmarking aid — generate to the token limit |

Message content may be a plain string or an array of parts: `text`, `image_url`, `video_url`,
and `input_audio`. Media parts require `--vision` and an artifact with a vision tower; sources
may be local paths, HTTP(S) URLs, or base64 data URIs.

### Reasoning models

Reasoning is separated from the answer rather than left inline. Non-streaming responses carry it
as `message.reasoning_content` with `content` holding the answer alone; streaming emits
`reasoning_content` deltas before `content` deltas. This is the DeepSeek/vLLM convention that
Open WebUI, Chatbox and similar clients already render.

`--no-thinking` disables reasoning server-wide; `--preserve-thinking` keeps closed-turn
reasoning in later prompts; per-request `chat_template_kwargs.enable_thinking` overrides both.

### Not supported

These are refused with a `400` and a specific error code rather than silently ignored:

| Field | Reason |
|---|---|
| `n` > 1 | One completion per request (`n_not_supported`) |
| `functions`, `function_call` | Legacy pre-`tools` API (`tools_not_supported`) |
| `response_format` other than `{"type":"text"}` | No constrained decoding (`response_format_not_supported`) |
| Message role `function` | Use role `tool` (`unsupported_role`) |

### Streaming shape

The first chunk carries the assistant role, then `reasoning_content` deltas, then `content`
deltas, then a final chunk with `finish_reason` and an empty delta. With
`stream_options.include_usage`, content chunks carry `usage: null` and one usage-only chunk
(empty `choices`) is emitted before `data: [DONE]`.

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","stream":true,
       "stream_options":{"include_usage":true},
       "messages":[{"role":"user","content":"Count to five."}]}'
```

## Sleep mode

On a server started with `--enable-sleep-mode`, the model can release its VRAM
without shutting down, mirroring vLLM's endpoints:

```bash
curl -X POST -d '' http://localhost:8080/sleep      # level 1 (the only level)
curl -X POST -d '' http://localhost:8080/wake_up
curl http://localhost:8080/is_sleeping
```

Sleeping waits for in-flight requests to finish, then copies the model's device
memory — weights, KV cache, every piece of state — into pinned host RAM and
releases the physical VRAM. Waking copies it back at PCIe speed and the server
resumes exactly where it was: identical outputs, prefix cache intact, no
recapture. Measured on a 27B (30 GiB of device state): sleeping leaves ~800 MiB
resident, waking takes ~0.6 s. The first sleep also allocates the pinned host
backup, which takes several seconds once; the backup is weights-plus-cache
sized, so budget host RAM accordingly.

While asleep, `/health` answers normally and generation requests get a 503
naming `/wake_up`. Send an empty body (`-d ''`) with the bare POSTs — a POST
with neither body nor `Content-Length` waits out a read timeout before the
server acts.

## LoRA adapters at runtime

On a server started with `--enable-lora`, adapters can be added and removed
without a restart, mirroring vLLM's endpoints:

```bash
curl -X POST http://localhost:8080/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "my-tune", "lora_path": "/path/to/peft/adapter"}'

curl -X POST http://localhost:8080/v1/unload_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "my-tune"}'
```

A loaded adapter appears in `/v1/models` and is selected per request by naming
it in `model`. Loading validates the adapter fully before it becomes routable —
wrong shapes, unsupported modules, a taken name, or exhausted `--max-loras`
slots are refused with the reason, and a failed load leaves nothing behind.
Unloading zeroes the adapter's contribution immediately; a request already in
flight that selected it finishes against the base model rather than reading
freed weights.

## Responses API

`POST /v1/responses` accepts `input` (string or typed items), `instructions`,
`previous_response_id`, `metadata`, `tools`, `tool_choice`, `store`, and `stream`, and emits
semantic SSE events when streaming. Stored state is **process-local** — it does not survive a
restart and is not shared between replicas — and bounded by `--response-store-max-records`
(1024) and `--response-store-max-mib` (256). Pass `store: false` for stateless use.

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","input":"Summarise paged attention.","store":false}'
```

## Anthropic Messages

`POST /v1/messages` and `POST /v1/messages/count_tokens` implement the Anthropic wire format
against the same engine, including its SSE event sequence and `system` handling. Point an
Anthropic SDK at the server's base URL.

## Embeddings

`POST /v1/embeddings` is served by `surogate serve --embed`, a separate process from the
generative server — an embedding model has no KV cache, sampler or decode loop, so it gets its
own small server. All four OpenAI `input` forms are accepted: one string, an array of strings,
one array of token ids, or an array of those.

```bash
curl http://127.0.0.1:8413/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-300m","input":["a first sentence","a second one"]}'
```

The response is the standard `{"object":"list","data":[{"object":"embedding","index":0,
"embedding":[...]}],"usage":{...}}` shape. Vectors are mean-pooled, projected and L2-normalised,
so cosine similarity is a dot product. Token-id input is useful when benchmarking against
another engine: it keeps the measurement on the model rather than on two tokenizers.
