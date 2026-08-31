# OpenAI-compatible API

`surogate-engine` speaks the OpenAI wire format natively — point any OpenAI client at it by
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
| `GET` | `/health` | Readiness probe |
| `POST` | `/v1/embeddings` | Embeddings — served by the **encoder** binary, not `surogate-engine` |

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

`POST /v1/embeddings` is served by the **encoder binary** (`sinfer_embedding_server`), not by
`surogate-engine` — an embedding model has no KV cache, sampler or decode loop, so it gets its
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
