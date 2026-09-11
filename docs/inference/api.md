# OpenAI-compatible API

Connect an OpenAI-compatible client to `surogate serve` by setting its base URL. The server
also supports the Anthropic Messages API. Supported features and limits are listed below.
The examples use `--served-model-name qwen3.6-27b` when starting the server.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")
print(client.chat.completions.create(
    model="qwen3.6-27b",
    messages=[{"role": "user", "content": "Suggest three easy vegetarian dinners."}],
).choices[0].message.content)
```

`--api-key KEY` enables authentication with either `Authorization: Bearer KEY` or `x-api-key:
KEY`; without it authentication is disabled. `/health` and CORS preflight requests are exempt.
`--cors` enables browser cross-origin requests. `--served-model-name ID` overrides the model id
the server reports, which is how you keep a client's hard-coded model string working.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat Completions, streaming or not |
| `POST` | `/v1/chat/completions/tokens` | Chat completions with an optional `tokens` array for exact prompt ids |
| `POST` | `/v1/completions` | Raw text completion without a chat template |
| `POST` | `/tokenize` | Render and tokenize `messages` or a raw `prompt` without generating |
| `POST` | `/v1/responses` | Responses API |
| `GET` | `/v1/responses/{id}` | Fetch a stored response |
| `DELETE` | `/v1/responses/{id}` | Delete a stored response |
| `POST` | `/v1/responses/{id}/cancel` | Background cancellation is unsupported; see [Responses API](#responses-api) |
| `GET` | `/v1/responses/{id}/input_items` | List the input items of a stored response |
| `POST` | `/v1/responses/input_tokens` | Count input tokens without generating |
| `POST` | `/v1/responses/compact` | Returns `400 compaction_not_supported` |
| `GET` | `/v1/models`, `/v1/models/{id}` | Model listing |
| `POST` | `/v1/messages` | **Anthropic** Messages API |
| `POST` | `/v1/messages/count_tokens` | **Anthropic** token counting |
| `POST` | `/sleep` | Free a model's GPU memory while saving it in system RAM; requires `--enable-sleep-mode` |
| `POST` | `/wake_up` | Restore a sleeping model |
| `GET` | `/is_sleeping` | Sleep state (`?model=NAME`) |
| `POST` | `/v1/load_lora_adapter` | Load a PEFT adapter at runtime (`--enable-lora`; `?model=NAME` targets a specific model on multi-model servers) |
| `POST` | `/v1/unload_lora_adapter` | Unload an adapter by name (`?model=NAME`) |
| `GET` | `/health` | Process health; also answers while a model sleeps |
| `GET` | `/kv_stats` | Cache memory use and active or waiting requests per model |
| `GET` | `/metrics` | Prometheus counters and gauges for throughput, requests, KV, weights, and sleep state |
| `POST` | `/v1/embeddings` | Embeddings — served by `surogate serve --embed`, a separate process |

On a server with several models, add `?model=NAME` to sleep, wake, sleep-state, and adapter
management requests to select the model.

## Chat Completions

### Supported request fields

| Field | Notes |
|---|---|
| `model` | Selects a served model or one of its named LoRA adapters |
| `messages` | Roles `system`, `user`, `assistant`, `tool` |
| `stream`, `stream_options.include_usage` | SSE; with `include_usage` a dedicated usage chunk precedes `[DONE]` |
| `max_completion_tokens`, `max_tokens` | `max_completion_tokens` wins; default from `--default-max-tokens` (8192) |
| `temperature`, `top_p`, `top_k`, `min_p` | Request values override sampling defaults; `top_k: -1` or `0` disables the request's top-k limit |
| `repetition_penalty` | Multiplicative penalty; must be positive; `1` disables it |
| `presence_penalty`, `frequency_penalty` | Adjust repetition penalties for previously used tokens |
| `seed` | Sets the sampler seed; an omitted seed uses the server seed or a fresh per-request seed |
| `stop` | String or array of strings |
| `logit_bias` | Map token-id strings to biases in `[-100,100]`; applies to greedy and sampled generation |
| `tools`, `tool_choice` | Function tools only; `none`, `auto`, `required`, or a named function object. Automatic choice requires `--enable-auto-tool-choice` |
| `response_format` | `text`, `json_object`, or `json_schema`; see [Structured output](#structured-output) |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; the loaded template must support the requested value |
| `chat_template_kwargs.enable_thinking` | Per-request thinking toggle; also accepted as top-level `enable_thinking` |
| `chat_template_kwargs.preserve_thinking` | Keep earlier assistant reasoning in later prompts; also accepted as top-level `preserve_thinking` |
| `ignore_eos` | Ignore the model's stop tokens; explicit stop strings and context limits still apply |
| `min_tokens` | Suppress stop tokens until this many tokens have been generated; stop strings remain active |
| `tokens` | Nonempty array of exact prompt token ids to use instead of the formatted messages; `messages` is still required |
| `add_generation_prompt` | Whether the template appends the assistant-generation prefix; defaults to `true` |
| `logprobs`, `return_token_ids` | Raw token details for text and tool-call Chat Completion responses, including SSE; see below |

If the server was started with `--greedy`, temperature is always zero even when a request
specifies another value.

Message content may be a plain string or an array of `text`, `image_url`, and `video_url` parts.
Images and video require `--vision` and a supported vision model; sources may be
local paths, HTTP(S) URLs, or base64 data URIs. `input_audio` is refused with
`modality_not_supported`.

For automatic tool choice, start the server with `--enable-auto-tool-choice` and a matching
`--tool-call-parser` (default `qwen3_xml`; `hermes`, `spark25`, `llama3_json`, and `llama4_json` are also
accepted). The model generates tool calls, and your application executes them. Validate the
arguments before use: the server does not guarantee that they match the tool's schema.

`logprobs: true` returns each generated token's log-probability and text in `choices[].logprobs.content`;
`top_logprobs` entries are empty. `return_token_ids: true` adds the exact prompt and completion
ids. These details cover the raw generated sequence, including reasoning and tool-call syntax,
for both text and tool-call responses. With `stream: true`, the server sends the token details
together in a chunk with an empty `delta` after generation and before the finish chunk.
Unavailable log-probabilities are returned as `null`.

### Reasoning models

With a matching reasoning parser, non-streaming responses put reasoning in
`message.reasoning_content` and the answer in `message.content`. Streaming responses use
`reasoning_content` and `content` deltas respectively.

`--no-thinking` sets the default thinking request to off where the template supports a toggle;
per-request `enable_thinking` overrides that setting. Some models always reason and cannot
disable it. Unsupported per-request toggles or effort values are refused. `--preserve-thinking`
keeps earlier assistant reasoning in later prompts; the request's `preserve_thinking` overrides
that default. If you supply both a top-level setting and its `chat_template_kwargs` equivalent,
their values must agree.

### Not supported

These are refused with a `400` and a specific error code rather than silently ignored:

| Field | Reason |
|---|---|
| `n` > 1 | One completion per request (`n_not_supported`) |
| `functions`, `function_call` | Legacy pre-`tools` API (`tools_not_supported`) |
| Message role `function` | Use role `tool` (`unsupported_role`) |
| `prompt_logprobs` requesting prompt scoring | Prompt token scoring is unsupported |
| Unsupported content parts, including `input_audio` | `modality_not_supported` |

Models without a chat template use `/v1/completions`; chat generation endpoints return
`chat_not_supported`.

### Structured output

Chat completions support `response_format: {"type":"json_object"}` for a JSON object, or
`json_schema` to specify its contents:

```json
{
  "model": "my-model",
  "messages": [{"role": "user", "content": "Is the task complete?"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"complete": {"type": "boolean"}},
        "required": ["complete"],
        "additionalProperties": false
      }
    }
  }
}
```

Supported schemas include nested objects and arrays, required properties, additional-property
rules, enums and constants, numeric bounds and `multipleOf`, array and string length limits,
string patterns, recognized formats, `anyOf`, and local `$ref` references. `allOf` intersections
and `oneOf` alternatives are supported when their constraints can be combined or proven
mutually exclusive.

JSON Schema support is partial. The following restrictions apply to both chat completions
`response_format` and Responses `text.format`, including streaming requests. Unsupported
constraints return HTTP `400` before generation; they are not silently ignored.

| Limitation | Meaning and suggested alternative |
|---|---|
| Overlapping `oneOf` alternatives | `oneOf` requires exactly one alternative to match. Alternatives that could both match are unsupported. Use distinct required `const` values to distinguish alternatives, or use `anyOf` only if matching more than one alternative is acceptable. |
| Some `allOf` intersections | Not every combination of constraints can be enforced together. Where possible, combine compatible requirements into a single schema. |
| Conditional schemas and general negation | `if`/`then`/`else` and general `not` constraints are unsupported. Express allowed values or supported alternatives directly, or validate these rules in your application. |
| `uniqueItems: true` | Array element uniqueness cannot be enforced during generation. Validate uniqueness in your application and retry if needed. |
| Remote `$ref` references | References outside the submitted schema are unsupported. Include the referenced schemas under `$defs` and use local references such as `#/$defs/address`. |

Removing an unsupported constraint also removes its generation guarantee. If you simplify a
schema to make a request acceptable, validate the result against your original requirements
before using it.

The constraint applies while generating, including streaming. An answer stopped by a token or
context limit can still be incomplete; check `finish_reason` before parsing it. Use sufficient
`max_tokens`, keep `ignore_eos` false and `min_tokens` at zero, and omit custom `stop` strings and active tools. JSON is
returned as answer content without a separate reasoning response.

Structured requests work on single- and multi-GPU servers, including MTP and DFlash speculative
decoding. The speed benefit depends on how often the proposed tokens satisfy the requested
format and are accepted by the target model.

The Responses API supports the same constraints through `text.format`. Unlike chat completions,
place `name`, `strict`, and `schema` directly inside the format object:

```json
{
  "model": "my-model",
  "input": "Is the task complete?",
  "text": {
    "format": {
      "type": "json_schema",
      "name": "answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"complete": {"type": "boolean"}},
        "required": ["complete"],
        "additionalProperties": false
      }
    }
  }
}
```

Use `"text": {"format": {"type": "json_object"}}` when any JSON object is sufficient.
Both formats work with streaming and are preserved when retrieving a stored response.

### Streaming shape

The role chunk is sent immediately before the first output item, so receiving it reflects
generation progress. Reasoning is emitted as `reasoning_content` deltas and answer text as
`content` deltas; parsed tool calls are emitted when generation finishes. A final chunk carries
`finish_reason` and an empty delta. With
`stream_options.include_usage`, content chunks carry `usage: null` and one usage-only chunk
(empty `choices`) is emitted before `data: [DONE]`.

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","stream":true,
       "stream_options":{"include_usage":true},
       "messages":[{"role":"user","content":"Count to five."}]}'
```

## Raw completions and tokenization

`POST /v1/completions` tokenizes `prompt` exactly as written, with no chat template. It accepts
a string or a one-element string array and supports streaming, sampling, stops, and adapter
selection. Batched prompts and token arrays in `prompt` are refused. Requests for `echo`,
`logprobs`, `suffix`, or `best_of > 1` are unsupported on this endpoint.

`POST /tokenize` accepts either `messages` or a raw `prompt` and returns `count`,
`max_model_len`, and `tokens`. Set `with_token_strings: true` to include `token_strs`, or
`add_generation_prompt: false` to render a chat fragment without the assistant-generation
prefix. The ids use the loaded model's tokenizer and template.

`/v1/chat/completions/tokens` uses the same schema as `/v1/chat/completions`: include
`messages` and a nonempty `tokens` array. The explicit ids become the model's prompt, which
allows a client to extend an exact previous token sequence without re-rendering its history.

## Sleep mode

Start with `--enable-sleep-mode` to free a model's GPU memory without shutting down the server:

```bash
curl -X POST -d '' http://localhost:8080/sleep
curl -X POST -d '' http://localhost:8080/wake_up
curl http://localhost:8080/is_sleeping
```

Manual sleep waits for active requests to finish, then saves the model and its cache in
system RAM. Waking restores them, including reusable prompts. Allow enough RAM for the model
and its cache. The first sleep on a single-model server can take longer while this memory is
prepared; servers with several models prepare it at startup.

While asleep, `/health` still answers normally. A single-model server returns `503` for
generation requests until you call `/wake_up`. A server with several models can make room
and wake the requested model automatically while the request waits. Send the empty body
(`-d ''`) shown above with sleep and wake requests to avoid a read-timeout delay.

## Monitoring memory and requests

Use `/kv_stats` to check cache memory use and how many requests are running or waiting:

```bash
curl http://127.0.0.1:8080/kv_stats
```

The response has a `models` array with an entry for each served model. Useful fields include:

| Field | Meaning |
|---|---|
| `model`, `sleeping` | Model name and sleep state |
| `running_requests`, `waiting_requests` | Requests generating or waiting to start |
| `kv_capacity_tokens` | Configured cache capacity in tokens |
| `weights_bytes` | Model weight size in bytes |
| `pool_bytes` | Full configured cache size |
| `in_use_bytes` | Cache space in use |
| `mapped_bytes` | Cache memory currently allocated on the GPU |

Cache memory grows with demand by default, so `mapped_bytes` can be smaller than `pool_bytes`.
It may remain above `in_use_bytes` because the server keeps some memory ready for reuse.
Use `/metrics` for Prometheus monitoring of requests, throughput, memory, and sleep state.

## LoRA adapters at runtime

Start with `--enable-lora` to add and remove adapters without restarting:

```bash
curl -X POST http://localhost:8080/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "my-tune", "lora_path": "/path/to/peft/adapter"}'

curl -X POST http://localhost:8080/v1/unload_lora_adapter \
  -H 'Content-Type: application/json' \
  -d '{"lora_name": "my-tune"}'
```

A loaded adapter appears in `/v1/models` and is selected per request by naming
it in `model`. Loading an existing name replaces that adapter. Loading fails with an
explanation if the adapter is incompatible or no slot is available for a new name.
The default rank limit is 32; use
`--max-lora-rank` at startup for larger supported adapters.

Replacement and unloading wait for requests already using that adapter to finish.
Those requests keep their original policy for their entire response. New requests for the
same adapter wait during the update; requests for other adapters and the base model can
continue. After unloading, requests still waiting for that adapter receive an error.

Waiting for active requests is bounded by `--pending-timeout-ms`. An update that times out returns HTTP 503
and leaves the current adapter unchanged. An incompatible replacement also leaves it
unchanged. Replacing an adapter works with `--max-loras 1`.

## Responses API

`POST /v1/responses` accepts `input` (string or typed items), `instructions`,
`previous_response_id`, `metadata`, `tools`, `tool_choice`, `store`, and `stream`, and emits
semantic SSE events when streaming. Stored state is **process-local** — it does not survive a
restart and is not shared between replicas — and bounded by `--response-store-max-records`
(1024) and `--response-store-max-mib` (256). Pass `store: false` for stateless use.

Background execution and compaction are unsupported. `/v1/responses/compact` returns
`400 compaction_not_supported`. `/v1/responses/{id}/cancel` returns
`400 background_not_supported` for a stored id and `404` for an unknown id; it does not cancel
foreground generation. Disconnecting the client cancels its active generation request.

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","input":"Suggest three easy vegetarian dinners.","store":false}'
```

## Anthropic Messages

`POST /v1/messages` and `POST /v1/messages/count_tokens` support Anthropic-style requests,
including streaming and a top-level `system` prompt. Point your Anthropic client at
`http://127.0.0.1:8080`.

## Embeddings

Start a separate embeddings server with `surogate serve --embed`; see the
[GPU and CPU examples](serving-models.md#embedding-model-cpu-and-gpu). `POST /v1/embeddings`
accepts one string, an array of strings, one array of token ids, or an array of token-id arrays.

```bash
curl http://127.0.0.1:8413/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-300m","input":["a first sentence","a second one"]}'
```

The response contains a `data` array with one item per input. Each item has an `index` and an
`embedding` vector. Vectors are normalized, so you can compare them using a dot product.
