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
`--api-key-file PATH` reads the key from a file instead: one line of visible ASCII, surrounding
whitespace ignored. This keeps the key out of the process's command line, which every local user
can read in `/proc`. The server refuses `--api-key` and `--api-key-file` together, and warns when
other users can read the file.
`--cors` enables browser cross-origin requests. `--served-model-name ID` overrides the model id
the server reports, which is how you keep a client's hard-coded model string working.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat Completions, streaming or not |
| `POST` | `/v1/chat/completions/tokens` | Chat completions with an optional `tokens` array for exact prompt ids |
| `POST` | `/v1/completions` | Raw text completion without a chat template |
| `POST` | `/v1/decisions` (stable v1; aliases `/api/alpha/decisions`, `/api/v1/decisions`) | Decisions API: several single-token questions over one shared state, with opt-in [thinking](decisions.md#thinking) for unsure questions; see [decisions.md](decisions.md) |
| `POST` | `/tokenize` | Render and tokenize `messages` or a raw `prompt` without generating |
| `POST` | `/v1/responses` | Responses API |
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
| `GET` | `/health` | Process health; also answers while a model sleeps. `503` once an inference engine has stopped (see below) |
| `GET` | `/kv_stats` | Cache memory use and active or waiting requests per model |
| `GET` | `/metrics` | Prometheus counters and gauges for throughput, requests, KV, weights, and sleep state |
| `POST` | `/v1/embeddings` | Embeddings — served by `surogate serve --embed`, a separate process |

On a server with several models, add `?model=NAME` to sleep, wake, sleep-state, and adapter
management requests to select the model.

If the GPU runs out of memory, the server does not stop. When it happens while a request is being
admitted, only that request fails. When it happens in the middle of a batch, every request being
generated at that moment fails. Failed requests get `429` with error code `server_overloaded`,
which clients may retry. Requests still waiting in the queue, and later ones, are served as
usual. An engine that fails in a way it cannot recover from makes `/health`
answer `503` with `"status": "unavailable"`. About two seconds later the process exits with
status 1, so a supervisor such as systemd can restart it and health checks stop routing to it.

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
| `parallel_decoding` | Opt into independent boolean/enum classification on JSON-schema chat requests; see [Parallel constrained decoding](#parallel-constrained-decoding) |
| `parallel_tool_calls` | Defaults to `true`; `false` returns at most one tool call |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`; the loaded template must support the requested value |
| `chat_template_kwargs.enable_thinking` | Per-request thinking toggle; also accepted as top-level `enable_thinking` |
| `chat_template_kwargs.preserve_thinking` | Keep earlier assistant reasoning in later prompts; also accepted as top-level `preserve_thinking` |
| `ignore_eos` | Ignore the model's stop tokens; explicit stop strings and context limits still apply |
| `min_tokens` | Suppress stop tokens until this many tokens have been generated; stop strings remain active |
| `tokens` | Nonempty array of exact prompt token ids to use instead of the formatted messages; `messages` is still required |
| `add_generation_prompt` | Whether the template appends the assistant-generation prefix; defaults to `true` |
| `logprobs`, `top_logprobs`, `prompt_logprobs`, `return_token_ids` | Raw token details for text and tool-call Chat Completion responses, including SSE; see below |

If the server was started with `--greedy`, temperature is always zero even when a request
specifies another value.

Message content may be a plain string or an array of `text`, `image_url`, and `video_url` parts.
Images and video require `--vision` and a supported vision model; sources may be
local paths, HTTP(S) URLs, or base64 data URIs. `input_audio` is refused with
`modality_not_supported`.

For automatic tool choice, start the server with `--enable-auto-tool-choice` and a matching
`--tool-call-parser` (default `qwen3_xml`; `hermes`, `spark25`, `muse_glimmer`, `llama3_json`, and `llama4_json` are also
accepted). The model generates tool calls, and your application executes them.

Tool choice follows these rules for Chat Completions and Responses:

| Choice | Behavior |
|---|---|
| `none` | Tools are disabled, and their argument schemas are not enforced. |
| `auto`, with no tool marked `strict: true` | The model freely chooses an answer or tool calls. Argument schemas are not enforced. |
| `auto`, with at least one tool marked `strict: true` | Normal text remains allowed. When the model starts a tool call, its name and arguments are constrained. Tools explicitly marked `strict: false` remain exempt from argument-schema enforcement; tools with omitted `strict` use their schemas. |
| `required` | A completed response must contain a tool call. Argument schemas apply unless a tool explicitly sets `strict: false`. |
| Named function | A completed response calls the selected function once. Its argument schema applies unless it explicitly sets `strict: false`. |

As in vLLM, automatic choice with `parallel_tool_calls: false` returns only the first call if
multiple calls were generated. The setting does not change automatic generation; usage and
raw token details still cover the whole generated sequence.

Constrained tool arguments support nested objects and arrays, required properties,
additional-property rules, enums, constants, numeric bounds, array length limits, `anyOf`,
and local references. Declare required fields in `properties`. Their schema support is narrower
than JSON answer formats: string patterns, string length limits, formats, `multipleOf`,
`allOf`, `oneOf`, and assertion siblings alongside `$ref` or `anyOf` are currently rejected.
Unsupported tool constraints return HTTP `400` with `tool_schema_not_supported`; they are
not silently ignored. Schemas on disabled or unconstrained tools are not compiled.

Constrained tools require `ignore_eos: false`, `min_tokens: 0`, and no custom stop strings.
If generation reaches its output or context limit, the call may be incomplete; no executable
tool call is returned from that incomplete response. Validate unconstrained arguments in your
application before executing them.

When tool-derived constraints are active, they take precedence over `response_format`
(or Responses `text.format`), matching vLLM. With unconstrained automatic tools, a JSON answer
format still applies to the generated answer.

`logprobs: true` returns each generated token's log-probability, text, and bytes in
`choices[].logprobs.content`. Add `top_logprobs: 5` for the five most likely alternatives at
each position; counts from 0 through 20 are supported. The selected token is always scored,
even when it is outside those alternatives.

`prompt_logprobs: 5` returns a top-level `prompt_logprobs` array aligned with the prompt
tokens. Its first entry is `null`, because the first token has no preceding context. Each
remaining entry maps token ids to `logprob`, `rank`, and `decoded_token`, including the actual
prompt token and up to five alternatives. Use `prompt_logprobs: 0` to score only the actual
prompt tokens. `return_token_ids: true` includes the exact prompt and completion ids.

Probabilities use the full model distribution before temperature, penalties, logit bias, or
sampling constraints, matching vLLM's default raw log-probabilities. They are not probabilities
renormalized over the tokens allowed by a schema or sampling filter. Generated scores include
reasoning and tool-call syntax. Ordinary, MTP, and DFlash generation support these fields,
including pipeline serving.

Numerical rounding can cause small differences between scores returned during generation
and scores from a separate prompt-scoring request. Use the generated-token scores when
you need the probabilities from the original rollout.

Requesting and retaining scores adds work and uses system memory. Repeated or extended
prompts can reuse cached scores when the model, adapter, and prefix match and the cached entries contain enough alternatives for the
new request. Missing scores are computed as needed; an unscored prefix may require a fresh
prefill. Cached scores are discarded when their prefix is evicted or the adapter is replaced.

With `stream: true`, generated-token scores arrive incrementally in chunks with an empty
`delta`. Concatenate `choices[].logprobs.content` across those chunks. Prompt scores accompany
the first score chunk; requested token-id arrays are sent once before the finish chunk.

### Usage and cached tokens

Every response reports `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens` and
`prompt_tokens_details.cached_tokens`. The last is the part of the prompt the prefix cache
supplied instead of prefilling, which a price for cached input applies to. `prompt_tokens`
still counts the whole prompt, cached part included. `/v1/completions` reports the same
fields. When streaming with `stream_options.include_usage`, the usage chunk carries them too.
The number equals `prefix_cache_hit_tokens` in the request's `request_done` log record
(`--request-log-jsonl`), clamped to the prompt. The Responses API reports the same number as
`usage.input_tokens_details.cached_tokens`, and Anthropic Messages as
`usage.cache_read_input_tokens`.

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
`max_tokens`, keep `ignore_eos` false and `min_tokens` at zero, and omit custom `stop` strings. JSON is
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
Both formats work with streaming and non-streaming requests.

### Parallel constrained decoding

Set `parallel_decoding: true` on `/v1/chat/completions` to classify several independent fields
from the same conversation. Each field receives its highest-probability allowed value, and
`choices[0].message.content` contains the assembled JSON object. No separate checkpoint is
required. Use an instruction-tuned model suited to your classification task.

```json
{
  "model": "my-model",
  "messages": [{"role": "user", "content": "Our production server is down. Please help immediately."}],
  "parallel_decoding": true,
  "temperature": 1,
  "max_tokens": 128,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "triage",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "urgent": {"type": "boolean", "description": "Does this need immediate attention?"},
          "category": {"type": "string", "enum": ["outage", "billing", "other"]}
        },
        "required": ["urgent", "category"],
        "additionalProperties": false
      }
    }
  }
}
```

With the OpenAI Python client, pass the opt-in through
`extra_body={"parallel_decoding": True}`. Start the server with `--max-num-seqs 4` or higher
to evaluate fields concurrently. The shared conversation is processed once; each field then
continues from it. Increasing the sequence limit allows more fields to run together, up to 64.
A server with one sequence slot also accepts these requests. Classification needs GPU cache
space for the shared conversation and the field continuations.
Speedups depend on the model, prompt length, number of fields, and available memory; benchmark
your workload.

The response also includes `parallel_decoding.fields`. Each field contains its selected
`value` and a `probabilities` array of `{ "value": ..., "probability": ... }` entries.
Probabilities sum to one over that field's allowed choices. They describe the model's
constrained choice distribution, not calibrated confidence or a guarantee of correctness.
Multi-token choices, including choices that share prefixes, are supported.

Fields are evaluated independently. Use ordinary structured output when one field should depend
on another. This mode accepts a flat object with 1–64 required properties, each a boolean or an
enum of up to 256 distinct scalar values. Set `additionalProperties: false`. Nested objects,
arrays, optional fields, references, and other schema constraints return HTTP `400`. Very long
choices or large combined schemas can exceed the request limit; shorten them or split the request.

Classification always chooses the highest-probability value. `temperature` follows the normal
request/CLI/model default precedence and controls the choice probabilities; zero uses unscaled
probabilities. Sampling filters and repetition penalties are not used. Explicit non-neutral
filters, penalties, or `logit_bias` return HTTP `400`. Custom stops, tool calls, media, explicit
token prompts, and token-level logprobs/IDs are unavailable in this mode. Give `max_tokens`
enough room for any allowed result; an insufficient budget returns HTTP `400`.

With `stream: true`, the complete JSON is delivered after classification finishes, followed by
field probabilities on the final chunk. Usage counts the shared prompt once plus field-query
suffixes, and counts the returned JSON as completion tokens. Ordinary JSON-schema requests
keep their existing behavior when the opt-in is absent or false.

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
selection. Set `logprobs` to an integer from 0 through 20 to receive token probabilities and
that many alternatives in `choices[].logprobs`. `prompt_logprobs` and `return_token_ids` work
as described above. Streaming delivers score chunks incrementally; concatenate their token
and probability arrays. Text offsets continue across chunks. Batched prompts and token arrays
in `prompt` are refused. Requests for `echo`, `suffix`, or `best_of > 1` are unsupported on
this endpoint.

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

### Request ids and the request log

A caller, such as a gateway, can name each request with an `X-Request-Id` header. The server
echoes the header on the response, refusals included. It also writes the id as
`client_request_id` into every `request_start`, `request_done`, `request_rejected` and
`request_error` record of `--request-log-jsonl`. The protocols covered are Chat Completions,
Completions, Anthropic Messages, Responses and decisions. Records of requests sent without the
header have `client_request_id: null`.

Only ids of 1 to 128 visible ASCII characters are used. Any other value is ignored, neither
echoed nor logged, so a header cannot inject text into the log. The server warns about this once
on the console, without the value. Header values are percent-decoded, so an id must not contain
`%`; UUIDs and similar ids are unaffected. A request that carries more than one `X-Request-Id`
header gets no id at all. A gateway should therefore replace the client's header with its own,
not add a second one. With `--cors`, browsers may send the header and read the echoed one.

Every request ends in exactly one terminal record: `request_done`, `request_error` or
`request_rejected`. A request refused before generation ends in `request_rejected`. This includes
a request whose client left while it was being prepared. A request refused during preparation
has no `request_start`.

A streamed request whose client leaves after that still ends in `request_done`. This includes a
client that leaves before the first event, and a parallel-decoding stream. The record carries
the prompt tokens, the cached tokens (`prefix_cache_hit_tokens`) and the completion tokens
generated before generation stopped. A caller that never received the final usage chunk can
settle the request from that record. Two details:

- `finish_reason` is usually `"cancelled"`. It can also be the normal reason, for example
  `"stop"`, when generation finished before the server noticed the client had left.
- `prompt_tokens` and `computed_prefill_tokens` count the whole prompt, even when the client
  left before its prefill finished. A dropped parallel-decoding stream reports
  `completion_tokens: 0`.

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
`metadata`, `tools`, `tool_choice`, and `stream`, and emits semantic SSE events when streaming.
Requests are stateless: include the complete conversation history in `input`, including prior
assistant outputs and tool results. The server does not store conversations or provide
response retrieval, deletion, or input-item listing endpoints.

Omit `store` or set it to `false`. `store: true` and non-null `previous_response_id` return
HTTP `400`. Responses report `store: false`; their IDs cannot be used to continue a conversation.

Responses accepts `tool_choice: "auto"`, `"none"`, `"required"`, or a named function such as
`{"type":"function","name":"weather"}`. Define function tools with `name`, `parameters`, and
optional `strict` directly on the tool object. Both strict tools and `parallel_tool_calls: false`
work with streaming and non-streaming requests; the tool-choice rules above apply unchanged.

For generated-token probabilities, set `include: ["message.output_text.logprobs"]` and
optionally `top_logprobs` (0–20). Scores appear on the output-text content part and in the
`response.output_text.done` event when streaming. Incremental scores also arrive in
`response.output_text.delta` events, sometimes with an empty text delta; concatenate their
`logprobs` arrays. The done event contains the complete array. Scores cover the raw generated
tokens, including reasoning and tool syntax, using the same probabilities as Chat Completions.

Background execution and compaction are unsupported. `/v1/responses/compact` returns
`400 compaction_not_supported`. Disconnecting the client cancels its active generation request.

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","input":"Suggest three easy vegetarian dinners.","store":false}'
```

## Anthropic Messages

`POST /v1/messages` and `POST /v1/messages/count_tokens` support Anthropic-style requests,
including streaming and a top-level `system` prompt. Point your Anthropic client at
`http://127.0.0.1:8080`.

Usage follows Anthropic's convention:

- `input_tokens` excludes the prompt tokens the prefix cache supplied.
- `cache_read_input_tokens` reports those tokens. Add the two to get the whole prompt.
- `cache_creation_input_tokens` is always 0, because nothing is written to a cache on the
  client's behalf.

When streaming, `message_start` is sent once the prompt is prefilled, before the first content
block, so it already carries the cache split. The final `message_delta` repeats the cumulative
usage with the same split. A stream that fails before its prompt is prefilled, for example on a
queue timeout, sends only an `error` event and no `message_start`.

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
