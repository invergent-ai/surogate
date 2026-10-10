# Inference CLI

```bash
surogate serve <model> [options...]                       # OpenAI/Anthropic HTTP server
surogate serve --generate <model> --prompt "..."          # one-shot generation
surogate serve --embed <model> [--frontend DIR]           # embeddings server
```

`<model>` is a Hugging Face repo id, a local safetensors directory, or a GGUF file. The first
start prepares and caches the model; later starts reuse that preparation. Keep source GGUF
files in place while using their cache entries.

```bash
surogate serve Qwen/Qwen3.6-27B
surogate serve ~/models/qwen3.6-27b-hf/
surogate serve ~/models/qwen3.6-27b-Q4_K_M.gguf
```

From a source checkout, build serving support with `make serve-build`.
`surogate serve --engine-help` prints server help. Add `--generate` or `--embed` to see the
options for those modes. Help does not prepare or download a model. Options may appear before
or after the model, and value options accept both `--flag value` and `--flag=value`.

## Server options

### Endpoint and access

| Flag | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | Bind address; use `0.0.0.0` to accept remote connections |
| `--port N` | `8080` | HTTP port |
| `--api-key KEY` | none | Require this key through a bearer token or `x-api-key` header |
| `--api-key-file PATH` | none | Read that key from a file instead, keeping it out of the process's command line |
| `--openrouter-models-file PATH` | none | Serve an operator-supplied [OpenRouter catalog](api.md#openrouter-provider-catalog) at `/openrouter/v1/models` |
| `--served-model-name ID` | model argument | Name clients use in the `model` field |
| `--cors` | off | Allow browser cross-origin requests |
| `--max-request-mib N` | 384 | Maximum request body size |
| `--request-log-jsonl FILE` | none | Append request records to this file |
| `--log-stats-interval-ms N` | 5000 | Throughput log interval; `0` disables |

### Context and KV cache

The context limit controls how much text a request can use. The KV cache holds information
from earlier tokens; its capacity affects how many requests can run together.

| Flag | Default | Meaning |
|---|---|---|
| `--max-model-len N\|auto` | `auto` | Context limit per request; `auto` fits available GPU memory up to the model's maximum |
| `--kv-capacity N\|auto` | follows context | Total cache capacity in tokens; `auto` sizes from free GPU memory, leaving 1024 MiB |
| `--kv-cache-dtype auto\|fp8\|bf16\|int8` | `auto` | Choose cache precision automatically for the model, or force a specific format |
| `--kv-cache-dtype-skip-layers L,...` | none | Keep the listed attention layers' cache at BF16 |
| `--elastic-kv` | on | Grow cache memory use with demand |
| `--no-elastic-kv` | off | Reserve the full cache in GPU memory instead of growing memory use with demand |
| `--elastic-kv-overcommit` | off | Let several models share unused GPU memory for their caches |
| `--gpu-memory-limit-mib N` | whole card | Cap everything the server holds on each GPU; automatic sizing fits in it (see [Sharing a GPU](#sharing-a-gpu)) |
| `--no-cache` | off | Rebuild the prepared model cache on disk |
| `--no-prefix-reuse` | off | Disable reuse of compatible earlier prompts |
| `--enable-prefix-caching`, `--no-enable-prefix-caching` | enabled | Alternative spellings for enabling or disabling prompt reuse |
| `--enforce-eager` | off | Disable CUDA graphs for debugging |
| `--batch-invariant` | off | Make each request's results independent of the other requests batched with it (see [Reproducible results](#reproducible-results)) |

With `auto`, the cache grows with demand up to the capacity it was sized for. The 1024 MiB left
over stays free for CUDA graphs and working buffers. The cache goes past its capacity only when
its pages are too scattered to fit, and then only while that 1024 MiB is still free.

When `--max-model-len` is automatic, omitted `--kv-capacity` also defaults to `auto`. When
context is explicit, omitted cache capacity defaults to that same token count. Use
`--kv-capacity auto` explicitly to make more cache available for simultaneous requests.

Compatible conversation context is cached automatically, including after a response
ends on EOS or a stop string. Completed conversations can also be reused after an
unrelated request. Each model has a 512 MiB host cache budget per GPU; conversations
that exceed that budget are not retained there. Older entries
can be released as new requests arrive. Changing earlier messages, requesting additional prompt
logprobs, or a cache miss can require some or all of the prompt to be processed again.
Use `--no-prefix-reuse` to disable reuse.

Cache precision `auto` selects BF16 for models such as Qwen3 and Llama, and FP8 for hybrid
models such as Qwen3.5/3.6/3.8, including with DFlash. FP8 uses half the cache storage of BF16.

With `--elastic-kv-overcommit`, `--kv-capacity` becomes a guaranteed minimum; `auto` guarantees
enough for one full-context request per model. See [Serving models](serving-models.md#several-models-on-one-gpu).

### Reproducible results

By default a request's logits, and so its log-probabilities and greedy tokens, can change in the
last bits with the other requests the server batches it with: a BF16 projection's algorithm
follows the number of tokens in the step, some fused kernels split their sums by it, and a long
prompt that shares a step is cut where the step's token budget runs out. The differences are
rounding-level, but a sensitive model turns them into visible ones: on Qwen3.5-0.8B, scoring
sentences sixteen at a time moved sentence log-probabilities by up to 4 nats against scoring
them one at a time, and changed some greedy answers.

`--batch-invariant` removes that dependence for BF16 text models. Every BF16 projection runs a
kernel whose per-value reduction order is fixed by the matrix alone; the GDN gating projection
takes a fixed-order kernel; and prompts are cut at fixed positions (every
`--max-num-batched-tokens` less 128 tokens, 1920 by default, for up to 128 sequences) whatever
shares their steps.
A request then returns the same bits alone, among sixteen, or in any arrival order. It implies
`--enforce-eager` and `--no-prefix-reuse` (a prefix hit continues from another request's state),
and cannot be combined with `--spec`. Other weight formats keep their own kernels (the GGUF ones
already compute a token the same way at every width; FP8, NVFP4 and W8 are not verified), and
images and pipelines across several GPUs are not covered.

Measured on an RTX 5090 with Qwen3.5-0.8B, against the same eager, no-reuse server without the
flag: short-prompt scoring 5% slower one at a time and 9% slower sixteen at a time, long-prompt
scoring 12% slower, sixteen concurrent generations unchanged, one generation stream 26% slower.

### Sharing a GPU

`--gpu-memory-limit-mib N` gives the server a budget of N MiB on each of its GPUs. The budget
covers weights, cache, working buffers and the CUDA context. Automatic sizing then works within
it instead of within the card's free memory:

- `--kv-capacity auto` and `--max-model-len auto`;
- `--host-moe-layers auto`, automatic expert slots and pipelines;
- the optional repacked weight copies.

An explicit `--kv-capacity`, or the capacity that an explicit `--max-model-len` implies, must fit
the budget with 1024 MiB left over, or the server refuses to start. So must the weights and
their load staging.

The cache never grows past the capacity it was sized for. The 1024 MiB left over absorbs
working buffers that grow later, so the server stays within its limit.

Several models in one server share the budget. To share a GPU between processes, give every
process on it a limit, and keep the limits below the card's memory less what the driver reserves.
Processes without a limit, such as other programs, can still take memory the others counted on.
`--expert-slots N` is not checked against the limit.

The server measures its own usage through NVML. Inside a container that hides host process ids,
NVML cannot attribute usage to the process. The server then counts everything allocated on the
device since it started, plus 512 MiB for the CUDA context, and prints a warning. Run such
containers with `--pid=host` for an exact figure.

With a limit set, the `device_free_bytes` reported by `/kv_stats` and `/metrics` is what the
server may still allocate.

### Devices

| Flag | Default | Meaning |
|---|---|---|
| `--device N` | 0 | Use one GPU |
| `--devices A,B,...` | — | Split a supported model across the listed GPUs, one pipeline stage per GPU |
| `--data-parallel` | off | With `--devices`, run a whole copy of the model on each listed GPU instead |

`--devices` takes precedence over `--device`; its first GPU is also used for model preparation.
With one GPU, preparation follows `--device`. `SUROGATE_CONVERT_DEVICE` overrides the preparation
device when needed. GPU indices follow `CUDA_VISIBLE_DEVICES`.

All supported text-generation families can use `--devices`. MTP and DFlash can also use
multiple GPUs when the prepared model includes compatible draft weights. The maximum
number of GPUs depends on the checkpoint; if the requested split is rejected, use fewer GPUs.

Additional models can share a GPU, use different GPUs, or use their own GPU groups. They
inherit the primary model's placement unless you override it:

```bash
surogate serve /path/to/primary.sinfer --devices 0,1 --enable-sleep-mode \
  --model second=/path/to/second.sinfer,devices=2:3 \
  --model small=/path/to/small.sinfer,device=4
```

Within `--model`, separate GPU indices with colons. Sleep and wake apply to every GPU used
by the selected model. The scheduler accounts for available memory on each GPU.

#### Pipeline or data-parallel

`--devices` on its own is pipeline parallelism: each GPU holds a contiguous range of layers,
and every token passes through all of them in turn. It is how a model too large for one GPU is
served, and it gives the longest context, because every card's memory left after the weights
holds cache. It does not add throughput: at any moment most of the cards wait for the one
working on the current layers.

When the model fits one GPU, add `--data-parallel`. Each listed GPU then runs its own complete
engine, all under one model name, and requests are spread across them:

```bash
surogate serve /path/to/model --devices 0,1,2,3,4,5 --data-parallel --max-num-seqs 16
```

- A conversation goes back to the replica that served its previous turn, which holds that
  prompt in its cache. Without that, a later turn of a long agentic conversation would be
  prefilled again on another card. A replica that already carries more than 1.25 times the
  average load gives the request to the least busy replica instead.
- A client that manages its own placement can send `X-data-parallel-rank: N` (vLLM's header)
  to send a request to replica N, counted from 0 in `--devices` order.
- Each replica has its own `--max-num-seqs`, request queue and cache: six replicas with
  `--max-num-seqs 16` serve 96 requests at once.
- Adapter loads, `/sleep` and `/wake_up` apply to every replica. `/v1/models` lists the model
  once; `/kv_stats` and `/metrics` report each replica separately, with a `replica` field or label.
- Extra models need an explicit `device=` or `devices=` placement under `--data-parallel`.

For a model that fits one GPU, this is much faster than a pipeline: measured on six RTX PRO
6000 cards with an agentic workload, six single-GPU engines decoded about 40 times more tokens
per second than one six-stage pipeline.

For independent replicas under separate names, add the same artifact as an extra model,
for example `--model replica=/path/to/primary.sinfer,device=2`. Select the replica with the
request's `model` field; each replica has its own request capacity and cache.

### Host offload

Use system RAM for part of a model when it does not fit in GPU memory. This reduces GPU memory
requirements but can make responses slower. The machine needs enough RAM to hold the offloaded
weights; that memory cannot be swapped out while the model is loaded.

| Flag | Default | Meaning |
|---|---|---|
| `--host-moe-layers N\|auto\|all` | off | Move experts from N MoE layers to RAM; `auto` chooses enough to fit on one GPU or each GPU in a multi-GPU run |
| `--gpu-layers N\|all`, `-ngl N`, `--n-gpu-layers N` | all | Keep the first N decoder layers on the GPU; `0` offloads all decoder layers, and `all` keeps them resident |
| `--offload-vision` | off | Store the image encoder and projector weights in system RAM |
| `--offload-embeddings` | off | Store the token embeddings in system RAM |
| `--offload-output-head` | off | Store the output-head weights in system RAM |

Whole-layer and component offload work on one or multiple GPUs. Combine the flags as needed;
GPU computation and the decode cache still need GPU memory. When a model shares its embeddings
and output head, offloading either one moves their shared weights to RAM.
MoE-specific options apply only to MoE models and only affect experts that are offloaded.

For a mixture-of-experts (MoE) model, try `--host-moe-layers` first. It generally transfers
less data than offloading whole layers.

```bash
surogate serve models/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --device 0 --host-moe-layers all --max-model-len 2048
```

### MoE expert cache

Every supported MoE generation model can cache offloaded experts on the GPU and use CPU
cores for some expert computation. Enable expert offload with `--host-moe-layers` or offload
whole layers with `--gpu-layers` before using these settings.

| Flag | Default | Meaning |
|---|---|---|
| `--expert-slots N` | automatic | Number of experts cached on the GPU; omitted or `0` sizes the cache from available memory. An explicit count must hold at least the experts selected for one token |
| `--host-expert-bank auto\|w8\|q4` | `auto` | Precision of offloaded experts. `w8` uses eight bits throughout; `q4` uses four bits, saving RAM but potentially reducing quality |
| `--cpu-moe-share F\|auto` | off | Fraction of expert work sent to CPU cores; `auto` measures the machine at startup |
| `--cpu-moe-prefill-share F` | 0 | CPU share during prompt processing; `0` disables it |
| `--cpu-moe-min-tokens N` | model default | Minimum token count before CPU sharing is used |

CPU shares must be numbers from `0` to `1`; decode also accepts `auto`. They apply to cache
misses, so a fully cached expert does not need CPU execution.

Automatic host precision keeps four-bit, five-bit, and wider source weights at suitable
precisions. Force `q4` only when you want the RAM saving from reducing wider weights to four bits.

### Scheduling

| Flag | Default | Meaning |
|---|---|---|
| `--max-num-seqs N` | 1 | Maximum active requests per model |
| `--max-num-batched-tokens N` | 2048 | Prompt tokens processed at a time; must be a positive multiple of 128 |
| `--max-pending-requests N` | 16 | Additional requests allowed to wait |
| `--pending-timeout-ms N` | 30000 | Time allowed for model wake-up, prompt preparation, and waiting to start generation |
| `--default-max-tokens N` | 8192 | Output limit when a request omits it |

Set `--max-num-seqs 256` (or higher) to allow more than 128 active requests per model.
More simultaneous requests need more memory and may increase response latency; pair this
setting with `--kv-capacity auto`. The pending timeout does not limit how long an
already-running response may take.
Wake waits count toward the pending-request limit. If the pending timeout expires,
the request returns HTTP 503; disconnecting while waiting releases its queue slot.

Incoming prompts can share a batch with requests already generating, including requests
using different LoRA adapters, DFlash, and images or video. On one GPU without `--spec`, prompts
that arrive together while nothing is generating are prefilled together too, up to
`--max-num-batched-tokens` per step: a burst of short prompts, such as one-question
`/v1/decisions`, costs a few steps rather than one step each. A decision, or a chat request with
`parallel_decoding`, takes the queue places its questions need, up to the smaller of their count,
`--max-num-seqs` and 64, before any of it runs. When the queue is busy it waits for them, in
arrival order and within `--pending-timeout-ms` (then HTTP 503), rather than run its shared prefix
and find no room for its questions. While it waits, the places it needs are kept for it, so other
requests can get HTTP 429 before the queue is full; `surogate_requests{state="reserving"}` counts
the requests waiting this way. Lower `--max-num-batched-tokens`
to reduce the time spent on each prompt chunk when streaming responsiveness matters;
larger values can improve prompt throughput. Active text requests continue generating throughout
image and video processing, on one or multiple GPUs. Sharing the GPU can delay image responses;
lower image resolution or fewer video frames can improve latency. DFlash needs room in the batch
for both prompt tokens and the proposed tokens it checks.

### Speculative decoding

Speculative decoding can make generation faster by proposing several tokens for the model to
check together. It works best when those proposals are often accepted. Measure it with your
workload: more simultaneous requests or heavy CPU offload can reduce the benefit.

| Flag | Meaning |
|---|---|
| `--spec mtp` | Enable MTP on a supported model; supports one or multiple GPUs |
| `--spec dflash` | Use a compatible separate drafter on one or multiple GPUs; supports BF16 or FP8 caches and can be combined with `--vision`. Qwen3.5 and the Gemma 4 mixture take the drafter at preparation; see [Preparing a DFlash pair](serving-models.md#preparing-a-dflash-pair) |
| `--dflash-model PATH` | Matching separate Muse-Glimmer DFlash GGUF; see [Muse-Glimmer](#muse-glimmer) |
| `--draft-tokens N` | Number of proposed tokens: 1–5 for MTP, 1–15 for DFlash; required with `--spec`. With `--spec-adaptive`, sets the maximum |
| `--spec-adaptive` | Adjust DFlash draft length to measured throughput. Can temporarily stop drafting and retry it later. Off by default |
| `--spec-max-lanes N\|all` | MTP checks drafts only while at most N requests are decoding. Default (or `0`) is 1, or `all` on a DGX Spark (GB10) outside a pipeline; `all` keeps checking at every concurrency level. Does not affect DFlash |
| `--lm-head-draft` | Draft with the model's smaller draft vocabulary (shortlist head). MTP already does this by default when the model provides one |
| `--full-head-draft` | Draft MTP proposals with the full output head instead of the shortlist head. Greedy output is the same either way, and sampled output keeps the model's distribution |

For workloads where DFlash acceptance or concurrency varies, use `--spec dflash --draft-tokens 15 --spec-adaptive`. The engine learns from completed decode rounds, so short requests may finish before it has enough measurements. It keeps separate measurements for different batch sizes and context lengths. Adaptive mode uses more GPU memory and can take longer to start. Calibration and periodic retries add some overhead; compare throughput on your workload. This option also works with vision prompts, BF16 or FP8 caches, and pipeline serving. Omitting `--spec-adaptive` keeps the requested draft length fixed. Changing draft lengths can also change individual token scores or greedy wording through numerical rounding, especially with an FP8 cache.

MTP needs the checkpoint's MTP weights, which some community exports omit. DFlash
needs a compatible drafter included during model preparation. Missing draft weights produce
a startup error. See [Preparing a DFlash pair](serving-models.md#preparing-a-dflash-pair).

With DFlash, `--kv-cache-dtype fp8` reduces cache memory for both the target and drafter.
Use `--kv-cache-dtype bf16` for higher cache precision. FP8 can change generated output and
draft acceptance, so compare memory use and generation speed on your workload.

For image or video conversations, prepare the pair with the target's vision weights and
start the server with `--vision --spec dflash`. Text-only requests can use the same server.
Draft acceptance and speed depend on the prompt; compare with ordinary decoding for your workload.

### Serving several models from one process

Use `--model name=path[,key=value...]` for each additional model. Currently, the additional
model's path must be a prepared `.sinfer` file from the serving cache. Prepare the model
separately first, then use its cache path; see [Serving models](serving-models.md#several-models-on-one-gpu).
The first, positional model still accepts a repo id, safetensors directory, or GGUF.

| Setting | Meaning |
|---|---|
| `--model name=path` | Add a model; clients select it with `model: "name"` |
| `kv-tokens=N` | Override this model's cache budget; required with `--no-elastic-kv` |
| `max-num-seqs=N` | Override its maximum simultaneous requests |
| `device=N` | Place this model on one GPU |
| `devices=A:B:...` | Split this model across these GPUs |
| `max-model-len=N` | Override its context limit |
| `spec=mtp\|dflash` | Enable speculation for this model; defaults its draft token count to 3 |
| `draft-tokens=N`, `spec-max-lanes=N\|all`, `spec-adaptive=true\|false` | Override this model's speculation settings |
| `lora=name:path` | Add an adapter to this model; repeatable |
| `priority=high\|normal\|low` | Priority when models compete for memory; defaults to `normal` |
| `--model-priority high\|normal\|low` | Set the first model's priority |

Put per-model settings after its path, separated by commas. Omitted context, concurrency,
and cache settings inherit from the first model. Speculation is off unless enabled for each
additional model. `/v1/models` lists all served model and adapter names.

With `--enable-sleep-mode`, the models do not all need to fit in GPU memory at once. Requests
for a sleeping model wait while the server frees room and wakes it. Less recently used,
lower-priority idle models are preferred for sleeping. After a grace period, a busy model may
be paused for an equal- or higher-priority request, then resumed when memory is available.
Higher-priority models stay ready longer after use, but idle models can still be put to sleep.

Allow enough system RAM for the saved models. Multi-model startup prepares this memory in
advance. Management endpoints accept `?model=NAME` to select a particular model.

### Sleep mode

| Flag | Default | Meaning |
|---|---|---|
| `--enable-sleep-mode` | off | Enable `/sleep` and `/wake_up` to release and restore GPU memory while preserving model state across its GPUs |

See the [API guide](api.md#sleep-mode) for commands and behavior while asleep.

### LoRA adapters

Serve PEFT adapters beside the base model. A request selects an adapter by putting its name in
`model`; the base model's name selects the unadapted model. Different adapters can serve
requests at the same time.

| Flag | Default | Meaning |
|---|---|---|
| `--enable-lora` | off | Enable adapters, including loading them later through the API |
| `--lora-modules name=path,...` | — | Adapter directories to load at startup |
| `--max-loras N` | 1 | Maximum loaded adapters per model |
| `--max-lora-rank N` | 32 | Largest adapter rank accepted |

LoRA and DoRA adapters support the following modules, where they exist in the checkpoint:

| Model or component | Adapter modules |
|---|---|
| Llama, dense Qwen layers, Gemma 3 and Gemma 4 | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Token embeddings and output head | `embed_tokens` (or the checkpoint’s `embedding`, `tok_embeddings`, `word_embeddings`), `lm_head` |
| Gemma 4 E2B/E4B per-layer inputs | `per_layer_input_gate`, `per_layer_projection` |
| Qwen3.5/3.6 and Qwen3.8-Flash-Next attention | `q_proj` (including its attention gate), `k_proj`, `v_proj`, `o_proj` |
| Qwen hybrid linear attention | `linear_attn.in_proj_qkv`, `linear_attn.in_proj_z`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `linear_attn.out_proj` |
| Spark-X2.5 | `q_k_v_proj`, `g_proj`, `out_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LFM2, LFM2-MoE and the text decoder of LFM2-VL | Attention projections, `conv.in_proj`, `conv.out_proj`, and dense `feed_forward.w1`, `w2`, `w3` |
| Qwen vision towers | Patch projection, attention QKV/output, MLP projections, merger and deepstack projections |
| LFM2-VL vision tower | Patch projection, attention Q/K/V/output, MLP projections and multimodal projector |
| Gemma 3/4 vision | Patch, attention and MLP projections; Gemma 4 output projection |
| Routed experts on supported MoE models | Each expert's `gate_proj`, `up_proj`, `down_proj`; LFM2 expert names `w1`, `w3`, `w2` are also accepted |
| MoE router and shared expert | `mlp.gate` (or the checkpoint's `router.proj` / `feed_forward.gate`), `mlp.shared_expert_gate`, and the shared expert's `gate_proj`, `up_proj`, `down_proj`, where present |

Keep the checkpoint's module paths, including expert numbers. Surogate's fused
`gate_up_proj` adapter exports are also accepted. Expert adapters use separate A/B matrices
for each expert and support ranks up to 256. Adapters work with supported quantized base
models, including NVFP4.
Use the smallest `--max-lora-rank` and `--max-loras` that fit your adapters, especially for MoE models.

Expert adapters support batched prefill and CPU expert computation. CPU weight offloading,
`--cpu-moe-share` and `--cpu-moe-prefill-share` remain available with adapters enabled.

Adapters may include DoRA magnitudes, LoRA B biases, saved base biases, and per-module rank
or alpha overrides. Set `--max-lora-rank` to cover the largest rank used by any module.
Vision adapters require a model served with `--vision`; embedding and output-head adapters
must match the served vocabulary.

Saved full embedding and output-head weights, including `modules_to_save` exports, are
also supported. These consume additional GPU memory for each loaded adapter.
Full replacements of other modules and targets outside the supported
modules still need `surogate merge` before serving. If an adapter contains unsupported weights,
loading fails with a message naming the affected tensor or module and explaining the next step.
Runtime loading also writes a warning to the server log. Unsupported weights are never silently
skipped; a rejected replacement leaves the currently loaded adapter available.

Additional models have their own adapters through `lora=name:path` in `--model`. Every model
and adapter name must be unique. The runtime load/unload endpoints accept `?model=NAME`; see
[LoRA adapters at runtime](api.md#lora-adapters-at-runtime).

### Sampling defaults

For each sampling field, the precedence is:

1. Explicit request parameter.
2. Server CLI setting.
3. The model's `generation_config.json`.
4. Built-in family default.

The generation config supplies `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`,
`presence_penalty`, and `frequency_penalty`. Its values apply to both thinking modes; missing
or `null` fields retain the family default for that mode. For GGUF models, the file's optional
sampling metadata supplies the generation defaults. If neither source provides a field,
the family default applies.

`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, and `--frequency-penalty`
set server defaults. `--seed` sets the random seed, which can also be overridden per request.
`--greedy` always forces temperature zero, including when a request asks for another value.
Requests also accept `repetition_penalty`.

Sampling honors the requested `top_k` without a fixed candidate cap; `top_k: 0` leaves the
whole vocabulary available. `min_p` removes tokens with low probability relative to the most
likely token, and `top_p` keeps the smallest remaining prefix reaching the requested probability
mass. These filters also apply during speculative decoding.

### Thinking

Without a flag, each model thinks or not as its chat template does by default (Qwen and
Granite think, Gemma 4 answers directly). `--thinking` or `--no-thinking` sets that default
for every request on a model whose template has the toggle.
`--preserve-thinking` keeps earlier assistant reasoning in later prompts. These settings are
independent and can be overridden per request. Supported reasoning-effort values depend on
the model's chat template.

| Flag | Default | Meaning |
|---|---|---|
| `--reasoning-parser NAME` | `qwen3` | Read reasoning in the model's output format; also accepts `deepseek_r1`, `glm4_moe`, `think`, or `none`/`off` |
| `--tool-call-parser NAME` | `qwen3_xml` | Read tool calls; also accepts `hermes`, `spark25`, `muse_glimmer`, `llama3_json`, `llama4_json`, or `none`/`off` |
| `--enable-auto-tool-choice` | off | Allow the model to choose a tool automatically; requires an enabled tool parser |
| `--chat-template FILE` | model template | Use this Jinja file to format chat prompts |

### Decisions calibration

| Flag | Default | Meaning |
|---|---|---|
| `--decision-temperature T` | `1` | Calibration temperature of the [decisions endpoint](decisions.md#calibration-temperature): every answer is read from `softmax(option logits / T)` |
| `--decision-attempts N` | `3` | How many times a [decisions](decisions.md#non-finite-logits) request runs while its option logits come out non-finite (1..16; `1` returns the error at once) |

`T` must be a finite number greater than zero (anything else is refused at startup); `1`
returns the model's own distribution unchanged. It applies to every decisions request
the process answers: every question type, every route alias, every served model (including
those added with `--model`) and every adapter. `T > 1` softens an overconfident model and
`T < 1` sharpens an underconfident one. The chosen option never changes, only the
probabilities and what is computed from them (`confidence`, `noul`, a score's expected
level). Other endpoints are unaffected, and it is independent of `--temperature`, which the
decisions readout never uses. The value appears in the `server_start` and decisions request
records of `--request-log-jsonl`, and in the startup log when it is not 1.

Fit `T` on a held-out calibration split, never on evaluation data, and state the recommended
value in the model card; see [Choosing the temperature](decisions.md#choosing-the-temperature).

```bash
surogate serve ./models/decision-model/ --decision-temperature 2.5
```

### Vision

`--vision` enables images and video on supported models. `--media-cache-mib N` (default 1024)
sets how much processed media to retain for reuse; `0` disables retention. `--media-live-mib N`
(default 2048) limits memory for media currently being processed or used by requests.
`--media-preprocess-threads N` chooses processing threads; `0` selects automatically, up to 16.

Qwen3-VL dense and MoE, vision-enabled Gemma 3 and Gemma 4, and LFM2-VL/LFM2.5-VL
checkpoints support image requests. Video requests sample frames; Gemma 3 and LFM2-VL
receive those frames as a sequence of images. For example:

```bash
surogate serve Qwen/Qwen3-VL-2B-Instruct --vision --port 8080
surogate serve Qwen/Qwen3-VL-30B-A3B-Instruct --vision --devices 0,1 --port 8080
surogate serve google/gemma-3-4b-it --vision --port 8080
surogate serve google/gemma-4-E2B-it --vision --offload-vision --port 8080
```

Choose one command and size the GPU group for the checkpoint. Send media through the
[chat API](api.md#chat-completions) or Responses API. Without `--vision`, the same checkpoint
serves text prompts.

For a Qwen3-VL, Gemma 3/4, or LFM2-VL GGUF, supply the matching vision projector from the same model release:

```bash
surogate serve /models/Qwen3-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj /models/mmproj-BF16.gguf --vision --port 8080
```

Dense and MoE GGUFs are supported, including split text files; pass the first shard.
If exactly one compatible `mmproj*.gguf` is beside the text model, it is selected automatically.
Otherwise, use `--mmproj` explicitly. Keep all source GGUF files available after preparation.
Supported projector quantization is preserved during preparation. Use `--offload-vision`
to reduce GPU memory further, and add `--offload-embeddings --offload-output-head` when needed.
LFM2-VL text weights use the same 8-bit serving format as LFM2; the other families retain
supported GGUF text quantization.
Qwen3-VL image and video serving has been checked with the 2B and 30B-A3B GGUF checkpoints.
The complete 235B-A22B checkpoint has not yet been tested.

### Muse-Glimmer

Muse-Glimmer-30B GGUF supports text, image, and video chat. Download the text GGUF and its
matching `mmproj` from [the model release](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF), then run:

```bash
surogate serve /models/Muse-Glimmer-30B-UD-Q4_K_XL.gguf \
  --mmproj /models/mmproj-Muse-Glimmer-30B-BF16.gguf --vision \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer
```

Add `--devices 0,1` to split the model across two GPUs. Omit `--vision` for text-only
serving. Reasoning is returned separately in `reasoning_content`.

Send videos using `video_url` in the chat API. The first conversion downloads a small
video-weight file from the original model release; subsequent starts use the local cache.

To enable the separate DFlash assistant, also download `dflash-kquant.gguf` from the
same GGUF release and add:

```bash
--spec dflash --dflash-model /models/dflash-kquant.gguf --draft-tokens 15
```

With `--spec dflash`, the assistant is found automatically when it is the only DFlash
GGUF beside the main model. DFlash works with text, images, video, and constrained
tool calls on one or multiple GPUs. A smaller `--draft-tokens` value can improve speed
when the assistant's suggestions are frequently rejected.

Tools support `auto`, `none`, `required`, named tool choice, and strict argument schemas.
The general [tool schema limits](api.md) also apply to Muse-Glimmer.

## `--generate`: one-shot

```bash
surogate serve --generate Qwen/Qwen3.6-27B --prompt "Suggest three easy vegetarian dinners." --max-new 256
```

Answer content goes to stdout; reasoning and diagnostics go to stderr. Use `> answer.txt` to
save only the answer. This mode uses `--max-context` instead of `--max-model-len`, and
`--kv-dtype` instead of `--kv-cache-dtype`.

| Flag | Meaning |
|---|---|
| `--prompt <text>` / `--messages <file.json>` | Input |
| `--max-new N` | Maximum new tokens |
| `--prefill-chunk N` | Prompt-processing size; default 2048 |
| `--stop <text>`, `--stop-token-id N`, `--reasoning-stop <text>` | Repeatable stop conditions |
| `--raw-output`, `--print-token-ids` | Verbatim output or token ids |
| `--prefill-warmup` | Run the supplied prompt once before timing; the measured run processes the prompt again |
| `--no-cuda-graph` | Disable CUDA graphs for debugging |
| `--reasoning-effort minimal\|low\|medium\|high\|xhigh\|max` | Reasoning setting, where the model supports it |

## `--embed`: encoder models

```bash
surogate serve --embed <model.gguf> --device 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | Bind address |
| `--port N` | 8413 | HTTP port |
| `--device N\|cpu` | `0` | GPU number, or `cpu` |
| `--served-model-name NAME` | model argument | Model name accepted by requests and listed by `/v1/models` |
| `--frontend DIR` | automatic | Optional tokenizer override containing `tokenizer.model` and `tokenizer_config.json` |

EmbeddingGemma and Harrier (270M, 0.6B, 27B) Q8_0 GGUF files include their tokenizer, so no separate download or `--frontend`
is required. If both tokenizer files are beside the GGUF, they are used instead. An explicit
`--frontend` takes precedence; keep it in the command when using a custom tokenizer.

### CPU environment

| Variable | Setting | Purpose |
|---|---|---|
| `OMP_WAIT_POLICY` | `ACTIVE` | Reduce delays between CPU operations |
| `OMP_NUM_THREADS` | Physical cores of one NUMA node | Keep work on one CPU socket; avoid counting SMT threads as extra cores |
| `SINFER_CPU_GEMM` | unset | Use automatic selection, or force `builtin`, `onednn`, or `zendnn` when available in the build |

Pin CPU and memory use to the same node with `numactl --cpunodebind=0 --membind=0`.
See [CPU embedding examples](serving-models.md#on-cpu).

## Environment

| Variable | Meaning |
|---|---|
| `SUROGATE_SERVE_CACHE` | Prepared-model cache directory; default `~/.cache/surogate/serve` |
| `SUROGATE_CONVERT_DEVICE` | Override the preparation device, such as `cuda:1` or `cpu`; otherwise follows the serving GPU |
| `SUROGATE_SERVE_ELASTIC_KV_HEADROOM_MIB` | GPU memory kept free when sharing caches across models; default 1024 MiB |
| `SUROGATE_SERVE_DFLASH_PACKED_PREFILL` | `1` lets a DFlash server prefill several waiting prompts in one round, as a server without speculation does; off by default. See [Preparing a DFlash pair](serving-models.md#preparing-a-dflash-pair) |
| `SUROGATE_SERVE_MOE_TRTLLM_FUSED_FINALIZE` | `1` lets NVFP4 mixture experts add their outputs in the GEMM epilogue: up to about 3% faster, but the same request can then get slightly different answers from run to run; off by default |
| `SUROGATE_SERVE_PREFILL_GRAPH_BUDGET_MIB` | GPU memory for prefill and mixed-round CUDA graphs captured while serving; new shapes run without a graph once it is spent; default 512 MiB, 0 captures none |

## Public admission limits

The server can reject excess POST requests with HTTP 429 and `Retry-After`, before JSON
parsing or media preparation. Limits are shared by all models and route aliases in the process.
Authentication runs first; unauthenticated requests do not consume the bucket. Health checks,
model discovery and CORS preflights remain available.

| Flag | Default | Meaning |
|---|---|---|
| `--rate-limit-rps R` | disabled | Token bucket refill rate in requests/second (0.001–1,000,000) |
| `--rate-limit-burst N` | 1 | Maximum accumulated requests; starts full |
| `--max-inflight-requests N` | disabled | Concurrent POSTs, from admission through response delivery, including streams |
| `--max-thinking-requests N` | disabled | Concurrent decisions with `thinking: true`, across all decisions aliases |
| `--max-image-requests N` | disabled | Concurrent decisions carrying images, across all decisions aliases |

For a Gemma 4 mixture with `--vision`, `--gemma-image-tokens 280|560|1120` overrides the
artifact's per-image soft-token budget. Omission preserves the artifact default. Video frame
budgets are unchanged. Unsupported targets or an artifact with a smaller vision envelope are
rejected at startup.

Accepted malformed requests consume rate capacity but release their in-flight slot when their
error response completes. A full concurrency gate does not spend a rate token. Thinking requests
also consume the general bucket. `Retry-After` is an integer number of seconds; for concurrency
rejections it is a retry hint of one second, not a reservation. Clients should retry with jitter.

These limits protect preparation and inference. Place a reverse proxy with connection, body,
timeout and per-client limits in front of the loopback server for public service: the native HTTP
worker queue does not bound accepted sockets. Only expose the inference routes needed by clients;
keep administration and metrics private. Use `--api-key-file` to keep the secret out of process args.
