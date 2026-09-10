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
| `--served-model-name ID` | model identity | Name clients use in the `model` field |
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
| `--no-cache` | off | Rebuild the prepared model cache on disk |
| `--no-prefix-reuse` | off | Disable reuse of compatible earlier prompts |
| `--enable-prefix-caching`, `--no-enable-prefix-caching` | enabled | Alternative spellings for enabling or disabling prompt reuse |
| `--rewrite-checkpoints`, `--no-rewrite-checkpoints` | off | Enable or disable extra memory for faster editing and resending of the last turn |
| `--enforce-eager` | off | Disable CUDA graphs for debugging |

When `--max-model-len` is automatic, omitted `--kv-capacity` also defaults to `auto`. When
context is explicit, omitted cache capacity defaults to that same token count. Use
`--kv-capacity auto` explicitly to make more cache available for simultaneous requests.

Cache precision `auto` selects BF16 for models such as Qwen3 and Llama, and FP8 for hybrid
models such as Qwen3.5/3.6/3.8. FP8 uses half the cache storage of BF16. DFlash requires BF16.

With `--elastic-kv-overcommit`, `--kv-capacity` becomes a guaranteed minimum; `auto` guarantees
enough for one full-context request per model. See [Serving models](serving-models.md#several-models-on-one-gpu).

### Devices

| Flag | Default | Meaning |
|---|---|---|
| `--device N` | 0 | Use one GPU |
| `--devices A,B,...` | — | Split a supported model across the listed GPUs |

`--devices` takes precedence over `--device`; its first GPU is also used for model preparation.
With one GPU, preparation follows `--device`. `SUROGATE_CONVERT_DEVICE` overrides the preparation
device when needed. GPU indices follow `CUDA_VISIBLE_DEVICES`.

Multiple GPUs are supported for GLM-5.3-Flash, Qwen3.8 Flash-Next, Qwen3.5/3.6/3.8 dense
models, and Qwen3.5/3.6 MoE models. MTP can be used with these models when their checkpoint
includes compatible MTP weights.

DFlash, additional models through `--model`, and sleep mode currently require a single GPU.
To run independent replicas, start separate servers on different devices and ports.

### Host offload

Use system RAM for part of a model when it does not fit in GPU memory. This reduces GPU memory
requirements but can make responses slower. The machine needs enough RAM to hold the offloaded
weights; that memory cannot be swapped out while the model is loaded.

| Flag | Default | Meaning |
|---|---|---|
| `--host-moe-layers N\|auto\|all` | off | Move experts from N MoE layers to RAM; `auto` chooses enough to fit on one GPU or each GPU in a multi-GPU run |
| `--gpu-layers N\|all`, `-ngl N`, `--n-gpu-layers N` | all | Keep the first N decoder layers on the GPU; `0` offloads all decoder layers, and `all` keeps them resident |

Whole-layer offload works with every supported generation model. Embeddings, the output head,
and any enabled image encoder still need GPU memory. MoE-specific options apply only to MoE
models and only affect experts that are offloaded.

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
| `--expert-slots N` | automatic | Number of experts cached on the GPU; omitted or `0` sizes the cache from available memory. An explicit count must hold at least one layer’s experts |
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
| `--max-num-seqs N` | 1 | Maximum simultaneous requests, from 1 to 128 |
| `--max-num-batched-tokens N` | 2048 | Prompt tokens processed at a time; must be a positive multiple of 128 |
| `--max-pending-requests N` | 16 | Additional requests allowed to wait |
| `--pending-timeout-ms N` | 30000 | Time allowed for prompt preparation and waiting to start generation |
| `--default-max-tokens N` | 8192 | Output limit when a request omits it |

Raise `--max-num-seqs` when serving multiple users. More simultaneous requests need more
memory; pair it with `--kv-capacity auto`. The pending timeout does not limit how long an
already-running response may take.

### Speculative decoding

Speculative decoding can make generation faster by proposing several tokens for the model to
check together. It works best when those proposals are often accepted. Measure it with your
workload: more simultaneous requests or heavy CPU offload can reduce the benefit.

| Flag | Meaning |
|---|---|
| `--spec mtp` | Enable MTP on a supported model; supports one or multiple GPUs |
| `--spec dflash` | Use a compatible separate drafter; requires one GPU, `--kv-cache-dtype bf16`, and no `--vision` |
| `--draft-tokens N` | Number of proposed tokens: 1–5 for MTP, 1–15 for DFlash; required with `--spec` |
| `--spec-max-lanes N\|all` | MTP checks drafts only while at most N requests are decoding. Default (or `0`) is 1; `all` keeps checking at every concurrency level. Does not affect DFlash |
| `--lm-head-draft` | Use a smaller draft vocabulary when the checkpoint provides one |

MTP needs the checkpoint's MTP weights, which some community exports omit. DFlash
needs a compatible drafter included during model preparation. Missing draft weights produce
a startup error.

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
| `max-model-len=N` | Override its context limit |
| `spec=mtp\|dflash` | Enable speculation for this model; defaults its draft token count to 3 |
| `draft-tokens=N`, `spec-max-lanes=N\|all` | Override this model's speculation settings |
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
| `--enable-sleep-mode` | off | Enable `/sleep` and `/wake_up` to release and restore GPU memory while preserving model state; one GPU only |

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
modules still need `surogate merge` before serving. Unsupported tensors are rejected without
partially applying the adapter.

Additional models have their own adapters through `lora=name:path` in `--model`. Every model
and adapter name must be unique. The runtime load/unload endpoints accept `?model=NAME`; see
[LoRA adapters at runtime](api.md#lora-adapters-at-runtime).

### Sampling defaults

`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--frequency-penalty`,
and `--seed` override the model's defaults. Request fields override individual server settings.
`--greedy` always forces temperature zero, including when a request asks for another value.
Requests also accept `repetition_penalty`.

### Thinking

`--no-thinking` requests answers without reasoning when the model supports disabling it.
`--preserve-thinking` keeps earlier assistant reasoning in later prompts. These settings are
independent and can be overridden per request. Supported reasoning-effort values depend on
the model's chat template.

| Flag | Default | Meaning |
|---|---|---|
| `--reasoning-parser NAME` | `qwen3` | Read reasoning in the model's output format; also accepts `deepseek_r1`, `glm4_moe`, `think`, or `none`/`off` |
| `--tool-call-parser NAME` | `qwen3_xml` | Read tool calls; also accepts `hermes`, `spark25`, `llama3_json`, `llama4_json`, or `none`/`off` |
| `--enable-auto-tool-choice` | off | Allow the model to choose a tool automatically; requires an enabled tool parser |
| `--chat-template FILE` | model template | Use this Jinja file to format chat prompts |

### Vision

`--vision` enables images and video on supported models. `--media-cache-mib N` (default 1024)
sets how much processed media to retain for reuse; `0` disables retention. `--media-live-mib N`
(default 2048) limits memory for media currently being processed or used by requests.
`--media-preprocess-threads N` chooses processing threads; `0` selects automatically, up to 16.

For example, serve a dense Qwen3-VL checkpoint with image and video inputs:

```bash
surogate serve Qwen/Qwen3-VL-2B-Instruct --vision --port 8080
```

Send media through the [chat API](api.md#chat-completions). Without `--vision`, the same
checkpoint serves text prompts. Qwen3-VL-MoE and Qwen3-VL GGUF files are not supported yet.

### Responses state

`--response-store-max-records N` (1024) and `--response-store-max-mib N` (256) limit stored
Responses API conversations. They are lost when the server restarts and are not shared across
separate servers.

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
surogate serve --embed <model.gguf> --frontend <hf-snapshot-dir> --device 0
```

| Flag | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | Bind address |
| `--port N` | 8413 | HTTP port |
| `--device N\|cpu` | `0` | GPU number, or `cpu` |
| `--frontend DIR` | beside the GGUF | Model directory containing `tokenizer.model` and `tokenizer_config.json` for preparation |

`--frontend` is needed during preparation only; it can be omitted once the model is cached.

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
