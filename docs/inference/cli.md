# Inference CLI

Serving is a subcommand of the same `surogate` CLI that runs training:

```bash
surogate serve <model> [options...]                       # OpenAI/Anthropic HTTP server
surogate serve --generate <model> --prompt "..."          # one-shot generation to stdout
surogate serve --embed <model> [--frontend DIR]           # /v1/embeddings for an encoder model
```

`<model>` is a **Hugging Face repo id, a local safetensors directory, or a GGUF file**. The
first load converts it into a local cache; later loads are instant.

```bash
surogate serve Qwen/Qwen3.6-27B
surogate serve ~/models/qwen3.6-27b-hf/
surogate serve ~/models/qwen3.6-27b-Q4_K_M.gguf
```

From a source checkout the engine binaries come from `make serve-build`. They are internal
implementation, not a user interface — `surogate serve` resolves and executes them, and no
Python (and no Python CUDA context) stays in the serving process.

`surogate serve --engine-help` prints the full option surface of whichever binary the mode
selects, which is always the canonical list.

## Server options

The most common ones; `--engine-help` has the rest.

### Endpoint and access

| Flag | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | Bind address |
| `--port N` | `8080` | Bind port |
| `--api-key KEY` | none | Require this bearer token |
| `--served-model-name ID` | model identity | Override the reported model id |
| `--cors` | off | Allow cross-origin browser requests |
| `--max-request-mib N` | 384 | Body cap, enforced before JSON parsing |
| `--request-log-jsonl FILE` | none | Append full-precision request records |
| `--log-stats-interval-ms N` | 5000 | Periodic throughput log; `0` disables |

### Context and KV cache

| Flag | Default | Meaning |
|---|---|---|
| `--max-model-len N\|auto` | 8192 | Context length, capped by the model's trained maximum |
| `--kv-capacity N\|auto` | 8192 | KV pool size in tokens; `auto` sizes from free VRAM, leaving 1024 MiB |
| `--kv-cache-dtype auto\|fp8\|bf16\|int8` | `fp8` | Cache precision; `auto` = fp8 (e4m3) |
| `--kv-cache-dtype-skip-layers L,...` | none | Hold these full-attention layers at BF16 |
| `--no-prefix-reuse` | off | Disable compatible-prefix caching |
| `--rewrite-checkpoints` | off | Keep a per-lane GDN checkpoint so an edited last turn resumes from its prefix |
| `--enforce-eager` | off | Skip CUDA graph capture (debugging) |

### Devices

| Flag | Default | Meaning |
|---|---|---|
| `--device N` | 0 | Single GPU |
| `--devices A,B,...` | — | Pipeline stages across GPUs, one stage per card |

### MoE offload

For models whose experts exceed VRAM.

| Flag | Default | Meaning |
|---|---|---|
| `--expert-slots N` | off | Device LRU slot cache; enabling it turns on the pinned host bank |
| `--host-expert-bank w8\|q4` | `q4` with slots | Host bank precision; `w8` restores the artifact's format |
| `--cpu-moe-share F\|auto` | off | Share of routed expert work on host cores; `auto` measures host vs PCIe rates at startup |
| `--cpu-moe-prefill-share F` | 0.5 | Separate share during prefill; `0` disables |
| `--cpu-moe-min-tokens N` | target default | Skip the CPU split below this token count |

### Scheduling

| Flag | Default | Meaning |
|---|---|---|
| `--max-num-seqs N` | 1 | Concurrent lanes — raise it for any real serving load |
| `--max-num-batched-tokens N` | 2048 | Prefill chunk size; must be a positive multiple of 128 |
| `--max-pending-requests N` | 16 | Admission queue depth |
| `--pending-timeout-ms N` | 30000 | Queue wait before rejection |
| `--default-max-tokens N` | 8192 | Used when a request omits `max_tokens` |

### Speculative decoding

Draft tokens are proposed cheaply and verified by the full model, so output is
identical to non-speculative decoding — it just arrives faster. Measured 2.2–2.5×
on decode at a draft window of 3.

| Flag | Meaning |
|---|---|
| `--spec mtp` | Multi-token prediction, using the model's own MTP block. Draft window 1–5. |
| `--spec dflash` | A separate trained drafter. Draft window 1–15; needs a bf16 KV cache, and is not combinable with `--vision`. |
| `--draft-tokens N` | Tokens proposed per round |
| `--lm-head-draft` | Propose through the reduced draft head instead of the full output head |

Availability depends on what the model carries. `mtp` needs the checkpoint's MTP
(`nextn`) block — most first-party checkpoints have it, and community exports
frequently strip it. `dflash` needs a drafter checkpoint that is converted in
alongside the model. Either way a model without one refuses at startup, naming
what is missing, rather than silently serving unaccelerated.

### LoRA adapters

Serve PEFT adapters beside the base model, several at once, each addressable by
name. A request selects one by putting the adapter's name in `model`; the base
id keeps meaning the unadapted model, and requests for different adapters share
a batch. Adapters can also be loaded and unloaded at runtime through
`POST /v1/load_lora_adapter` and `POST /v1/unload_lora_adapter`, without a
restart — see the API page.

| Flag | Default | Meaning |
|---|---|---|
| `--enable-lora` | off | Prepare the adapter machinery. Valid with zero adapters named — they can arrive later through the runtime endpoints. |
| `--lora-modules name=path,...` | — | PEFT adapter directories to load at startup, each addressable as `name` |
| `--max-loras N` | 1 | Resident adapter slots |
| `--max-lora-rank N` | 16 | Largest adapter rank accepted |

Adapters apply to `q_proj`, `k_proj`, `v_proj`, `o_proj`, and — on dense-MLP
models — `down_proj`. A module the server cannot apply is refused at load with
the reason, rather than skipped: an adapter only partly applied is neither the
base model nor the fine-tune. Two refusals you may meet: `gate_proj`/`up_proj`
are fused and consumed inside the SwiGLU projection, and Mixture-of-Experts
models route their MLP through per-expert weights; in both cases, merging the
adapter into the checkpoint before conversion (`surogate merge`) serves its full
effect.

Adapters run under CUDA graphs and with quantized (e.g. NVFP4) base weights;
the delta is computed in BF16 beside the base projection either way.

### Sampling defaults

`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--frequency-penalty`,
`--seed`, `--greedy` (forces temperature 0, exact argmax).

Defaults come from the loaded model and the resolved thinking mode; these flags override
individual values, and request fields override the flags.

### Thinking

`--no-thinking` disables reasoning; `--preserve-thinking` retains closed-turn assistant
reasoning in later prompts.

### Vision

`--vision` enables media input and loads the fixed vision allocations. `--media-cache-mib N`
(1024; `0` disables retained reuse), `--media-live-mib N` (2048), `--media-preprocess-threads N`
(0 = auto, at most 16).

### Responses state

`--response-store-max-records N` (1024) and `--response-store-max-mib N` (256) bound the
process-local Responses store.

## `--generate`: one-shot

```bash
surogate serve --generate Qwen/Qwen3.6-27B --prompt "Explain paged attention." --max-new 256
```

Answer content goes to stdout, reasoning and diagnostics to stderr, so `> answer.txt` keeps only
the answer. This mode has its own spellings for a few options — `--max-context` rather than
`--max-model-len`, `--kv-dtype` rather than `--kv-cache-dtype` — plus:

| Flag | Meaning |
|---|---|
| `--prompt <text>` / `--messages <file.json>` | Input |
| `--max-new N` | Tokens to generate |
| `--prefill-chunk N` | Prefill chunk size (default 2048) |
| `--stop <text>`, `--stop-token-id N`, `--reasoning-stop <text>` | Repeatable stop conditions |
| `--raw-output`, `--print-token-ids` | Verbatim output / token ids |
| `--prefill-warmup` | Warm the prefill path before timing |
| `--no-cuda-graph` | Eager decode |
| `--reasoning-effort low\|medium\|xhigh` | Thinking budget |

## `--embed`: encoder models

```bash
surogate serve --embed <model.gguf> --frontend <hf-snapshot-dir> [--device N|cpu]
```

| Flag | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | Bind address |
| `--port N` | 8413 | Bind port |
| `--device N\|cpu` | `0` | GPU ordinal, or `cpu` for the host backend |
| `--frontend DIR` | beside the GGUF | Tokenizer source for conversion (`tokenizer.model`, `tokenizer_config.json`) |

`--frontend` is consumed during conversion and is unnecessary once the model is cached.

### CPU environment

The CPU backend is tuned by environment, not flags:

| Variable | Recommended | Why |
|---|---|---|
| `OMP_WAIT_POLICY` | `ACTIVE` | Everything runs on one OpenMP team, so spinning covers the microsecond gaps between phases. Must be an environment variable — libgomp reads it before `main`. |
| `OMP_NUM_THREADS` | physical cores of one NUMA node | Every matmul ends in a barrier, so the slowest thread sets the pace: SMT siblings contend for ports, and cross-socket threads wait on interconnect. |
| `SINFER_CPU_GEMM` | unset | `builtin`, `onednn` or `zendnn` to override the vendor GEMM choice. |

Pin the process to match: `numactl --cpunodebind=0 --membind=0`. On a 2×EPYC 9124 host, 16
threads on one node measured 2.15× faster than 64 threads spanning both sockets.

## Environment

| Variable | Meaning |
|---|---|
| `SUROGATE_SERVE_CACHE` | Converted-weights cache (default `~/.cache/surogate/serve`) |
| `SUROGATE_CONVERT_DEVICE` | Device used for conversion (e.g. `cuda`, `cpu`) |
