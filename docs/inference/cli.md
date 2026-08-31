# Inference CLI

Three binaries, built from `csrc/build-serve`:

| Binary | What it is |
|---|---|
| `surogate-engine` | The HTTP server — OpenAI + Anthropic endpoints |
| `surogate-engine-cli` | One-shot generation to stdout; the tool for parity checks and quick probes |
| `sinfer_embedding_server` | The encoder server — `/v1/embeddings`, GPU or CPU |

```bash
cmake --build csrc/build-serve --parallel 32 \
  --target surogate-engine surogate-engine-cli sinfer_embedding_server
```

Name the targets explicitly: a bare `ninja` in that directory builds `all`, which does **not**
include `surogate-engine`. Never rebuild while an engine process is live — the binary is mmapped
and relinking it will SIGBUS the running server.

## `surogate-engine`

```bash
surogate-engine <model.sinfer> [options]
```

### Endpoint and access

| Flag | Default | Meaning |
|---|---|---|
| `--host H` | `127.0.0.1` | Bind address |
| `--port N` | `8080` | Bind port |
| `--api-key KEY` | none | Require this bearer token |
| `--served-model-name ID` | artifact id | Override the reported model id |
| `--cors` | off | Allow cross-origin browser requests |
| `--max-request-mib N` | 384 | Body cap, enforced before JSON parsing |
| `--request-log-jsonl FILE` | none | Append full-precision request records |
| `--log-stats-interval-ms N` | 5000 | Periodic throughput log; `0` disables |

### Context and KV cache

| Flag | Default | Meaning |
|---|---|---|
| `--max-model-len N\|auto` | 8192 | Context length, capped by the model's trained maximum |
| `--kv-capacity N\|auto` | 8192 | KV pool size in tokens; `auto` sizes from free VRAM, leaving 1024 MiB of headroom |
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
| `--cpu-moe-share F\|auto` | off | Share of routed expert work computed on host cores; `auto` measures host vs PCIe rates at startup |
| `--cpu-moe-prefill-share F` | 0.5 | Separate share during prefill; `0` disables |
| `--cpu-moe-min-tokens N` | — | Skip the CPU split below this token count |

### Scheduling

| Flag | Default | Meaning |
|---|---|---|
| `--max-num-seqs N` | 1 | Concurrent lanes — raise it for any real serving load |
| `--max-num-batched-tokens N` | 2048 | Prefill chunk size; must be a positive multiple of 128 |
| `--max-pending-requests N` | 16 | Admission queue depth |
| `--pending-timeout-ms N` | 30000 | Queue wait before rejection |
| `--default-max-tokens N` | 8192 | Used when a request omits `max_tokens` |

### Speculative decoding

| Flag | Meaning |
|---|---|
| `--spec mtp\|dflash` | Draft head type |
| `--draft-tokens N` | Tokens proposed per round |
| `--lm-head-draft` | Draft through the LM head |

### Sampling defaults

`--temperature`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`,
`--frequency-penalty`, `--seed`, `--greedy` (forces temperature 0, exact argmax).

Defaults come from the loaded model and the resolved thinking mode; these flags override
individual values, and request fields override the flags.

### Thinking

`--no-thinking` disables reasoning; `--preserve-thinking` retains closed-turn assistant
reasoning in later prompts.

### Vision

`--vision` enables media input and loads the fixed vision allocations.
`--media-cache-mib N` (1024, `0` disables retained reuse), `--media-live-mib N` (2048, bounds
live BF16 patch payloads), `--media-preprocess-threads N` (0 = auto, at most 16).

### Responses state

`--response-store-max-records N` (1024) and `--response-store-max-mib N` (256) bound the
process-local Responses store.

## `surogate-engine-cli`

```bash
surogate-engine-cli <model.sinfer> (--prompt <text> | --messages <messages.json>) [options]
```

Answer content goes to stdout; reasoning and diagnostics go to stderr, so `> answer.txt` keeps
only the answer.

| Flag | Meaning |
|---|---|
| `--max-new N` | Tokens to generate |
| `--max-context N\|auto` | Context length (the CLI's spelling of `--max-model-len`) |
| `--prefill-chunk N` | Prefill chunk size (default 2048) |
| `--kv-dtype bf16\|int8` | Cache precision for this run |
| `--stop <text>`, `--stop-token-id N`, `--reasoning-stop <text>` | Repeatable stop conditions |
| `--raw-output`, `--print-token-ids` | Verbatim output / token ids, for parity work |
| `--prefill-warmup` | Warm the prefill path before timing |
| `--no-cuda-graph` | Eager decode |
| `--reasoning-effort low\|medium\|xhigh` | Thinking budget |

It also takes `--device`/`--devices`, the `--expert-slots` family, `--spec`/`--draft-tokens`,
`--vision`, and the same sampling flags as the server.

## `sinfer_embedding_server`

```bash
sinfer_embedding_server --artifact <model.sinfer> [--host H] [--port N] [--device N|cpu]
```

| Flag | Default | Meaning |
|---|---|---|
| `--artifact PATH` | required | The encoder artifact |
| `--host H` | `127.0.0.1` | Bind address |
| `--port N` | 8413 | Bind port |
| `--device N\|cpu` | `0` | GPU ordinal, or `cpu` for the host backend |

### CPU environment

The CPU backend is tuned by environment, not flags:

| Variable | Recommended | Why |
|---|---|---|
| `OMP_WAIT_POLICY` | `ACTIVE` | Everything runs on one OpenMP team, so spinning covers the microsecond gaps between phases. Must be an environment variable — libgomp reads it before `main`. |
| `OMP_NUM_THREADS` | physical cores of one NUMA node | Every matmul ends in a barrier, so the slowest thread sets the pace: SMT siblings contend for ports, and cross-socket threads wait on interconnect. |
| `SINFER_CPU_GEMM` | unset | `builtin`, `onednn` or `zendnn` to override the vendor choice. |

Pin the process to match: `numactl --cpunodebind=0 --membind=0`. On a 2×EPYC 9124 host, 16
threads on one node measured 2.15× faster than 64 threads spanning both sockets.

There is also `sinfer_embed_cli` (GPU) and `sinfer_cpu_embed_cli` (CPU) for one-shot embedding
of `--text` or `--tokens`, which is what the benchmarks drive.
