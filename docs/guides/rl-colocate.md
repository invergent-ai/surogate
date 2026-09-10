# Single-GPU GRPO

Native co-locate mode runs GRPO rollouts and LoRA training on one GPU. The base
weights are loaded once and shared, including embeddings and normalization
weights. Generation pauses during each training update, and the next batch uses
the updated adapter. Per-step adapter updates stay in GPU memory.

This mode supports the training families below, excluding **Nemotron**, using
**unquantized BF16 safetensors weights with LoRA**:

- Qwen3 and Qwen3.5, including their MoE variants and compatible Qwen3.6 checkpoints.
- Llama, MiniCPM5, and Spark-X2.5.
- Gemma 3 and Gemma 4, including E-series, unified, and MoE variants.
- LFM2/LFM2.5, LFM2-MoE, Qwen3-VL, and LFM2-VL.
- GPT-OSS and Laguna, when supplied as unquantized BF16 checkpoints.
- GLM-5.3-Flash text, including sparse DSA and persistent decode state; see the
  [tiny GLM fixture](../../examples/sft/glm/README.md) for tested settings and limitations.

Multimodal checkpoints use **text prompts only**. Experimental training definitions
such as DeepSeek-V4 and Flash-Next are not included.

Dense Qwen3 and Qwen3.5 use the optimized generation server. Other families generate
through the same model instance that performs training. This general path uses
continuous batching: concurrent requests submit token steps to one compute worker,
which batches their projections and MLPs. New requests join between rounds;
prefill is chunked so long prompts do not hold the worker for a complete rollout.
Each request retains its own attention, convolution and recurrent history.
Standard attention appends KV and attends across the batch with shared launches.

Attention history uses 128-token pages from a shared GPU pool, with independent
page tables per request. Completed requests return pages for reuse. GLM's pooled
indexer also uses paged history and reuses score scratch across 128-query tiles.
Its attention cache stores normalized MLA latents, reconstructing selected K/V
projections with the original BF16 and LoRA arithmetic. Recurrent and convolution
states use fixed-size request allocations. Context remains bounded by available
memory, `max_model_len`, and the configured request capacity (`max_num_seqs`).
The shared training executor also bounds context by the trainer's sequence length.

On the shared training path, `decode_cache_bytes` in `infer.yaml` sets a hard byte
budget for cache pages, page tables, recurrent/convolution states, and sampling
token counts. The default `0` chooses 25% of free VRAM after the trainer is
allocated. `decode_memory_bytes` separately bounds cache storage plus decode
activation arenas, extra temporary stacks, sampler scratch, and batch metadata.
Its default `0` selects 80% of free VRAM after trainer allocation. The trainer's
resident weights and existing execution buffers are outside this incremental
budget. It includes 64 MiB of reserved headroom for CUDA graph executables and
library allocations whose exact size cannot be queried in advance.

Admission reserves execution workspaces before advancing request histories. It
evicts unused decode shapes under memory pressure and splits execution into
smaller batches when a larger shape cannot fit. A request that cannot fit receives
a capacity error while other requests continue.
Non-streaming HTTP requests receive status 429. A stream that has already started
receives a JSON `error` event with a status code, followed by `[DONE]`.
Release or retry the rejected request after capacity becomes available.
There is no cache eviction, preemption,
CPU spill, prefix sharing, or sliding-window page recycling.

`trainer.get_decode_batch_stats()` reports active sessions, cached tokens, pages,
allocated/in-use pool bytes, auxiliary bytes, both budgets, workspace and headroom
reservations, page reuse, batch splits, and decode shape eviction/compilation/capture/replay
counts. The server summary reports batch sizes and decode rounds.
Low-level callers use `decode_batch_logits(session_ids,
input_ids, offsets, reset)` with flattened token chunks and one reset flag per
session, then `release_decode_sessions(session_ids)` when requests finish.
`decode_logits` and `get_decode_cache_stats` remain available for single-session
callers. An optimizer update or weight/adapter import invalidates every session;
training also releases unused pool pages. Native decode keeps up to eight compiled
batch/chunk shapes in an LRU cache with separate activation arenas. With
`use_cuda_graphs: true`, one-token shapes warm up, then capture and replay stateless
segments. Attention, recurrent state updates, dynamic MoE operations and the final
generation head run eagerly. Training's captured graphs retain their buffers.
Convolution, recurrent state updates, the GLM indexer and the vocabulary projection
process batches together. GLM reconstructs selected latents in tiles of up to
eight queries, preserving its BF16/LoRA arithmetic and attention reduction order.
Decode does not allocate the training-only 256 MiB backward replay arena per shape.

The HTTP server samples on GPU and transfers only selected tokens and requested
log-probabilities. Temperature, top-k/top-p/min-p, repetition/presence/frequency
penalties, logit bias and minimum-length stop-token blocking are supported.
Log-probabilities describe the temperature-scaled policy **before** penalties and
filtering, as required by GRPO. Each request owns its seeded random stream; ties
prefer smaller token IDs. Low-level callers can use `decode_batch_sample` with one
sampling dictionary per session and a `uniform` draw in `[0, 1)`, after checking
`admit_decode_sessions`. A nonzero sampling result `status` applies only to that row.

This path supports text completions, streaming, and token-based multi-turn conversations with function
tools. Structured `response_format` decoding remains unsupported on this path.

For GLM, native co-locate automatically enables consistent rollout/scoring
arithmetic before allocating the trainer: recurrent FLA KDA forward, fixed-order
dense and expert GEMMs, deterministic BF16 forward additions, and stable sparse
attention reduction slots even before top-k fills. This avoids
batch-size rounding differences changing nearly tied expert or sparse-attention
selections during GRPO scoring. Backward uses the FLA chunk kernels with FP32
intermediates. Training forward sacrifices token parallelism for agreement with
decode; ordinary SFT keeps its parallel chunk forward. `doc_masking: true` is
required. Low-level Python users can select this mode with
`options.glm_rollout_parity = True` before constructing `SurogateTrainer`.

Set `long_context: true` and `lora_dropout: 0` in the training config to tile dense
MLP activations during scoring and updates. GLM also tiles routed and shared expert
MLPs in resident BF16 execution with `ep_size: 1`. Indexer query tiling and the
latent decode cache are automatic. See [long-context memory](long-context.md).

Choose a model and sequence length that fit your GPU together with training
activations and optimizer state. The entire base must fit on one GPU, including all
experts for MoE models. Training buffers remain reserved throughout the run.

Create `train.yaml`:

```yaml
model: Qwen/Qwen3.5-0.8B
output_dir: ./outputs/shared-grpo
gpus: 1
per_device_train_batch_size: 1
sequence_len: 2048
max_steps: 20
learning_rate: 1e-4
lr_scheduler_type: constant
recipe: bf16
lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

Create `infer.yaml`:

```yaml
backend: surogate
model: Qwen/Qwen3.5-0.8B
enable_lora: true
max_model_len: 2048
max_num_seqs: 4
port: 8007
```

Create `orch.yaml`:

```yaml
model:
  name: Qwen/Qwen3.5-0.8B
  lora_adapter: default
  lora_rank: 16
  lora_alpha: 32
env:
  - id: markdown-table-qa
batch_size: 4
rollouts_per_example: 4
sequence_len: 2048
max_steps: 20
use_token_client: true
sampling:
  max_tokens: 1024
  temperature: 1.0
  top_p: 1.0
output_dir: ./outputs/shared-grpo/run_default
```

Run it on your chosen GPU:

```bash
CUDA_VISIBLE_DEVICES=0 surogate grpo-colocate \
    --train train.yaml --infer infer.yaml --orch orch.yaml
```

To use another supported model, replace the model in all three files with its
checkpoint and select the LoRA targets from that model's training example.
`lora_target_modules: [all]` selects the targets supported by its training definition.
For MoE models, also set `lora_dtype: bf16`; expert LoRA training currently requires
BF16 adapters. Official quantized GPT-OSS and FP8 Laguna checkpoints must first be
exported as BF16; their quantized formats cannot be used in this mode.

Use the same model in all three files, matching `max_steps` in training and
orchestration, and an orchestrator output directory directly inside the training
output directory. Start each run in a fresh directory. The runner connects the
orchestrator to the local server and sets synchronous generation automatically.

Reduce `max_num_seqs` to lower serving concurrency and memory use. On the shared
training path, `decode_cache_bytes` also caps persistent request cache storage.
Use `decode_memory_bytes` to cap that storage together with decode workspaces.
Reduce `sequence_len` and `max_model_len` together to lower memory use. Set `sequence_len` in both the
training and orchestrator configs. Keep enough generation tokens for the model
to finish its answer and receive a useful reward.

The run writes `shared_weights.jsonl` in the training output directory. It reports
the shared base size and policy version at each phase. Base
upload bytes and serving base allocation bytes should both be zero. Normal final
adapter saving and training checkpoints still work; per-step broadcast folders
contain readiness markers only.

Quantized checkpoints, QLoRA, full fine-tuning, image/video prompts,
Nemotron, multiple GPUs, CPU weight offload, QeRL weight noise, and
checkpoint resume are not yet supported by native co-locate mode. Use the split-GPU runner for those.

## Agentic tool rollouts

Every supported family above accepts OpenAI function tools. Use a Verifiers
`ToolEnv` (or an environment providing tool schemas and executing tool calls) and
set `use_token_client: true`. The server returns structured `tool_calls` with
JSON argument strings and `finish_reason: tool_calls`. The environment executes
the function and supplies the next tool message. A runnable local example is
[`examples/grpo/tools-orch.yaml`](../../examples/grpo/tools-orch.yaml).

The shared training server chooses the parser from the checkpoint's actual chat
template, including custom and `tool_use` variants:

| Checkpoint protocol | Families using it |
|---|---|
| JSON inside `<tool_call>` | Qwen3, Qwen3-MoE, Qwen3-VL |
| Function/parameter XML | Qwen3.5 and its variants |
| Function/param XML | MiniCPM5 (including checkpoints identifying as Llama) |
| Argument-key/value XML | GLM-5.3-Flash, Spark-X2.5, Laguna |
| JSON or Python function lists | Llama tool templates, Gemma 3 tool templates |
| Python function lists with control tokens | LFM2/LFM2.5, MoE and VL text variants |
| Gemma tool and channel tokens | Gemma 4 variants |
| Harmony channels and tool handoff | GPT-OSS |

Templates without tool support get a JSON tool prompt and adapted tool history;
checkpoints without a chat template use a basic text template. This provides the
interface for base models too; it does not supply learned tool-use capability.
The optimized Qwen server enables automatic tool choice and accepts both Qwen
JSON and function/parameter XML responses.

Raw completion IDs and per-token policy log probabilities include reasoning,
tool syntax and stop tokens. Parsing changes only the HTTP message fields.
The token client appends tool results while retaining earlier generated IDs;
tool results have a zero loss mask. Its template bridge retains function names
for GPT-OSS/Gemma and a stable dummy reasoning span for Qwen. It is installed
in the orchestrator and its environment workers.

Tool calls are exposed only after a complete, normally terminated generation.
Incomplete, malformed, unknown-function or length-truncated calls remain text.
Parallel calls are supported by formats that allow them. On the shared training
path, `parallel_tool_calls: false` prevents returning multiple executable calls.
That path supports `tool_choice: auto` and `none`; required/named choices and
schema-constrained decoding are unavailable. Function `strict` metadata is
accepted for Verifiers compatibility, but does not enable constrained sampling;
the environment handles argument validation and execution errors.
Tool arguments are buffered until completion in streaming responses; Harmony
chat fields are also buffered, while raw token/logprob events continue to stream.
Multimodal tool results remain unsupported; use text or text-only content parts.
