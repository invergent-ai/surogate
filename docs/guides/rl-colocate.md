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

Generation uses continuous batching, so new requests can join while other
requests are generating. Active generation has priority, while waiting prompts
continue to make progress. Long prompts are processed in smaller chunks to keep
the server responsive.

Prompt caching reuses previously processed text, which can reduce work when GRPO
requests several completions for the same prompt or continues a conversation.
Each request keeps its own sampling settings and seed. Cached prompts are cleared
before training updates so subsequent rollouts use the updated policy.
For models with sliding-window attention, older history is released as the window
advances. Full-attention models still need their complete history.

Context is limited by available GPU memory, `max_model_len`, and the training
`sequence_len`. Use `max_num_seqs` to control the number of concurrent requests.
The entire model and its training buffers must fit alongside generation.

Dense Qwen3 and Qwen3.5 normally use the optimized generation server. The following
`infer.yaml` settings apply to models using the shared training server:

| Setting | Default | When to change it |
|---|---|---|
| `decode_prefill_chunk` | `256` | Lower the maximum prompt tokens processed per round to improve responsiveness during long prompts. The server also reduces chunks automatically when busy or short of memory. |
| `decode_prefix_entries` | `32` | Increase the number of cached prompt prefixes for more reuse, at the cost of memory. Set `0` to disable prompt caching. |
| `decode_cache_bytes` | `0` | Cap generation cache memory in bytes. `0` selects 25% of free GPU memory after the trainer is loaded. |
| `decode_memory_bytes` | `0` | Cap total generation memory in bytes. `0` selects 80% of free GPU memory after the trainer is loaded. |

These memory budgets are additional to the model and training memory already in
use. The server discards unused cached prompts and reduces batches when memory
is tight. If a request still cannot fit, it receives HTTP 429; a streaming request
reports an error and ends. Other requests continue. Retry when capacity becomes
available, or reduce concurrency or context length. Active requests cannot be
paused and moved to CPU to make room.

Text completions, streaming, and token-based multi-turn conversations with
function tools are supported. Sampling options include temperature,
top-k/top-p/min-p, repetition/presence/frequency penalties, logit bias, and minimum
completion length. Returned log-probabilities describe the temperature-scaled
policy before penalties and filtering, as required by GRPO. Structured
`response_format` decoding remains unsupported on this path.

For GLM, native co-locate automatically keeps rollout and scoring log-probabilities
consistent. This can make training slower than ordinary SFT. Set
`doc_masking: true` in the training config.

Set `long_context: true` and `lora_dropout: 0` to reduce memory use during scoring
and training with long sequences. See [long-context memory](long-context.md).

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
