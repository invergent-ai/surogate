# Serving examples

Run commands from the repository root with Surogate installed (source builds need
`make serve-build`). [launch.sh](launch.sh) starts one foreground process; stop it
before starting another scenario on the same port. [client.py](client.py) uses the
Python standard library and sends requests to an already-running server.

```bash
bash examples/serve/launch.sh chat
# In another terminal:
python examples/serve/client.py chat
python examples/serve/client.py stream
python examples/serve/client.py tools
```

`MODEL` selects a Hugging Face repo, local safetensors directory, or local GGUF.
For split GGUFs, pass the first shard and keep all shards together. Prepared caches
retain references to the source GGUF: keep those source files in place.
`PORT` changes the port. Generation scenarios serve the name `demo`.
Additional arguments pass through to `surogate serve`.

## Launch recipes

| Scenario | Demonstrates | Inputs / hardware |
|---|---|---|
| `chat` | Chat, streaming, thinking controls, automatic function tools | Qwen3-0.6B by default; one supported GPU |
| `concurrent` | Continuous batching, chunked prefill, FP8 KV, elastic allocation, prefix reuse, edited-turn checkpoints, queues and request logs | Qwen3.5-0.8B; one GPU |
| `lora` | Startup PEFT adapter and runtime load/unload | Train `examples/training/runtime-lora.yaml` first |
| `vision` | Image/video input and media caching | Qwen3-VL-2B safetensors; one GPU |
| `offload` | Whole decoder layers in host RAM | `GPU_LAYERS=8` by default; GPU still required |
| `moe` | Expert offload, automatic GPU expert cache, CPU/GPU expert compute | Set `MODEL` to a supported MoE; sufficient host RAM |
| `multi-gpu` | Layer pipeline on two GPUs | Qwen3.5-4B by default; `DEVICES=0,1` |
| `multi-model` | Named models, priorities, shared spare KV memory, automatic sleep/wake | Set `SECOND_MODEL` to a prepared artifact; one GPU and host RAM |
| `mtp` | Multi-token prediction, concurrency cutoff | Set `MODEL` to a supported MTP checkpoint |
| `dflash` | Separate speculative drafter | Prepared target + drafter artifact; one GPU, BF16 KV, text only |
| `generate` | One-shot generation with answer on stdout | Qwen3-0.6B; no HTTP server |
| `embeddings` | EmbeddingGemma on GPU or CPU | Set `MODEL` to GGUF and `FRONTEND` to its tokenizer directory |

Hardware and checkpoint formats vary by family. See the repository's
[model coverage](../../docs/inference/index.md) and [CLI reference](../../docs/inference/cli.md).
The launch defaults are illustrative resource budgets, not capacity guarantees.

### Checkpoint formats and placement

```bash
# Local BF16 or FP8 safetensors directory; precision is detected.
MODEL=./models/my-checkpoint bash examples/serve/launch.sh chat
# A prequantized NVFP4 export requires a supported Blackwell GPU.
MODEL=nvidia/Qwen3.6-27B-NVFP4 bash examples/serve/launch.sh concurrent
# GGUF, including automatic discovery of the remaining split shards.
MODEL=./models/model-00001-of-00006.gguf bash examples/serve/launch.sh moe
MODEL=./models/large-moe.gguf bash examples/serve/launch.sh moe --devices 0,1
bash examples/serve/launch.sh multi-gpu
```

Whole-layer offload still computes on the GPU. MoE CPU sharing applies to offloaded
experts. `--host-expert-bank q4` trades host precision for RAM; `--expert-slots N`
sets an explicit cache size that must hold at least one layer's experts.

### Runtime adapters and merged models

```bash
surogate sft examples/training/runtime-lora.yaml
bash examples/serve/launch.sh lora
# In another terminal, select the startup adapter by name:
python examples/serve/client.py chat --model tuned
python examples/serve/client.py unload-lora
python examples/serve/client.py load-lora --adapter "$(pwd)/outputs/training/runtime-lora"
```

Adapter paths refer to the server's filesystem. Wait for an adapter's requests to
finish before unloading. The runtime example uses attention projections; broader
adapter support depends on the family and weight format. For a standalone model,
merge any supported training adapter, then optionally quantize it:

```bash
surogate merge --base-model Qwen/Qwen3-0.6B \
  --checkpoint-dir ./outputs/training/runtime-lora --output ./outputs/serve/merged
surogate quantize --model ./outputs/serve/merged \
  --output ./outputs/serve/merged-Q4_K_M.gguf --type q4_k_m
MODEL=./outputs/serve/merged-Q4_K_M.gguf bash examples/serve/launch.sh chat
```

`quantize` uses the packaged llama.cpp converter/quantizer. Source builds need the
quantizer target; allow disk space for an intermediate BF16 GGUF. A merged
safetensors directory can also be served directly.

### Multiple models and sleep

Prepare the second model once by starting it with `surogate serve Qwen/Qwen3.5-0.8B`,
then stop that process and copy the printed cache path into `SECOND_MODEL`:

```bash
SECOND_MODEL=/absolute/path/to/prepared-small.sinfer bash examples/serve/launch.sh multi-model
python examples/serve/client.py chat --model small
python examples/serve/client.py sleep --model small
python examples/serve/client.py wake --model small
python examples/serve/client.py status
```

Additional models currently require prepared `.sinfer` files. Both models use one
GPU. `--enable-sleep-mode` keeps sleeping model state in system RAM. A multi-model
server wakes a model on demand; a single-model server requires an explicit wake.
For a fixed KV reservation, use `--no-elastic-kv` and specify `kv-tokens=N` for every
additional model instead of shared elastic KV.

### Speculation

```bash
MODEL=./models/checkpoint-with-mtp bash examples/serve/launch.sh mtp
# Supported multi-GPU families also allow MTP:
MODEL=./models/checkpoint-with-mtp bash examples/serve/launch.sh mtp --devices 0,1
```

MTP needs actual draft weights; community exports may omit them. Add `--lm-head-draft`
only when the checkpoint also includes a reduced draft vocabulary. Compare with and
without speculation at the concurrency you expect to serve.

DFlash preparation is an advanced source-checkout workflow. Supply a compatible
Qwen3.5/3.6 MoE target and DFlash drafter checkpoint; both must already be downloaded:

```bash
python -m surogate.serve.convert.qwen3_5_moe.convert \
  --model ./models/target --dflash-model ./models/compatible-drafter \
  --out ./outputs/target-dflash.sinfer --no-vision
MODEL=./outputs/target-dflash.sinfer bash examples/serve/launch.sh dflash
```

The `.sinfer` input here is an internal prepared artifact. DFlash requires a single
GPU, BF16 KV and no vision. Ordinary model preparation does not add a drafter.

### Images, video, and embeddings

```bash
bash examples/serve/launch.sh vision
python examples/serve/client.py image --media ./photo.jpg
python examples/serve/client.py video --media ./clip.mp4
```

The client sends local files as base64 data URIs; URLs also work. Large media can
exceed the request or media budgets. Vision coverage depends on the family;
Qwen3-VL GGUF and Qwen3-VL-MoE are currently unsupported.

```bash
MODEL=./models/embeddinggemma-300M-Q8_0.gguf FRONTEND=./models/embeddinggemma-300m \
  bash examples/serve/launch.sh embeddings
python examples/serve/client.py embeddings --base-url http://127.0.0.1:8413 --model embeddinggemma-300m
# CPU alternative: requires AVX-512; choose the physical cores of one NUMA node.
OMP_WAIT_POLICY=ACTIVE OMP_NUM_THREADS=16 DEVICE=cpu \
MODEL=./models/embeddinggemma-300M-Q8_0.gguf FRONTEND=./models/embeddinggemma-300m \
  bash examples/serve/launch.sh embeddings
```

The embeddings process is separate from generation. `FRONTEND` supplies
`tokenizer.model` and `tokenizer_config.json` for preparation.

## API examples

With `launch.sh chat` running, choose any client command:

| Command | Request behavior |
|---|---|
| `chat` | Sampling controls, generated-token logprobs and exact token IDs |
| `stream` | SSE deltas, reasoning fields when present, usage and completion marker |
| `reasoning` | Per-request thinking enabled; response separates reasoning and answer |
| `tools` | Auto tool selection, argument validation, local function and tool-result round trip |
| `completions` | Raw text without a chat template, greedy decoding and stop string |
| `tokens` | Tokenization with token strings, then generation from exact prompt IDs |
| `responses` | Stored conversation, continuation by ID, retrieval, input items/count, deletion |
| `anthropic` | Messages and message token counts |
| `concurrent` | Eight simultaneous requests with a common prefix; use the concurrent launch |
| `status` | Health, model names, KV statistics and Prometheus metrics |

```bash
python examples/serve/client.py responses
python examples/serve/client.py anthropic
python examples/serve/client.py tokens
```

The server has no constrained JSON/schema decoding or prompt-logprob scoring;
RULER judges and distillation teachers need the external services described in their
examples. `top_logprobs` alternatives are unavailable in native generation.
Responses state is local to the running process; background execution and
compaction are unsupported.

For authentication, export the same `SUROGATE_API_KEY` in server and client terminals.
The launch script passes it as `--api-key`; the client sends a bearer header.
`--host 0.0.0.0` enables remote access and `--cors` enables browser cross-origin access.
`/health` remains unauthenticated. Inspect `outputs/serve/requests.jsonl` after the
concurrent scenario; `/metrics` and `/kv_stats` expose throughput and memory use.

```bash
bash examples/serve/launch.sh chat --enable-sleep-mode --preserve-thinking
python examples/serve/client.py sleep
python examples/serve/client.py wake
bash examples/serve/launch.sh generate > answer.txt
```

Use `--enforce-eager` (server) or `--no-cuda-graph` (one-shot) when investigating CUDA
graph behavior. The complete set of flags is available through
`surogate serve --engine-help` and its `--generate` / `--embed` variants.
