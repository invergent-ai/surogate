# Serving models

`surogate serve <model>` accepts a Hugging Face repo id, a local safetensors directory, or a
GGUF file. The first start prepares the model and saves the result under
`~/.cache/surogate/serve`; later starts reuse it.

```bash
surogate serve Qwen/Qwen3.6-27B --port 8080
```

The examples below cover quantized models, models larger than GPU memory, several models across
GPUs, and embeddings. See the [CLI reference](cli.md) for all options.

## NVFP4 model

NVFP4 checkpoints use four-bit weights and require a supported Blackwell GPU. Surogate detects
the format automatically; no quantization flag is needed.

```bash
surogate serve nvidia/Qwen3.6-27B-NVFP4 \
  --served-model-name qwen3.6-27b \
  --host 0.0.0.0 --port 8080 \
  --max-model-len 8192 --kv-capacity auto --max-num-seqs 32
```

A local directory works the same way:

```bash
surogate serve ~/models/qwen3.6-27b-nvfp4/ \
  --served-model-name qwen3.6-27b --port 8080 --max-num-seqs 32
```

Once the server is ready, send a request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Hello"}]}'
```

Adjust the settings to your workload:

- **Simultaneous requests:** `--max-num-seqs` defaults to 1. Raise it to serve more users at
  once. `--kv-capacity auto` uses available GPU memory for the conversation cache.
- **Context length:** `--max-model-len` defaults to `auto`. Set a number, as above, to choose
  a specific limit for each request. Longer conversations need more memory.
- **Cache precision:** the default `auto` chooses FP8 for this Qwen3.6 model. Use
  `--kv-cache-dtype bf16` for BF16 precision; it needs more memory.
- **Generation speed:** try `--spec mtp --draft-tokens 3` when the checkpoint includes MTP
  weights. The benefit depends on the prompts, hardware, and number of simultaneous requests.
  By default, MTP checks drafts only while one request is decoding.

## GGUF model

Point `surogate serve` at the `.gguf` file:

```bash
surogate serve ~/models/Qwen3.6-27B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 --max-num-seqs 16 --kv-capacity auto
```

Supported GGUF weights are read from the source files without creating another full copy.
Prepared files go under `~/.cache/surogate/serve`, or the directory set by
`SUROGATE_SERVE_CACHE`. Keep the source GGUF files at their original paths while using the
cache. For a split GGUF, pass the **first shard**; the rest are found automatically.

To serve an adapter you trained here as a merged model, merge it first. You can serve the
merged directory directly:

```bash
surogate merge --base-model Qwen/Qwen3.5-0.8B --checkpoint-dir out/step_00000050 --output merged
surogate serve merged
```

Or quantize the merged model before serving:

```bash
surogate quantize --model merged --output merged-Q4_K_M.gguf --type q4_k_m
surogate serve merged-Q4_K_M.gguf
```

## MiniCPM5

Serve the Hugging Face checkpoint directly:

```bash
surogate serve openbmb/MiniCPM5-1B --port 8080
```

For GGUF, download a file from [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
and pass its local path:

```bash
surogate serve ~/models/MiniCPM5-2B-Q4_K_M.gguf --port 8080
```

The GGUF includes its tokenizer and chat template. Use `--no-thinking` for direct answers,
or set `"chat_template_kwargs": {"enable_thinking": false}` in an individual chat request.
Set `enable_thinking` to `true` to request reasoning output.

## Spark-X2.5

Serve either [Spark-X2.5-1.7B](https://huggingface.co/XHToken/Spark-X2.5-1.7B) or
[Spark-X2.5-4B](https://huggingface.co/XHToken/Spark-X2.5-4B) directly:

```bash
surogate serve XHToken/Spark-X2.5-1.7B --port 8080 --enable-auto-tool-choice --tool-call-parser spark25
surogate serve XHToken/Spark-X2.5-4B --port 8080 --enable-auto-tool-choice --tool-call-parser spark25
```

The `spark25` parser enables tool calling through the chat API. LoRA adapters are not yet
supported for these models.

Choose one command. The first start downloads the checkpoint and prepares its serving cache;
subsequent starts reuse that cache. A local checkpoint directory also works.

Use `--no-thinking` for direct answers. You can override this per chat request with
`"chat_template_kwargs": {"enable_thinking": true}` or `false`. Set `--max-model-len`
to the context length you need, for example `--max-model-len 8192`.

## LFM2-MoE and LFM2-VL

Serve an LFM2-MoE checkpoint directly:

```bash
surogate serve LiquidAI/LFM2-8B-A1B --port 8080
```

A local LFM2-MoE GGUF also works. GGUFs are prepared as 8-bit serving weights, so a
lower-bit download needs more disk space and GPU memory after preparation.

For images, use an LFM2-VL or LFM2.5-VL safetensors checkpoint with `--vision`:

```bash
surogate serve LiquidAI/LFM2-VL-450M --vision --port 8080
```

Send images through the chat API using `image_url` content parts, as shown in the
[API guide](api.md). Large images are resized or split automatically using the checkpoint's
processor settings. Multiple images and text-only requests are supported. Video input and
VL GGUF files are not supported yet.

These models run on one GPU. Merge trained LoRA adapters into the checkpoint before serving
them; loading adapters separately is not supported for LFM2-MoE or LFM2-VL.

## A model larger than the card

Use system RAM for part of a model when its weights do not fit in GPU memory. This requires
enough RAM for the offloaded weights and usually makes generation slower.

For a supported mixture-of-experts (MoE) model, start with `--host-moe-layers`:

```bash
surogate serve models/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf \
  --device 0 --host-moe-layers all --max-model-len 2048
```

`--host-moe-layers all` puts all routed expert weights in system RAM; use a number to offload
fewer MoE layers. `--host-moe-layers auto` chooses enough offload to fit, on one GPU or across
several GPUs. Every supported generation model also accepts `--gpu-layers N`: it keeps the
first N decoder layers on the GPU and uses RAM for the rest. `--gpu-layers 0` offloads every
decoder layer; `--gpu-layers all` keeps them on the GPU. Other model weights and the request
cache still need GPU memory. For MoE models, offloading just the experts usually gives a better
speed tradeoff.

Offloaded weight memory cannot be swapped out, so leave enough RAM for the operating system
and other applications. Loading large offloaded models also takes time on every start, even
when model preparation is cached.

### Using CPU cores and an expert cache

Every supported MoE generation model can cache frequently used offloaded experts on the GPU
and send some expert computation to CPU cores. For example:

```bash
surogate serve ~/models/Qwen3.8-Flash-Next-00001-of-00004.gguf \
  --port 8080 \
  --max-num-seqs 16 --max-model-len 4096 --kv-capacity auto \
  --host-moe-layers all --cpu-moe-share auto
```

The expert cache sizes itself from available GPU memory. Start with the automatic settings;
use `--expert-slots N` if you need to choose its size explicitly. Raising concurrency or
context length leaves less memory for this cache.

`--cpu-moe-share auto` measures the machine at startup to choose how much work to send to the
CPU. `--cpu-moe-prefill-share` controls the CPU share during prompt processing; `0` disables
that share.

Offloaded experts use an automatically selected precision based on their source weights.
`--host-expert-bank q4` forces four-bit storage to save system RAM, but can reduce quality when
the source uses higher precision. `--host-expert-bank w8` uses eight-bit storage and more RAM.

### Using several GPUs

Use `--devices` to spread a supported model across several cards:

```bash
surogate serve ~/models/Qwen3.8-Flash-Next-00001-of-00004.gguf \
  --devices 0,1,2,3,4,5,6,7 \
  --max-num-seqs 64 --max-model-len 2048 --kv-capacity auto
```

Memory requirements depend on the model, context length, and concurrency. Add
`--host-moe-layers auto` when the model needs additional system RAM. Supported models can
also use MTP across multiple GPUs if their checkpoints include the required draft weights.
See [Devices](cli.md#devices) for the supported families. DFlash also supports pipeline serving.

### Preparing a DFlash pair

Use a drafter trained for the exact target checkpoint. For example,
[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) has a matching
[z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) drafter.
Download both checkpoints, then prepare them together:

```bash
python -m surogate.serve.convert.qwen3_5.convert \
  --model /path/to/Qwen3.5-4B \
  --dflash-model /path/to/Qwen3.5-4B-DFlash \
  --out /path/to/qwen35-4b-dflash.sinfer --no-vision

surogate serve /path/to/qwen35-4b-dflash.sinfer \
  --devices 0,1 --spec dflash --draft-tokens 3 --kv-cache-dtype bf16
```

Use `--device 0` for a single GPU. Each pipeline GPU needs room for its share of the target,
the drafter, and cache. Sleep/wake and completed-turn cache reuse remain available.

## Several models on one GPU

Use `--model name=path` to add a model to the same server. Clients select one by its name in
the request's `model` field.

Additional models currently require a prepared `.sinfer` cache file. To prepare one, start
it separately first:

```bash
surogate serve Qwen/Qwen3.5-0.8B --served-model-name small
```

Note the cache path printed during preparation, then stop that server. Use that path in place
of `/path/to/cached-small.sinfer` below. Cache files normally live in
`~/.cache/surogate/serve`.

```bash
surogate serve nvidia/Qwen3.6-27B-NVFP4 --served-model-name big \
  --model small=/path/to/cached-small.sinfer,max-num-seqs=16,max-model-len=8192 \
  --kv-capacity auto --max-model-len 8192 --max-num-seqs 16 --port 8080
```

The first model accepts a repo id, local safetensors directory, or GGUF as usual. Additional
models use cache paths. Check `/v1/models` for all available names.

To place an additional model on another GPU, append `,device=1` to its `--model` value.
For a separate GPU group, use `,devices=2:3`. Without a placement override, it uses the
primary model's GPUs. You can load the same artifact under different names to serve
independent replicas from this one server. See [Devices](cli.md#devices) for examples.

### Cache memory grows with demand

By default, each model's conversation cache uses GPU memory as requests need it, instead of
reserving its full capacity immediately. The server can retain reusable prompts after a
request finishes, so memory use does not necessarily drop to zero when the model is idle.

Use `/kv_stats` to inspect cache memory use. `--no-elastic-kv` reserves the full cache instead;
with that setting, each additional model needs an explicit `kv-tokens=N` budget.

### Sharing unused cache memory

Add `--elastic-kv-overcommit` to let models use spare GPU memory beyond their individual cache
budgets:

```bash
surogate serve nvidia/Qwen3.6-27B-NVFP4 --served-model-name big \
  --model small=/path/to/cached-small.sinfer,max-num-seqs=16,max-model-len=8192 \
  --elastic-kv-overcommit --kv-capacity auto --max-model-len 8192 --max-num-seqs 16
```

With this option, `--kv-capacity` sets each model's guaranteed minimum. `auto` reserves enough
for one full-context request per model; additional requests share available memory. When
memory is tight, new requests may wait until others finish. Requests that wait too long can
expire according to `--pending-timeout-ms`.

### Models that do not fit together

Add `--enable-sleep-mode` to let the server save models in system RAM and wake them when
requested. This also requires enough RAM for their saved state. A request for a sleeping
model waits while the server makes room and restores it.

Give frequently used models `priority=high` in their `--model` settings, or use
`--model-priority high` for the first model. Lower-priority idle models are preferred for
sleeping. See the [CLI reference](cli.md#serving-several-models-from-one-process) for how
priorities affect waiting requests.

## Embedding model, CPU and GPU

Start an embeddings server with `--embed`. EmbeddingGemma can run on an NVIDIA GPU or on an
AVX-512-capable CPU.

### On GPU

```bash
surogate serve --embed ~/models/embeddinggemma-300M-Q8_0.gguf \
  --frontend ~/models/embeddinggemma-300m \
  --host 0.0.0.0 --port 8413 --device 0
```

`--frontend` points at a Hugging Face model directory containing the tokenizer files. It is
needed only during preparation and can be omitted once the model is cached.

### On CPU

Use `--device cpu`. On a machine with 16 physical cores in one NUMA node, for example:

```bash
OMP_WAIT_POLICY=ACTIVE OMP_NUM_THREADS=16 \
numactl --cpunodebind=0 --membind=0 \
  surogate serve --embed ~/models/embeddinggemma-300M-Q8_0.gguf \
    --frontend ~/models/embeddinggemma-300m \
    --host 0.0.0.0 --port 8413 --device cpu
```

Set `OMP_NUM_THREADS` to the physical cores of one NUMA node, without counting SMT threads as
extra cores. Keeping CPU and memory use on the same node can improve performance. Adjust
`16` and the node number to match your machine.

### Sending an embedding request

The API request is the same on either device:

```bash
curl http://127.0.0.1:8413/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-300m","input":"the capital of France"}'
```

Use a GPU when embedding speed is the priority and a card is available. CPU serving is useful
when the GPU is busy with text generation, on CPU-only hosts, or for modest retrieval workloads.
