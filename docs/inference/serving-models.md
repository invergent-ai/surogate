# Serving models

`surogate serve <model>` takes a Hugging Face repo id, a local safetensors directory, or a GGUF
file, and turns it into a running endpoint. Conversion happens once, transparently, into
`~/.cache/surogate/serve`; every later start reuses it.

```bash
surogate serve Qwen/Qwen3.6-27B --port 8080
```

Three worked examples below: an NVFP4 checkpoint, a GGUF checkpoint, and an embedding model on
both GPU and CPU.

## NVFP4 model

NVFP4 is the 4-bit float format Blackwell's tensor cores consume directly (E2M1 weights, one
E4M3FN scale per 16 values). There is no separate flag for it: the quantization is read out of
the checkpoint's own `quantization_config`, and the matching converter is selected for you.

```bash
surogate serve nvidia/Qwen3.6-27B-NVFP4 \
  --host 0.0.0.0 --port 8080 \
  --max-model-len 8192 \
  --kv-capacity auto \
  --max-num-seqs 32
```

A local directory works the same way:

```bash
surogate serve ~/models/qwen3.6-27b-nvfp4/ --port 8080 --max-num-seqs 32
```

First run prints its conversion progress and writes the cache entry; the second starts in
seconds. Then:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Hello"}]}'
```

Notes:

- `--max-num-seqs` defaults to **1**. Raise it for any real serving load; lanes and the KV pool
  compete for the same memory, so `--kv-capacity auto` is the easy pairing.
- The KV cache is fp8 regardless of weight format. `--kv-cache-dtype bf16` for a full-precision
  cache.
- Speculative decoding is worth turning on at low concurrency: `--spec mtp --draft-tokens 3`
  measured 2.2–2.5× on decode, with byte-identical output. It needs the checkpoint's MTP block,
  and says so at startup if the model lacks one.

## GGUF model

Point `surogate serve` at the `.gguf` file. Nothing is copied and nothing is requantised: the
K-quant superblocks are read from the file as it stores them, `Q8_0` tensors are rearranged into
the same numbers the engine's 8-bit format holds, and what lands beside the file is a small index
naming the stretches each tensor comes from. For a 22 GB `Q4_K_M` that index is about 70 MB and
takes ~18 seconds to build once.

```bash
surogate serve ~/models/Qwen3.6-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --max-num-seqs 16
```

For a split GGUF, pass the **first shard** — the rest are found automatically.

A model you trained here is not a GGUF yet. Merge the adapter and quantize it first, then serve
the result the same way:

```bash
surogate merge --base-model Qwen/Qwen3.5-0.8B --checkpoint-dir out/step_00000050 --output merged
surogate quantize --model merged --output merged-Q4_K_M.gguf --type q4_k_m
surogate serve merged-Q4_K_M.gguf
```

### MoE larger than VRAM

A mixture-of-experts model whose experts do not fit on the card turns on the offload wing:

```bash
surogate serve ~/models/Qwen3.8-Flash-Next-00001-of-00004.gguf \
  --port 8080 \
  --max-num-seqs 16 --max-model-len 4096 --kv-capacity auto \
  --cpu-moe-share auto
```

- The device expert cache sizes itself: it takes what the card has left after the weights,
  the KV floor for `--max-num-seqs` lanes and the runtime's own reservation, so 64 lanes
  simply get a smaller pool. `--expert-slots N` still fixes it by hand.
- The pinned host bank holds the GGUF's experts decoded to Q4G32AM (about 88 GB for
  Flash-Next; built at load, ~70 s). The host worker pool follows the machine's load: one
  pinned worker per physical core on an idle box, fewer and unpinned when other jobs are
  running.
- `--cpu-moe-share auto` measures host-memory versus PCIe rates at startup and splits routed
  expert work accordingly. `--cpu-moe-prefill-share 0.7` suits a single user; `0` turns the
  prefill split off.
- `--host-expert-bank w8` restores the artifact's 8-bit bank if you would rather spend host RAM
  than accept the 4-bit requantisation.
- **Watch host memory.** A model in this tier pins its expert bank in RAM and can approach
  300 GB between that and page cache. Size the box for it, and run one such process at a time.

Across several GPUs instead, one pipeline stage per card:

```bash
surogate serve ~/models/Qwen3.8-Flash-Next-00001-of-00004.gguf \
  --devices 0,1,2,3,4,5,6,7 \
  --max-num-seqs 64 --max-model-len 2048 --kv-capacity auto --expert-slots 3072
```

Each card materialises only its own layers and holds every expert of those layers locally.

## Several models on one GPU

`--model name=path` serves an additional model beside the primary, each in its own engine
inside one process, so requests for different models genuinely share the GPU:

```bash
surogate serve nvidia/Qwen3.6-27B-NVFP4 --served-model-name big \
  --model small=nvidia/Qwen3.5-4B-NVFP4,max-num-seqs=16,max-model-len=8192 \
  --kv-capacity auto --max-model-len 8192 --max-num-seqs 16 --port 8080
```

### The KV pool is elastic

By default a model's KV pool is a **virtual span**: it is laid out at its planned size, but
only the pages a request actually reaches hold VRAM, mapped in 2 MiB-per-plane granules as
sequences grow and returned once they finish. A small reserve (four granules) is kept mapped
ahead of demand so a request never waits on the driver. Throughput is the same as with a static
pool — measured on this 27B + 4B pair, both busy at 8 + 8 users: 313 + 626 tok/s elastic
against 318 + 635 static — and what changes is what sits idle:

| | static pool | elastic pool |
|---|---|---|
| free VRAM after the 27B starts | 9.75 GiB | 13.56 GiB |
| KV held with both models busy | 6.0 GiB provisioned | 0.62 + 0.31 GiB mapped |
| KV held by the 27B asleep | 4 GiB | 192 MiB |

`GET /kv_stats` shows the numbers per model. `--no-elastic-kv` restores the static arena, in
which case every extra model must state its budget with `kv-tokens=N`.

### Sharing the room: `--elastic-kv-overcommit`

Elastic pools still each fit the GPU on their own: their caps are checked against free memory
at startup and never sum past it, so one model's idle cache is not available to another's
requests. `--elastic-kv-overcommit` makes each model's `--kv-capacity` a **guaranteed floor**
(`auto` = one full-context request) and admits every page past it against the memory the GPU
actually has free, shared by all models on the device:

```bash
surogate serve nvidia/Qwen3.6-27B-NVFP4 --served-model-name big \
  --model small=nvidia/Qwen3.5-4B-NVFP4,max-num-seqs=16,max-model-len=8192 \
  --elastic-kv-overcommit --kv-capacity auto --max-model-len 8192 --max-num-seqs 16
```

When the GPU runs short, a request waits in its model's queue (the ordinary
`--pending-timeout-ms` applies) rather than failing, the models trim their reserves, give back
their prefix caches, and resume admitting as running requests finish — measured with two
engines pushing 4 GB of KV demand into 2.4 GB of room for two minutes: every request
completed, none rejected, both models holding a share, and full throughput back once the
burst ended. A headroom (`SUROGATE_SERVE_ELASTIC_KV_HEADROOM_MIB`, default 1024) is never
given to KV: CUDA graphs are captured lazily and need it. Overcommit changes when a request
runs, never how it runs — the output is identical.

With `--enable-sleep-mode` as well, models that do not fit together at all are swapped by the
scheduler; see the [CLI page](cli.md#serving-several-models-from-one-process) for priorities
and preemption.

## Embedding model, CPU and GPU

Embedding models take the encoder path — one forward, no KV cache, no sampler, no CUDA graphs —
so they are served with `--embed`. The same model runs on either device.

### On GPU

```bash
surogate serve --embed ~/models/embeddinggemma-300M-Q8_0.gguf \
  --frontend ~/models/embeddinggemma-300m \
  --host 0.0.0.0 --port 8413 --device 0
```

`--frontend` points at the model's Hugging Face snapshot, which supplies the tokenizer the GGUF
does not carry in the form the converter wants. It is needed only for the first (converting) run.

### On CPU

Same command, `--device cpu`, plus the OpenMP environment — those settings matter more than any
flag:

```bash
OMP_WAIT_POLICY=ACTIVE OMP_NUM_THREADS=16 \
numactl --cpunodebind=0 --membind=0 \
  surogate serve --embed ~/models/embeddinggemma-300M-Q8_0.gguf \
    --host 0.0.0.0 --port 8413 --device cpu
```

Size `OMP_NUM_THREADS` to the **physical cores of one NUMA node** and pin to that node. Every
matmul ends in a barrier, so the slowest thread sets the pace: SMT siblings contend for the same
execution ports, and a thread on the far socket waits on the interconnect. On a 2×EPYC 9124
(32 physical cores, 2 nodes), 16 pinned threads beat 64 unpinned ones by 2.15×.

Either way the request is identical:

```bash
curl http://127.0.0.1:8413/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-300m","input":"the capital of France"}'
```

### Which device?

Measured at 512 tokens, single stream, on this reference host (RTX 5090 / 2×EPYC 9124):

| | per request | vs llama.cpp |
|---|---|---|
| GPU | 7.4 ms | 1.35× faster |
| CPU (16 pinned cores) | 70.4 ms | 2.6× faster |

The GPU is roughly 10× faster per request, so it wins whenever a card is free. The CPU path
exists because embedding work often has nowhere else to go — a card busy serving an LLM, a
CPU-only host, or a retrieval service whose throughput needs are modest. At those sizes the host
answers in tens of milliseconds while leaving the GPU alone.
