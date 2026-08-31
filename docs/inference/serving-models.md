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
- Speculative decoding helps most at low concurrency: `--spec dflash --draft-tokens 3` when the
  model carries a draft head.

## GGUF model

Point `surogate serve` at the `.gguf` file. GGUF conversion is mostly *no work*: GGML `Q8_0` and
the engine's internal 8-bit format are the same layout, so those tensors move across bit-exactly
with no dequantize and no GPU. Only K-quantised tensors need a real requantisation pass.

```bash
surogate serve ~/models/Qwen3.6-27B-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --max-num-seqs 16
```

For a split GGUF, pass the **first shard** — the rest are found automatically.

### MoE larger than VRAM

A mixture-of-experts model whose experts do not fit on the card turns on the offload wing:

```bash
surogate serve ~/models/Qwen3.8-Flash-Next-00001-of-00004.gguf \
  --port 8080 \
  --max-num-seqs 16 --max-model-len 2048 --kv-capacity auto \
  --expert-slots 3000 \
  --cpu-moe-share auto
```

- `--expert-slots N` sizes the device LRU cache of experts and turns on the pinned host bank.
  Use fewer slots (≈2000) at 32–64 lanes so CUDA graphs and the KV pool still fit.
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
