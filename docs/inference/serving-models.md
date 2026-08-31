# Serving models

Every model follows the same two steps: **convert once** to a `.sinfer` artifact, then **serve**
it. Conversion needs Python and (for some families) a GPU; serving needs neither.

Three worked examples below: an NVFP4 checkpoint, a GGUF checkpoint, and an embedding model on
both GPU and CPU.

## NVFP4 model

NVFP4 is the 4-bit float format Blackwell's tensor cores consume directly (E2M1 weights, one
E4M3FN scale per 16 values). The converter takes **two** sources: the NVFP4 checkpoint for the
quantised projections, and the BF16 base checkpoint for everything NVFP4 does not cover — norms,
embeddings, and the head.

```bash
python3 -m surogate.serve.tools.convert.qwen3_6_27b.convert_nvfp4 \
  --model       /models/Qwen3.6-27B/base-hf-bf16 \
  --nvfp4-model /models/Qwen3.6-27B/vllm-nvfp4-bf16 \
  --out         /artifacts/qwen3_6_27b_nvfp4.sinfer \
  --device cuda
```

Serve it:

```bash
surogate-engine /artifacts/qwen3_6_27b_nvfp4.sinfer \
  --host 0.0.0.0 --port 8080 \
  --max-model-len 8192 \
  --kv-capacity auto \
  --max-num-seqs 32
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"Hello"}]}'
```

Notes:

- `--kv-capacity auto` sizes the KV pool from free VRAM, leaving 1024 MiB of headroom. Give
  `--max-num-seqs` a real value: lanes and the KV pool compete for the same memory.
- The KV cache defaults to fp8 regardless of weight format. `--kv-cache-dtype bf16` if you want
  a full-precision cache.
- Speculative decoding is worth trying on a dense model at low concurrency:
  `--spec dflash --draft-tokens 3` if the artifact carries a draft head.

## GGUF model

GGUF conversion is mostly *no work*: GGML `Q8_0` and the artifact's `W8G32_F16S` are the same
numeric format, so those tensors repack bit-exactly straight off a memmap — no dequantize, no
GPU. Only K-quantised tensors need a real requantisation pass.

This example is Qwen3.8 Flash-Next, a MoE served from a split GGUF. Pass the **first shard**;
the converter finds the rest. `--frontend` is a directory holding the tokenizer, chat template
and configs, which get embedded into the artifact.

```bash
python3 -m surogate.serve.tools.convert.qwen4exp.convert \
  --gguf     /models/Qwen3.8-Flash-Next-GGUF/model-00001-of-00004.gguf \
  --frontend /models/Qwen3.8-Flash-Next-frontend \
  --out      /artifacts/qwen4exp.sinfer \
  --device cuda
```

This model's experts do not fit in one card's VRAM, so serving turns on the offload wing:

```bash
surogate/serve/tools/run_guarded.sh --mem 300G -- \
  surogate-engine /artifacts/qwen4exp.sinfer \
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
- **`run_guarded.sh` is not optional here.** One Flash-Next process peaks near 300 GB of host
  memory between the pinned bank and page cache; the guard caps it in a cgroup, refuses a second
  serving process, and forces `numactl --interleave=all`. Without it, an unlucky run takes the
  machine down rather than itself.

An 8-GPU pipeline instead of offload:

```bash
numactl --interleave=all surogate-engine /artifacts/qwen4exp.sinfer \
  --devices 0,1,2,3,4,5,6,7 \
  --max-num-seqs 64 --max-model-len 2048 --kv-capacity auto --expert-slots 3072
```

Each card takes one pipeline stage, materialises only its own layers, and holds every expert of
those layers in its local pool.

## Embedding model, CPU and GPU

Embedding models take the encoder path — one forward, no KV cache, no sampler, no CUDA graphs —
and get their own server binary. The same artifact serves on either device.

Convert (this one is GGUF-native; `--frontend` supplies the tokenizer):

```bash
python3 -m surogate.serve.tools.convert.gemma_embedding.convert \
  --gguf     /models/embeddinggemma-300M-Q8_0.gguf \
  --frontend /models/embeddinggemma-300m \
  --out      /artifacts/embeddinggemma_300m.sinfer
```

99.78 % of the parameters repack bit-exactly; the whole conversion is a few seconds and needs no
GPU.

### On GPU

```bash
sinfer_embedding_server \
  --artifact /artifacts/embeddinggemma_300m.sinfer \
  --host 0.0.0.0 --port 8413 --device 0
```

### On CPU

Same binary, `--device cpu`, plus the OpenMP environment — the settings matter more than any
flag:

```bash
OMP_WAIT_POLICY=ACTIVE OMP_NUM_THREADS=16 \
numactl --cpunodebind=0 --membind=0 \
  sinfer_embedding_server \
    --artifact /artifacts/embeddinggemma_300m.sinfer \
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
