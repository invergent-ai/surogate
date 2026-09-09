# Training and serving examples

Run commands from the repository root with Surogate installed. Model IDs download
weights on first use; gated checkpoints require Hugging Face access. Training needs
a supported NVIDIA GPU. GPU counts are explicit in the larger recipes, and memory
use depends on model, precision, context and batch size.

```bash
surogate sft examples/sft/qwen3/qwen3-lora-bf16.yaml
# Serve the trained adapter as a merged model:
surogate merge --base-model Qwen/Qwen3-0.6B \
  --checkpoint-dir ./outputs/sft/qwen3/qwen3-lora-bf16 --output ./outputs/qwen3-merged
surogate serve ./outputs/qwen3-merged --served-model-name demo
# Another terminal:
python examples/serve/client.py chat
```

Examples use separate output directories and start fresh by default. To resume,
follow the [checkpoint instructions](training/README.md#continue-an-adapter-stack-an-adapter-or-resume-a-run).
The short budgets and bundled toy datasets demonstrate workflows; they are not
quality or hardware-capacity guarantees.

## Feature map

| Feature | Start here |
|---|---|
| Random-initialized pretraining; NorMuon | [Pretraining](pt/README.md), [config](pt/qwen3.yaml) |
| Continued pretraining on raw text | [Continued pretraining](pt/qwen3-continue.yaml), run with `surogate sft` |
| Full-parameter SFT; AdamW and FP32 masters | [Full fine-tuning](training/full-finetune.yaml) |
| LoRA, BF16/FP8/NVFP4 compute | [Model and precision recipes](#model-and-precision-recipes) |
| Online BnB NF4, FP8 and NVFP4 QLoRA; prequantized checkpoints | [Qwen3 recipes](sft/qwen3), [precision guidance](training/README.md) |
| Import/continue and stack adapters; checkpoint resume, merge and export | [Adapter lifecycle](training/README.md#continue-an-adapter-stack-an-adapter-or-resume-a-run) |
| Local/hub datasets, instruction/chat mapping, mixing, packing and validation | [Dataset examples](datasets/README.md), [split chat columns](sft/reverse-text-qwen3.yaml) |
| Vision fine-tuning and text-only multimodal backbones | [Qwen3-VL](sft/qwen3vl), [Qwen3.5](sft/qwen35) |
| DPO, differing-span masking, length normalization and reference-free loss | [Preference training](dpo/README.md) |
| Offline top-K distillation, local/API teachers and tokenizer transplantation | [Distillation](distillation/README.md) |
| Split-GPU, colocated and separate-process GRPO | [GRPO](grpo/README.md) |
| RL evaluation, difficulty filtering, temperature scheduling and resume | [GRPO orchestrator](grpo/orch.yaml), [guide](grpo/README.md) |
| LLM judge rewards: replace, add or metrics only | [RULER](ruler/README.md) |
| On-policy teacher rewards, multi-turn diagnostics, adaptive rollout depth | [TurnOPD](turnopd/README.md) |
| Native data parallelism, ZeRO sharding and communication overlap | [Multi-GPU](training/multi-gpu.yaml) |
| Multi-node training with Ray | [Ray config](sft/qwen3/qwen3-lora-bf16-ray.yaml), [setup](training/README.md#multiple-gpus-and-large-models) |
| Dispatch pipeline parallelism with streamed frozen weights | [27B dispatch recipe](sft/qwen35/qwen36-text-lora-fp8-pp.yaml) |
| Expert parallelism, routing losses, balancing and expert offload | [Qwen3.5/3.6 MoE](sft/qwen35moe), [Qwen3 MoE QLoRA](sft/qwen3moe/qwen3moe-lora-qbnb.yaml) |
| CPU weights, gradients, optimizer, quantized weights and activation offload | [CPU training](training/cpu-offload.yaml), [sharded FP8](training/multi-gpu.yaml) |
| Long context, tiled MLP, recomputation and chunked loss/attention | [Long context](training/long-context.yaml) |
| Chunked-sequence training | [Laguna](sft/laguna/laguna-s-lora-fp8.yaml), [Qwen3.6 MoE](sft/qwen35moe/qwen36moe-lora-fp8.yaml) |
| Adaptive LR, early stopping, token budgets, schedules, W&B/Aim and local metrics | [Adaptive training](training/README.md#data-validation-and-monitoring) |
| HF/GGUF serving, quantization and runtime adapters | [Serving](serve/README.md#runtime-adapters-and-merged-models) |
| Chat/Completions/Responses/Anthropic APIs, streaming, tools and tokenization | [API client](serve/README.md#api-examples) |
| Concurrency, prompt caching, elastic KV, authentication, health and metrics | [Serving launches](serve/README.md#launch-recipes) |
| CPU layer/expert offload, expert cache and CPU/GPU compute sharing | [Placement](serve/README.md#checkpoint-formats-and-placement) |
| Multiple GPUs, multiple named models, priorities and sleep/wake | [Serving launches](serve/README.md#launch-recipes) |
| MTP and DFlash speculation | [Speculation](serve/README.md#speculation) |
| Image/video requests and GPU/CPU embeddings | [Media and embeddings](serve/README.md#images-video-and-embeddings) |

## Model and precision recipes

Every YAML below runs with `surogate sft <path>`. Each family retains examples for
different execution paths or checkpoint formats. A larger model generally needs
more GPUs, host offload or quantization; inspect the selected config before launch.

| Family | Recipes | GPU count in configs |
|---|---|---|
| Gemma 4 text backbones | [12b-lora-fp8](sft/gemma4/gemma4-12b-lora-fp8.yaml), [e2b-lora-bf16](sft/gemma4/gemma4-e2b-lora-bf16.yaml), [e2b-lora-fp8](sft/gemma4/gemma4-e2b-lora-fp8.yaml) | 1 |
| GPT-OSS MXFP4 | [lora-mxfp4](sft/gpt-oss/gptoss-lora-mxfp4.yaml) | 4 |
| Laguna-S | [lora-fp8](sft/laguna/laguna-s-lora-fp8.yaml) | 8 |
| LFM2.5 | [lora-bf16](sft/lfm2/lfm25-lora-bf16.yaml), [lora-fp8](sft/lfm2/lfm25-lora-fp8.yaml) | 1 |
| Llama 3.2 | [lora-bf16](sft/llama/llama32-lora-bf16.yaml) | 1 |
| MiniCPM5 | [lora-bf16](sft/minicpm5/minicpm5-lora-bf16.yaml) | 1 |
| Nemotron / Cascade | [cascade2-qlora-bnb](sft/nemotron3/nemotron-cascade2-qlora-bnb.yaml), [nano3-nvfp4](sft/nemotron3/nemotron-nano3-nvfp4.yaml), [nano3-qlora-bnb](sft/nemotron3/nemotron-nano3-qlora-bnb.yaml) | 1 |
| Qwen3 dense | [lora-bf16-ray](sft/qwen3/qwen3-lora-bf16-ray.yaml), [lora-bf16](sft/qwen3/qwen3-lora-bf16.yaml), [lora-fp4](sft/qwen3/qwen3-lora-fp4.yaml), [lora-fp8](sft/qwen3/qwen3-lora-fp8.yaml), [lora-nvfp4](sft/qwen3/qwen3-lora-nvfp4.yaml), [lora-prequant-fp8](sft/qwen3/qwen3-lora-prequant-fp8.yaml), [lora-qbnb](sft/qwen3/qwen3-lora-qbnb.yaml), [lora-qfp4](sft/qwen3/qwen3-lora-qfp4.yaml), [lora-qfp8](sft/qwen3/qwen3-lora-qfp8.yaml) | 1, 2 nodes × 4 |
| Qwen3.5 / 3.6 dense | [text-lora-bf16](sft/qwen35/qwen35-text-lora-bf16.yaml), [text-lora-fp8](sft/qwen35/qwen35-text-lora-fp8.yaml), [qwen35vl-lora-bf16](sft/qwen35/qwen35vl-lora-bf16.yaml), [qwen36-text-lora-fp8-pp](sft/qwen35/qwen36-text-lora-fp8-pp.yaml), [qwen36-text-lora-fp8](sft/qwen35/qwen36-text-lora-fp8.yaml), [qwen36-text-lora-nvfp4](sft/qwen35/qwen36-text-lora-nvfp4.yaml) | 1, 4 |
| Qwen3.5 / 3.6 MoE | [qwen35moe-lora-fp8](sft/qwen35moe/qwen35moe-lora-fp8.yaml), [qwen35moe-lora-qbnb](sft/qwen35moe/qwen35moe-lora-qbnb.yaml), [qwen36moe-lora-fp8](sft/qwen35moe/qwen36moe-lora-fp8.yaml) | 2, 4 |
| Qwen3 MoE | [lora-qbnb](sft/qwen3moe/qwen3moe-lora-qbnb.yaml), [nvfp4](sft/qwen3moe/qwen3moe-nvfp4.yaml) | 1, 4 |
| Qwen3-VL | [lora-bf16](sft/qwen3vl/qwen3vl-lora-bf16.yaml), [text-lora-bf16](sft/qwen3vl/qwen3vl-text-lora-bf16.yaml) | 1 |
| Spark-X2.5 | [lora-bf16](sft/spark/spark-lora-bf16.yaml) | 1 |

`fp4` denotes the NVFP4 compute recipe; `qfp4` denotes online NVFP4 base quantization.
`prequant-fp8`, `nvfp4` model IDs and GPT-OSS MXFP4 examples load already quantized
weights. Filenames alone do not specify compute precision: check `recipe`.
Text-backbone recipes set `train_vision: false`; image recipes set it to `true`.

## Scope and maintenance

The library covers supported user workflows. QeRL noise is rejected by the native
RL runners; replay loss/config fields are not wired into a complete CLI workflow.
Native serving lacks constrained structured outputs and prompt-token scoring, so
RULER and distillation explicitly use external compatible services. Experimental
architecture definitions with deferred training components are not advertised as
ready-to-run recipes. See the [compatibility guide](../docs/appendix/compatibility.md).

One-step/benchmark duplicates and the GPU thermal soak were removed. The few configs
used by regression/capture tooling moved to [training fixtures](../tests/fixtures/training).
Laguna uses one chunked-sequence recipe, reverse-text uses one model, and TurnOPD uses
one student with configurable depth budgeting. To do a smoke run, copy a recipe and
change its step/evaluation/save budget as described in the training guide.
