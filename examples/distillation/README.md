# Knowledge distillation examples

Offline top-K logit distillation: a teacher model is run once over the
tokenized training data (`.kd` sidecars next to the token shards), then the
student trains against `ce_weight * CE + kd_weight * tau^2 * KL(teacher || student)`.
The teacher and student must share a tokenizer.

## What's in this directory

| File | Purpose |
|---|---|
| [`qwen3-kd.yaml`](qwen3-kd.yaml) | Qwen3-1.7B teacher → Qwen3-0.6B student, LoRA + bf16, one config for both steps. |

## Run

```bash
# Step 1: capture teacher top-K logprobs (writes train-NNN.bin.kd sidecars)
surogate distill-capture examples/distillation/qwen3-kd.yaml

# Step 2: train the student against the sidecars
surogate sft examples/distillation/qwen3-kd.yaml
```

## Remote teacher

If the teacher is too big to load next to your capture GPU, serve it with vLLM
and let capture query the API (set `distillation.teacher_model` to the served
model name):

```bash
vllm serve Qwen/Qwen3-1.7B --max-logprobs 64   # must be >= distillation.top_k
surogate distill-capture examples/distillation/qwen3-kd.yaml --api-base http://localhost:8000/v1
```

## Cross-tokenizer variant

If the teacher uses a different tokenizer, transplant it onto the student
first (requires `pip install mergekit`), point `model:` at the transplanted
directory, then run the same two steps. Optionally restore the native
tokenizer afterwards and follow with a short healing SFT:

```bash
surogate transplant-tokenizer --student Qwen/Qwen3-8B --teacher deepseek-ai/DeepSeek-V3 \
    --output ./qwen3-8b-dsv3-vocab
# ... capture + sft with model: ./qwen3-8b-dsv3-vocab ...
surogate transplant-tokenizer --restore ./qwen3-8b-dsv3-vocab/transplant_manifest.json \
    --student ./output/merged --output ./qwen3-8b-distilled-native
```

See the [Knowledge Distillation guide](../../docs/guides/distillation.md) for
the full `distillation:` reference, loss details, sidecar format,
cross-tokenizer transplantation, and troubleshooting.
