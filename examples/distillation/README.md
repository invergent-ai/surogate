# Offline knowledge distillation

[qwen3-kd.yaml](qwen3-kd.yaml) captures top-64 Qwen3-1.7B teacher distributions and
trains a Qwen3-0.6B LoRA student against `ce_weight * CE + kd_weight * tau² * KL`.
Teacher and student share a tokenizer. Run from the repository root:

```bash
surogate distill-capture examples/distillation/qwen3-kd.yaml
surogate sft examples/distillation/qwen3-kd.yaml
```

Capture tokenizes the data and writes `.kd` sidecars next to the token shards in
`outputs/distillation/qwen3-kd`. Training reads the same shards. Keep the tokenizer,
packing, sequence length and data unchanged between capture and training; recapture
when they change. The teacher is only loaded during capture. `ce_weight` defaults
to `1 - kd_weight`; set `kd_weight: 1.0` and `ce_weight: 0.0` for pure KL supervision.

## Remote teacher

Use a separate vLLM environment to serve prompt logprobs, then run capture from the
Surogate environment. Native Surogate serving does not implement this scoring API.

```bash
vllm serve Qwen/Qwen3-1.7B --max-logprobs 64
surogate distill-capture examples/distillation/qwen3-kd.yaml --api-base http://localhost:8000/v1
surogate sft examples/distillation/qwen3-kd.yaml
```

Set `distillation.teacher_model` to the served model name. To persist remote settings
in a copy of the config, add them inside the existing `distillation` block:

```yaml
teacher_api_base: http://localhost:8000/v1
teacher_api_key_var: VLLM_API_KEY
teacher_api_concurrency: 8
teacher_api_timeout: 1200
```

The server must return at least `top_k` prompt candidates. Export `VLLM_API_KEY` if
required. This API is distinct from the [on-policy teacher endpoint](../turnopd/README.md).

## Different tokenizers

Install the optional `mergekit` dependency in an appropriate environment, then
transplant the teacher vocabulary onto the student:

```bash
surogate transplant-tokenizer --student Qwen/Qwen3-0.6B --teacher meta-llama/Llama-3.2-1B \
  --output ./outputs/distillation/transplanted
```

This teacher requires Hugging Face access. Copy `qwen3-kd.yaml`, set
`model: ./outputs/distillation/transplanted`, set
`distillation.teacher_model: meta-llama/Llama-3.2-1B`, and choose a new output directory
such as `./outputs/distillation/cross-tokenizer`. Run capture and SFT on that copy.
Merge the trained adapter before restoring the original tokenizer:

```bash
surogate merge --base-model ./outputs/distillation/transplanted \
  --checkpoint-dir ./outputs/distillation/cross-tokenizer \
  --output ./outputs/distillation/cross-tokenizer-merged
surogate transplant-tokenizer --restore ./outputs/distillation/transplanted/transplant_manifest.json \
  --student ./outputs/distillation/cross-tokenizer-merged --output ./outputs/distillation/restored
```

Follow with a short healing SFT using the restored model and newly tokenized data.
See the [distillation guide](../../docs/guides/distillation.md) for constraints,
temperature interpretation and transplant behavior.
