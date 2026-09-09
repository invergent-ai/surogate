# GRPO training

The maintained example trains Qwen3-0.6B with the local `markdown-table-qa` reward
environment. The same configs support split GPUs or one shared GPU:

```bash
surogate grpo --train examples/grpo/train.yaml --infer examples/grpo/infer.yaml \
  --orch examples/grpo/orch.yaml --infer-gpus 0 --trainer-gpus 1
```

See [the GRPO example](../../../examples/grpo/README.md) for colocation,
separate-process launches, evaluation, metrics and resume. Additional workflows
cover [RULER judge rewards](../../../examples/ruler/README.md) and
[multi-turn on-policy distillation](../../../examples/turnopd/README.md).

The [reverse-text SFT recipe](../../../examples/sft/reverse-text-qwen3.yaml) is an
independent demonstration of split prompt/completion columns and automatic adapter
merging; it is not a prerequisite for this GRPO task.
