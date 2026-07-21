# Quickstart: Preference Fine-Tuning (DPO)

This runs an offline **Direct Preference Optimization** job. DPO teaches a model to
prefer a *chosen* response over a *rejected* one for the same prompt — directly, with
no reward model and no rollouts. You supply a static dataset of `{prompt, chosen,
rejected}` triples and run a single command.

## 1) Prepare preference data

A `type: preference` dataset is JSONL with three required fields per row. `prompt` may be a
string or a chat `messages` list; `chosen`/`rejected` are the two competing assistant
continuations. Chat rows may also set `enable_thinking`:

```json
{"prompt": "Scrie corect în limba română.", "chosen": "Ei mergeau acasă.", "rejected": "Ei mergerăm acasă."}
```

Loss is applied only to the response tokens. Minimal pairs — where `chosen` and
`rejected` differ in as little as a single word — give the cleanest signal, because
the model learns the preferred token rather than incidental style or length.

## 2) Write a config

```yaml
# dpo.yaml
model: ./path/to/start_checkpoint        # the model to fine-tune (also the DPO reference)
output_dir: ./out_dpo

lora: true
lora_rank: 16
lora_alpha: 32
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

recipe: fp8-hybrid
optimizer: adamw_8bit
learning_rate: 5.0e-7
lr_scheduler_type: constant
gpus: 1
per_device_train_batch_size: 8           # must be even — each pair is 2 rows (B/2 pairs/step)
gradient_accumulation_steps: 1
sequence_len: 1024
max_steps: 80
save_steps: 20

loss:
  type: dpo
  dpo_beta: 0.1                           # KL temperature; higher pushes harder from the reference
  length_norm: false                      # enable when chosen/rejected lengths differ a lot
  span_mask: false                        # score only disjoint differing token spans
  reference_free: false                  # optimize the absolute chosen/rejected gap
  target_margin: 0.0                     # requires reference_free; beta-scaled target gap

datasets:
  - path: ./pairs.jsonl
    type: preference
```

DPO inherits all SFT settings (LoRA, recipe, optimizer, schedule, checkpoint resume
and retention). The base model is frozen; only the LoRA adapter is trained. In
standard DPO, the **reference** is the frozen start checkpoint and is evaluated
inline on each exact micro-batch. This keeps FP8 activation scaling aligned with
the policy forward. `reference_free: true` skips that extra reference forward.

## 3) Run

```bash
CUDA_VISIBLE_DEVICES=0 surogate dpo dpo.yaml
```

For multiple GPUs, set `gpus: N`; distinct pairs are sharded across the GPUs
(data-parallel) and gradients are averaged.

## 4) What to expect

```
step=0 dpo_loss=0.6931 acc=0.000 margin=0.0000 ...
step=1 dpo_loss=0.6871 acc=1.000 margin=0.0121 ...
```

- **Step 0** logs `dpo_loss ≈ 0.6931 = log 2` with `margin = 0`. This is the identity
  check: at initialization the policy equals the reference (LoRA `B` is zero-init), so
  the preference margin is exactly zero. If you don't see ~0.693 at step 0, something
  is wrong with the reference or the data layout.
- As training proceeds, `dpo_loss` decreases, `margin` goes positive, and `acc`
  (fraction of pairs with `margin > 0`) rises toward 1.0.

## 5) Outputs

- `output_dir/step_{:08}/adapter_model.safetensors` — LoRA checkpoints at `save_steps`;
  older checkpoints are removed according to `save_total_limit`.
- `output_dir/final_adapter/` — the final LoRA adapter.

Merge an adapter into a standalone model with:

```bash
surogate merge --base-model ./path/to/start_checkpoint \
  --checkpoint-dir ./out_dpo/step_00000040 --output ./out_dpo_eval
```

## Tips

- **`dpo_beta`** is the main knob. Too high overfits the preference (and can hurt
  general quality); too low barely moves the model. `0.1` is a reasonable start.
- **Build minimal pairs.** Broad, full-rewrite pairs leak style/length; prefer pairs
  that isolate the exact behavior you want to change.
- **Small learning rate.** The preference gradient is small; `gradient_dtype` and
  the trainable LoRA weights use FP32 so it is not lost to BF16 rounding. The frozen
  base does not receive a wasteful FP32 master copy.

## See also

- [Training Modes](training-modes.md)
- [Quickstart: GRPO](quickstart-grpo.md)
- [Config reference](../reference/config.md)
- [CLI reference](../reference/cli.md)
