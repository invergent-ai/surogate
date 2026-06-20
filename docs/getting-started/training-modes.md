# Training Modes

Surogate supports five practical ways to adapt a model:

1) **Pretraining / Continued Pretraining (PT)**
2) **Full Fine-Tuning**
3) **LoRA / QLoRA (Adapter Fine-Tuning)**
4) **RL Fine-Tuning (GRPO)**
5) **Preference Fine-Tuning (DPO)**

They differ in *which parameters are updated*, *how much data you need*, and *how much compute/VRAM you’ll spend*.

---

## Quick decision guide

| Goal | Recommended mode |
| --- | --- |
| Train a base model from scratch on large text | **Pretraining** |
| Continue training a base model on more text (domain adaptation) | **Continued pretraining** |
| Maximize quality for a specific task/domain and you can afford it | **Full fine-tuning** |
| Fast adaptation, smaller GPU, cheaper runs, easy iteration | **LoRA** |
| Same as LoRA but you’re VRAM-limited | **QLoRA** |
| Improve reasoning, math, coding via reward-based training | **GRPO** |
| Compress a larger model into a smaller one from the same family | **SFT + [Knowledge Distillation](../guides/distillation.md)** |
| Steer behavior from chosen-vs-rejected preference pairs (offline) | **DPO** |

---

## 1) Pretraining / Continued Pretraining (PT)

### How it works
**Pretraining** updates *all* (or nearly all) model weights by predicting the next token on large-scale raw text.

- **What updates?** Base model weights (full model training)
- **Typical data:** Raw text corpora (`type: text` datasets)
- **Typical run shape:** Lots of tokens, long runs, throughput-focused

In Surogate, you typically run PT with:

```bash
surogate pt path/to/config.yaml
```

### When to use it
- You’re training a new base model.
- You want **domain adaptation** via continued pretraining (e.g., legal/medical/finance corpora).
- You care about general capabilities and broad distribution learning.

### Tradeoffs
- Highest compute + longest wall-clock.
- Requires the most data.

---

## 2) Full Fine-Tuning (update all weights)

### How it works
**Full fine-tuning** starts from a pretrained checkpoint and updates the full parameter set on a task/domain dataset.

- **What updates?** Base model weights (and optionally embeddings / lm_head depending on your config)
- **Typical data:** Instruction or conversation datasets
- **Typical run shape:** Fewer tokens than PT, but still heavy VRAM/compute

In Surogate, full fine-tuning is usually done through the SFT workflow with LoRA disabled:

```yaml
lora: false
```

(and then run via `surogate sft ...`).

### When to use it
- You need the *best possible* task/domain performance.
- You can afford the VRAM/compute of updating all weights.
- You don’t need easy “adapter swapping” across many downstream tasks.

### Tradeoffs
- More expensive and heavier than LoRA.
- Harder to maintain multiple downstream variants (each run produces a full checkpoint).

---

## 3) LoRA / QLoRA (adapter fine-tuning)

### How LoRA works
**LoRA** freezes the base model weights and trains small low-rank adapter matrices inserted into selected linear layers.

- **What updates?** Only LoRA adapter parameters
- **Base weights:** Frozen (unchanged)
- **Typical data:** Instruction / conversation / task datasets

In config, you enable LoRA with:

```yaml
lora: true
lora_rank: 16
lora_alpha: 32
```

### When to use LoRA
- You want fast iteration and lower cost.
- You want multiple downstream specializations (adapters are easy to store, ship, and swap).
- You’re experimenting and want quick turnarounds.

### What is QLoRA?
**QLoRA** is LoRA plus quantization of the frozen base model weights to reduce VRAM.

- **What updates?** LoRA adapters (still)
- **Base weights:** Frozen and stored in a quantized format (FP8 / FP4 / NF4)
- **When to use it:** When VRAM is the bottleneck

Surogate supports:
- `qlora_fp8` (SM89+)
- `qlora_fp4` (SM100+ Blackwell)
- `qlora_bnb` (NF4 via BitsAndBytes, broad compatibility)

---

## 4) RL Fine-Tuning (GRPO)

### How it works
**GRPO** (Group Relative Policy Optimization) is a reinforcement learning method that improves a model by generating rollouts, scoring them with a reward function, and updating the policy to favor higher-reward responses.

- **What updates?** LoRA adapter parameters (base model is frozen)
- **Typical data:** Prompts + reward environments (math, code, custom verifiers)
- **Typical run shape:** Iterative rollout → reward → gradient loop

GRPO coordinates three components in a single command:

```bash
surogate grpo --train train.yaml --infer infer.yaml --orch orch.yaml
```

### When to use it
- You want to improve reasoning, math, or coding capabilities beyond what SFT achieves.
- You have a reward signal (verifier, unit tests, reference answers) rather than just demonstration data.
- You want to optimize for outcome quality rather than imitating a fixed dataset.

### Tradeoffs
- Requires a reward environment (built-in or custom).
- More complex pipeline than SFT (inference server + orchestrator + trainer).
- Training dynamics are less predictable — requires monitoring importance ratios and masking fractions.

---

## 5) Preference Fine-Tuning (DPO)

### How it works
**DPO** (Direct Preference Optimization) teaches a model to prefer a *chosen* response over a *rejected* one for the same prompt, directly — no reward model and no rollouts. It is **offline**: you supply a static dataset of `{prompt, chosen, rejected}` triples, and the loss raises the implicit reward of `chosen` relative to `rejected`:

```
margin = β · [ (logπθ − logπref)(chosen) − (logπθ − logπref)(rejected) ]
loss   = −log σ(margin)
```

- **What updates?** LoRA adapter parameters (base model is frozen)
- **Reference model:** the start checkpoint, captured once (LoRA disabled) and cached
- **Typical data:** `type: preference` datasets (`{prompt, chosen, rejected}`)
- **Typical run shape:** a few hundred gradient steps over preference pairs

In Surogate, DPO runs from a single config:

```bash
surogate dpo path/to/config.yaml
```

```yaml
loss:
  type: dpo
  dpo_beta: 0.1
  length_norm: false   # enable when chosen/rejected lengths differ a lot
datasets:
  - path: pairs.jsonl
    type: preference
```

### When to use it
- You have **paired preferences** (chosen vs rejected), not a reward function or pure demonstrations.
- You want to fix a specific behavior with **minimal, targeted contrasts** (e.g. correct vs incorrect word forms) without the cost/instability of RL.
- You want something simpler than GRPO: no inference server, no orchestrator.

### Tradeoffs
- Needs preference pairs; quality depends heavily on pair construction.
- `dpo_beta` controls how hard the policy is pushed from the reference — too high overfits, too low barely moves.
- Offline only (pairs are fixed; no on-policy regeneration).

---

## Practical recommendations

- Start with **LoRA (bf16 recipe)** unless you have a strong reason not to.
- Use **QLoRA** when you can’t fit the base model + activations comfortably.
- Use **Full fine-tuning** when you want maximum quality and have budget.
- Use **(Continued) pretraining** for domain adaptation on large raw text.
- Use **GRPO** when you have a reward signal and want to go beyond imitation learning.
- Use **DPO** when you have chosen-vs-rejected preference pairs and want offline preference steering without a reward model.

---

## See also

- [Quickstart: Pretraining](quickstart-pretraining.md)
- [Quickstart: Supervised Fine-Tuning](quickstart-sft.md)
- [Quickstart: GRPO](quickstart-grpo.md)
- [Quickstart: DPO](quickstart-dpo.md)
- [Configuration](../guides/configuration.md)
- [Precision & recipes](../guides/precision-and-recipes.md)
- [QLoRA](../guides/qlora.md)
- [Knowledge Distillation](../guides/distillation.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.mdx)
