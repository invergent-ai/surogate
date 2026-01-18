---
title: BlockSpec Cookbook
---

# BlockSpec Cookbook

This page shows how to compose new transformer block variants using the declarative `BlockSpec`
pipeline (BlockBuilder → BlockExecutor). It’s intentionally short and practical.

## The core idea

You describe a block as an ordered list of primitive ops:

```
LN1 -> QKV -> (QKNorm) -> RoPE -> Attention -> AttnOut
  -> ResidualAdd | ResidualLN2
  -> LN2 -> MLPUp -> SwiGLU -> MLPDown
```

**ASCII diagram (dense pre‑norm):**

```
residual ──► LN1 ─► QKV ─► (QKNorm) ─► RoPE ─► Attn ─► AttnOut ─┐
                                                             │
residual ─────────────────────────────────────────────────────┼─► Residual+LN2 ─► MLPUp ─► SwiGLU ─► MLPDown
                                                             │
                                                        residual_att
```

The runtime calls:
- `BlockBuilder::build(config, layer_idx)` → returns `BlockSpec`
- `BlockExecutor::{forward|backward|recompute}(spec, …)` → runs the ops

You generally **do not call BlockBuilder manually**. It’s wired into the modular block path
when `ModelOptions::use_modular_blocks = true`.

---

## Available ops (current)

Order matters:

- `LN1`
- `QKV`
- `QKNorm` (optional)
- `RoPE`
- `Attention`
- `AttnOut`
- `ResidualAdd` (parallel residual style)
- `ResidualLN2` (fused residual add + norm)
- `LN2`
- `MLPUp`
- `SwiGLU`
- `MLPDown`

---

## Variant recipes

### 1) LLaMA / standard dense pre‑norm

**Config:**
- `config.use_parallel_residual = false`
- `config.use_qk_norm = false`

**Spec:**
```
LN1, QKV, RoPE, Attention, AttnOut, ResidualLN2, MLPUp, SwiGLU, MLPDown
```

This is the default dense spec returned by `BlockBuilder::dense()`.

---

### 2) Qwen3‑style (dense + QK norm + QKV bias)

**Config:**
- `config.use_qk_norm = true`
- `config.UseQKVBias = true` (weight layout; not a spec op)

**Spec:**
```
LN1, QKV, QKNorm, RoPE, Attention, AttnOut, ResidualLN2, MLPUp, SwiGLU, MLPDown
```

Notes:
- QK norm is injected by `BlockBuilder` if `config.use_qk_norm` is true.
- QKV bias is handled by the weights (not a spec op).

---

### 3) GPT‑NeoX / parallel residual

**Config:**
- `config.use_parallel_residual = true`
- `config.use_qk_norm = false` (or true if desired)

**Spec:**
```
LN1, QKV, (QKNorm), RoPE, Attention, AttnOut, ResidualAdd,
LN2, MLPUp, SwiGLU, MLPDown
```

**Semantics:**
Both attention and MLP are computed from the same residual stream (parallel), then combined.

---

## Where to add new variants

If you want a new variant:
1. Update `BlockBuilder` (most cases).
2. If you need a new op, add to:
   - `BlockOp` in `csrc/src/modules/composite/block_spec.h`
   - Implement it in `BlockExecutor` (forward/backward/recompute)

---

## How to enable the BlockSpec path

```cpp
ModelOptions options;
options.use_modular_blocks = true;
```

Then select the variant via `ModelConfig`:

```cpp
config.use_parallel_residual = true; // GPT‑NeoX
config.use_qk_norm = true;           // Qwen3‑style
```

The model will automatically use BlockBuilder + BlockExecutor.

