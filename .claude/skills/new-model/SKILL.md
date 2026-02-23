---
name: new-model
description: Implement a new model architecture using the Surogate Python DSL. Use when the user wants to add support for a new HuggingFace model (e.g., "add Gemma2 support", "implement DeepSeek model", "add new architecture"). Guides through creating block, model, and HF mapping definitions.
argument-hint: [model-name-or-hf-repo]
---

# Implement a New Model Architecture in Surogate DSL

You are implementing a new model architecture for the Surogate training framework using its Python DSL.
The model name or HuggingFace repo is: **$ARGUMENTS**

## Overview

Surogate models are defined entirely in Python DSL. No C++ changes are needed unless the model requires new primitive operations. The implementation requires:

1. **Block definition** (`surogate/dsl/blocks/<model>.py`) - transformer layer computation
2. **Model definition** (`surogate/dsl/models/<model>.py`) - full model stacking blocks
3. **Module reuse** - compose from existing modules in `surogate/dsl/modules/`
4. **HF integration** - config mapping + weight mapping from HuggingFace checkpoints

## Step-by-Step Process

### Step 1: Research the HuggingFace Model

Before writing any code, thoroughly understand the target architecture:

1. **Read the HF config** (`config.json`) to identify:
   - `architectures` field (e.g., `["Qwen3ForCausalLM"]`)
   - `model_type` field (e.g., `"qwen3"`)
   - All config fields: hidden_size, num_layers, num_heads, intermediate_size, etc.
   - Special fields: attention_bias, use_qk_norm, hybrid patterns, MoE params, etc.

2. **Read the HF model code** (`modeling_*.py`) to understand:
   - Layer structure (pre-norm? post-norm? parallel attention+MLP?)
   - Attention type (MHA, GQA, MQA; with/without RoPE, QK-norm)
   - MLP type (SwiGLU, GELU, ReLU^2; fused gate+up?)
   - Special blocks (MoE, Mamba/SSM, hybrid patterns)
   - Weight naming convention (e.g., `model.layers.{i}.self_attn.q_proj.weight`)

3. **Identify which existing Surogate modules can be reused**:
   - `GQAAttention` / `Qwen3Attention` (with QK-norm) - for standard GQA attention
   - `SwiGLUMLP` - for SwiGLU MLP (gate+up fused)
   - `MoEExpertsGated` / `MoESharedExpert` - for MoE layers
   - `Mamba2Mixer` / `SimpleMLP` - for hybrid Mamba architectures
   - `RMSNorm` - for RMS normalization
   - `Linear` - for generic linear projections

### Step 2: Determine Architecture Type

Classify the model as one of:

| Type | Example | Block Class | Key Differences |
|------|---------|-------------|-----------------|
| **Dense** | Llama, Qwen3 | Single block type | Uniform attention+MLP layers |
| **MoE** | Qwen3-MoE, DeepSeek | Single MoE block | Experts replace MLP, add router |
| **Hybrid** | Nemotron-H | Multiple block types | Interleaved Mamba/Attention/MLP/MoE |
| **VL** | Qwen3-VL | Dense + vision | MultiRoPE, vision encoder |

### Step 3: Create the Block Definition

Create `surogate/dsl/blocks/<model>.py`. The block defines ONE transformer layer.

#### Required Imports

```python
from __future__ import annotations

from ..tensor_type import Tensor
from ..decorators import block, forward, Param, Activation, Gradient
from ..graph_builder import graph
from ..dim import Dim, B, T
```

#### Block Structure Template

```python
@block
class MyModelBlock:
    """<Model> transformer block."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-6,
        # Add model-specific params here
    ):
        # Store ALL constructor params as attributes
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.eps = eps

        # Typed dimensions (short symbolic names for C++ ShapeEnv)
        self.C = Dim("C")        # d_model
        self.Hq = Dim("Hq")      # num_query_heads
        self.Hkv = Dim("Hkv")    # num_kv_heads
        self.D = Dim("D")        # head_size
        self.M = Dim("M")        # d_ff
        self.MaxSeq = Dim("MaxSeq")

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D
        self.MUp = 2 * self.M  # For SwiGLU gate+up fused

    # === Parameters ===
    # (see Parameter Rules below)

    # === Activation Slots ===
    # (see Activation Slot Rules below)

    # === Gradient Slots ===
    # (see Gradient Slot Rules below)

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        with graph() as g:
            # ... computation graph ...
            return out, residual_out
```

#### Parameter Declaration Rules

```python
# Layer norms — never quantized
ln1_weight = Param(Tensor["C"])
ln2_weight = Param(Tensor["C"])

# Attention weights
qkv_weight = Param(Tensor["QKV", "C"])
qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")       # Conditional
out_weight = Param(Tensor["C", "AttnDim"])

# QK-norm weights (conditional)
q_norm_weight = Param(Tensor["D"], when="use_qk_norm")
k_norm_weight = Param(Tensor["D"], when="use_qk_norm")

# RoPE frequencies (frozen, FP32)
rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

# MLP weights (SwiGLU: gate+up fused)
mlp_up_weight = Param(Tensor["MUp", "C"])
mlp_down_weight = Param(Tensor["C", "M"])

# MoE expert weights (with offloading)
router_weight = Param(Tensor["E", "C"])
experts_gate_up = Param(Tensor["E", "MUp", "C"], offload_group="moe_experts")
experts_down = Param(Tensor["E", "C", "M"], offload_group="moe_experts")

# Mamba SSM params (never quantized)
A_log = Param(Tensor["H", "fp32"], quantizable=False)
conv_weight = Param(Tensor["D_conv", "K"], quantizable=False)
dt_bias = Param(Tensor["H", "fp32"], quantizable=False)
```

**Key rules:**
- Norm weights: always add `quantizable=False` (optional since norm is auto-detected, but explicit is better)
- RoPE freqs: `frozen=True` (precomputed, not trained)
- Conditional params: use `when="attr_name"` for bool attrs, `when=lambda self: expr` for complex conditions
- MoE experts: add `offload_group="moe_experts"` for CPU offloading support
- SSM/Mamba params: mark `quantizable=False` for A_log, conv_weight, dt_bias, gated_norm_weight

#### Activation Slot Rules

Every intermediate tensor in the forward pass needs an activation slot declaration. This controls memory management and gradient checkpointing.

```python
# Standard recomputable activation (most common)
ln1 = Activation(
    Tensor["B", "T", "C"],
    aliases=["ln1_flat"],                                    # Alternative runtime names
    recompute=True,                                          # Can recompute in backward
    recompute_from=["res_ffn", "ln1_rstd", "@param:ln1_weight"],  # Inputs for recompute
    recompute_op="rmsnorm_apply_saved",                      # Kernel to use
    recompute_policy="always",                               # always | fft_only | lora_only
    share_policy="when_recomputed",                          # Share across layers when recomputed
)

# Saved FP32 statistics (always save per-layer)
ln1_rstd = Activation(
    Tensor["B", "T"], dtype="fp32", save=True,
    share_policy="per_layer",
)

# Attention output (save for backward, recompute only in FFT)
att = Activation(
    Tensor["B", "T", "AttnDim"],
    save=True,
    recompute=True,
    recompute_group="attn_fwd",                              # Multi-output recompute group
    recompute_outputs=["att", "lse"],                        # Both outputs recomputed together
    recompute_from=["qkv_rope"],
    recompute_op="flash_attention",
    recompute_attrs={"attn_impl": "cudnn"},
    recompute_policy="fft_only",                             # Only recompute in FFT, not LoRA
    share_policy="fft_share",                                # Share in FFT, per-layer in LoRA
)

# LoRA injection point
att_out = Activation(
    Tensor["B", "T", "C"],
    recompute=True,
    recompute_from=["att", "@param:out_weight"],
    recompute_op="matmul",
    recompute_attrs={"matmul_op": "attn_out", "transpose": "NT"},
    recompute_policy="always",
    lora_targets=["o"],                                      # LoRA target names
    share_policy="when_recomputed",
)

# Block output residual (managed by residual manager)
res_ffn = Activation(
    Tensor["B", "T", "C"],
    aliases=["residual_ffn"],
    share_policy="per_layer",                                # Managed by residual manager
)
```

**Recompute reference prefixes:**
- `@param:name` — references a model parameter
- `@global:name` — references a global activation (e.g., `@global:freq_cis`)
- `@input:name` — references a forward input (e.g., `@input:position_ids`)
- `?@param:name` — optional parameter (may not exist based on config)
- No prefix — references another activation slot in the same block

#### Gradient Slot Rules

One gradient slot per activation that receives gradients in the backward pass:

```python
d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
d_mlp_up = Gradient(Tensor["B", "T", "MUp"], gradient_of="mlp_up")
d_res_ffn = Gradient(Tensor["B", "T", "C"], gradient_of="res_ffn")
```

#### Forward Pass Pattern (Dense Block)

The standard pre-norm residual pattern:

```python
@forward
def forward(self, x, residual, position_ids):
    with graph() as g:
        # 1. Pre-attention norm (fused with residual add)
        res_ffn, ln1, ln1_rstd = g.fused_residual_rmsnorm(
            residual, x, "ln1_weight", eps=self.eps,
            res_out_name="res_ffn", y_name="ln1", rstd_name="ln1_rstd",
        )

        # 2. QKV projection
        ln1_flat = g.view(ln1, shape=[B * T, self.C], out_name="ln1_flat")
        qkv_flat = g.matmul(ln1_flat, "qkv_weight", transpose="NT", out_name="qkv_flat")
        qkv = g.view(qkv_flat, shape=[B, T, self.Hq + 2 * self.Hkv, self.D], out_name="qkv")

        # 3. QK-Norm + RoPE (or just RoPE)
        if self.use_qk_norm:
            qkv_rope, q_rstd, k_rstd = g.qkv_qk_norm_rope(
                qkv, "q_norm_weight", "k_norm_weight", "rope_freqs", position_ids,
                eps=self.eps, out_name="qkv_rope", q_rstd_name="q_rstd", k_rstd_name="k_rstd",
            )
        else:
            qkv_rope = g.rope(qkv, "rope_freqs", position_ids, out_name="qkv_rope")

        # 4. Flash Attention
        att, lse = g.flash_attention(qkv_rope, causal=True, out_name="att", lse_name="lse")

        # 5. Output projection
        att_flat = g.view(att, shape=[B * T, self.AttnDim], out_name="att_flat")
        att_out_flat = g.matmul(att_flat, "out_weight", transpose="NT", out_name="att_out_flat")
        att_out = g.view(att_out_flat, shape=[B, T, self.C], out_name="att_out")

        # 6. Pre-MLP norm (fused with residual add)
        res_att, ln2, ln2_rstd = g.fused_residual_rmsnorm(
            res_ffn, att_out, "ln2_weight", eps=self.eps,
            res_out_name="res_att", y_name="ln2", rstd_name="ln2_rstd",
        )

        # 7. MLP (SwiGLU)
        ln2_flat = g.view(ln2, shape=[B * T, self.C], out_name="ln2_flat")
        mlp_up_flat = g.matmul(ln2_flat, "mlp_up_weight", transpose="NT", out_name="mlp_up_flat")
        mlp_up = g.view(mlp_up_flat, shape=[B, T, self.MUp], out_name="mlp_up")
        mlp_act = g.swiglu(mlp_up, out_name="swiglu")
        mlp_act_flat = g.view(mlp_act, shape=[B * T, self.M], out_name="swiglu_flat")
        out_flat = g.matmul(mlp_act_flat, "mlp_down_weight", transpose="NT", out_name="mlp_down_flat")
        out = g.view(out_flat, shape=[B, T, self.C], out_name="mlp_down")

        return out, res_att
```

#### Forward Pass Pattern (MoE Block)

Replace MLP section with MoE routing + expert execution:

```python
# After attention output and pre-MLP norm...

# MoE routing
ln2_flat = g.view(ln2, shape=[B * T, self.C], out_name="ln2_flat")
router_logits = g.matmul(ln2_flat, "router_weight", transpose="NT", out_name="router_logits")
router_probs = g.moe_softmax(router_logits, out_name="router_probs")
# Or for sigmoid routing:
# router_probs = g.moe_sigmoid(router_logits, out_name="router_probs")
weights, indices = g.moe_topk(router_probs, top_k=self.num_experts_per_tok,
                               normalize=True, out_name="moe_weights", indices_name="moe_indices")

# Expert execution
permuted, scatter = g.moe_permute(ln2_flat, indices, top_k=self.num_experts_per_tok,
                                   out_name="moe_permuted", scatter_name="scatter_indices")
gate_up = g.moe_grouped_gemm_gate_up(permuted, "experts_gate_up", scatter,
                                      out_name="moe_gate_up")
act = g.swiglu(gate_up, out_name="moe_act")
down = g.moe_grouped_gemm_down(act, "experts_down", scatter, out_name="moe_down")
moe_out = g.moe_unpermute(down, weights, indices, top_k=self.num_experts_per_tok,
                           out_name="moe_out")

# Optional: shared expert
if self.use_shared_expert:
    shared_up = g.matmul(ln2_flat, "shared_expert_gate_up", transpose="NT")
    shared_act = g.swiglu(shared_up)
    shared_down = g.matmul(shared_act, "shared_expert_down", transpose="NT")
    moe_out = g.add(moe_out, shared_down, out_name="moe_combined")

out = g.view(moe_out, shape=[B, T, self.C], out_name="moe_block_out")
```

### Step 4: Create the Model Definition

Create `surogate/dsl/models/<model>.py`. The model stacks blocks with embeddings and LM head.

#### Required Imports

```python
from __future__ import annotations

from ..tensor_type import Tensor, Array
from ..decorators import model, forward, hf_config, Param, Activation, Gradient
from ..graph_builder import graph
from ..hf import build_dense_block_mappings  # Or build_moe_mappings, etc.
```

#### Model Template

```python
@model
@hf_config(
    architecture="MyModelForCausalLM",     # From HF config.json "architectures"
    model_type="my_model",                  # From HF config.json "model_type"
    d_model="hidden_size",                  # Map DSL param -> HF config field
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    use_qkv_bias="attention_bias",
    # Add model-specific fields
)
class MyModel:
    """<Model> description."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        n_layers: int = 32,
        num_query_heads: int = 32,
        num_kv_heads: int = 8,
        d_ff: int = 14336,
        max_seq: int = 8192,
        head_size: int = 128,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = False,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_ff = d_ff
        self.max_seq = max_seq
        self.head_size = head_size
        self.eps = eps
        self.use_qkv_bias = use_qkv_bias
        self.use_qk_norm = use_qk_norm

        # Derived
        self.D = head_size if head_size > 0 else d_model // num_query_heads

    # === Model weights ===
    embedding = Param(Tensor["vocab_size", "d_model"], hf_mapping="model.embed_tokens.weight")
    blocks = Param(Array["n_layers", "MyModelBlock"])
    final_norm = Param(Tensor["d_model"], hf_mapping="model.norm.weight")
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight")
    # If lm_head is tied to embedding:
    # lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping=tied_to("embedding"))

    # === IO slots ===
    token_ids = Activation(Tensor["B", "T"], dtype="int32", scope="global")
    position_ids = Activation(Tensor["T"], dtype="int32", scope="global")
    targets = Activation(Tensor["B", "T"], dtype="int32", scope="global", aliases=["labels"])
    freq_cis = Activation(Tensor["max_seq", "D", 2], dtype="fp32", scope="global",
                          aliases=["rope_freqs"])

    # === Global activations ===
    x0 = Activation(Tensor["B", "T", "d_model"], aliases=["encoded"], scope="global")
    residual0 = Activation(Tensor["B", "T", "d_model"], scope="global")
    xN = Activation(Tensor["B", "T", "d_model"], scope="global")
    residualN = Activation(Tensor["B", "T", "d_model"], scope="global")
    residual_final = Activation(Tensor["B", "T", "d_model"], scope="global")
    xF = Activation(Tensor["B", "T", "d_model"], aliases=["ln_final"], scope="global")
    xF_flat = Activation(Tensor["B * T", "d_model"], scope="global")
    ln_final_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True, scope="global")
    loss = Activation(Tensor["B * T"], dtype="fp32", scope="global", aliases=["losses"])

    # === Global gradients ===
    d_loss = Gradient(Tensor["B * T"], dtype="fp32", gradient_of="loss", scope="global")
    d_xF = Gradient(Tensor["B", "T", "d_model"], gradient_of="xF", scope="global")
    d_xN = Gradient(Tensor["B", "T", "d_model"], gradient_of="xN", scope="global")
    d_residualN = Gradient(Tensor["B", "T", "d_model"], gradient_of="residualN", scope="global")
    d_x0 = Gradient(Tensor["B", "T", "d_model"], gradient_of="x0", scope="global")

    # === HF Block Mappings ===
    # For dense models:
    _hf_block_mappings_ = build_dense_block_mappings(attn_module=MyAttention)
    # For MoE models:
    # _hf_block_mappings_ = {
    #     **build_norm_mappings(),
    #     **build_attn_mappings(attn_module=MyAttention),
    #     **build_moe_mappings(include_shared=True),
    # }

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor["T", "int32"],
        targets: Tensor["B", "T", "int32"],
    ) -> Tensor["B * T", "fp32"]:
        with graph() as g:
            x0 = g.embedding(token_ids, "embedding")
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            xN, residualN = g.call(
                "StackedBlocks",
                x0, residual0, position_ids,
                num_outputs=2,
                blocks="blocks",
                n_layers=self.n_layers,
            )

            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN, xN, "final_norm", eps=self.eps,
                res_out_name="residual_final",
                y_name="xF",
                rstd_name="ln_final_rstd",
            )

            xF_flat = g.view(xF, shape=["B * T", "d_model"], out_name="xF_flat")
            loss = g.fused_lm_head_loss(xF_flat, "lm_head", targets,
                                        compute_accuracy=True, out_name="loss")
            return loss
```

### Step 5: HF Weight Mapping

#### Module-Level Defaults

Each module (`GQAAttention`, `SwiGLUMLP`, etc.) has `_hf_mapping_defaults_` using `{prefix}` placeholders:

```python
# GQAAttention._hf_mapping_defaults_ =
{
    "qkv_weight": fuse(
        "{prefix}.q_proj.weight",
        "{prefix}.k_proj.weight",
        "{prefix}.v_proj.weight",
        dim=0,
    ),
    "out_weight": "{prefix}.o_proj.weight",
}

# SwiGLUMLP._hf_mapping_defaults_ =
{
    "up_weight": fuse(
        "{prefix}.up_proj.weight",
        "{prefix}.gate_proj.weight",
        dim=0,
    ),
    "down_weight": "{prefix}.down_proj.weight",
}
```

#### Composing Block Mappings

Use helper functions in `surogate/dsl/hf.py`:

```python
from surogate.dsl.hf import (
    build_dense_block_mappings,  # Full dense block (norm + attn + mlp)
    build_norm_mappings,         # Just norms
    build_attn_mappings,         # Just attention
    build_mlp_mappings,          # Just MLP
    build_moe_mappings,          # Router + experts + optional shared
    build_mamba_mappings,        # Mamba2 mixer
    build_simple_mlp_mappings,   # SimpleMLP (relu2 etc.)
    expand_module_mapping,       # Custom module expansion
)

# Dense model:
_hf_block_mappings_ = build_dense_block_mappings(
    layer_prefix="model.layers.{layer}",   # Default
    attn_module=Qwen3Attention,            # Custom attention with QK-norm
    mlp_module=SwiGLUMLP,                  # Default
)

# MoE model:
_hf_block_mappings_ = {
    **build_norm_mappings(layer_prefix="model.layers.{layer}"),
    **build_attn_mappings(layer_prefix="model.layers.{layer}", attn_module=Qwen3Attention),
    **build_moe_mappings(layer_prefix="model.layers.{layer}", include_shared=True),
}
```

#### Weight Transformation Functions

```python
from surogate.dsl.hf import fuse, split, transform, tied_to, stack_experts

# Fuse Q, K, V into QKV
fuse("...q_proj.weight", "...k_proj.weight", "...v_proj.weight", dim=0)

# Split fused gate_up
split("...gate_up_proj.weight", ranges=[(0, d_ff)], dim=0)  # gate only
split("...gate_up_proj.weight", ranges=[(d_ff, 2*d_ff)], dim=0)  # up only

# Transform (transpose, permute)
transform("model.embed_tokens.weight", fn="transpose")

# Tied weights
tied_to("embedding")

# Stack MoE experts
stack_experts("...experts.{expert}.down_proj.weight")
stack_experts("...experts.{expert}.gate_proj.weight", fuse_gate_up=True)
```

### Step 6: Register the Model

Add imports in `surogate/dsl/blocks/__init__.py` and `surogate/dsl/models/__init__.py`:

```python
# In surogate/dsl/blocks/__init__.py
from . import my_model  # noqa: F401

# In surogate/dsl/models/__init__.py
from . import my_model  # noqa: F401
```

### Step 7: Register in Core (Optional but recommended)

Add model info in `surogate/core/model/models/` for CLI support (chat templates, model detection, etc.).

### Step 8: Verify

Run the DSL compiler to verify the model compiles correctly:

```python
from surogate.dsl import compile_model_for_hf
import json

# Load HF config
with open("path/to/config.json") as f:
    config = json.load(f)

result = compile_model_for_hf("MyModelForCausalLM", config)
if result.get("success"):
    print("Model compiled successfully!")
else:
    print("Errors:", result.get("errors"))
```

## Reference: Available Graph Operations

### Matrix Operations
- `g.matmul(a, b, transpose="NT")` - Matrix multiply
- `g.matmul_bias(a, b, bias, transpose="NT")` - Matmul + bias
- `g.batched_matmul(a, b, transpose="NT")` - Batched matmul

### Normalization
- `g.rmsnorm(x, weight, eps=1e-6)` -> (y, rstd)
- `g.fused_residual_rmsnorm(residual, x, weight, eps=1e-6)` -> (res_out, y, rstd)
- `g.layernorm(x, weight, bias, eps=1e-5)` -> (y, mean, rstd)

### Attention
- `g.flash_attention(qkv, causal=True)` -> (out, lse)
- `g.flash_attention_qkv(q, k, v, causal=True)` -> (out, lse)
- `g.rope(qkv, freqs, pos_ids)` - Rotary position embeddings
- `g.mrope(qkv, freqs, pos_ids, mrope_section=[...])` - Multi-RoPE
- `g.qkv_qk_norm_rope(qkv, q_norm, k_norm, freqs, pos_ids, eps)` -> (qkv, q_rstd, k_rstd)
- `g.qkv_qk_norm(qkv, q_norm, k_norm, eps)` -> (qkv, q_rstd, k_rstd)

### Activations
- `g.swiglu(x)` - SwiGLU: silu(gate) * up
- `g.silu(x)` / `g.relu(x)` / `g.relu2(x)` / `g.gelu(x)`
- `g.silu_mul(gate, up)` - silu(gate) * up (separate inputs)

### MoE
- `g.moe_softmax(logits)` / `g.moe_sigmoid(logits)` - Router activation
- `g.moe_topk(probs, top_k=K, normalize=True)` -> (weights, indices)
- `g.moe_permute(x, indices, top_k=K)` -> (permuted, scatter)
- `g.moe_unpermute(x, weights, indices, top_k=K)` - Combine expert outputs
- `g.moe_grouped_gemm_gate_up(x, weights, scatter)` - Fused gate+up GEMM
- `g.moe_grouped_gemm_down(x, weights, scatter)` - Down projection GEMM

### Mamba2 SSM
- `g.mamba_split_proj(projected, intermediate_size=..., conv_dim=..., num_heads=..., head_dim=...)`
- `g.mamba_conv1d(x, weight, bias, activation="silu")`
- `g.mamba_ssm_scan(hidden, dt, A_log, B, C, D, dt_bias, ...)`
- `g.mamba_gated_rmsnorm(x, gate, weight, eps, n_groups, norm_before_gate)`
- `g.mamba_combine_scan(states, conv_weight, ..., num_heads, head_dim, ...)` - Fused conv+scan

### Tensor Ops
- `g.view(x, shape=[B * T, self.C])` - Reshape
- `g.transpose(x, dim0, dim1)` / `g.permute(x, dims=[...])` / `g.contiguous(x)`
- `g.split(x, split_size=..., dim=...)` / `g.concat(a, b, dim=...)`
- `g.add(a, b)` / `g.mul(a, b)` / `g.scale(x, factor=...)` / `g.bias_add(x, bias)`

### Embedding & Loss
- `g.embedding(token_ids, "embedding")` - Embedding lookup
- `g.fused_lm_head_loss(x, "lm_head", targets, compute_accuracy=True)` - Fused LM head + CE loss

### Initialization
- `g.zeros(shape=[...], dtype="bf16")` / `g.ones(shape=[...])` / `g.fill(shape=[...], value=0.5)`

## Reference: Existing Model Files

Study these files for patterns:

| Architecture | Block | Model | Key Features |
|-------------|-------|-------|--------------|
| Dense (Qwen3) | `surogate/dsl/blocks/qwen3.py` | `surogate/dsl/models/qwen3.py` | QK-norm, SwiGLU, standard pattern |
| Dense (Llama) | `surogate/dsl/blocks/llama.py` | `surogate/dsl/models/llama.py` | No QK-norm, standard pattern |
| MoE (Qwen3) | `surogate/dsl/blocks/qwen3_moe.py` | `surogate/dsl/models/qwen3_moe.py` | Softmax routing, shared expert |
| MoE (GPT-OSS) | `surogate/dsl/blocks/gpt_oss.py` | `surogate/dsl/models/gpt_oss.py` | Sigmoid routing, correction bias |
| Hybrid | `surogate/dsl/blocks/nemotron_h.py` | `surogate/dsl/models/nemotron_h.py` | Mamba+Attn+MLP+MoE blocks |
| VL | `surogate/dsl/blocks/qwen3_vl.py` | `surogate/dsl/models/qwen3_vl.py` | MultiRoPE, vision encoder |

## Checklist

Before considering the implementation complete, verify:

- [ ] Block `__init__` stores ALL constructor params as `self.xxx` attributes
- [ ] All dimensions use `Dim("name")` with short symbolic names
- [ ] All parameters have correct shapes and conditions (`when=`)
- [ ] Activation slots cover every intermediate tensor in forward
- [ ] Gradient slots exist for every activation that receives gradients
- [ ] `out_name` is set on every graph operation to match activation slot names
- [ ] `@hf_config` maps ALL relevant HF config fields
- [ ] `_hf_block_mappings_` covers all block parameters
- [ ] Model-level HF mappings cover embedding, final_norm, lm_head
- [ ] Block and model are imported in respective `__init__.py` files
- [ ] The forward pass matches HF implementation (check norm positions, activation types, residual patterns)
- [ ] No RoPE if the model doesn't use it (e.g., Nemotron-H attention)
- [ ] Correct activation function (SwiGLU vs GELU vs ReLU^2)
- [ ] Correct routing type for MoE (softmax vs sigmoid, with/without correction bias)
