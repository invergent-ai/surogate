---
title: "DSL Language"
---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Python Decorator Syntax](#2-python-decorator-syntax)
3. [Type System](#3-type-system)
4. [Module Definitions](#4-module-definitions)
5. [Block Definitions](#5-block-definitions)
6. [Model Definitions](#6-model-definitions)
7. [Primitive Definitions](#7-primitive-definitions)
8. [Graph Builder API](#8-graph-builder-api)
9. [HuggingFace Integration](#9-huggingface-integration)
10. [Activation Slots](#10-activation-slots)
11. [Compilation Pipeline](#11-compilation-pipeline)
12. [Diagnostics](#12-diagnostics)
13. [Examples](#13-examples)

---

## 1. Introduction

### 1.1 Purpose

The Module DSL is a Python decorator-based domain-specific language for defining neural network architectures with explicit computation graphs. It compiles to a JSON IR that targets Surogate's high-performance C++/CUDA execution engine.

### 1.2 Design Goals

1. **Python-native**: Uses standard Python class syntax with decorators and type annotations
2. **Explicit computation**: Forward graphs are explicitly defined using a graph builder API (backward graphs are not compiled yet; see Section 11)
3. **HuggingFace compatible**: First-class support for weight mapping, config translation, and checkpoint import/export
4. **Shape metadata**: Tensor shapes/dtypes are annotated with symbolic dimensions and serialized into the JSON IR (no full static shape validation yet)
5. **Memory control metadata**: Activation slots and save/recompute annotations are captured and serialized for the runtime to use

### 1.3 Non-Goals

- Automatic differentiation (autograd)
- Dynamic control flow (while loops with data-dependent conditions)
- Eager execution mode

### 1.4 Terminology

| Term                | Definition                                                    |
| ------------------- | ------------------------------------------------------------- |
| **Module**          | A reusable computation unit with parameters and forward graph |
| **Block**           | A transformer layer with attention + MLP and activation slots |
| **Model**           | A top-level architecture with HuggingFace integration         |
| **Primitive**       | A built-in CUDA kernel wrapper                                |
| **Graph Builder**   | Context manager API for defining computation graphs           |
| **Activation Slot** | Pre-declared tensor buffer for forward/backward passes        |

---

## 2. Python Decorator Syntax

### 2.1 Core Decorators

The DSL uses Python class decorators to define neural network components:

```python
from surogate.dsl import module, block, model, primitive, param, forward, backward

@module       # Define a reusable module (Linear, RMSNorm, etc.)
@block        # Define a transformer block with activation slots
@model        # Define a top-level model with HF integration
@primitive    # Define a CUDA kernel wrapper
```

### 2.2 Parameter Decorators

```python
from surogate.dsl import param, Param, tied_to, save, recompute

@param                           # Decorator style for parameters
Param(Tensor["C", "D"])          # Class attribute style for parameters
@tied_to("embedding")            # Tie parameter to another (method-style @param only)
@save("x", "weight")             # Forward save list
@recompute("hidden")             # Forward recompute list
```

**Note**:
- `@save(...)` / `@recompute(...)` can be placed either above or below `@forward` (both orders work).
- The compiled `forward.save` / `forward.recompute` lists are the union of decorator directives and any explicit `g.save(...)` / `g.mark_recompute(...)` calls in the `graph()` context (Section 8.14).

### 2.3 HuggingFace Decorators

```python
from surogate.dsl import hf_config, hf_mapping, hf_export

@hf_config(architecture="Qwen3ForCausalLM", ...)  # Map HF config
@hf_mapping(embedding="model.embed_tokens.weight", ...)  # Import mapping
@hf_mapping.indexed("blocks", ...)  # Indexed block mapping (default index var is "{layer}")
@hf_export(...)  # Export mapping
```

### 2.4 Module Structure Decorators

```python
from surogate.dsl import abstract, extends

@abstract     # Mark module as abstract (interface only)
@extends("BaseModule")  # Inherit from another module
```

---

## 3. Type System

### 3.1 Tensor Type Annotations

Tensor shapes are annotated using the `Tensor` subscript syntax:

```python
from surogate.dsl import Tensor

# Basic tensor annotation
Tensor["B", "T", "C"]                    # Symbolic dimensions
Tensor["B", "T", 4096]                   # Mixed symbolic/concrete
Tensor["B", "T", "C", "fp32"]            # With explicit dtype
Tensor["C", "D"] | None                  # Optional tensor

# Dtype as last element (if string in known dtype set)
Tensor["B", "T", "bf16"]                 # bfloat16 tensor
Tensor["B", "T", "fp8_e4m3"]             # FP8 E4M3 tensor
```

**Note**: Annotations typically use string dimensions (e.g., `"B"`, `"T"`, `"C"`) because Python evaluates type hints at class definition time before `self` is available.
The implementation also accepts `Dim`, `DimExpr`, and integer dimensions inside `Tensor[...]`, but method annotations generally cannot reference `self` (e.g., `Tensor[self.C]` will not work).

### 3.2 Supported Dtypes

| Dtype      | Description                    |
| ---------- | ------------------------------ |
| `bf16`     | bfloat16 (default)             |
| `fp32`     | float32                        |
| `fp16`     | float16                        |
| `fp8_e4m3` | FP8 E4M3 (forward activations) |
| `fp8_e5m2` | FP8 E5M2 (backward gradients)  |
| `fp4_e2m1` | FP4 E2M1 (Blackwell+)          |
| `int8`     | 8-bit integer                  |
| `int32`    | 32-bit integer                 |

### 3.3 Array Type Annotations

For repeated elements (e.g., stacked layers):

```python
from surogate.dsl import Array

Array["n_layers", "DenseTransformerBlock"]  # Symbolic size
Array[8, "ExpertMLP"]                       # Concrete size
```

### 3.4 Dimension System

The DSL provides first-class `Dim` and `DimExpr` objects for dimension arithmetic:

```python
from surogate.dsl import Dim, B, T

# Predefined batch/sequence dimensions
B  # Batch dimension
T  # Sequence length

# Create typed dimensions bound to config parameters
C = Dim("d_model")
H = Dim("num_heads")
D = C // H  # DimExpr: computed dimension

# Use in graph operations (not annotations)
x_flat = g.view(x, shape=[B * T, self.C])
```

**Important**:
- Annotations typically use strings: `Tensor["B", "T", "C"]` (string expressions like `"D // 2"` are allowed and preserved in the IR).
- Graph operations can use `Dim` / `DimExpr` objects: `g.view(x, shape=[B * T, self.C])`.

### 3.5 Optional Types

```python
# Optional parameter (may be None)
bias: Tensor["O"] | None

# Conditional parameter (included when condition is true)
bias = Param(Tensor["O"], when="use_bias")
```

---

## 4. Module Definitions

### 4.1 Basic Module Structure

```python
from surogate.dsl import module, param, Param, forward, save, Tensor
from surogate.dsl import graph, Dim, B, T

@module
class Linear:
    """Linear projection: y = x @ W^T (+ bias)."""

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        # Typed dimensions bound to config parameters
        self.C = Dim("in_dim")
        self.O = Dim("out_dim")

    # Parameters - class attribute style
    weight = Param(Tensor["O", "C"])
    bias = Param(Tensor["O"], when="use_bias")

    @save("x")
    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "O"]:
        with graph() as g:
            x_flat = g.view(x, shape=[B * T, self.C])

            if self.use_bias:
                y_flat = g.matmul_bias(x_flat, "weight", "bias", transpose="NT")
            else:
                y_flat = g.matmul(x_flat, "weight", transpose="NT")

            y = g.view(y_flat, shape=[B, T, self.O])
            return y
```

### 4.2 Parameter Declaration Styles

**Class attribute style (recommended):**

```python
@module
class MyModule:
    weight = Param(Tensor["O", "C"])
    bias = Param(Tensor["O"], when="use_bias")
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)
    blocks = Param(Array["n_layers", "TransformerBlock"])
```

**Decorator style (alternative):**

```python
@module
class MyModule:
    @param
    def weight(self) -> Tensor["O", "C"]:
        ...

    @param(condition=lambda self: self.use_bias)
    def bias(self) -> Tensor["O"]:
        ...

    @param(frozen=True)
    def rope_freqs(self) -> Tensor["MaxSeq", "D // 2", 2, "fp32"]:
        ...
```

### 4.3 Param Class Options

```python
Param(
    param_type,        # Tensor[...] or Array[...]
    when="condition",  # Condition string (attribute name) or callable
    frozen=True,       # Precomputed, not trained
    hf_mapping="...",  # HuggingFace weight path
)
```

**Note**: tied parameters (`ParamKind.TIED`) are currently only expressible via the method-style `@param` + `@tied_to(...)` decorators, not via the class-attribute `Param(...)` helper.

### 4.4 Abstract Modules

```python
from surogate.dsl import abstract, module

@abstract
@module
class BaseAttention:
    """Abstract attention interface."""

    def __init__(self, d_model: int, num_heads: int):
        ...

    qkv_weight = Param(Tensor["QKV", "C"])
    out_weight = Param(Tensor["C", "AttnDim"])

    # No forward implementation - must be provided by subclass
```

### 4.5 Module Inheritance

```python
from surogate.dsl import extends, module

@extends("BaseAttention")
@module
class LlamaAttention:
    def __init__(self, d_model, num_heads, max_seq):
        super().__init__(d_model, num_heads)
        self.max_seq = max_seq

    # Add new parameter
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"])

    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        # Implementation
        ...
```

---

## 5. Block Definitions

Blocks are modules representing transformer layers with activation slot declarations.

### 5.1 Block Structure

```python
from surogate.dsl import block, forward, Param, Activation, Gradient, Tensor
from surogate.dsl import graph, Dim, B, T

@block
class DenseTransformerBlock:
    """Pre-norm dense transformer block with attention + MLP."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = False,
    ):
        self.d_model = d_model
        # ... store config

        # Typed dimensions
        self.C = Dim("C")
        self.Hq = Dim("Hq")
        self.Hkv = Dim("Hkv")
        self.D = Dim("D")
        self.M = Dim("M")

        # Derived dimensions (DimExpr)
        self.QKV = (self.Hq + 2 * self.Hkv) * self.D
        self.AttnDim = self.Hq * self.D

    # LayerNorm weights
    ln1_weight = Param(Tensor["C"])
    ln2_weight = Param(Tensor["C"])

    # Attention weights
    qkv_weight = Param(Tensor["QKV", "C"])
    qkv_bias = Param(Tensor["QKV"], when="use_qkv_bias")
    out_weight = Param(Tensor["C", "AttnDim"])
    q_norm_weight = Param(Tensor["D"], when="use_qk_norm")
    k_norm_weight = Param(Tensor["D"], when="use_qk_norm")
    rope_freqs = Param(Tensor["MaxSeq", "D // 2", 2, "fp32"], frozen=True)

    # MLP weights
    mlp_up_weight = Param(Tensor["MUp", "C"])
    mlp_down_weight = Param(Tensor["C", "M"])

    # Activation slots (see Section 10)
    ln1 = Activation(Tensor["B", "T", "C"], aliases=["ln1_flat"], recompute=True)
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True)
    qkv = Activation(Tensor["B", "T", "QKV"], aliases=["qkv_flat", "qkv_biased"])
    # ... more activation slots

    # Gradient slots
    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    # ... more gradient slots

    @forward
    def forward(
        self,
        x: Tensor["B", "T", "C"],
        residual: Tensor["B", "T", "C"],
        position_ids: Tensor["T", "int32"],
    ) -> tuple[Tensor["B", "T", "C"], Tensor["B", "T", "C"]]:
        """Returns (out, residual_out)."""
        with graph() as g:
            # Pre-attention LayerNorm (fused with residual)
            res_ffn, ln1_out, ln1_rstd = g.fused_residual_rmsnorm(
                residual, x, "ln1_weight", eps=self.eps,
                res_out_name="res_ffn",
                y_name="ln1",
                rstd_name="ln1_rstd",
            )
            # ... rest of implementation
            return out, res_att
```

---

## 6. Model Definitions

Models are top-level architectures with HuggingFace integration.

### 6.1 Model Structure

```python
from surogate.dsl import model, forward, hf_config, Param, Activation, Gradient
from surogate.dsl import Tensor, Array, graph
from surogate.dsl import fuse

@model
@hf_config(
    architecture="Qwen3ForCausalLM",
    model_type="qwen3",
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    use_qkv_bias="attention_bias",
)
class Qwen3Model:
    """Qwen3 model using Qwen3Block."""

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 1024,
        n_layers: int = 28,
        num_query_heads: int = 16,
        num_kv_heads: int = 8,
        d_ff: int = 3072,
        max_seq: int = 40960,
        head_size: int = 128,
        eps: float = 1e-6,
        use_qkv_bias: bool = False,
        use_qk_norm: bool = True,
    ):
        # Store config
        self.vocab_size = vocab_size
        self.d_model = d_model
        # ...

    # Model weights
    embedding = Param(Tensor["vocab_size", "d_model"],
                      hf_mapping="model.embed_tokens.weight")
    blocks = Param(Array["n_layers", "Qwen3Block"])
    final_norm = Param(Tensor["d_model"], hf_mapping="model.norm.weight")
    lm_head = Param(Tensor["vocab_size", "d_model"], hf_mapping="lm_head.weight")

    # IO slots
    token_ids = Activation(Tensor["B", "T"], dtype="int32", scope="global")
    position_ids = Activation(Tensor["T"], dtype="int32", scope="global")
    targets = Activation(Tensor["B", "T"], dtype="int32", scope="global",
                         aliases=["labels"])

    # Global activation slots
    x0 = Activation(Tensor["B", "T", "d_model"], aliases=["encoded"], scope="global")
    xF = Activation(Tensor["B", "T", "d_model"], aliases=["ln_final"], scope="global")
    loss = Activation(Tensor["B * T"], dtype="fp32", scope="global")

    # HF block mappings (fuse Q, K, V projections)
    _hf_block_mappings_ = {
        "ln1_weight": "model.layers.{layer}.input_layernorm.weight",
        "qkv_weight": fuse(
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            dim=0,
        ),
        "out_weight": "model.layers.{layer}.self_attn.o_proj.weight",
        # ...
    }

    @forward
    def forward(
        self,
        token_ids: Tensor["B", "T", "int32"],
        position_ids: Tensor["T", "int32"],
        targets: Tensor["B", "T", "int32"],
    ) -> Tensor["B * T", "fp32"]:
        with graph() as g:
            # Embedding lookup
            x0 = g.embedding(token_ids, "embedding")

            # Initialize residual stream
            residual0 = g.zeros(shape=["B", "T", "d_model"], dtype="bf16")

            # Stacked blocks
            xN, residualN = g.call(
                "StackedBlocks",
                x0, residual0, position_ids,
                num_outputs=2,
                blocks="blocks",
                n_layers=self.n_layers,
            )

            # Final norm + LM head + loss
            residual_final, xF, ln_final_rstd = g.fused_residual_rmsnorm(
                residualN, xN, "final_norm", eps=self.eps,
            )
            xF_flat = g.view(xF, shape=["B * T", "d_model"])
            loss = g.fused_lm_head_loss(xF_flat, "lm_head", targets)

            return loss
```

---

## 7. Primitive Definitions

Primitives wrap CUDA kernels with type-safe signatures.

### 7.1 Basic Primitive

```python
from surogate.dsl import primitive, save, Tensor

@primitive(impl="kernels.matmul")
def matmul(
    A: Tensor["M", "K"],
    B: Tensor["K", "N"],
    *,
    transpose: str = "NN",
    accumulate: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> Tensor["M", "N"]:
    """Matrix multiplication: C = alpha * op(A) @ op(B) + beta * C."""
    ...

@matmul.backward
@save("A", "B")
def matmul_backward(
    d_C: Tensor["M", "N"],
    A: Tensor["M", "K"],
    B: Tensor["K", "N"],
) -> tuple[Tensor["M", "K"], Tensor["K", "N"]]:
    """Backward pass for matmul."""
    ...
```

**Status**: `@primitive.backward` registers backward signature metadata, but `compile_model` currently serializes only forward graphs. Backward graphs are not emitted into the JSON IR in v0.1.0.

### 7.2 Primitive Categories

The DSL provides registered `@primitive` operations:

| Category      | Primitives                                                                                                                    |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Matrix ops    | `matmul`, `batched_matmul`                                                                                                    |
| Normalization | `rmsnorm`, `fused_residual_rmsnorm`                                                                                           |
| Activations   | `swiglu`, `silu`, `relu2`, `silu_mul`                                                                                         |
| Attention     | `flash_attention`, `rope`, `qkv_qk_norm_rope`                                                                                 |
| Tensor ops    | `view`, `transpose`, `split`, `concat`                                                                                        |
| Elementwise   | `add`, `mul`, `scale`, `bias_add`                                                                                             |
| Embedding     | `embedding`                                                                                                                   |
| Losses        | `fused_lm_head_loss`                                                                                                          |
| MoE           | `moe_softmax`, `moe_sigmoid`, `moe_topk`, `moe_permute`, `moe_unpermute`, `moe_grouped_gemm_gate_up`, `moe_grouped_gemm_down` |
| Init          | `zeros`, `ones`, `fill_normal`                                                                                                |

> **Note:** `GraphBuilder` methods (Section 8) provide a superset of operations including convenience methods like `relu()`, `gelu()`, `softmax()`, `layernorm()`, `permute()` that are not registered as `@primitive` definitions. The `@primitive` decorator is used for operations with explicit backward implementations that map directly to C++ kernels.
>
> **Implementation note:** the current Python compiler does not validate that an op used in `GraphBuilder` has a registered `@primitive` spec; unknown ops can be emitted via `g.custom(...)` and are resolved by the C++ runtime.

---

## 8. Graph Builder API

The `graph()` context manager provides a fluent API for building computation graphs.

### 8.1 Basic Usage

```python
from surogate.dsl import forward, graph, Tensor, B, T

@forward
def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
    with graph() as g:
        # Operations return GraphRef objects
        x_flat = g.view(x, shape=[B * T, self.C])
        y_flat = g.matmul(x_flat, "weight", transpose="NT")
        y = g.view(y_flat, shape=[B, T, self.C])
        return y
```

### 8.2 GraphRef

Operations return `GraphRef` objects that can be used as inputs to other operations:

```python
with graph() as g:
    a = g.input("x")           # Register input
    b = g.matmul(a, "weight")  # Returns GraphRef
    c = g.relu(b)              # Use GraphRef as input
    g.output(c)                # Register output
```

### 8.3 Matrix Operations

```python
# Matrix multiply
y = g.matmul(a, b, transpose="NT", accumulate=False, alpha=1.0, beta=0.0)

# Matmul with fused bias
y = g.matmul_bias(a, b, bias, transpose="NT")

# Batched matmul
y = g.batched_matmul(a, b, transpose="NN")
```

### 8.4 Normalization

```python
# RMSNorm (returns y, rstd)
y, rstd = g.rmsnorm(x, "weight", eps=1e-6)

# Fused residual + RMSNorm (returns residual_out, y, rstd)
res_out, y, rstd = g.fused_residual_rmsnorm(
    residual, x, "weight", eps=1e-6,
    res_out_name="res",
    y_name="ln",
    rstd_name="rstd",
)

# LayerNorm (returns y, mean, rstd)
y, mean, rstd = g.layernorm(x, "weight", "bias", eps=1e-5)
```

### 8.5 Activations

```python
y = g.swiglu(x)                    # SwiGLU: silu(gate) * up
y = g.silu(x)                      # SiLU (Swish)
y = g.relu(x)                      # ReLU
y = g.relu2(x)                     # ReLU squared
y = g.gelu(x)                      # GELU
y = g.softmax(x, dim=-1)           # Softmax
y = g.silu_mul(gate, up)           # silu(gate) * up
```

### 8.6 Attention

```python
# FlashAttention (packed QKV, returns out, lse)
out, lse = g.flash_attention(qkv, causal=True, softmax_scale=None, window_size=None)

# FlashAttention with separate Q, K, V
out, lse = g.flash_attention_qkv(q, k, v, causal=True)

# RoPE
qkv_rope = g.rope(qkv, "rope_freqs", position_ids, rotary_dim="D")

# Fused QK-Norm + RoPE (returns qkv_out, q_rstd, k_rstd)
qkv_out, q_rstd, k_rstd = g.qkv_qk_norm_rope(
    qkv, "q_norm_weight", "k_norm_weight", "rope_freqs", position_ids, eps=1e-6
)
```

### 8.7 Tensor Manipulation

```python
# Reshape
y = g.view(x, shape=[B * T, self.C])
y = g.view(x, shape=["B", "T", "C"])  # String dims are preserved in the IR for the runtime shape environment

# Transpose
y = g.transpose(x, dim0=0, dim1=1)

# Permute
y = g.permute(x, dims=[0, 2, 1, 3])

# Split (returns tuple)
a, b = g.split(x, split_size=[d1, d2], dim=0)

# Concat
y = g.concat(a, b, c, dim=0)

# Copy
y = g.copy(x)

# Make contiguous
y = g.contiguous(x)
```

### 8.8 Elementwise Operations

```python
c = g.add(a, b)          # Element-wise add
c = g.mul(a, b)          # Element-wise multiply
y = g.scale(x, factor=0.5)  # Scale by constant
y = g.bias_add(x, bias)  # Add bias
```

### 8.9 Embedding and Loss

```python
# Embedding lookup
embedded = g.embedding(token_ids, "embedding_weight")

# Fused LM head + cross-entropy loss
loss = g.fused_lm_head_loss(xF_flat, "lm_head", targets, compute_accuracy=False)
```

### 8.10 Initialization

```python
z = g.zeros(shape=["B", "T", "C"], dtype="bf16")
o = g.ones(shape=["B", "T", "C"], dtype="bf16")
f = g.fill(shape=["B", "T", "C"], value=0.5, dtype="bf16")
```

> **Note:** The `@primitive` decorator provides `fill_normal()` for random initialization, but GraphBuilder exposes `fill()` for constant-value tensors.

### 8.11 MoE Operations

```python
# Router
probs = g.moe_softmax(logits)
probs = g.moe_sigmoid(logits)
weights, indices = g.moe_topk(probs, top_k=8, normalize=True)

# Permutation
permuted, scatter_indices = g.moe_permute(x, indices, top_k=8)

# Grouped GEMM
gate_up = g.moe_grouped_gemm_gate_up(permuted, "experts_gate_up", scatter_indices)
down = g.moe_grouped_gemm_down(activated, "experts_down", scatter_indices)

# Unpermute and combine
output = g.moe_unpermute(down, weights, indices, top_k=8)
```

### 8.12 Module Calls

```python
# Call a submodule or custom operation
xN, residualN = g.call(
    "StackedBlocks",
    x0, residual0, position_ids,
    num_outputs=2,
    blocks="blocks",
    n_layers=self.n_layers,
)

# Custom operation
y = g.custom("my_kernel", x, w, num_outputs=1, attr1=value1)
```

### 8.13 Output Naming

Most operations accept an `out_name` parameter to explicitly name outputs:

```python
y = g.matmul(x, "weight", transpose="NT", out_name="qkv_flat")
res_out, y, rstd = g.fused_residual_rmsnorm(
    residual, x, "weight",
    res_out_name="residual",
    y_name="ln1",
    rstd_name="ln1_rstd",
)
```

### 8.14 Memory Directives and Saved Tensor Access

The `GraphBuilder` supports explicit save/recompute lists (in addition to the `@save` / `@recompute` decorators):

```python
with graph() as g:
    # Mark tensors to save / recompute (names or GraphRef)
    g.save("x", "rstd")
    g.mark_recompute("ln1")

    # Access a tensor saved from forward (for backward graphs; metadata only for now)
    x_saved = g.saved("x")  # yields a GraphRef named "saved.x"
```

These lists are serialized into the JSON IR as `forward.save` / `forward.recompute`.

---

## 9. HuggingFace Integration

### 9.1 Config Mapping

The `@hf_config` decorator maps HuggingFace config fields to DSL constructor parameters:

```python
@model
@hf_config(
    architecture="Qwen3ForCausalLM",    # HF architecture name
    model_type="qwen3",                  # HF model_type
    # Parameter mappings: DSL_param = "hf_config_field"
    d_model="hidden_size",
    n_layers="num_hidden_layers",
    num_query_heads="num_attention_heads",
    num_kv_heads="num_key_value_heads",
    d_ff="intermediate_size",
    vocab_size="vocab_size",
    max_seq="max_position_embeddings",
    head_size="head_dim",
    eps="rms_norm_eps",
    use_qkv_bias="attention_bias",
)
class Qwen3Model:
    ...
```

### 9.2 Weight Mapping

Direct mapping to HF checkpoint paths:

```python
@hf_mapping(
    embedding="model.embed_tokens.weight",
    final_norm="model.norm.weight",
    lm_head="lm_head.weight",
)
```

### 9.3 Indexed Mapping for Blocks

For block arrays with layer indices:

```python
@hf_mapping.indexed("blocks",
    ln1_weight="model.layers.{layer}.input_layernorm.weight",
    ln2_weight="model.layers.{layer}.post_attention_layernorm.weight",
    out_weight="model.layers.{layer}.self_attn.o_proj.weight",
)
```

### 9.4 Weight Transformations

**Fuse multiple tensors:**

```python
from surogate.dsl import fuse

# Fuse separate Q, K, V projections into combined QKV
qkv_weight = fuse(
    "model.layers.{layer}.self_attn.q_proj.weight",
    "model.layers.{layer}.self_attn.k_proj.weight",
    "model.layers.{layer}.self_attn.v_proj.weight",
    dim=0,
)
```

**Split tensor:**

```python
from surogate.dsl import split

# Extract part of a fused tensor
gate_weight = split(
    "model.layers.{layer}.mlp.gate_up_proj.weight",
    ranges=[(0, 2048)],
    dim=0,
)
```

**Transform tensor:**

```python
from surogate.dsl import transform

# Apply transformation (transpose, permute_qkv, etc.)
lm_head = transform("model.embed_tokens.weight", fn="transpose")
```

**Tie weights:**

```python
from surogate.dsl import hf_tied_to

# Tie lm_head to embedding
lm_head = hf_tied_to("embedding")
```

`hf_tied_to(...)` serializes to a structured mapping payload: `{"type": "tied_to", "target": "embedding"}`.

**Stack experts (MoE):**

```python
from surogate.dsl.hf import stack_experts

# Stack per-expert weights into batched tensor
experts_down = stack_experts(
    "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
    num_experts=64,
)

# With fused gate+up
experts_gate_up = stack_experts(
    "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
    fuse_gate_up=True,
)
```

---

## 10. Activation Slots

Activation slots pre-declare tensor buffers for forward/backward passes, eliminating hardcoded tensor-to-slot mappings in the C++ runtime.

### 10.1 Activation Class

```python
from surogate.dsl import Activation, Tensor

Activation(
    tensor_type,                      # Tensor[...] annotation
    dtype="fp32",                     # Override dtype
    aliases=["alt_name"],             # Alternative names
    save=True,                        # Save for backward
    recompute=True,                   # Recompute in backward
    recompute_from=["a", "b"],        # Tensors to recompute from
    recompute_op="rmsnorm",           # Operation to recompute with
    recompute_attrs={...},            # Operation attributes
    recompute_policy="fft_only",      # When to recompute
    recompute_group="attn_fwd",       # Group multiple slots into one recompute op
    recompute_outputs=["att", "lse"], # Named outputs produced by recompute op
    lora_targets=["q", "k", "v"],     # Which LoRA adapters touch this slot (metadata)
    shares_with="other_slot",         # Share memory with another slot
    share_policy="when_recomputed",   # Cross-layer sharing policy
    when="use_qk_norm",               # Condition for optional slots
    scope="block",                    # "block", "global", "gradient", "global_gradient"
    description="...",                # Documentation
)
```

**Note (dtype defaults)**: `Activation(...)` and `Gradient(...)` default to the dtype in their `Tensor[...]` annotation (which defaults to `bf16`). They do not currently default to a runtime-selected “activation dtype” when `dtype` is omitted.

### 10.2 Gradient Class

```python
from surogate.dsl import Gradient, Tensor

Gradient(
    tensor_type,                      # Tensor[...] annotation
    gradient_of="ln1",                # Forward activation this is gradient of
    dtype="fp32",                     # Override dtype
    shares_with="d_other",            # Share memory
    alias_of="d_something",           # Alias another slot
    when="condition",                 # Condition for optional
    scope="gradient",                 # "gradient" or "global_gradient"
    description="...",                # Documentation
)
```

### 10.3 Activation Scopes

| Scope             | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `block`           | Per-layer activation (in SimplifiedLayerActivations) |
| `global`          | Global activation (in NonBlockActivations)           |
| `gradient`        | Per-layer gradient (in SimplifiedLayerGradients)     |
| `global_gradient` | Global gradient (in NonBlockGradientBuffers)         |

### 10.4 Memory Hints

| Hint         | Description                            |
| ------------ | -------------------------------------- |
| `persistent` | Keep in memory across forward/backward |
| `save`       | Save for backward pass                 |
| `recompute`  | Can be recomputed in backward          |
| `temporary`  | Stack-allocated, freed after use       |
| `shared`     | Shares memory with another slot        |

**Note**: in v0.1.0, `Activation(...)`/`Gradient(...)` declarations do not provide an explicit way to request `temporary`; the compiler currently derives `memory_hint` from `save`, `recompute`, and `shares_with`. `temporary` is reserved for future use.

### 10.5 Share Policies

| Policy            | Description                                    |
| ----------------- | ---------------------------------------------- |
| `per_layer`       | Always allocate per-layer (no sharing)         |
| `when_recomputed` | Share when slot will be recomputed in backward |
| `always_share`    | Always share across layers                     |
| `fft_share`       | Share only in FFT mode (not LoRA)              |
| `lora_share`      | Share only in LoRA mode (not FFT)              |

### 10.6 Example

```python
@block
class TransformerBlock:
    # Forward activations
    ln1 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln1_flat"],
        recompute=True,
        recompute_from=["res_ffn", "ln1_rstd", "@param:ln1_weight"],
        recompute_op="rmsnorm_apply_saved",
        recompute_policy="fft_only",
    )
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True)
    qkv = Activation(Tensor["B", "T", "QKV"], aliases=["qkv_flat", "qkv_biased"])
    lse = Activation(Tensor["B", "Hq", "T"], dtype="fp32", save=True)

    # Conditional slots
    q_rstd = Activation(Tensor["B", "T", "Hq"], dtype="fp32", save=True,
                        when="use_qk_norm")

    # Gradient slots
    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
```

---

## 11. Compilation Pipeline

### 11.1 Pipeline Overview

```
Python Classes (decorated)
        ↓ (decorators.py extracts specs)
Specification Objects (ModuleSpec, BlockSpec, ModelSpec)
        ↓ (py_compiler.py executes @forward under a patched graph() to capture GraphBuilder nodes)
Intermediate Representation (ModuleIR + GraphIR)
        ↓ (py_compiler.py serialization)
JSON IR
        ↓
C++ Runtime Execution
```

**Current limitations**:
- Only forward graphs are compiled and serialized. `@backward` graphs and `@primitive.backward` metadata are collected/registered but not compiled into the JSON IR yet.
- Shape/dtype annotations are serialized as metadata; the compiler does not perform full static shape checking.

### 11.2 Specification Types

| Spec Type              | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| `ModuleSpec`           | Reusable module specification              |
| `BlockSpec`            | Transformer block with activation layout   |
| `ModelSpec`            | Top-level model with HF integration        |
| `PrimitiveSpec`        | CUDA kernel wrapper specification          |
| `ParamSpec`            | Parameter (weight/submodule) specification |
| `ForwardSpec`          | Forward pass specification                 |
| `BackwardSpec`         | Backward pass specification                |
| `ActivationSlotSpec`   | Activation tensor slot specification       |
| `ActivationLayoutSpec` | Complete activation layout                 |
| `HFConfigSpec`         | HuggingFace config mapping                 |
| `HFMappingSpec`        | HuggingFace weight mapping                 |

### 11.3 Compiler API

```python
from surogate.dsl import (
    compile_model,
    compile_model_for_hf,
    get_model_spec,
    get_block_spec,
    get_module_spec,
    list_registered_models,
    list_registered_blocks,
    list_registered_modules,
)

# Compile model to JSON IR
json_ir = compile_model("Qwen3Model", config_json)

# Compile from HF architecture name
json_ir = compile_model_for_hf("Qwen3ForCausalLM", config_json)

# Get specifications
model_spec = get_model_spec("Qwen3Model")
block_spec = get_block_spec("DenseTransformerBlock")
module_spec = get_module_spec("Linear")

# List registered components
models = list_registered_models()    # ["Qwen3Model", "LlamaModel", ...]
blocks = list_registered_blocks()    # ["DenseTransformerBlock", ...]
modules = list_registered_modules()  # ["Linear", "RMSNorm", ...]
```

### 11.4 Registry

Decorated classes are automatically registered:

```python
from surogate.dsl.decorators import (
    _module_registry,
    _block_registry,
    _model_registry,
    _primitive_registry,
)
```

Or use the unified registry:

```python
from surogate.dsl import registry

spec = registry.get_module("Linear")
spec = registry.get_block("DenseTransformerBlock")
spec = registry.get_model("Qwen3Model")
spec = registry.get_any("SomeComponent")
```

---

## 12. Diagnostics

The compiler emits structured diagnostics using error codes (`E…`) and warning codes (`W…`).

### 12.1 How Diagnostics Are Reported

- `compile_model(...)` / `compile_model_for_hf(...)` return JSON with:
  - `success: true|false`
  - `errors: [{code, message, hint?, location?}]` (when `success=false`)
  - `warnings: [{code, message, location?}]` (when present)
- Set `raise_on_error=True` to raise a `DSLError` instead of returning a JSON error payload.

### 12.2 Currently Emitted Codes (Python Compiler)

| Code | Meaning                    | Typical trigger                                                                            |
| ---- | -------------------------- | ------------------------------------------------------------------------------------------ |
| E001 | Syntax error               | Could not capture a forward graph; internal compiler failure wrapped as DSL syntax error   |
| E002 | Undefined identifier       | Unknown model/block name; no DSL model found for a given HF architecture                   |
| E003 | Type mismatch              | `StackedBlocks` param type mismatch or input arity mismatch                                |
| E008 | Invalid annotation         | Class passed to `compile_model(...)` is not a DSL `@model`                                 |
| E010 | Invalid weight mapping     | Invalid HF mapping spec (unknown mapping type, bad `stack_experts` pattern, etc.)          |
| E012 | Missing required parameter | Model missing `@forward`; `StackedBlocks` missing `n_layers`; missing compiled block graph |
| E013 | Invalid fusion pattern     | Invalid `fuse(...)` / `split(...)` spec                                                    |

| Code | Meaning                           | Typical trigger                                                    |
| ---- | --------------------------------- | ------------------------------------------------------------------ |
| W001 | User definition shadows primitive | Model/block/module name collides with a registered primitive name  |
| W004 | Unused saved tensor               | `@save(...)` lists a tensor name not present in the compiled graph |

### 12.3 Reserved Codes

Additional codes (`E004–E007`, `E009`, `E011`, `E014–E027`, `W002`, `W003`, `W005`) are defined in `surogate/dsl/errors.py` but are not yet emitted by the Python compiler in v0.1.0.

---

## 13. Examples

### 13.1 Simple Linear Module

```python
from surogate.dsl import module, forward, save, Param, Tensor, graph, Dim, B, T

@module
class Linear:
    """Linear projection: y = x @ W^T (+ bias)."""

    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = False):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.C = Dim("in_dim")
        self.O = Dim("out_dim")

    weight = Param(Tensor["O", "C"])
    bias = Param(Tensor["O"], when="use_bias")

    @save("x")
    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "O"]:
        with graph() as g:
            x_flat = g.view(x, shape=[B * T, self.C])
            if self.use_bias:
                y_flat = g.matmul_bias(x_flat, "weight", "bias", transpose="NT")
            else:
                y_flat = g.matmul(x_flat, "weight", transpose="NT")
            y = g.view(y_flat, shape=[B, T, self.O])
            return y
```

### 13.2 RMSNorm Module

```python
from surogate.dsl import module, forward, Param, Tensor, graph

@module
class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

    weight = Param(Tensor["d_model"])

    @forward
    def forward(self, x: Tensor["B", "T", "d_model"]) -> Tensor["B", "T", "d_model"]:
        with graph() as g:
            y, rstd = g.rmsnorm(x, "weight", eps=self.eps)
            return y
```

### 13.3 SwiGLU MLP Module

```python
from surogate.dsl import module, forward, save, Param, Tensor, graph, Dim, B, T

@module
class SwiGLUMLP:
    """SwiGLU MLP: down(swiglu(up(x)))."""

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        self.C = Dim("d_model")
        self.M = Dim("d_ff")

    # up_weight is [2*d_ff, d_model] for fused gate+up
    up_weight = Param(Tensor["2 * M", "C"])
    down_weight = Param(Tensor["C", "M"])

    @save("x", "up")
    @forward
    def forward(self, x: Tensor["B", "T", "C"]) -> Tensor["B", "T", "C"]:
        with graph() as g:
            x_flat = g.view(x, shape=[B * T, self.C])
            up_flat = g.matmul(x_flat, "up_weight", transpose="NT")
            up = g.view(up_flat, shape=[B, T, 2 * self.M])
            act = g.swiglu(up)
            act_flat = g.view(act, shape=[B * T, self.M])
            out_flat = g.matmul(act_flat, "down_weight", transpose="NT")
            out = g.view(out_flat, shape=[B, T, self.C])
            return out
```

### 13.4 Complete Model Example

See [Section 6](#6-model-definitions) for a full Qwen3Model example.

---

## Appendix A: File Structure

```
surogate/dsl/
├── __init__.py          # Package exports
├── decorators.py        # @module, @block, @model, @primitive, @param, etc.
├── tensor_type.py       # Tensor[...], Array[...] annotations
├── dim.py               # Dim, DimExpr, B, T
├── types.py             # Dtype, Shape, TensorTypeSpec
├── graph_builder.py     # graph(), GraphBuilder, GraphRef
├── specs.py             # ModuleSpec, BlockSpec, ModelSpec, etc.
├── hf.py                # fuse(), split(), transform(), tied_to(), stack_experts()
├── ir.py                # GraphIR, ModuleIR, ScheduleIR
├── ir_builder.py        # Convenience wrappers (e.g., HF → IR)
├── py_lowering.py       # Spec → IR lowering
├── py_compiler.py       # IR → JSON compilation
├── py_registry.py       # Global registry
├── registry.py          # HF architecture registry
├── errors.py            # DSLError, error codes
├── primitives/          # @primitive definitions
├── modules/             # @module definitions (Linear, RMSNorm, etc.)
├── blocks/              # @block definitions (DenseTransformerBlock, etc.)
└── models/              # @model definitions (Qwen3Model, LlamaModel, etc.)
```

