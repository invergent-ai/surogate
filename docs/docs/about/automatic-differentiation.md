---
title: Automatic Differentiation
---

Surogate's DSL includes a powerful **Ahead-Of-Time (AOT) automatic differentiation system** that derives backward computation graphs from forward graphs. This enables automatic gradient computation for training without manually implementing backward passes.

## Overview

The autodiff system operates on the DSL's intermediate representation (IR) graphs. Given a forward computation graph, it produces a corresponding backward graph that computes gradients with respect to all differentiable inputs and parameters.

### Graph-Based vs. Tape-Based Autodiff

Unlike tape-based autograd engines (PyTorch, TinyGrad, JAX eager mode), which record operations dynamically during execution and construct the backward pass on-the-fly, Surogate's autodiff works **ahead-of-time** on static computation graphs. This fundamental difference enables significant performance advantages:

**Compile-Time Optimization**: The backward graph is derived once during model compilation, not rebuilt on every training iteration. This eliminates runtime overhead for gradient graph construction and enables whole-graph optimizations (fusion, memory planning, dead code elimination).

**Explicit Memory Management**: By analyzing both forward and backward graphs together, the system precisely determines which activations must be saved (`saved.*` references), which can be recomputed, and when buffers can be freed. Tape-based systems must conservatively retain all intermediate tensors until the backward pass completes.

**Kernel Fusion Opportunities**: Static graphs expose multi-operation patterns that can be fused into specialized CUDA kernels (e.g., `fused_lm_head_loss_backward` combines matmul + cross-entropy gradient). Tape-based systems struggle to identify fusion opportunities across dynamically recorded operations.

**Zero Python Overhead**: Once compiled, training runs entirely in C++/CUDA without Python interpreter involvement. Tape-based systems incur per-operation Python dispatch overhead during both forward and backward passes.

**Activation Checkpointing Integration**: The `save` list produced by `compute_required_saves()` integrates directly with memory planning, enabling efficient activation checkpointing strategies determined at compile time rather than runtime.

The trade-off is reduced flexibility: dynamic control flow and data-dependent shapes require explicit graph variants, whereas tape-based systems handle these naturally. For transformer training workloads with static shapes and fixed layer counts, the performance benefits are substantial.

**Key Features:**

- **Graph-level differentiation**: Operates on entire computation graphs, not individual operations
- **Backward rule registry**: Extensible registry of differentiation rules for each operation type
- **Automatic tensor saving**: Detects which intermediate tensors must be saved for the backward pass
- **Gradient accumulation**: Handles multi-use tensors with automatic gradient accumulation
- **Non-differentiable detection**: Automatically skips integer tensors and index arrays
- **Stop gradients**: Supports freezing parameters (e.g., for LoRA training)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Forward Graph                                    │
│  inputs: {x, position_ids}                                              │
│  params: {embed_weight, ln_weight, qkv_weight, ...}                     │
│  operations: [embedding, rmsnorm, matmul, rope, attention, ...]         │
│  outputs: {loss}                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   derive_backward_graph()     │
                    │                               │
                    │  1. Build tensor→producer map │
                    │  2. Propagate needs_grad      │
                    │  3. Reverse topological order │
                    │  4. Apply backward rules      │
                    │  5. Accumulate gradients      │
                    │  6. Compute required saves    │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Backward Graph                                   │
│  inputs: {d_loss}                                                       │
│  outputs: {d_embed_weight, d_ln_weight, d_qkv_weight, ...}              │
│  operations: [fused_lm_head_loss_backward, rmsnorm_backward, ...]       │
│  save: [x, qkv, attention_out, lse, ...]                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### BackwardRuleRegistry

A singleton registry that maps operation types to their differentiation rules:

```cpp
// Get the global registry
auto& registry = BackwardRuleRegistry::instance();

// Register a custom backward rule
registry.register_rule("my_op", my_op_backward);

// Check if a rule exists
bool has_rule = registry.has_rule("matmul");

// Get all registered operations
std::vector<std::string> ops = registry.registered_ops();
```

### BackwardRuleContext

Context passed to each backward rule containing all information needed to generate backward operations:

```cpp
struct BackwardRuleContext {
    const Operation& fwd_op;           // The forward operation
    const std::vector<std::string>& d_outputs;  // Output gradients
    const std::string& d_output;       // Primary output gradient name
    const std::vector<std::string>& d_inputs;   // Input gradient names to produce
    const ShapeEnv& shape_env;         // Shape environment
    int& op_counter;                   // For unique op IDs
    const Graph* forward_graph;        // Full forward graph

    // Helper methods
    bool needs_grad(size_t idx) const; // Check if input needs gradient
    bool is_param(const std::string& name) const;  // Is this a parameter?
    bool is_input(const std::string& name) const;  // Is this a graph input?
};
```

### DeriveBackwardOptions

Configuration options for backward graph derivation:

```cpp
struct DeriveBackwardOptions {
    std::string loss_name = "loss";       // Tensor to differentiate from
    bool auto_save = true;                // Auto-detect saves
    std::vector<std::string> extra_saves; // Additional tensors to save
    bool accumulate_grads = true;         // Accumulate for multi-use tensors
    std::string grad_prefix = "d_";       // Gradient naming prefix
    std::vector<std::string> stop_gradients;  // Non-differentiable tensors
};
```

## Derivation Algorithm

The `derive_backward_graph()` function implements reverse-mode automatic differentiation through a five-step process:

### Step 1: Build Producer Map

The algorithm first constructs a mapping from each tensor name to the index of the operation that produces it. This allows efficient backward traversal through the computation graph. Graph inputs and parameters are marked with a special sentinel value indicating they are produced externally.

### Step 2: Propagate `needs_grad`

Starting from the loss tensor (or all output tensors if no loss is specified), the algorithm performs a backward traversal to determine which tensors require gradients. Using a work queue, it visits each tensor that needs a gradient and marks all inputs to the operation that produced it as also needing gradients. This continues recursively until all reachable tensors are marked.

Tensors are skipped during propagation if they are:
- Non-differentiable (integer dtypes, index arrays)
- Listed in the stop-gradients set (for freezing parameters)

### Step 3: Process Operations in Reverse

The algorithm processes the forward operations in reverse topological order (from loss to inputs). For each operation whose outputs require gradients:

1. Retrieve the registered backward rule for the operation type
2. Construct a `BackwardRuleContext` containing forward operation info and gradient tensor names
3. Invoke the backward rule to generate backward operations
4. Append the generated operations to the backward graph

Operations whose outputs don't need gradients are skipped entirely (dead code elimination).

### Step 4: Gradient Accumulation

When a tensor is used as input to multiple operations (fan-out), it receives gradient contributions from each consumer. The algorithm tracks which tensors already have gradients and automatically inserts addition operations to accumulate the contributions.

For example, if tensor `x` is used by both `matmul` and `add`, the algorithm generates `d_x_from_matmul` and `d_x_from_add`, then creates `d_x_accum = d_x_from_matmul + d_x_from_add` to combine them. The gradient map is updated to point to the accumulated gradient.

### Step 5: Compute Required Saves

After the backward graph is constructed, the algorithm scans all backward operations to identify tensors referenced with the `saved.` prefix. These are forward activations that must be preserved for use during the backward pass. The list of required saves is extracted by removing the prefix and collecting unique tensor names


## Backward Rules

### Rule Structure

Each backward rule is a function that generates backward operations:

```cpp
using BackwardRule = std::function<std::vector<Operation>(const BackwardRuleContext&)>;
```

### Example: Matrix Multiplication

```cpp
// Forward: C = A @ B
// Backward: dA = dC @ B.T, dB = A.T @ dC
std::vector<Operation> matmul_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    const std::string& A = fwd.inputs[0];
    const std::string& B = fwd.inputs[1];
    const std::string& dC = ctx.d_output;

    // Determine tensor references for backward
    // Parameters available at backward time; activations must be saved
    std::string A_for_dB = ctx.is_param(A) ? A : saved_ref(A);
    std::string B_for_dA = ctx.is_param(B) ? B : saved_ref(B);

    ops.push_back(make_operation(
        "matmul_backward",
        {dC, A_for_dB, B_for_dA},
        {ctx.d_inputs[0], ctx.d_inputs[1]},
        {{"transpose", fwd.attrs["transpose"]}}));

    return ops;
}
```

### Example: RMSNorm

```cpp
// Forward: y, rstd = rmsnorm(x, weight, eps)
// Backward: dx, dweight = rmsnorm_backward(dy, x, weight, rstd)
std::vector<Operation> rmsnorm_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::string x = fwd.inputs[0];
    std::string weight = fwd.inputs[1];
    std::string rstd = fwd.outputs[1];  // Saved from forward

    ops.push_back(make_operation(
        "rmsnorm_backward",
        {ctx.d_output, saved_ref(x), weight, saved_ref(rstd)},
        {ctx.d_inputs[0], ctx.d_inputs[1]},
        copy_attrs(fwd.attrs, {"eps"})));

    return ops;
}
```

## Registered Operations

The following operations have built-in backward rules:

### Core Operations

| Operation     | Backward Rule                    | Saved Tensors                        |
| ------------- | -------------------------------- | ------------------------------------ |
| `matmul`      | dA = dC @ B.T, dB = A.T @ dC     | A (if activation), B (if activation) |
| `matmul_bias` | Same as matmul + dBias = sum(dC) | A, B                                 |
| `add`         | dA = dC, dB = dC                 | None                                 |
| `multiply`    | dA = dC * B, dB = dC * A         | A, B                                 |

### Normalization

| Operation                | Backward Rule                     | Saved Tensors      |
| ------------------------ | --------------------------------- | ------------------ |
| `rmsnorm`                | Fused RMSNorm backward            | x, rstd            |
| `fused_residual_rmsnorm` | Fused residual + RMSNorm backward | residual_out, rstd |

### Activations

| Operation  | Backward Rule                  | Saved Tensors |
| ---------- | ------------------------------ | ------------- |
| `silu`     | dx = dy * silu'(x)             | x             |
| `gelu`     | dx = dy * gelu'(x)             | x             |
| `swiglu`   | d_gate, d_up = swiglu_backward | gate_up       |
| `bias_add` | dx = dy, d_bias = sum(dy)      | None          |

### Attention

| Operation          | Backward Rule                    | Saved Tensors           |
| ------------------ | -------------------------------- | ----------------------- |
| `rope`             | Inverse RoPE rotation            | freqs, position_ids     |
| `qkv_qk_norm_rope` | Combined QK-Norm + RoPE backward | qkv_out, q_rstd, k_rstd |
| `flash_attention`  | Flash attention backward         | out, lse, qkv           |
| `attention`        | Attention backward               | q, k, v, out, lse       |
| `softmax`          | dx = y * (dy - sum(dy * y))      | y                       |

### Tensor Operations

| Operation           | Backward Rule          | Saved Tensors          |
| ------------------- | ---------------------- | ---------------------- |
| `view` / `reshape`  | Reshape gradient back  | None (uses shape_like) |
| `zeros`             | No gradient (constant) | None                   |
| `identity` / `copy` | dx = dy                | None                   |

### Loss Functions

| Operation            | Backward Rule                       | Saved Tensors   |
| -------------------- | ----------------------------------- | --------------- |
| `cross_entropy`      | d_logits = softmax - one_hot        | logits          |
| `fused_lm_head_loss` | Combined LM head + CE loss backward | xF_flat, weight |

### MoE Operations

| Operation                  | Backward Rule                            | Saved Tensors                                |
| -------------------------- | ---------------------------------------- | -------------------------------------------- |
| `moe_sigmoid`              | d_logits = d_probs * probs * (1 - probs) | probs                                        |
| `moe_softmax`              | Softmax backward                         | probs                                        |
| `moe_topk`                 | Scatter d_weights to selected positions  | probs, indices                               |
| `moe_permute`              | Inverse permutation                      | scatter_indices                              |
| `moe_grouped_gemm_gate_up` | Grouped GEMM backward                    | inp, weights, scatter_indices                |
| `moe_grouped_gemm_down`    | Grouped GEMM backward                    | inp, weights, scatter_indices                |
| `moe_unpermute`            | d_expert_out, d_routing_weights          | expert_out, routing_weights, scatter_indices |

## Non-Differentiable Tensors

The system automatically detects non-differentiable tensors:

### By Data Type

- `INT32`, `INT8`, `BYTE` tensors are non-differentiable

### By Name Pattern

- `rope_freqs` - RoPE frequency tensors
- `scatter_indices`, `routing_indices`, `gather_indices` - MoE index tensors
- `expert_offsets` - Expert offset arrays

### By Stop Gradient List

```cpp
DeriveBackwardOptions options;
options.stop_gradients = {"base_weight", "frozen_param"};  // Freeze these
auto backward = derive_backward_graph(forward, options);
```

## Saved Tensor Convention

Tensors that must be preserved from forward for backward use the `saved.` prefix:

```cpp
// In backward rule
std::string x_ref = saved_ref("x");  // Returns "saved.x"

// In backward operation
ops.push_back(make_operation(
    "silu_backward",
    {d_output, "saved.x"},  // Reference saved tensor
    {d_input}));
```

The `compute_required_saves()` function scans the backward graph to determine which tensors need saving.

## Usage Example

```cpp
#include "dsl/autodiff.h"

// Load forward graph
dsl::Graph forward = load_forward_graph(...);

// Configure backward derivation
dsl::DeriveBackwardOptions options;
options.loss_name = "loss";
options.auto_save = true;
options.stop_gradients = {"embed_tokens.weight"};  // Freeze embeddings

// Derive backward graph
dsl::Graph backward = dsl::derive_backward_graph(forward, options);

// backward.operations contains all backward ops
// backward.save contains tensors to preserve from forward
// backward.inputs contains gradient inputs (d_loss)
// backward.outputs contains parameter gradients
```

## Adding Custom Backward Rules

To add a backward rule for a new operation:

```cpp
#include "dsl/autodiff.h"

namespace dsl {

std::vector<Operation> my_op_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        // Generate backward operations
        ops.push_back(make_operation(
            "my_op_backward_" + std::to_string(ctx.op_counter++),
            "my_op_backward",
            "my_op_backward",
            {ctx.d_output, saved_ref(fwd.inputs[0])},
            {ctx.d_inputs[0]}));
    }

    return ops;
}

// Register at initialization
void register_custom_rules() {
    auto& reg = BackwardRuleRegistry::instance();
    reg.register_rule("my_op", my_op_backward);
}

} // namespace dsl
```

## Integration with Training

The autodiff system integrates with Surogate's training pipeline:

1. **DSL Compilation**: Forward graph defined in Python DSL → IR JSON
2. **Backward Derivation**: `derive_backward_graph()` generates backward IR
3. **Kernel Dispatch**: Executor maps operations to CUDA kernels
4. **Memory Planning**: `save` list determines activation checkpointing

This enables end-to-end training with automatic gradient computation while maintaining the performance benefits of hand-optimized CUDA kernels.
