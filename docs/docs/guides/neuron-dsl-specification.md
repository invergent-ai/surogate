# Module DSL Language Specification

**Version**: 0.1.0
**Status**: Draft
**Authors**: Surogate Team
**Last Updated**: 2026-01-19

## Table of Contents

1. [Introduction](#1-introduction)
2. [Lexical Structure](#2-lexical-structure)
3. [Type System](#3-type-system)
4. [Declarations](#4-declarations)
5. [Graph Expressions](#5-graph-expressions)
6. [Primitive Operations](#6-primitive-operations)
7. [Annotations](#7-annotations)
8. [Weight Mapping](#8-weight-mapping)
9. [Compilation Model](#9-compilation-model)
10. [Standard Library](#10-standard-library)
11. [Examples](#11-examples)
12. [Appendix](#appendix)

---

## 1. Introduction

### 1.1 Purpose

The Module DSL is a domain-specific language for defining neural network architectures with explicit forward and backward computation graphs. It targets high-performance inference and training on NVIDIA GPUs without automatic differentiation.

### 1.2 Design Goals

1. **Explicit over implicit**: Both forward and backward passes are explicitly defined
2. **Performance transparency**: Memory allocation, fusion, and precision are controllable
3. **Composability**: Modules can be composed hierarchically
4. **Target independence**: DSL compiles to C++/CUDA, with potential future backends
5. **HuggingFace compatibility**: First-class support for weight mapping and config translation

### 1.3 Non-Goals

- Automatic differentiation (autograd)
- Dynamic control flow (while loops with data-dependent conditions)
- Eager execution mode

### 1.4 Terminology

| Term          | Definition                                                                     |
| ------------- | ------------------------------------------------------------------------------ |
| **Module**    | A reusable computation unit with parameters, forward graph, and backward graph |
| **Primitive** | A built-in operation with known forward/backward kernel implementations        |
| **Block**     | A module representing one transformer layer (attention + FFN)                  |
| **Model**     | A top-level module representing a complete architecture                        |
| **Graph**     | A directed acyclic graph of operations                                         |
| **Edge**      | A named tensor flowing between operations                                      |

---

## 2. Lexical Structure

### 2.1 Character Set

Source files are UTF-8 encoded. Identifiers use ASCII alphanumerics and underscores.

### 2.2 Comments

```
# Single-line comment

"""
Multi-line documentation comment.
Used for module/param documentation.
"""
```

### 2.3 Keywords

Reserved keywords:

```
module    block     model     primitive
forward   backward  params    let
in        out       if        else
graph     save      recompute pattern
impl      extends   abstract  import
true      false     None      constraint   
```

### 2.4 Identifiers

```ebnf
identifier     = letter (letter | digit | "_")* ;
letter         = "a".."z" | "A".."Z" | "_" ;
digit          = "0".."9" ;
type_var       = uppercase_letter (letter | digit)* ;  (* e.g., B, T, C *)
```

### 2.5 Literals

```ebnf
integer        = digit+ ;
float          = digit+ "." digit+ (("e" | "E") ("+" | "-")? digit+)? ;
string         = '"' (char | escape)* '"' ;
boolean        = "true" | "false" ;
```

### 2.6 Operators

```
->    # Data flow
=>    # Pattern matching / mapping
:     # Type annotation
=     # Assignment / default value
+     # Addition
-     # Subtraction
*     # Multiplication
/     # Division
//    # Integer division
%     # Modulo
?     # Optional (suffix)
@     # Annotation prefix
```

### 2.7 Delimiters

```
(  )    # Grouping, tuples, function calls
[  ]    # Array types, indexing, shape specs
{  }    # Blocks, dict literals
,       # Separator
.       # Member access
|       # Alternatives
```

---

## 3. Type System

### 3.1 Scalar Types

```
int       # 32-bit signed integer
float     # 32-bit floating point (used for shapes, constants)
bool      # Boolean
string    # UTF-8 string (metadata only)
```

### 3.2 Tensor Types

Tensors are the primary data type, using **bracket notation** for shape specification:

```
# Canonical form (used throughout the DSL)
[B, T, C]                        # Tensor with symbolic shape
[B, T, C, bf16]                  # Tensor with explicit dtype (dtype as last element after comma)

# Shape specification
[B, T, C]                        # Symbolic dimensions
[batch, seq, 4096]               # Mixed symbolic/concrete
[*, T, C]                        # Variadic batch dimensions (leading dimensions)
[B, *, C]                        # NOT allowed - variadic must be leading
```

**Note:** The `Tensor[...]` form is **deprecated**. Use bracket notation `[...]` directly. For clarity in documentation or when disambiguation is needed, use `Tensor<[B, T, C]>` as an explicit type annotation.

### 3.3 Dtype Specifiers

```
bf16      # bfloat16 (default for activations)
fp32      # float32
fp16      # float16
fp8_e4m3  # FP8 E4M3 (forward activations)
fp8_e5m2  # FP8 E5M2 (backward gradients)
fp4_e2m1  # FP4 E2M1 (Blackwell+)
int8      # 8-bit integer
int32     # 32-bit integer
```

### 3.4 Composite Types

```
# Tuple (heterogeneous, fixed-size)
([B, T, C], [B, T, C])           # Tuple of two tensors

# Optional
[B, T, C]?                       # May be None

# Array (homogeneous, for layer stacking) - two equivalent syntaxes:
[n_layers] × ModuleType          # Unicode multiplication sign (×)
[n_layers] x ModuleType          # ASCII 'x' alternative (preferred for tooling)
Array<n_layers, ModuleType>      # Explicit generic syntax (alternative)
```

**Note on Array Syntax:** The `×` (Unicode U+00D7) and ASCII `x` are interchangeable. The ASCII form is recommended for better editor/tooling compatibility. The `Array<N, T>` form is provided for contexts where the symbolic multiplication might be ambiguous.

### 3.5 Symbolic Dimensions

Symbolic dimensions are type variables representing runtime-determined sizes:

```
let:
  B: batch_dim                   # Binds B to batch dimension
  T: seq_dim                     # Binds T to sequence dimension
  C = d_model                    # C equals parameter d_model
  D = C // num_heads             # Computed dimension
```

Symbolic dimensions support:
- Arithmetic: `+`, `-`, `*`, `//`, `%`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=` (compile-time assertions)

#### 3.5.1 Compile-time Constraints

To prevent runtime hardware errors (e.g., division by zero or non-integer head dims), constraints can be defined in the `let:` block using a dedicated `constraint:` subsection:

**Syntax:**
```ebnf
constraint_section = "constraint:" INDENT { constraint_stmt } DEDENT ;
constraint_stmt    = const_expr "," string_literal ;   (* condition, error message *)
```

**Example:**
```
let:
  C = d_model
  H = num_heads
  D = C // H

  constraint:
    C % H == 0, "d_model must be divisible by num_heads"
    D <= 128, "Head dimension exceeds hardware limit"
    D > 0, "Head dimension must be positive"
```

**Semantics:**
- Constraints are evaluated at **compile time** (during resolution phase)
- The condition must be a `ConstExpr` (see §3.7 Expression Classes)
- If a constraint fails, compilation aborts with the provided error message
- Constraints are checked in declaration order

### 3.6 Type Inference

Types flow forward through graphs. Explicit annotations required for:
- Module input/output signatures
- Parameter declarations
- Ambiguous operations

### 3.7 Expression Classes

The DSL distinguishes between two classes of expressions to ensure compile-time safety and prevent accidental runtime-dependent logic in static contexts.

#### 3.7.1 ConstExpr (Compile-time Expressions)

`ConstExpr` values are fully determined at compile time. They can appear in:
- Shape specifications
- `if` guards (conditional compilation)
- Constraint conditions
- Annotation arguments
- Array sizes

**Valid `ConstExpr` forms:**
```
# Literals
42, 3.14, true, false, "string", None

# Module/model parameters (known at compile time)
d_model, num_heads, vocab_size

# Let-bound constants
D = C // H              # D is ConstExpr if C and H are ConstExpr

# Arithmetic on ConstExpr
d_model * 4, num_heads // 2, d_ff + 256

# Comparisons on ConstExpr (result is ConstExpr<bool>)
num_heads > 0, d_model % 8 == 0

# Pure intrinsics
sqrt(d_head), ceil_div(d_model, 8), min(a, b), max(a, b)

# Boolean operations
use_bias and use_norm, not is_causal
```

**NOT `ConstExpr`:**
```
# Tensor values (these are RuntimeExpr)
x.shape[0]              # Shape of a runtime tensor
reduce_sum(x)           # Result of tensor operation

# Data-dependent conditions
has_nan(x)              # Depends on tensor contents
```

#### 3.7.2 RuntimeExpr (Runtime Expressions)

`RuntimeExpr` values are determined at runtime and include all tensor operations. They can only appear in:
- Graph statements (data flow operations)
- Tensor computations

**The fundamental rule:** Tensor values are always `RuntimeExpr`. Operations on tensors produce `RuntimeExpr` results.

#### 3.7.3 Enforcement Rules

| Context | Required Expression Class | Example |
|---------|--------------------------|---------|
| Shape specification | `ConstExpr` | `[B, T, d_model]` |
| `if` guard | `ConstExpr<bool>` | `if use_bias:` |
| `constraint:` condition | `ConstExpr<bool>` | `C % H == 0` |
| Annotation argument | `ConstExpr` (usually) | `@shard(tp_size=8)` |
| Graph operation argument | `ConstExpr` or `RuntimeExpr` | `scale(1.0 / sqrt(d_head))` |
| Data flow source/dest | `RuntimeExpr` (tensor name) | `x -> op() -> y` |

**Compiler Error Example:**
```
module BadExample():
  forward:
    graph:
      # ERROR: if guard cannot depend on tensor shape (RuntimeExpr)
      if x.shape[0] > 32:    # E016: if guard must be ConstExpr
        x -> op_a() -> y
      else:
        x -> op_b() -> y
```

### 3.8 None and Optional Types

The `None` value represents the absence of a tensor or parameter.

**Rules:**
- `None` has type `None` which unifies with any optional type `T?`
- Optional types are written as `T?` (e.g., `[B, T, C]?`)
- Optional parameters can be omitted or passed as `None`
- Accessing a `None` value where a tensor is expected is a compile error

**Example:**
```
module LinearWithBias(in_dim, out_dim, use_bias: bool = true):
  params:
    weight: [out_dim, in_dim]
    bias: [out_dim]?              # Optional - may be None

  forward:
    graph:
      in -> matmul(weight) -> y
      if use_bias:                # ConstExpr guard
        (y, bias) -> add() -> out
      else:
        y -> out
```

---

## 4. Declarations

### 4.1 Module Declaration

A module is the fundamental unit of computation. Note that the `save:` and `recompute:` lists define the default memory behavior for all instances of this module, which can be overridden by instance-level annotations.



```
module ModuleName(param1: type = default, param2: type, ...):
  """Optional documentation string."""

  let:
    # Local bindings (computed from params)
    derived_value = expression

  params:
    # Learnable parameters
    weight_name: [shape]
    bias_name: [shape] if condition

  forward:
    in: tensor_type
    out: tensor_type

    graph:
      # Forward computation graph

    save: [x, y]       # Default: save these for backward
    recompute: [z]     # Default: recompute this during backward

  backward:
    d_out: tensor_type         # Gradient input
    d_in: tensor_type          # Gradient output

    graph:
      # Backward computation graph
```

### 4.2 Abstract Module

Abstract modules define interfaces without implementation:

```
abstract module BaseAttention(d_model: int, num_heads: int):
  params:
    qkv_weight: [3 * d_model, d_model]
    out_weight: [d_model, d_model]

  forward:
    in: [B, T, d_model]
    out: [B, T, d_model]
    # No graph - must be provided by concrete implementation
```

### 4.3 Module Inheritance

Modules can extend abstract or concrete modules:

```
module LlamaAttention(d_model, num_heads, max_seq) extends BaseAttention:
  params:
    # Inherits qkv_weight, out_weight
    +rope_freqs: [max_seq, d_model // num_heads // 2, 2]  # Add new param

  forward:
    graph:
      # Full implementation required
```

### 4.4 Block Declaration

Blocks are modules with a known residual pattern:

```
block BlockName(params...):
  """Block with standard residual connections."""

  params:
    norm1: RMSNormParams
    sublayer1: AttentionParams
    norm2: RMSNormParams
    sublayer2: MLPParams

  # High-level pattern (compiler generates forward/backward)
  pattern: sequential_residual
    sublayers:
      - (norm1, sublayer1)
      - (norm2, sublayer2)

  # OR explicit forward/backward (same as module)
  forward:
    ...
  backward:
    ...
```

### 4.5 Model Declaration

Models are top-level modules representing complete architectures:

```
model ModelName(
  vocab_size: int = 32000,
  d_model: int = 4096,
  n_layers: int = 32,
  ...
):
  """Complete model architecture."""

  params:
    embedding: [vocab_size, d_model]
    blocks: [n_layers] × BlockType(...)
    final_norm: [d_model]
    lm_head: tied_to(embedding)          # Weight tying

  forward:
    in: [B, T]                           # Token IDs (int32)
    out: [B, T, vocab_size]              # Logits

    graph:
      in -> Embedding(embedding) -> x0
      x0 -> StackedBlocks(blocks, n_layers) -> xN
      xN -> RMSNorm(final_norm) -> xF
      xF -> Linear(lm_head) -> out

  backward:
    ...

  # HuggingFace compatibility
  hf_config:
    architecture: "LlamaForCausalLM"
    mapping: ...
```

### 4.6 Primitive Declaration

Primitives define low-level operations with kernel bindings:

```
primitive matmul:
  """Matrix multiplication: C = A @ B with optional transpose."""

  params:
    transpose: enum(NN, NT, TN, TT) = NN
    accumulate: bool = false

  forward:
    in: (A: [M, K], B: [K, N])
    out: [M, N]
    impl: kernels.matmul

  backward:
    # Backward is defined in terms of forward
    d_A = matmul(d_C, B, transpose=NT)
    d_B = matmul(A, d_C, transpose=TN)
```

### 4.7 Import Declaration

Supports versioning to ensure stability as the standard library evolves.

```
import primitives                               # Import all primitives
import std.primitives.matmul.v1                 # Import specific version
import std.modules.attention.v2 as Attn         # Import specific version
import models.llama.LlamaAttention              # Import from another file
from models.qwen import QwenMLP as MLP          # Aliased import
```

#### 4.7.1 Import Resolution Rules

The compiler resolves imports following these rules:

**Search Order:**
1. Standard library (`std/`) — bundled with compiler
2. Project-local modules (relative to current file)
3. Explicit paths (absolute module paths)

**Version Resolution:**

| Import Form | Resolution |
|-------------|------------|
| `import std.primitives.matmul` | Latest stable version |
| `import std.primitives.matmul.v1` | Exactly version 1 |
| `import std.primitives.matmul.v2` | Exactly version 2 |

**Conflict Resolution:**

```
# CONFLICT: Two modules export same name
import models.llama.Attention        # Exports 'Attention'
import models.qwen.Attention         # Exports 'Attention' - ERROR E023

# RESOLUTION: Use aliases
import models.llama.Attention as LlamaAttention
import models.qwen.Attention as QwenAttention

# Or use qualified names
models.llama.Attention(...)
models.qwen.Attention(...)
```

**Name Shadowing Rules:**

| Scenario | Behavior |
|----------|----------|
| User module same name as primitive | User definition shadows primitive (warning W001) |
| Local name shadows import | Local definition wins (warning W002) |
| Import alias shadows existing name | Error E024 |

**Module Manifest:**

For reproducible builds, the compiler records resolved versions in a manifest:

```yaml
# module.lock (auto-generated)
version: 1
resolved_modules:
  std.primitives.matmul: v1.2.3
  std.primitives.rmsnorm: v1.1.0
  std.modules.attention: v2.0.1
  models.llama: local@abc123f
```

**Recommendations:**
- Always use explicit versions in production code
- Use `import ... as` when importing similar modules
- Avoid wildcard imports (`import module.*`) in library code

---

## 5. Graph Expressions

### 5.1 Data Flow Syntax

The primary graph syntax uses `->` for data flow:

```
source -> Operation(args) -> destination
```

Components:
- **source**: Input tensor name or tuple of names
- **Operation**: Primitive or module invocation
- **destination**: Output tensor name or tuple of names

### 5.2 Single Input/Output

```
x -> Linear(weight) -> y
```

### 5.3 Multiple Inputs

```
(a, b) -> Add() -> c
(q, k, v) -> FlashAttention() -> out
```

### 5.4 Multiple Outputs

```
x -> Split([d1, d2, d3]) -> (a, b, c)
x -> Fork(2) -> (branch1, branch2)
```

### 5.5 Chained Operations

```
x -> Linear(w1) -> ReLU() -> Linear(w2) -> y
```

Equivalent to:

```
x -> Linear(w1) -> t1
t1 -> ReLU() -> t2
t2 -> Linear(w2) -> y
```

### 5.6 Conditional Graphs

```
if condition:
  x -> OperationA() -> y
else:
  x -> OperationB() -> y
```

Conditions must be compile-time constants (module parameters).

### 5.7 Identity Passthrough

When a branch does nothing:

```
if use_norm:
  x -> RMSNorm(weight) -> y
else:
  x -> y                    # Identity / alias
```

### 5.8 Inline Expressions

For simple computations:

```
graph:
  x -> Linear(weight, bias=bias if use_bias else None) -> y
```

### 5.9 Graph Scoping

Tensors are scoped to their enclosing graph:

```
forward:
  graph:
    in -> Linear(w) -> hidden    # 'hidden' visible here
    hidden -> ReLU() -> out

backward:
  graph:
    # 'hidden' not visible - use saved.hidden
    saved.hidden -> ...
```

### 5.10 Saved Tensor Access

In backward graphs, access forward activations via `saved.`:

```
forward:
  graph:
    in -> Linear(w) -> out
  save: [in]

backward:
  graph:
    (d_out, saved.in) -> LinearBackward(w) -> d_in
```

### 5.11 Graph Semantics

This section formalizes the execution model for graphs to ensure consistent behavior and enable compiler optimizations.

#### 5.11.1 Single Static Assignment (SSA)

Graphs follow **SSA form** by default:

- Each tensor name is assigned **exactly once** within a scope
- Redefinition of a tensor name in the same scope is a compile error
- Different scopes (e.g., `if`/`else` branches) may define the same name

```
# VALID: Each name assigned once
forward:
  graph:
    in -> Linear(w1) -> h1
    h1 -> ReLU() -> h2
    h2 -> Linear(w2) -> out

# INVALID: Redefinition of 'x'
forward:
  graph:
    in -> Linear(w1) -> x
    x -> ReLU() -> x          # ERROR E017: Tensor 'x' already defined in this scope
```

**Exception for conditionals:** Both branches may define the same output name:
```
# VALID: 'y' defined in both branches, unified at join point
if use_norm:
  x -> RMSNorm(weight) -> y
else:
  x -> y                      # Identity (alias)
```

#### 5.11.2 Alias vs Copy Semantics

The identity operation `x -> y` creates an **alias**, not a copy:

| Syntax | Semantics | Memory |
|--------|-----------|--------|
| `x -> y` | Alias: `y` points to same buffer as `x` | No allocation |
| `x -> copy() -> y` | Copy: `y` is independent buffer | New allocation |
| `x -> view([...]) -> y` | Alias: view into same memory | No allocation |
| `x -> contiguous() -> y` | Copy if needed, else alias | Conditional |

**Aliasing Rules:**

1. **Alias creation:** `x -> y` makes `y` an alias of `x`
2. **Lifetime extension:** An alias extends the lifetime of the underlying buffer
3. **Modification propagation:** Modifying an alias modifies the original (via hooks)
4. **Optimization barrier:** Aliases constrain buffer reuse optimizations

**Explicit Alias Annotation:**
```
# Make aliasing explicit for documentation
x -> y @alias                  # Explicitly marks as alias
x -> copy() -> y @noalias      # Explicitly marks as independent copy
```

#### 5.11.3 Effects Model for Hooks

Hooks can have side effects that impact optimization. The `mode` parameter declares the effect:

| Hook Mode | Effect | Optimization Impact |
|-----------|--------|---------------------|
| `observe` | Read-only | None - can be reordered |
| `modify` | In-place mutation | Prevents buffer aliasing, forces materialization |
| `replace` | Returns new tensor | Prevents buffer reuse, requires allocation |

**Effect Propagation:**

```
# 'modify' hook creates an optimization barrier
x -> Linear(w) -> y @hook(MyHook, mode=modify)
y -> ReLU() -> z

# Compiler inserts implicit synchronization:
# 1. Compute y
# 2. Materialize y to concrete buffer (not aliased)
# 3. Call MyHook(y) - may modify in place
# 4. Continue with ReLU

# If hook were 'observe', y could potentially alias with other buffers
```

**Hook Effects on Fusion:**
- `observe` hooks: Can be fused with surrounding ops (probe inserted post-fusion)
- `modify` hooks: Act as fusion barriers - ops before and after cannot fuse
- `replace` hooks: Act as fusion barriers and require output buffer allocation

#### 5.11.4 Tensor Indexing

Tensor indexing produces views (aliases) into the original tensor:

```
# Indexing syntax
x[i]                          # Index along first dimension
x[:, j]                       # Index along second dimension
x[start:end]                  # Slice along first dimension
x[..., -1]                    # Ellipsis for remaining dimensions

# All indexing produces aliases
x[0] -> first_batch           # Alias into x
x[:, :seq_len] -> truncated   # Alias into x

# For independent copy:
x[0] -> copy() -> first_batch_copy
```

**Indexing Constraints:**
- Indices must be `ConstExpr` (no data-dependent indexing)
- Negative indices are allowed (`-1` = last element)
- Slices must have statically-known bounds

#### 5.11.5 Evaluation Order

Within a graph, operations execute in **topological order**:

1. Operations with no dependencies can execute in any order (or parallel)
2. An operation executes only after all its inputs are available
3. The compiler may reorder operations that have no data dependencies
4. Hooks with `modify`/`replace` mode act as sequence points

**Explicit Ordering:**
```
# These two independent operations may execute in any order:
a -> op1() -> b
c -> op2() -> d

# To enforce ordering, create a data dependency or use @barrier:
a -> op1() -> b @barrier
c -> op2() -> d              # Will execute after op1
```

### 5.12 Backward Pass Contracts

This section formalizes the rules for backward pass definition, gradient naming, and accumulation semantics.

#### 5.12.1 Backward Signature Structure

Every module with trainable parameters must define a backward pass with this structure:

```
backward:
  # Gradient inputs (from downstream)
  d_<output_name>: [output_shape]     # One per forward output

  # Gradient outputs (to upstream)
  d_<input_name>: [input_shape]       # One per forward input

  graph:
    # Gradient computation graph
    ...
```

**Naming Convention:**
- Gradient tensors use the prefix `d_` followed by the original tensor name
- `d_out` = gradient of loss w.r.t. `out`
- `d_weight` = gradient of loss w.r.t. `weight`

#### 5.12.2 Gradient Requirements

**Required Gradients:**

| Entity | Gradient Required? | Rule |
|--------|-------------------|------|
| Forward input | Yes (unless `@gradient(skip)`) | Must compute `d_<input>` for each input |
| Trainable parameter | Yes (unless `@frozen`) | Must compute `d_<param>` for each trainable param |
| Forward output | Provided by caller | `d_<output>` comes from downstream |
| Intermediate tensor | No | Internal gradients are not exposed |

**Omission Rules:**
- Omitting a required gradient is a compile error (E005)
- Use `@gradient(skip)` to explicitly skip gradient for an input
- Use `@frozen` to mark parameters that don't need gradients

```
module PartialBackward():
  params:
    weight: [out, in]                    # Needs gradient
    scale: [1] @frozen                   # No gradient needed

  forward:
    in: (x: [B, in], mask: [B] @gradient(skip))  # mask doesn't need gradient
    out: [B, out]
    graph:
      x -> Linear(weight) -> y
      (y, mask) -> apply_mask() -> out

  backward:
    d_out: [B, out]
    d_x: [B, in]                         # Required (x is not @gradient(skip))
    # d_mask not required (marked @gradient(skip))
    # d_scale not required (marked @frozen)

    graph:
      (d_out, mask) -> apply_mask_backward() -> d_y
      (d_y, weight) -> Linear_backward() -> d_x
      (d_y, saved.x) -> weight_grad() -> d_weight
```

#### 5.12.3 Gradient Accumulation

Parameter gradients accumulate across multiple uses:

**Default Behavior:**
- If a parameter is used multiple times in forward, its gradient contributions are summed
- The compiler generates accumulation automatically

```
module SharedWeight():
  params:
    weight: [D, D]

  forward:
    graph:
      x -> Linear(weight) -> h           # First use of weight
      h -> Linear(weight) -> out         # Second use of weight

  backward:
    graph:
      # Compiler understands weight is used twice
      # d_weight accumulates gradients from both uses
      (d_out, weight) -> Linear_backward() -> d_h
      (d_out, saved.h) -> weight_grad(accumulate=true) -> d_weight    # First contribution

      (d_h, weight) -> Linear_backward() -> d_x
      (d_h, saved.x) -> weight_grad(accumulate=true) -> d_weight      # Accumulates
```

**Explicit Accumulation Control:**
```
# Explicit accumulate=true when manually managing
(grad_source, input) -> matmul(transpose=TN, accumulate=true) -> d_weight

# accumulate=false (default) for first contribution
(grad_source, input) -> matmul(transpose=TN, accumulate=false) -> d_weight
```

#### 5.12.4 Gradient Storage Model

Parameter gradients are stored in a structured `Gradients` object:

```
# Conceptual structure (compiler-generated)
struct ModuleName_Gradients {
    Tensor d_weight;              # Gradient for 'weight' param
    Tensor d_bias;                # Gradient for 'bias' param (if exists)
    // ... one field per trainable parameter
};
```

**Access in Backward:**
- `d_<param_name>` refers to the gradient buffer for that parameter
- Gradients are zero-initialized at the start of backward
- Multiple writes with `accumulate=true` sum into the buffer

#### 5.12.5 Backward Graph Validation

The compiler validates backward graphs:

1. **Completeness Check:** All required gradients are computed
2. **Shape Check:** Gradient shapes match original tensor shapes
3. **Save Check:** All `saved.<name>` references exist in `save:` list
4. **Dependency Check:** No circular dependencies in gradient flow
5. **Type Check:** Gradient dtypes are compatible (or explicit cast)

**Validation Errors:**

| Code | Description |
|------|-------------|
| E005 | Missing required gradient |
| E006 | `saved.<name>` not in save list |
| E018 | Gradient shape mismatch |
| E019 | Circular gradient dependency |
| E020 | Gradient dtype incompatible |

#### 5.12.6 Automatic Backward Derivation

For simple modules composed only of primitives with known backward rules, the compiler can derive the backward graph automatically:

```
module SimpleReLUMLP(d_model, d_ff):
  params:
    w1: [d_ff, d_model]
    w2: [d_model, d_ff]

  forward:
    graph:
      in -> Linear(w1) -> h1
      h1 -> relu() -> h2
      h2 -> Linear(w2) -> out
    save: [in, h1, h2]

  # backward: OMITTED - compiler derives from primitive rules
```

**Auto-derivation Rules:**
- Only applies when all operations are primitives with known backward
- `save:` list must include all tensors needed for backward
- Compiler emits warning if auto-derivation may be suboptimal
- Complex control flow (conditionals) may prevent auto-derivation

**Opting Out:**
```
module ManualBackward():
  forward:
    ...
    save: [...]

  backward:
    # Explicit backward disables auto-derivation
    graph:
      ...
```

---

## 6. Primitive Operations

Primitives are the foundational operations of the DSL. Each primitive has:
- A well-defined mathematical operation
- Known forward and backward implementations
- Direct mapping to CUDA kernels

Primitives are **not user-definable** in standard DSL usage; they form the built-in operation set that modules compose.

---

### 6.1 Linear Algebra

#### 6.1.1 matmul

**Description:**
General matrix multiplication with optional transpose modes. This is the workhorse operation for all linear projections in neural networks. Supports accumulation into existing output for gradient computation.

**Mathematical Definition:**
```
C = α * op(A) @ op(B) + β * C

where op(X) = X      if transpose = N
      op(X) = X^T    if transpose = T
```

**Specification:**
```
primitive matmul:
  """
  General matrix multiplication: C = A @ B with configurable transposes.

  Transpose modes control how A and B are interpreted:
  - NN: C[m,n] = Σ_k A[m,k] * B[k,n]     (standard)
  - NT: C[m,n] = Σ_k A[m,k] * B[n,k]     (B transposed)
  - TN: C[m,n] = Σ_k A[k,m] * B[k,n]     (A transposed)
  - TT: C[m,n] = Σ_k A[k,m] * B[n,k]     (both transposed)
  """

  params:
    transpose: enum(NN, NT, TN, TT) = NN   # Transpose mode for (A, B)
    accumulate: bool = false                # If true, C += A@B; else C = A@B
    alpha: float = 1.0                      # Scaling factor for A@B
    beta: float = 0.0                       # Scaling factor for existing C

  forward:
    in: (A: [M, K], B: [K, N])
    out: C: [M, N]

  backward:
    # d_A = d_C @ B^T  (gradient w.r.t. first input)
    # d_B = A^T @ d_C  (gradient w.r.t. second input)
    d_A = matmul(d_C, B, transpose=NT)
    d_B = matmul(A, d_C, transpose=TN)

  impl:
    forward: kernels.matmul        # cuBLAS GEMM
    backward: kernels.matmul       # Same kernel, different transpose

  precision:
    supported: [fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, fp4_e2m1]
    accumulation: fp32             # Internal accumulation always in fp32
```

**Example Usage:**
```
# Linear projection: y = x @ W^T (TN mode because weight is [out, in])
module Linear(in_dim, out_dim):
  params:
    weight: [out_dim, in_dim]

  forward:
    graph:
      # TN: treat weight as transposed, so effectively x @ weight^T
      (in, weight) -> matmul(transpose=TN) -> out
    save: [in]

  backward:
    graph:
      # Gradient w.r.t. input: d_in = d_out @ weight (NN mode)
      (d_out, weight) -> matmul(transpose=NN) -> d_in

      # Gradient w.r.t. weight: d_weight = d_out^T @ in (TN mode, accumulate)
      (d_out, saved.in) -> matmul(transpose=TN, accumulate=true) -> d_weight

# QKV projection with fused output
module QKVProjection(d_model, qkv_dim):
  params:
    qkv_weight: [qkv_dim, d_model]   # Fused Q, K, V weights

  forward:
    graph:
      (in, qkv_weight) -> matmul(transpose=TN) -> qkv
```

---

#### 6.1.2 batched_matmul

**Description:**
Batched matrix multiplication for parallel computation across batch dimensions. Used in multi-head attention and expert-parallel MoE computations.

**Mathematical Definition:**
```
C[b, m, n] = Σ_k A[b, m, k] * B[b, k, n]   for each batch b
```

**Specification:**
```
primitive batched_matmul:
  """
  Batched matrix multiplication across leading dimensions.

  Supports broadcasting: if A has shape [B, M, K] and B has shape [K, N],
  B is broadcast across the batch dimension.
  """

  params:
    transpose: enum(NN, NT, TN, TT) = NN
    accumulate: bool = false

  forward:
    in: (A: [*, M, K], B: [*, K, N])    # * indicates batch dims
    out: C: [*, M, N]

  backward:
    d_A = batched_matmul(d_C, B, transpose=NT)
    d_B = batched_matmul(A, d_C, transpose=TN)

  impl:
    forward: kernels.batched_matmul    # cuBLAS batched GEMM or grouped GEMM
```

**Example Usage:**
```
# Attention score computation: scores = Q @ K^T
module AttentionScores(num_heads, d_head):
  forward:
    in: (q: [B, H, T, D], k: [B, H, T, D])
    out: scores: [B, H, T, T]

    graph:
      # Q @ K^T with NT transpose (K transposed)
      (q, k) -> batched_matmul(transpose=NT) -> raw_scores
      raw_scores -> scale(1.0 / sqrt(d_head)) -> scores

# Attention output: out = softmax(scores) @ V
module AttentionOutput():
  forward:
    graph:
      (attn_weights, v) -> batched_matmul(transpose=NN) -> out
```

---

#### 6.1.3 grouped_gemm

**Description:**
Grouped GEMM for MoE expert computation. Executes multiple matrix multiplications with different sizes in a single kernel launch, essential for efficient expert-parallel computation.

**Mathematical Definition:**
```
For each group g with n_g tokens:
  C_g[n_g, N] = A_g[n_g, K] @ B_g[K, N]^T
```

**Specification:**
```
primitive grouped_gemm:
  """
  Grouped GEMM for variable-size batches (MoE experts).

  Each group can have a different number of tokens but shares the same
  K and N dimensions. Groups are defined by offsets array.
  """

  params:
    transpose: enum(NN, NT, TN, TT) = TN

  forward:
    in: (
      input: [total_tokens, K],           # Concatenated inputs for all groups
      weights: [num_groups, N, K],        # Per-group weight matrices
      offsets: [num_groups + 1]           # Group boundaries (int32)
    )
    out: output: [total_tokens, N]

  backward:
    d_input = grouped_gemm(d_output, weights, offsets, transpose=NN)
    d_weights = grouped_gemm_weight_grad(input, d_output, offsets)

  impl:
    forward: kernels.moe_grouped_gemm
    backward: kernels.moe_grouped_gemm_backward
```

**Example Usage:**
```
# MoE expert forward pass
module MoEExperts(num_experts, d_model, d_ff):
  params:
    gate_up_weights: [num_experts, 2 * d_ff, d_model]
    down_weights: [num_experts, d_model, d_ff]

  forward:
    in: (permuted_tokens: [total_tokens, d_model], expert_offsets: [num_experts + 1])
    out: expert_outputs: [total_tokens, d_model]

    graph:
      # Up projection for all experts
      (permuted_tokens, gate_up_weights, expert_offsets)
        -> grouped_gemm(transpose=TN) -> gate_up

      gate_up -> split([d_ff, d_ff], dim=-1) -> (gate, up)
      (gate, up) -> swiglu() -> hidden

      # Down projection for all experts
      (hidden, down_weights, expert_offsets)
        -> grouped_gemm(transpose=TN) -> expert_outputs
```

---

### 6.2 Normalization

#### 6.2.1 rmsnorm

**Description:**
Root Mean Square Layer Normalization, the standard normalization in LLaMA-family models. Unlike LayerNorm, RMSNorm does not center the activations (no mean subtraction), making it computationally simpler while maintaining effectiveness.

**Mathematical Definition:**
```
RMS(x) = sqrt(mean(x²) + ε)
y = (x / RMS(x)) * weight
```

**Specification:**
```
primitive rmsnorm:
  """
  Root Mean Square Normalization.

  Normalizes input by its RMS value and applies learnable scale.
  More efficient than LayerNorm as it skips mean computation.

  Returns both normalized output and reciprocal std (rstd) for backward.
  """

  params:
    eps: float = 1e-6              # Epsilon for numerical stability

  forward:
    in: (x: [*, C], weight: [C])
    out: (y: [*, C], rstd: [*])    # rstd shape is x.shape[:-1]

  backward:
    in: (d_y: [*, C], x: [*, C], weight: [C], rstd: [*])
    out: (d_x: [*, C], d_weight: [C])

    # d_x involves chain rule through normalization
    # d_weight = sum(d_y * normalized_x, dim=batch)

  save: [x, rstd]

  impl:
    forward: kernels.rmsnorm_forward
    backward: kernels.rmsnorm_backward

  memory:
    # rstd is small (one float per token), always saved
    # x may be recomputed if checkpointing
    rstd_size: B * T * sizeof(float)
```

**Example Usage:**
```
# Pre-attention normalization
module PreAttentionNorm(d_model, eps=1e-6):
  params:
    weight: [d_model]

  forward:
    in: x: [B, T, d_model]
    out: y: [B, T, d_model]

    graph:
      (x, weight) -> rmsnorm(eps=eps) -> (y, rstd)

    save: [x, rstd]

  backward:
    graph:
      (d_y, saved.x, weight, saved.rstd) -> rmsnorm_backward() -> (d_x, d_weight)

# Standalone usage in a block
block TransformerBlock(...):
  forward:
    graph:
      in -> rmsnorm(ln1_weight, eps=1e-6) -> (ln1_out, ln1_rstd)
      ln1_out -> attention(...) -> attn_out
      ...
```

---

#### 6.2.2 fused_residual_rmsnorm

**Description:**
Fused residual addition and RMSNorm in a single kernel. This is a critical optimization for transformer blocks, reducing memory bandwidth by avoiding a separate residual addition pass.

**Mathematical Definition:**
```
residual_out = residual + input
y = RMSNorm(residual_out, weight)
```

**Specification:**
```
primitive fused_residual_rmsnorm:
  """
  Fused residual addition + RMSNorm.

  Computes:
    1. residual_out = residual + input  (residual connection)
    2. y = rmsnorm(residual_out)        (normalization)

  This fusion saves one full tensor read/write compared to separate ops.
  Essential for memory-bound transformer training.
  """

  params:
    eps: float = 1e-6

  forward:
    in: (
      residual: [*, C],     # Running residual stream
      input: [*, C],        # Output from previous sublayer (attention/MLP)
      weight: [C]           # RMSNorm weight
    )
    out: (
      residual_out: [*, C], # Updated residual (residual + input)
      y: [*, C],            # Normalized output
      rstd: [*]             # Reciprocal std for backward
    )

  backward:
    in: (
      d_y: [*, C],              # Gradient from downstream
      d_residual_next: [*, C],  # Gradient flowing through residual stream
      residual_out: [*, C],
      weight: [C],
      rstd: [*]
    )
    out: (
      d_residual: [*, C],   # Gradient to previous residual
      d_input: [*, C],      # Gradient to sublayer output
      d_weight: [C]
    )

    # Gradients split: both d_residual and d_input receive the combined gradient

  save: [residual_out, rstd]

  impl:
    forward: kernels.fused_residual_rmsnorm_forward
    backward: kernels.fused_residual_rmsnorm_backward
```

**Example Usage:**
```
# Standard transformer block residual pattern
block DenseTransformerBlock(d_model, ...):
  forward:
    inputs:
      x: [B, T, d_model]           # Previous MLP output
      residual: [B, T, d_model]    # Running residual

    graph:
      # First sublayer: attention
      # Fused: residual_mid = residual + x; ln1_out = norm(residual_mid)
      (residual, x, ln1_weight)
        -> fused_residual_rmsnorm(eps=1e-6)
        -> (residual_mid, ln1_out, ln1_rstd)

      ln1_out -> attention(...) -> attn_out

      # Second sublayer: MLP
      # Fused: residual_out = residual_mid + attn_out; ln2_out = norm(residual_out)
      (residual_mid, attn_out, ln2_weight)
        -> fused_residual_rmsnorm(eps=1e-6)
        -> (residual_out, ln2_out, ln2_rstd)

      ln2_out -> mlp(...) -> out

    save: [residual, residual_mid, ln1_out, ln1_rstd, attn_out, ln2_out, ln2_rstd]

  backward:
    graph:
      # MLP backward
      (d_out, saved.ln2_out, mlp_params) -> mlp_backward() -> d_ln2_out

      # Second fused residual+norm backward
      (d_ln2_out, d_residual, saved.residual_out, ln2_weight, saved.ln2_rstd)
        -> fused_residual_rmsnorm_backward()
        -> (d_residual_mid, d_attn_out, d_ln2_weight)

      # Attention backward
      (d_attn_out, saved.ln1_out, attn_params) -> attention_backward() -> d_ln1_out

      # First fused residual+norm backward
      (d_ln1_out, d_residual_mid, saved.residual, ln1_weight, saved.ln1_rstd)
        -> fused_residual_rmsnorm_backward()
        -> (d_residual_out, d_x, d_ln1_weight)
```

---

#### 6.2.3 layernorm

**Description:**
Standard Layer Normalization with mean centering. Less common in modern LLMs but still used in some architectures.

**Mathematical Definition:**
```
μ = mean(x)
σ² = var(x)
y = (x - μ) / sqrt(σ² + ε) * weight + bias
```

**Specification:**
```
primitive layernorm:
  """
  Standard Layer Normalization with centering.

  Unlike RMSNorm, this subtracts the mean before normalization.
  Includes optional bias parameter.
  """

  params:
    eps: float = 1e-5

  forward:
    in: (x: [*, C], weight: [C], bias: [C]?)
    out: (y: [*, C], mean: [*], rstd: [*])

  backward:
    in: (d_y, x, weight, mean, rstd)
    out: (d_x, d_weight, d_bias?)

  save: [x, mean, rstd]

  impl:
    forward: kernels.layernorm_forward
    backward: kernels.layernorm_backward
```

**Example Usage:**
```
# GPT-2 style block with LayerNorm
module GPT2Block(d_model, eps=1e-5):
  params:
    ln1_weight: [d_model]
    ln1_bias: [d_model]

  forward:
    graph:
      (in, ln1_weight, ln1_bias) -> layernorm(eps=eps) -> (ln1_out, mean, rstd)
```

---

### 6.3 Activations

#### 6.3.1 swiglu

**Description:**
SiLU-Gated Linear Unit, the standard activation in LLaMA, Qwen, and most modern LLMs. Combines a gating mechanism with the SiLU (Swish) activation function for improved expressiveness.

**Mathematical Definition:**
```
SwiGLU(gate, up) = SiLU(gate) * up
                 = (gate * sigmoid(gate)) * up
```

**Specification:**
```
primitive swiglu:
  """
  SiLU-Gated Linear Unit activation.

  Takes two inputs of the same shape:
  - gate: passed through SiLU activation
  - up: multiplied element-wise with activated gate

  This is the activation function used between up and down projections
  in LLaMA-style MLP blocks.
  """

  forward:
    in: (gate: [*, D], up: [*, D])
    out: [*, D]

    # Mathematically: silu(gate) * up
    # where silu(x) = x * sigmoid(x)

  backward:
    in: (d_out: [*, D], gate: [*, D], up: [*, D])
    out: (d_gate: [*, D], d_up: [*, D])

    # d_up = d_out * silu(gate)
    # d_gate = d_out * up * d_silu(gate)
    # where d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

  save: [gate, up]   # Both needed for backward

  impl:
    forward: kernels.silu_mul_forward
    backward: kernels.silu_mul_backward

  fusion:
    # Can be fused with preceding split and following matmul
    patterns:
      - [split, swiglu] -> fused_split_swiglu
      - [swiglu, matmul] -> fused_swiglu_matmul
```

**Example Usage:**
```
# Standard SwiGLU MLP
module SwiGLU_MLP(d_model, d_ff):
  params:
    up_weight: [2 * d_ff, d_model]    # Fused gate + up projection
    down_weight: [d_model, d_ff]

  forward:
    in: [B, T, d_model]
    out: [B, T, d_model]

    graph:
      # Project to 2x intermediate size (gate and up fused)
      in -> matmul(up_weight, transpose=TN) -> gate_up

      # Split into gate and up paths
      gate_up -> split([d_ff, d_ff], dim=-1) -> (gate, up)

      # SwiGLU activation
      (gate, up) -> swiglu() -> hidden

      # Down projection
      hidden -> matmul(down_weight, transpose=TN) -> out

    save: [in]
    recompute: [gate_up, gate, up, hidden]   # Recompute in backward to save memory

  backward:
    graph:
      # Recompute forward activations
      recompute:
        saved.in -> matmul(up_weight, transpose=TN) -> gate_up
        gate_up -> split([d_ff, d_ff], dim=-1) -> (gate, up)
        (gate, up) -> swiglu() -> hidden

      # Down projection backward
      (d_out, down_weight) -> matmul(transpose=NN) -> d_hidden
      (hidden, d_out) -> matmul(transpose=TN, accumulate=true) -> d_down_weight

      # SwiGLU backward
      (d_hidden, gate, up) -> swiglu_backward() -> (d_gate, d_up)

      # Concat gradients for fused up projection
      (d_gate, d_up) -> concat(dim=-1) -> d_gate_up

      # Up projection backward
      (d_gate_up, up_weight) -> matmul(transpose=NN) -> d_in
      (saved.in, d_gate_up) -> matmul(transpose=TN, accumulate=true) -> d_up_weight
```

---

#### 6.3.2 geglu

**Description:**
GELU-Gated Linear Unit, an alternative to SwiGLU using GELU activation for the gating.

**Mathematical Definition:**
```
GeGLU(gate, up) = GELU(gate) * up
                = gate * Φ(gate) * up
where Φ is the standard Gaussian CDF
```

**Specification:**
```
primitive geglu:
  """
  GELU-Gated Linear Unit activation.

  Similar to SwiGLU but uses GELU instead of SiLU for gating.
  Used in some GPT variants and PaLM.
  """

  forward:
    in: (gate: [*, D], up: [*, D])
    out: [*, D]

  backward:
    in: (d_out, gate, up)
    out: (d_gate, d_up)

  save: [gate, up]

  impl:
    forward: kernels.gelu_mul_forward
    backward: kernels.gelu_mul_backward
```

**Example Usage:**
```
# GeGLU MLP variant
module GeGLU_MLP(d_model, d_ff):
  params:
    up_weight: [2 * d_ff, d_model]
    down_weight: [d_model, d_ff]

  forward:
    graph:
      in -> matmul(up_weight, transpose=TN) -> gate_up
      gate_up -> split([d_ff, d_ff], dim=-1) -> (gate, up)
      (gate, up) -> geglu() -> hidden    # GeGLU instead of SwiGLU
      hidden -> matmul(down_weight, transpose=TN) -> out
```

---

#### 6.3.3 silu

**Description:**
Sigmoid Linear Unit (also known as Swish), a smooth approximation to ReLU with self-gating properties.

**Mathematical Definition:**
```
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

**Specification:**
```
primitive silu:
  """
  Sigmoid Linear Unit (Swish) activation.

  Smooth, non-monotonic activation that often outperforms ReLU.
  Self-gated: the input gates itself through sigmoid.
  """

  forward:
    in: x: [*]
    out: [*]

  backward:
    in: (d_out: [*], x: [*])
    out: d_x: [*]

    # d_x = d_out * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
    #     = d_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))

  save: [x]

  impl:
    forward: kernels.silu_forward
    backward: kernels.silu_backward
```

**Example Usage:**
```
# Simple MLP with SiLU (non-gated)
module SimpleMLP(d_model, d_ff):
  params:
    up_weight: [d_ff, d_model]      # Note: not 2*d_ff (no gating)
    down_weight: [d_model, d_ff]

  forward:
    graph:
      in -> matmul(up_weight, transpose=TN) -> hidden_pre
      hidden_pre -> silu() -> hidden
      hidden -> matmul(down_weight, transpose=TN) -> out

# Mamba gating (silu applied to gate path)
module MambaGating():
  forward:
    in: (gate: [B, T, D], scan_out: [B, T, D])
    out: [B, T, D]

    graph:
      gate -> silu() -> gate_act
      (gate_act, scan_out) -> mul() -> out
```

---

#### 6.3.4 relu

**Description:**
Rectified Linear Unit, the classic activation function. Simple and computationally efficient but can suffer from "dying ReLU" problem.

**Mathematical Definition:**
```
ReLU(x) = max(0, x)
```

**Specification:**
```
primitive relu:
  """
  Rectified Linear Unit activation.

  Simple thresholding at zero. Gradient is 1 for positive inputs, 0 otherwise.
  Memory-efficient: only need to store sign mask for backward.
  """

  forward:
    in: x: [*]
    out: [*]

  backward:
    in: (d_out: [*], x: [*])
    out: d_x: [*]

    # d_x = d_out * (x > 0)

  save: [x]    # Or just the sign mask for memory efficiency

  impl:
    forward: kernels.relu_forward
    backward: kernels.relu_backward
```

**Example Usage:**
```
# Classic MLP with ReLU
module ReLU_MLP(d_model, d_ff):
  forward:
    graph:
      in -> matmul(up_weight, transpose=TN) -> hidden_pre
      hidden_pre -> relu() -> hidden
      hidden -> matmul(down_weight, transpose=TN) -> out
```

---

#### 6.3.5 relu2

**Description:**
Squared ReLU activation, used in Nemotron and some other architectures. Provides stronger gradient signal for positive values.

**Mathematical Definition:**
```
ReLU²(x) = max(0, x)² = ReLU(x)²
```

**Specification:**
```
primitive relu2:
  """
  Squared ReLU activation: ReLU(x)².

  Stronger activation than standard ReLU, with quadratic growth
  for positive inputs. Used in Nemotron-H architecture.
  """

  forward:
    in: x: [*]
    out: [*]

  backward:
    in: (d_out: [*], x: [*])
    out: d_x: [*]

    # d_x = d_out * 2 * relu(x) * (x > 0)
    #     = d_out * 2 * max(0, x)

  save: [x]

  impl:
    forward: kernels.relu2_forward
    backward: kernels.relu2_backward
```

**Example Usage:**
```
# Nemotron-style MLP with ReLU²
module NemotronMLP(d_model, d_ff):
  params:
    up_weight: [d_ff, d_model]
    down_weight: [d_model, d_ff]

  forward:
    graph:
      in -> matmul(up_weight, transpose=TN) -> hidden_pre
      hidden_pre -> relu2() -> hidden     # Squared ReLU
      hidden -> matmul(down_weight, transpose=TN) -> out
```

---

#### 6.3.6 gelu

**Description:**
Gaussian Error Linear Unit, a smooth activation that weights inputs by their percentile under a Gaussian distribution.

**Mathematical Definition:**
```
GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
```

**Specification:**
```
primitive gelu:
  """
  Gaussian Error Linear Unit activation.

  Smooth activation that multiplies input by its Gaussian CDF value.
  Two variants: exact (using erf) and approximate (using tanh).
  """

  params:
    approximate: bool = true    # Use tanh approximation (faster)

  forward:
    in: x: [*]
    out: [*]

  backward:
    in: (d_out: [*], x: [*])
    out: d_x: [*]

  save: [x]

  impl:
    forward: kernels.gelu_forward
    backward: kernels.gelu_backward
```

**Example Usage:**
```
# BERT-style MLP with GELU
module BERT_MLP(d_model, d_ff):
  forward:
    graph:
      in -> matmul(up_weight, transpose=TN) -> hidden_pre
      hidden_pre -> gelu(approximate=true) -> hidden
      hidden -> matmul(down_weight, transpose=TN) -> out
```

---

#### 6.3.7 softmax

**Description:**
Softmax normalization, converts logits to probability distribution. Critical for attention and classification.

**Mathematical Definition:**
```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

**Specification:**
```
primitive softmax:
  """
  Softmax normalization along specified dimension.

  Numerically stable implementation using max subtraction.
  Returns probability distribution that sums to 1.
  """

  params:
    dim: int = -1              # Dimension to normalize over

  forward:
    in: x: [*]
    out: [*]

  backward:
    in: (d_out: [*], out: [*])     # Note: uses output, not input
    out: d_x: [*]

    # d_x = out * (d_out - sum(d_out * out, dim))

  save: [out]    # Save output, not input (more efficient for backward)

  impl:
    forward: kernels.softmax_forward
    backward: kernels.softmax_backward
```

**Example Usage:**
```
# Classification head
module ClassificationHead(d_model, num_classes):
  params:
    classifier: [num_classes, d_model]

  forward:
    in: [B, d_model]              # Pooled representation
    out: [B, num_classes]         # Probabilities

    graph:
      in -> matmul(classifier, transpose=TN) -> logits
      logits -> softmax(dim=-1) -> out

# Manual attention (not using flash_attention)
module ManualAttention():
  forward:
    graph:
      (q, k) -> batched_matmul(transpose=NT) -> scores
      scores -> scale(1.0 / sqrt(d_head)) -> scaled_scores
      scaled_scores -> softmax(dim=-1) -> attn_weights
      (attn_weights, v) -> batched_matmul() -> out
```

---

### 6.4 Attention

#### 6.4.1 flash_attention

**Description:**
Memory-efficient fused attention using the FlashAttention algorithm. Computes exact attention while dramatically reducing memory usage by never materializing the full attention matrix.

**Mathematical Definition:**
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

**Specification:**
```
primitive flash_attention:
  """
  Memory-efficient fused attention (FlashAttention-2).

  Computes scaled dot-product attention without materializing the
  full [T, T] attention matrix. Essential for long sequences.

  Supports:
  - Grouped Query Attention (GQA): Hq != Hkv with automatic broadcast
  - Causal masking: future tokens masked
  - Variable sequence lengths via cu_seqlens

  Returns log-sum-exp (lse) for numerically stable backward pass.
  """

  params:
    causal: bool = true            # Apply causal mask
    softmax_scale: float? = None   # Default: 1/sqrt(d_head)
    window_size: int? = None       # Sliding window attention (Mistral)
    alibi_slopes: [H]? = None      # ALiBi position bias slopes

  forward:
    in: (
      q: [B, Hq, T, D],            # Query: B=batch, Hq=query heads, T=seq, D=head_dim
      k: [B, Hkv, T, D],           # Key: Hkv=kv heads (Hkv <= Hq for GQA)
      v: [B, Hkv, T, D]            # Value
    )
    out: (
      out: [B, Hq, T, D],          # Attention output
      lse: [B, Hq, T]              # Log-sum-exp for backward
    )

  backward:
    in: (
      d_out: [B, Hq, T, D],
      q: [B, Hq, T, D],
      k: [B, Hkv, T, D],
      v: [B, Hkv, T, D],
      out: [B, Hq, T, D],
      lse: [B, Hq, T]
    )
    out: (
      d_q: [B, Hq, T, D],
      d_k: [B, Hkv, T, D],
      d_v: [B, Hkv, T, D]
    )

  save: [q, k, v, out, lse]

  impl:
    forward: kernels.flash_attention_forward
    backward: kernels.flash_attention_backward

  memory:
    # Memory usage is O(T) instead of O(T²)
    workspace: O(B * H * T)
    saved: O(B * H * T)    # lse
```

**Example Usage:**
```
# Full self-attention module
module CausalSelfAttention(d_model, num_heads, num_kv_heads, max_seq):
  let:
    d_head = d_model // num_heads
    q_dim = num_heads * d_head
    kv_dim = num_kv_heads * d_head

  params:
    qkv_weight: [q_dim + 2 * kv_dim, d_model]
    out_weight: [d_model, d_model]

  forward:
    in: [B, T, d_model]
    out: [B, T, d_model]

    graph:
      # QKV projection
      in -> matmul(qkv_weight, transpose=TN) -> qkv
      qkv -> split([q_dim, kv_dim, kv_dim], dim=-1) -> (q_flat, k_flat, v_flat)

      # Reshape to heads
      q_flat -> view([B, T, num_heads, d_head]) -> transpose(1, 2) -> q
      k_flat -> view([B, T, num_kv_heads, d_head]) -> transpose(1, 2) -> k
      v_flat -> view([B, T, num_kv_heads, d_head]) -> transpose(1, 2) -> v

      # Apply RoPE (separate primitive)
      (q, k) -> rope(d_head, max_seq) -> (q_rope, k_rope)

      # Flash Attention
      (q_rope, k_rope, v) -> flash_attention(causal=true) -> (attn_out, lse)

      # Reshape back and output projection
      attn_out -> transpose(1, 2) -> view([B, T, d_model]) -> attn_flat
      attn_flat -> matmul(out_weight, transpose=TN) -> out

    save: [in, qkv, q_rope, k_rope, v, attn_out, lse]

# Sliding window attention (Mistral style)
module SlidingWindowAttention(...):
  forward:
    graph:
      ...
      (q_rope, k_rope, v) -> flash_attention(causal=true, window_size=4096) -> (attn_out, lse)
      ...
```

---

#### 6.4.2 rope

**Description:**
Rotary Position Embedding (RoPE), the standard position encoding for modern LLMs. Encodes position information through rotation in the complex plane, enabling length extrapolation.

**Mathematical Definition:**
```
For position p and dimension pair (2i, 2i+1):
  θ_i = p / (base^(2i/d))

  [q_rot[2i]  ]   [cos(θ_i)  -sin(θ_i)] [q[2i]  ]
  [q_rot[2i+1]] = [sin(θ_i)   cos(θ_i)] [q[2i+1]]
```

**Specification:**
```
primitive rope:
  """
  Rotary Position Embedding.

  Applies rotation to query and key vectors based on position.
  Rotation angles are precomputed in freq_cis tensor.

  Two layout modes:
  - interleaved: pairs are (x[0], x[1]), (x[2], x[3]), ...
  - non-interleaved: pairs are (x[0], x[d/2]), (x[1], x[d/2+1]), ...
  """

  params:
    interleaved: bool = false      # Dimension layout

  forward:
    in: (
      q: [B, H, T, D],
      k: [B, Hkv, T, D],
      freqs: [T, D // 2, 2]        # Precomputed (cos, sin) pairs
    )
    out: (
      q_rot: [B, H, T, D],
      k_rot: [B, Hkv, T, D]
    )

  backward:
    in: (d_q_rot, d_k_rot, freqs)
    out: (d_q, d_k)

    # Backward is same rotation with negated sin (inverse rotation)

  # No learnable parameters - freqs are precomputed constants
  save: []    # Only need freqs, which is a constant

  impl:
    forward: kernels.rope_forward
    backward: kernels.rope_backward
```

**Example Usage:**
```
# RoPE application in attention
module RoPEAttention(d_model, num_heads, max_seq, rope_theta=10000.0):
  let:
    d_head = d_model // num_heads

  # freq_cis is precomputed at model init, not a learned param
  constants:
    freq_cis: precompute_rope_freqs(max_seq, d_head, rope_theta)

  forward:
    graph:
      # After QKV projection and reshape to heads...
      q -> view([B, num_heads, T, d_head]) -> q_heads
      k -> view([B, num_kv_heads, T, d_head]) -> k_heads

      # Apply RoPE
      (q_heads, k_heads, freq_cis) -> rope(interleaved=false) -> (q_rope, k_rope)

      # Continue to attention...

# Custom RoPE with YaRN scaling (for length extension)
module YaRNAttention(d_model, num_heads, max_seq, scale_factor=4.0):
  constants:
    freq_cis: precompute_yarn_freqs(max_seq, d_head, scale_factor)

  forward:
    graph:
      (q_heads, k_heads, freq_cis) -> rope() -> (q_rope, k_rope)
```

---

#### 6.4.3 qk_norm

**Description:**
QK Normalization, normalizes query and key vectors before RoPE application. Used in Qwen2.5 and other models to stabilize attention at scale.

**Specification:**
```
primitive qk_norm:
  """
  Per-head RMSNorm for query and key vectors.

  Applied after QKV projection, before RoPE. Stabilizes attention
  scores by normalizing Q and K independently per head.
  """

  params:
    eps: float = 1e-6

  forward:
    in: (
      q: [B, H, T, D],
      k: [B, Hkv, T, D],
      q_weight: [D],
      k_weight: [D]
    )
    out: (
      q_norm: [B, H, T, D],
      k_norm: [B, Hkv, T, D],
      q_rstd: [B, H, T],
      k_rstd: [B, Hkv, T]
    )

  backward:
    in: (d_q_norm, d_k_norm, q, k, q_weight, k_weight, q_rstd, k_rstd)
    out: (d_q, d_k, d_q_weight, d_k_weight)

  save: [q, k, q_rstd, k_rstd]

  impl:
    forward: kernels.qk_norm_forward
    backward: kernels.qk_norm_backward
```

**Example Usage:**
```
# Qwen-style attention with QK norm
module QwenAttention(d_model, num_heads, num_kv_heads, max_seq):
  params:
    qkv_weight: [qkv_dim, d_model]
    q_norm_weight: [d_head]
    k_norm_weight: [d_head]
    out_weight: [d_model, d_model]

  forward:
    graph:
      # QKV projection
      in -> matmul(qkv_weight, transpose=TN) -> qkv
      qkv -> split_and_reshape() -> (q, k, v)

      # QK Normalization (before RoPE)
      (q, k, q_norm_weight, k_norm_weight) -> qk_norm() -> (q_norm, k_norm, q_rstd, k_rstd)

      # RoPE on normalized Q and K
      (q_norm, k_norm) -> rope() -> (q_rope, k_rope)

      # Attention
      (q_rope, k_rope, v) -> flash_attention() -> (attn_out, lse)
```

---

### 6.5 Tensor Manipulation

#### 6.5.1 split

**Description:**
Splits a tensor into multiple parts along a specified dimension. Zero-cost operation that returns views into the original tensor.

**Specification:**
```
primitive split:
  """
  Split tensor into multiple parts along a dimension.

  Zero-cost operation: returns views (pointer arithmetic only).
  Sum of sizes must equal the dimension being split.
  """

  params:
    sizes: [int]           # Size of each output chunk
    dim: int = -1          # Dimension to split along

  forward:
    in: x: [*, N, *]       # N = sum(sizes) at dimension dim
    out: tuple([*, sizes[i], *] for i in range(len(sizes)))

  backward:
    in: (d_outputs...)     # Tuple of gradients
    out: d_x = concat(d_outputs, dim=dim)

  # Zero-cost: no data movement
  impl:
    forward: pointer_arithmetic
    backward: concat  # Backward requires actual concat
```

**Example Usage:**
```
# Split fused QKV projection
module SplitQKV(d_model, num_heads, num_kv_heads):
  let:
    d_head = d_model // num_heads
    q_dim = num_heads * d_head
    kv_dim = num_kv_heads * d_head

  forward:
    in: qkv: [B, T, q_dim + 2 * kv_dim]
    out: (q: [B, T, q_dim], k: [B, T, kv_dim], v: [B, T, kv_dim])

    graph:
      qkv -> split([q_dim, kv_dim, kv_dim], dim=-1) -> (q, k, v)

# Split gate and up in SwiGLU
module SwiGLU_Activation(d_ff):
  forward:
    in: gate_up: [B, T, 2 * d_ff]
    out: [B, T, d_ff]

    graph:
      gate_up -> split([d_ff, d_ff], dim=-1) -> (gate, up)
      (gate, up) -> swiglu() -> out
```

---

#### 6.5.2 concat

**Description:**
Concatenates multiple tensors along a specified dimension. The inverse of split.

**Specification:**
```
primitive concat:
  """
  Concatenate tensors along a dimension.

  All tensors must have the same shape except along the concat dimension.
  """

  params:
    dim: int = -1

  forward:
    in: (tensors...)        # Tuple of tensors to concatenate
    out: concatenated       # Shape: sum of input dims along concat axis

  backward:
    in: d_out
    out: (d_tensors...) = split(d_out, original_sizes, dim)

  impl:
    forward: kernels.concat_forward    # Requires memory copy
    backward: split                     # Zero-cost pointer arithmetic
```

**Example Usage:**
```
# Concat gradients for fused backward
module MLPBackward():
  backward:
    graph:
      # After SwiGLU backward, concat gradients for fused weight grad
      (d_gate, d_up) -> concat(dim=-1) -> d_gate_up
      (in, d_gate_up) -> matmul(transpose=TN, accumulate=true) -> d_up_weight

# Multi-head concat
module ConcatHeads(num_heads, d_head):
  forward:
    in: heads: [B, num_heads, T, d_head]
    out: [B, T, num_heads * d_head]

    graph:
      heads -> transpose(1, 2) -> [B, T, num_heads, d_head]
      -> view([B, T, num_heads * d_head]) -> out
```

---

#### 6.5.3 view

**Description:**
Reshapes a tensor without copying data. Requires the tensor to be contiguous in memory.

**Specification:**
```
primitive view:
  """
  Reshape tensor to new shape (must be contiguous).

  Zero-cost operation: only changes metadata (shape, strides).
  Total number of elements must remain the same.
  Use -1 for one dimension to infer its size.
  """

  params:
    shape: [int]           # New shape (-1 for inference)

  forward:
    in: x: [*]
    out: [shape]

  backward:
    in: d_out: [shape]
    out: d_x = view(d_out, original_shape)

  # Zero-cost: metadata only
  impl:
    forward: metadata_only
    backward: metadata_only

  constraints:
    - product(shape) == product(x.shape)
    - x must be contiguous
```

**Example Usage:**
```
# Reshape for multi-head attention
module ReshapeToHeads(num_heads, d_head):
  forward:
    in: x: [B, T, num_heads * d_head]
    out: [B, num_heads, T, d_head]

    graph:
      x -> view([B, T, num_heads, d_head]) -> intermediate
      intermediate -> transpose(1, 2) -> out

# Flatten for linear projection
module Flatten():
  forward:
    in: x: [B, T, C]
    out: [B * T, C]

    graph:
      x -> view([B * T, C]) -> out

# Reshape MoE outputs
module UnflattenMoE():
  forward:
    in: (x: [B * T, C], B: int, T: int)
    out: [B, T, C]

    graph:
      x -> view([B, T, C]) -> out
```

---

#### 6.5.4 transpose

**Description:**
Swaps two dimensions of a tensor. May or may not require data movement depending on subsequent operations.

**Specification:**
```
primitive transpose:
  """
  Swap two dimensions of a tensor.

  Can be zero-cost (stride manipulation) if followed by contiguous-preserving ops,
  or may require explicit transposition kernel if materialization needed.
  """

  params:
    dim0: int
    dim1: int

  forward:
    in: x: [*, D0, *, D1, *]
    out: [*, D1, *, D0, *]     # dim0 and dim1 swapped

  backward:
    in: d_out
    out: d_x = transpose(d_out, dim0, dim1)   # Same transpose is its own inverse

  impl:
    forward: kernels.transpose or stride_manipulation
    backward: same
```

**Example Usage:**
```
# Multi-head attention reshape
module AttentionReshape():
  forward:
    graph:
      # [B, T, H, D] -> [B, H, T, D]
      x -> transpose(1, 2) -> out

# Batched matmul preparation
module PrepareForBatchedMatmul():
  forward:
    in: x: [B, T, H, D]
    out: [B, H, T, D]

    graph:
      x -> transpose(1, 2) -> out    # Move heads to batch-like dimension

# Weight transpose for backward
module LinearBackward():
  backward:
    graph:
      # d_input = d_output @ weight (need weight in correct orientation)
      weight -> transpose(0, 1) -> weight_t
      (d_output, weight_t) -> matmul() -> d_input
```

---

#### 6.5.5 permute

**Description:**
Generalized transpose that reorders all dimensions according to a specified order.

**Specification:**
```
primitive permute:
  """
  Reorder tensor dimensions according to specified permutation.

  Generalization of transpose to arbitrary dimension reordering.
  """

  params:
    dims: [int]            # New dimension order (permutation)

  forward:
    in: x: [D0, D1, ..., Dn]
    out: [D_dims[0], D_dims[1], ..., D_dims[n]]

  backward:
    in: d_out
    out: d_x = permute(d_out, inverse_permutation(dims))

  impl:
    forward: kernels.permute or stride_manipulation
```

**Example Usage:**
```
# Complex reshape for attention
module AttentionPermute():
  forward:
    # [B, T, H, D] -> [B, H, T, D]
    graph:
      x -> permute([0, 2, 1, 3]) -> out

# Mamba layout conversion
module MambaLayoutConvert():
  forward:
    # [B, D, T] -> [B, T, D]
    graph:
      x -> permute([0, 2, 1]) -> out
```

---

#### 6.5.6 contiguous

**Description:**
Ensures tensor is contiguous in memory, copying if necessary.

**Specification:**
```
primitive contiguous:
  """
  Make tensor contiguous in memory.

  If already contiguous, this is a no-op (zero-cost).
  If not contiguous (e.g., after transpose), copies to new contiguous buffer.
  """

  forward:
    in: x: [*]
    out: [*]              # Same shape, guaranteed contiguous

  backward:
    in: d_out
    out: d_x              # May need to "un-contiguous" or just pass through

  impl:
    forward: kernels.copy if not contiguous, else no-op
```

**Example Usage:**
```
# Ensure contiguous before view
module SafeReshape():
  forward:
    graph:
      x -> transpose(1, 2) -> x_t
      x_t -> contiguous() -> x_contig    # Needed because transpose makes non-contiguous
      x_contig -> view([B, -1]) -> out
```

---

### 6.6 Elementwise Operations

#### 6.6.1 add

**Description:**
Element-wise addition with broadcasting support. Critical for residual connections.

**Specification:**
```
primitive add:
  """
  Element-wise addition: c = a + b

  Supports broadcasting: smaller tensor is broadcast to match larger.
  Gradient flows equally to both inputs (with reduction for broadcast dims).
  """

  forward:
    in: (a: [*], b: [*])     # Shapes must be broadcast-compatible
    out: [*]                  # Shape is broadcast result

  backward:
    in: d_out
    out: (d_a, d_b)

    # d_a = d_out (possibly reduced if a was broadcast)
    # d_b = d_out (possibly reduced if b was broadcast)

  impl:
    forward: kernels.add_forward
    backward: kernels.add_backward
```

**Example Usage:**
```
# Residual connection
module ResidualAdd():
  forward:
    in: (residual: [B, T, C], sublayer_out: [B, T, C])
    out: [B, T, C]

    graph:
      (residual, sublayer_out) -> add() -> out

# Bias addition (with broadcast)
module AddBias():
  forward:
    in: (x: [B, T, C], bias: [C])
    out: [B, T, C]

    graph:
      (x, bias) -> add() -> out    # bias broadcasts over B and T
```

---

#### 6.6.2 mul

**Description:**
Element-wise multiplication with broadcasting support.

**Specification:**
```
primitive mul:
  """
  Element-wise multiplication: c = a * b

  Supports broadcasting. Gradients require saved inputs.
  """

  forward:
    in: (a: [*], b: [*])
    out: [*]

  backward:
    in: (d_out, a, b)
    out: (d_a, d_b)

    # d_a = d_out * b
    # d_b = d_out * a

  save: [a, b]

  impl:
    forward: kernels.mul_forward
    backward: kernels.mul_backward
```

**Example Usage:**
```
# Gating mechanism
module Gate():
  forward:
    in: (gate: [B, T, D], value: [B, T, D])
    out: [B, T, D]

    graph:
      (gate, value) -> mul() -> out

# Attention weight application
module WeightedSum():
  forward:
    in: (weights: [B, T, K], values: [B, T, K, D])
    out: [B, T, D]

    graph:
      # Expand weights for broadcast
      weights -> view([B, T, K, 1]) -> weights_exp
      (weights_exp, values) -> mul() -> weighted
      weighted -> reduce_sum(dim=2) -> out
```

---

#### 6.6.3 scale

**Description:**
Multiply tensor by a scalar constant. Common for attention score scaling.

**Specification:**
```
primitive scale:
  """
  Scale tensor by constant factor: y = x * factor

  More efficient than mul() when one operand is a scalar constant.
  """

  params:
    factor: float

  forward:
    in: x: [*]
    out: [*]

  backward:
    in: d_out
    out: d_x = d_out * factor

  # No save needed - factor is a constant parameter

  impl:
    forward: kernels.scale
    backward: kernels.scale    # Same operation
```

**Example Usage:**
```
# Attention score scaling
module AttentionScale(d_head):
  forward:
    in: scores: [B, H, T, T]
    out: [B, H, T, T]

    graph:
      scores -> scale(1.0 / sqrt(d_head)) -> out

# Gradient scaling for mixed precision
module GradientScale(loss_scale=1024.0):
  backward:
    graph:
      d_out -> scale(1.0 / loss_scale) -> d_out_scaled
```

---

#### 6.6.4 add3

**Description:**
Three-way addition, used in parallel residual architectures.

**Specification:**
```
primitive add3:
  """
  Three-way element-wise addition: d = a + b + c

  Used in parallel residual architectures (GPT-NeoX style) where
  attention and MLP outputs are added to residual simultaneously.
  """

  forward:
    in: (a: [*], b: [*], c: [*])
    out: [*]

  backward:
    in: d_out
    out: (d_a, d_b, d_c) = (d_out, d_out, d_out)

  impl:
    forward: kernels.add3_forward
```

**Example Usage:**
```
# Parallel residual block (GPT-NeoX style)
block ParallelTransformerBlock():
  forward:
    graph:
      # Attention and MLP run in parallel on same input
      in -> fork(2) -> (attn_path, mlp_path)

      attn_path -> rmsnorm(ln1_weight) -> attention() -> attn_out
      mlp_path -> rmsnorm(ln2_weight) -> mlp() -> mlp_out

      # Three-way addition: residual + attention + MLP
      (in, attn_out, mlp_out) -> add3() -> out
```

---

### 6.7 Reduction Operations

#### 6.7.1 reduce_sum

**Description:**
Sum reduction along specified dimensions.

**Specification:**
```
primitive reduce_sum:
  """
  Sum elements along specified dimensions.

  Used for loss computation, bias gradients, etc.
  """

  params:
    dims: [int]            # Dimensions to reduce
    keepdim: bool = false  # Keep reduced dims as size 1

  forward:
    in: x: [*]
    out: reduced           # Shape with dims removed (or size 1 if keepdim)

  backward:
    in: d_out
    out: d_x = broadcast(d_out, original_shape)

  impl:
    forward: kernels.reduce_sum
    backward: kernels.broadcast
```

**Example Usage:**
```
# Bias gradient computation
module BiasBackward():
  backward:
    in: d_out: [B, T, C]
    out: d_bias: [C]

    graph:
      d_out -> reduce_sum(dims=[0, 1]) -> d_bias

# Loss reduction
module MeanLoss():
  forward:
    in: per_token_loss: [B, T]
    out: scalar

    graph:
      per_token_loss -> reduce_sum(dims=[0, 1]) -> total
      total -> scale(1.0 / (B * T)) -> out
```

---

#### 6.7.2 reduce_mean

**Description:**
Mean reduction along specified dimensions.

**Specification:**
```
primitive reduce_mean:
  """
  Compute mean along specified dimensions.

  Equivalent to reduce_sum followed by division by count.
  """

  params:
    dims: [int]
    keepdim: bool = false

  forward:
    in: x: [*]
    out: reduced

  backward:
    in: d_out
    out: d_x = broadcast(d_out / count, original_shape)

  impl:
    forward: kernels.reduce_mean
```

**Example Usage:**
```
# RMS computation (part of RMSNorm)
module ComputeRMS(eps=1e-6):
  forward:
    in: x: [B, T, C]
    out: rms: [B, T]

    graph:
      x -> mul(x) -> x_sq              # x²
      x_sq -> reduce_mean(dims=[-1], keepdim=true) -> mean_sq
      mean_sq -> add(eps) -> mean_sq_eps
      mean_sq_eps -> sqrt() -> rms
```

---

#### 6.7.3 reduce_max

**Description:**
Maximum reduction along specified dimensions. Used in softmax for numerical stability.

**Specification:**
```
primitive reduce_max:
  """
  Find maximum value along specified dimensions.

  Returns both max values and indices (for backward).
  """

  params:
    dim: int
    keepdim: bool = false

  forward:
    in: x: [*]
    out: (values: reduced, indices: reduced)

  backward:
    in: (d_values, indices)
    out: d_x    # Gradient only flows to max elements

  save: [indices]

  impl:
    forward: kernels.reduce_max
```

---

### 6.8 Embedding Operations

#### 6.8.1 embedding

**Description:**
Look up embeddings from a weight matrix using integer indices. The entry point for token-to-vector conversion.

**Specification:**
```
primitive embedding:
  """
  Embedding lookup: gather rows from weight matrix by indices.

  Forward is a simple gather; backward is a sparse scatter-add.
  """

  forward:
    in: (indices: [*], weight: [V, D])    # indices are int32
    out: [*, D]

  backward:
    in: (d_out: [*, D], indices: [*])
    out: d_weight: [V, D]                 # Sparse update

    # d_weight[i] += sum of d_out where indices == i

  impl:
    forward: kernels.embedding_forward
    backward: kernels.embedding_backward  # Atomic scatter-add

  memory:
    # Backward doesn't need saved activations, only indices
    save: [indices]
```

**Example Usage:**
```
# Token embedding layer
module TokenEmbedding(vocab_size, d_model):
  params:
    weight: [vocab_size, d_model]

  forward:
    in: token_ids: [B, T]        # int32 token IDs
    out: [B, T, d_model]

    graph:
      (token_ids, weight) -> embedding() -> out

  backward:
    graph:
      (d_out, saved.token_ids) -> embedding_backward(weight) -> d_weight

# Full model embedding with weight tying
model LlamaModel(vocab_size, d_model, ...):
  params:
    embedding_weight: [vocab_size, d_model]
    lm_head: tied_to(embedding_weight)    # Shared weights

  forward:
    graph:
      # Input embedding
      token_ids -> embedding(embedding_weight) -> x0

      # ... transformer layers ...

      # Output projection (tied weights)
      xN -> matmul(lm_head, transpose=TN) -> logits
```

---

### 6.9 MoE (Mixture of Experts) Primitives

#### 6.9.1 moe_router

**Description:**
Computes routing decisions for Mixture of Experts. Determines which experts process each token and with what weight.

**Specification:**
```
primitive moe_router:
  """
  MoE routing: compute expert assignments and weights.

  Performs:
  1. Router projection: logits = x @ gate^T
  2. Softmax over experts
  3. Top-K selection
  4. Optional weight normalization

  Also computes auxiliary load-balancing loss.
  """

  params:
    top_k: int                    # Number of experts per token
    normalize: bool = true        # Normalize top-k weights to sum to 1
    aux_loss_coef: float = 0.01   # Load balancing loss coefficient
    use_sigmoid: bool = false     # Sigmoid (DeepSeek) vs softmax routing

  forward:
    in: (
      x: [B*T, C],                # Flattened token representations
      gate: [E, C]                # Router weight matrix (E = num_experts)
    )
    out: (
      weights: [B*T, top_k],      # Routing weights (sum to 1 if normalized)
      indices: [B*T, top_k],      # Selected expert indices (int32)
      aux_loss: scalar            # Load balancing loss
    )

  backward:
    in: (d_weights, x, gate, indices)
    out: (d_x, d_gate)

  save: [x, indices]

  impl:
    forward: kernels.moe_router_forward
    backward: kernels.moe_router_backward
```

**Example Usage:**
```
# Standard MoE routing
module MoERouter(d_model, num_experts, top_k=2):
  params:
    gate: [num_experts, d_model]

  forward:
    in: x: [B, T, d_model]
    out: (weights: [B*T, top_k], indices: [B*T, top_k], aux_loss: scalar)

    graph:
      x -> view([B*T, d_model]) -> x_flat
      (x_flat, gate) -> moe_router(top_k=top_k, normalize=true) -> (weights, indices, aux_loss)

    save: [x_flat, indices]

# Qwen3-MoE style (normalized top-k)
module Qwen3Router(d_model, num_experts, top_k=4):
  params:
    gate: [num_experts, d_model]

  forward:
    graph:
      (x_flat, gate) -> moe_router(top_k=top_k, normalize=true, aux_loss_coef=0.001)
        -> (weights, indices, aux_loss)
```

---

#### 6.9.2 moe_permute

**Description:**
Reorders tokens from batch order to expert-grouped order for efficient grouped GEMM.

**Specification:**
```
primitive moe_permute:
  """
  Permute tokens to expert-grouped order.

  Reorders tokens so that all tokens assigned to expert 0 come first,
  then expert 1, etc. This enables efficient grouped GEMM.

  Returns permuted tokens and scatter indices for unpermute.
  """

  forward:
    in: (
      x: [B*T, C],                      # Tokens in batch order
      indices: [B*T, top_k],            # Expert assignments from router
      expert_offsets: [num_experts + 1] # Cumulative count of tokens per expert
    )
    out: (
      permuted: [total_tokens, C],      # total_tokens = B*T*top_k
      scatter_indices: [total_tokens]   # For unpermute
    )

  backward:
    in: (d_permuted, scatter_indices)
    out: d_x = moe_unpermute(d_permuted, scatter_indices, ...)

  impl:
    forward: kernels.moe_permute_tokens
```

**Example Usage:**
```
# Permute tokens for expert processing
module MoEPermute(num_experts):
  forward:
    in: (x: [B*T, C], indices: [B*T, top_k])
    out: (permuted: [total_tokens, C], scatter_idx: [total_tokens])

    graph:
      # Compute expert counts and offsets
      indices -> moe_compute_offsets(num_experts) -> expert_offsets

      # Permute tokens to expert order
      (x, indices, expert_offsets) -> moe_permute() -> (permuted, scatter_idx)
```

---

#### 6.9.3 moe_unpermute

**Description:**
Inverse of moe_permute: reorders expert outputs back to batch order and combines with routing weights.

**Specification:**
```
primitive moe_unpermute:
  """
  Unpermute expert outputs and combine with routing weights.

  Performs:
  1. Scatter expert outputs back to batch order
  2. Weight outputs by routing weights
  3. Sum contributions from all top-k experts per token
  """

  forward:
    in: (
      expert_outputs: [total_tokens, C],  # Expert computation results
      weights: [B*T, top_k],              # Routing weights
      scatter_indices: [total_tokens]     # From moe_permute
    )
    out: combined: [B*T, C]

  backward:
    in: (d_combined, weights, scatter_indices)
    out: (d_expert_outputs, d_weights)

  impl:
    forward: kernels.moe_unpermute_and_combine
```

**Example Usage:**
```
# Complete MoE forward
module MoELayer(d_model, num_experts, d_ff, top_k=2):
  params:
    gate: [num_experts, d_model]
    expert_up: [num_experts, 2 * d_ff, d_model]
    expert_down: [num_experts, d_model, d_ff]

  forward:
    in: x: [B, T, d_model]
    out: [B, T, d_model]

    graph:
      x -> view([B*T, d_model]) -> x_flat

      # Routing
      (x_flat, gate) -> moe_router(top_k=top_k) -> (weights, indices, aux_loss)
      indices -> moe_compute_offsets(num_experts) -> expert_offsets

      # Permute to expert order
      (x_flat, indices, expert_offsets) -> moe_permute() -> (permuted, scatter_idx)

      # Expert computation (grouped GEMM)
      (permuted, expert_up, expert_offsets) -> grouped_gemm() -> gate_up
      gate_up -> split([d_ff, d_ff], dim=-1) -> (gate_act, up)
      (gate_act, up) -> swiglu() -> hidden
      (hidden, expert_down, expert_offsets) -> grouped_gemm() -> expert_out

      # Unpermute and combine
      (expert_out, weights, scatter_idx) -> moe_unpermute() -> combined
      combined -> view([B, T, d_model]) -> out
```

---

### 6.10 Mamba/SSM Primitives

#### 6.10.1 mamba_conv1d

**Description:**
Causal 1D convolution for Mamba blocks. Depthwise convolution with activation.

**Specification:**
```
primitive mamba_conv1d:
  """
  Causal depthwise 1D convolution with optional activation.

  Used in Mamba for short-range context mixing before selective scan.
  Supports incremental state for inference.
  """

  params:
    kernel_size: int = 4
    activation: enum(silu, none) = silu

  forward:
    in: (
      x: [B, D, T],                       # Input (channels-first)
      weight: [D, 1, kernel_size],        # Depthwise conv weights
      bias: [D]?,                         # Optional bias
      conv_state: [B, D, kernel_size-1]?  # Optional state for incremental
    )
    out: (
      y: [B, D, T],
      conv_state_out: [B, D, kernel_size-1]?
    )

  backward:
    in: (d_y, x, weight, bias)
    out: (d_x, d_weight, d_bias?)

  save: [x]

  impl:
    forward: kernels.mamba_causal_conv1d_forward
    backward: kernels.mamba_causal_conv1d_backward
```

**Example Usage:**
```
# Mamba convolution layer
module MambaConv(d_inner, conv_kernel=4):
  params:
    weight: [d_inner, 1, conv_kernel]
    bias: [d_inner]

  forward:
    in: x: [B, d_inner, T]
    out: [B, d_inner, T]

    graph:
      (x, weight, bias) -> mamba_conv1d(kernel_size=conv_kernel, activation=silu) -> (out, _)
```

---

#### 6.10.2 mamba_selective_scan

**Description:**
The core selective state space model operation in Mamba. Implements hardware-efficient parallel scan.

**Specification:**
```
primitive mamba_selective_scan:
  """
  Mamba selective scan (S6) operation.

  Implements the selective state space model:
    h[t] = A * h[t-1] + B[t] * x[t]
    y[t] = C[t] * h[t] + D * x[t]

  Where A, B, C are input-dependent (selective).
  Uses chunked parallel scan for efficiency.
  """

  params:
    chunk_size: int = 256

  forward:
    in: (
      u: [B, D, T],                # Input
      delta: [B, D, T],            # Time step (discretization)
      A: [D, N],                   # State transition (log space)
      B: [B, G, N, T],             # Input projection (G = groups)
      C: [B, G, N, T],             # Output projection
      D: [D],                      # Skip connection
      dt_bias: [D],                # Delta bias
      ssm_state: [B, D, N]?        # Optional initial state
    )
    out: (
      y: [B, D, T],
      ssm_state_out: [B, D, N]
    )

  backward:
    in: (d_y, u, delta, A, B, C, D, dt_bias, ssm_state, y)
    out: (d_u, d_delta, d_A, d_B, d_C, d_D, d_dt_bias)

  save: [u, delta, A, B, C, ssm_state, y]

  impl:
    forward: kernels.mamba_selective_scan_forward
    backward: kernels.mamba_selective_scan_backward
```

**Example Usage:**
```
# Complete Mamba block
module MambaBlock(d_model, d_state=16, d_conv=4, expand=2):
  let:
    d_inner = d_model * expand

  params:
    in_proj: [2 * d_inner + d_conv + 1, d_model]
    conv_weight: [d_inner, 1, d_conv]
    conv_bias: [d_inner]
    A_log: [d_inner, d_state]
    D: [d_inner]
    dt_bias: [d_inner]
    norm_weight: [d_inner]
    out_proj: [d_model, d_inner]

  forward:
    in: x: [B, T, d_model]
    out: [B, T, d_model]

    graph:
      # Input projection
      x -> matmul(in_proj, transpose=TN) -> proj
      proj -> mamba_split_proj() -> (gate, conv_in, delta)

      # Causal conv
      conv_in -> permute([0, 2, 1]) -> conv_in_t    # [B, D, T]
      (conv_in_t, conv_weight, conv_bias) -> mamba_conv1d() -> (conv_out, _)
      conv_out -> mamba_split_conv() -> (u, B_ssm, C_ssm)

      # Selective scan
      A_log -> exp_neg() -> A
      (u, delta, A, B_ssm, C_ssm, D, dt_bias)
        -> mamba_selective_scan() -> (scan_out, ssm_state)

      # Gate and normalize
      (gate, scan_out) -> silu_mul() -> gated
      gated -> group_rmsnorm(norm_weight) -> normed

      # Output projection
      normed -> permute([0, 2, 1]) -> normed_t    # [B, T, D]
      normed_t -> matmul(out_proj, transpose=TN) -> out
```

---

### 6.11 Utility Primitives

#### 6.11.1 zeros

**Description:**
Create a tensor filled with zeros.

**Specification:**
```
primitive zeros:
  """Create a tensor of zeros with specified shape and dtype."""

  params:
    shape: [int]
    dtype: dtype = bf16

  forward:
    out: [shape]

  backward:
    # No backward - this is a constant

  impl:
    forward: cudaMemset
```

#### 6.11.2 ones

**Description:**
Create a tensor filled with ones.

**Specification:**
```
primitive ones:
  """Create a tensor of ones with specified shape and dtype."""

  params:
    shape: [int]
    dtype: dtype = bf16

  forward:
    out: [shape]

  backward:
    # No backward - this is a constant
```

#### 6.11.3 fill

**Description:**
Create a tensor filled with a constant value.

**Specification:**
```
primitive fill:
  """Create a tensor filled with a constant value."""

  params:
    shape: [int]
    value: float
    dtype: dtype = bf16

  forward:
    out: [shape]
```

#### 6.11.4 copy

**Description:**
Copy a tensor to a new buffer.

**Specification:**
```
primitive copy:
  """Copy tensor to new memory location."""

  forward:
    in: x: [*]
    out: [*]    # Same shape, new buffer

  backward:
    in: d_out
    out: d_x = d_out    # Gradient passes through

  impl:
    forward: cudaMemcpy
```

---

## 7. Annotations

Annotations provide metadata hints to the compiler and runtime without changing the mathematical semantics of the computation. They control optimization strategies, memory management, distributed execution, and debugging facilities.

Annotations are **advisory** by default—the compiler may ignore them if they conflict with correctness constraints. Some annotations (marked as "binding") must be respected.

---

### 7.1 Syntax

**Basic Syntax:**
```
# On an operation output
source -> Operation() -> destination @annotation

# On an operation output with parameters
source -> Operation() -> destination @annotation(param=value, ...)

# Multiple annotations
source -> Operation() -> destination @annotation1 @annotation2(param=value)

# On parameter declarations
params:
  weight: [dim1, dim2] @annotation(param=value)
```

**Annotation Placement:**
- **Output annotations**: Applied after the `->` destination, affect the output tensor
- **Parameter annotations**: Applied after the shape declaration, affect storage/sharding
- **Block annotations**: Applied to entire graph sections

**Example:**
```
module Example():
  params:
    weight: [4096, 4096] @shard(column) @precision(storage=fp8_e4m3)

  forward:
    graph:
      in -> Linear(weight) -> out @dtype(bf16) @hook(AfterProjection)
```

---

### 7.2 Memory Annotations

Memory annotations control how tensors are stored, when they're freed, and whether they're recomputed during the backward pass. These are critical for managing GPU memory in large model training.

#### 7.2.0 Memory Policy Precedence

Multiple mechanisms can specify memory behavior for a tensor. The compiler resolves conflicts using this **precedence ladder** (highest priority first):

| Priority | Source | Example |
|----------|--------|---------|
| 1 (highest) | Instance-level `@memory` annotation | `x -> op() -> y @memory(recompute)` |
| 2 | Instance-level `@checkpoint` with `preserve` list | `@checkpoint(preserve=[attn_out])` |
| 3 | Block-level `@checkpoint` annotation | `block: @checkpoint(full)` |
| 4 | Module's `save:` list | `save: [x, y, z]` |
| 5 | Module's `recompute:` list | `recompute: [h1, h2]` |
| 6 (lowest) | Default policy | See below |

**Default Policy:**
- Tensors consumed by backward: `save` (unless recompute is more efficient)
- Intermediate tensors not in backward: `temporary`
- Parameters: `pin` (except with `@memory(offload)` or `@memory(stream)`)

**Recompute Frontier:**

When a tensor is marked for recompute, the compiler must identify the **recompute frontier**—the set of saved tensors from which the recomputed tensor can be derived.

```
module Example():
  forward:
    graph:
      in -> op1() -> a              # a will be recomputed
      a -> op2() -> b @memory(save) # b is saved (frontier)
      b -> op3() -> c @memory(recompute)  # c recomputed from b

    save: [in, b]                   # Explicit save list
    recompute: [a, c]               # Explicit recompute list

  # Recompute frontier for 'a': {in}
  # Recompute frontier for 'c': {b}
```

**Legality Constraints:**

1. A tensor marked `@memory(recompute)` must be derivable from saved tensors
2. If derivation is impossible (missing inputs), compiler emits E021
3. Circular recompute dependencies are forbidden (E022)

**Example Resolution:**
```
module PrecedenceExample():
  forward:
    graph:
      in -> op1() -> x @memory(save)         # Priority 1: explicit save
      x -> op2() -> y                        # Priority 4: in save: list -> save
      y -> op3() -> z @memory(recompute)     # Priority 1: explicit recompute
      z -> op4() -> w                        # Priority 6: default -> temporary

    save: [y]
    recompute: []                            # z overrides to recompute via annotation
```

---

#### 7.2.1 @memory

**Description:**
Controls the memory lifecycle of a tensor. Specifies whether a tensor should be saved for backward, recomputed, treated as temporary, or pinned in GPU memory.

**Syntax:**
```
@memory(mode)
@memory(mode, options...)
```

**Modes:**

| Mode        | Description                                                         | Use Case                             |
| ----------- | ------------------------------------------------------------------- | ------------------------------------ |
| `save`      | Save tensor for backward pass (default for tensors in `save:` list) | Needed for gradient computation      |
| `recompute` | Don't save; recompute in backward pass                              | Memory-constrained training          |
| `temporary` | Can be freed immediately after use                                  | Intermediate values not needed later |
| `pin`       | Keep in GPU memory, never offload                                   | Frequently accessed tensors          |
| `offload`   | Eligible for CPU offloading                                         | Large tensors accessed infrequently  |
| `stream`    | Stream from CPU/NVMe on demand                                      | ZeRO-3 / FSDP weight streaming       |

**Options:**

| Option     | Type | Description                              |
| ---------- | ---- | ---------------------------------------- |
| `priority` | int  | Offload priority (lower = offload first) |
| `prefetch` | bool | Prefetch before use if offloaded         |
| `async`    | bool | Use async memory operations              |

**Specification:**
```
annotation @memory:
  """
  Control tensor memory lifecycle.

  Affects when tensors are allocated, retained, and freed.
  Critical for memory optimization in large model training.
  """

  params:
    mode: enum(save, recompute, temporary, pin, offload, stream)
    priority: int = 0           # Offload priority
    prefetch: bool = true       # Prefetch if offloaded
    async: bool = true          # Async memory ops

  applies_to: [tensor_output, parameter]

  compiler_behavior:
    save: Include in activation checkpoint
    recompute: Add to recompute graph, exclude from save
    temporary: Mark for immediate deallocation after consumers run
    pin: Exclude from offload candidates
    offload: Add to offload candidate list with priority
    stream: Enable on-demand weight loading
```

**Example Usage:**
```
# Activation checkpointing: recompute attention instead of saving
module MemoryEfficientAttention(d_model, num_heads):
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv
      qkv -> split_reshape() -> (q, k, v)
      (q, k) -> rope() -> (q_rope, k_rope) @memory(recompute)
      (q_rope, k_rope, v) -> flash_attention() -> (attn_out, lse) @memory(recompute)
      attn_out -> Linear(out_weight) -> out

    # Only save input and output; recompute intermediates
    save: [in]

# Temporary intermediate that's immediately consumed
module SwiGLU_MLP(d_model, d_ff):
  forward:
    graph:
      in -> Linear(up_weight) -> gate_up @memory(temporary)
      gate_up -> split([d_ff, d_ff]) -> (gate, up) @memory(temporary)
      (gate, up) -> swiglu() -> hidden
      hidden -> Linear(down_weight) -> out

# Pin embedding table (accessed every forward)
model LlamaModel():
  params:
    embedding: [vocab_size, d_model] @memory(pin)

# Offload optimizer states with priority
module AdamWState():
  params:
    exp_avg: [param_size] @memory(offload, priority=1)
    exp_avg_sq: [param_size] @memory(offload, priority=2)

# ZeRO-3 style weight streaming
model LargeModel():
  params:
    blocks[{i}].attention.qkv_weight: [...] @memory(stream, prefetch=true)
```

---

#### 7.2.2 @checkpoint

**Description:**
Marks a boundary for activation checkpointing (gradient checkpointing). Operations within a checkpoint boundary will have their activations recomputed during the backward pass instead of being stored.

**Syntax:**
```
@checkpoint
@checkpoint(granularity=level)
```

**Granularity Levels:**

| Level       | Description                               |
| ----------- | ----------------------------------------- |
| `full`      | Checkpoint entire subgraph; recompute all |
| `selective` | Only checkpoint marked tensors            |
| `none`      | Disable checkpointing for this region     |

**Specification:**
```
annotation @checkpoint:
  """
  Activation checkpointing boundary.

  Tensors produced within a checkpointed region are not saved
  during forward; they are recomputed from the checkpoint
  boundary inputs during backward.

  Trades compute for memory: ~33% more compute, ~60-80% less activation memory.
  """

  params:
    granularity: enum(full, selective, none) = full
    preserve: [string] = []     # Tensor names to save even within checkpoint

  applies_to: [subgraph, module_call]

  compiler_behavior:
    - Insert checkpoint boundary marker
    - During forward: save only boundary inputs
    - During backward: replay forward from boundary, then backward
```

**Example Usage:**
```
# Checkpoint each transformer block
block DenseTransformerBlock():
  forward:
    graph:
      # Everything in this block will be recomputed in backward
      in -> attention_sublayer() -> x1 @checkpoint
      x1 -> mlp_sublayer() -> out @checkpoint

# Checkpoint at model level
model Llama2():
  forward:
    graph:
      token_ids -> embedding() -> x0

      # Checkpoint every N layers
      x0 -> blocks[0:8]() -> x8 @checkpoint
      x8 -> blocks[8:16]() -> x16 @checkpoint
      x16 -> blocks[16:24]() -> x24 @checkpoint
      x24 -> blocks[24:32]() -> x32 @checkpoint

      x32 -> final_norm() -> lm_head() -> out

# Selective checkpointing: save attention output but recompute MLP
block SelectiveCheckpointBlock():
  forward:
    graph:
      in -> attention() -> attn_out    # Will be saved
      attn_out -> mlp() -> out @checkpoint(granularity=selective, preserve=[attn_out])

# Disable checkpointing for debugging
block DebugBlock():
  forward:
    graph:
      in -> attention() -> mlp() -> out @checkpoint(granularity=none)
```

---

#### 7.2.3 @lifetime

**Description:**
Explicitly specifies the lifetime scope of a tensor, enabling more aggressive memory reuse through buffer aliasing.

**Syntax:**
```
@lifetime(scope)
@lifetime(start=op_name, end=op_name)
```

**Specification:**
```
annotation @lifetime:
  """
  Explicit tensor lifetime specification.

  Helps the compiler perform buffer aliasing: tensors with
  non-overlapping lifetimes can share the same memory buffer.
  """

  params:
    scope: enum(operation, sublayer, block, layer, global) = operation
    start: string? = None       # Operation where lifetime begins
    end: string? = None         # Operation where lifetime ends

  applies_to: [tensor_output]

  compiler_behavior:
    - Use lifetime info for memory planning
    - Tensors with non-overlapping lifetimes may alias
```

**Example Usage:**
```
# Tensor only needed within attention sublayer
module Attention():
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv @lifetime(scope=sublayer)
      qkv -> split() -> (q, k, v)
      # qkv buffer can be reused after split completes

# Explicit start/end lifetime
module ExplicitLifetime():
  forward:
    graph:
      in -> op1() -> a @lifetime(start=op1, end=op3)
      a -> op2() -> b
      b -> op3() -> c
      # 'a' is freed after op3, even though it's not directly used there
```

---

### 7.3 Hook Annotations

Hook annotations define injection points where external code (like LoRA adapters, debugging probes, or custom kernels) can intercept and modify tensor values during forward and backward passes.

---

#### 7.3.1 @hook

**Description:**
Registers a named hook point that external code can attach to. Hooks are called with the tensor value and can observe or modify it.

**Syntax:**
```
@hook(HookPoint)
@hook(HookPoint, mode=mode)
```

**Standard Hook Points:**

| Hook Point               | Location                     | Typical Use              |
| ------------------------ | ---------------------------- | ------------------------ |
| `AfterEmbedding`         | After token embedding        | Input perturbation       |
| `AfterQKVProjection`     | After QKV linear             | LoRA on Q, K, V          |
| `AfterQKNorm`            | After QK normalization       | Debugging                |
| `BeforeAttention`        | Before flash attention       | Attention mask injection |
| `AfterAttention`         | After attention output       | Attention analysis       |
| `AfterAttnOutProjection` | After output projection      | LoRA on attention output |
| `BeforeResidualAdd`      | Before residual connection   | Residual analysis        |
| `AfterResidualAdd`       | After residual connection    | Debugging                |
| `AfterMLPUpProjection`   | After MLP up/gate projection | LoRA on MLP              |
| `AfterMLPActivation`     | After activation function    | Activation analysis      |
| `AfterMLPDownProjection` | After MLP down projection    | LoRA on MLP output       |
| `AfterRouterProjection`  | After MoE router             | Router debugging         |
| `BeforeExpertCompute`    | Before MoE experts           | Expert analysis          |
| `AfterExpertCompute`     | After MoE experts            | Expert output analysis   |
| `BeforeLMHead`           | Before language model head   | Feature extraction       |
| `AfterLMHead`            | After LM head (logits)       | Logit manipulation       |

**Modes:**

| Mode      | Description                          |
| --------- | ------------------------------------ |
| `observe` | Read-only access to tensor (default) |
| `modify`  | Can modify tensor in-place           |
| `replace` | Can return replacement tensor        |

**Specification:**
```
annotation @hook:
  """
  Define a hook injection point.

  Hooks allow external code to observe or modify tensors at specific
  points in the computation graph. Essential for:
  - LoRA/adapter injection
  - Debugging and visualization
  - Custom interventions (activation patching, etc.)
  """

  params:
    point: HookPoint             # Named hook point
    mode: enum(observe, modify, replace) = observe
    priority: int = 0            # Execution order when multiple hooks

  applies_to: [tensor_output]

  runtime_behavior:
    - If no hook registered: no-op (zero overhead)
    - If hook registered: call hook function with tensor
    - observe: hook receives tensor, returns nothing
    - modify: hook receives tensor, modifies in-place
    - replace: hook receives tensor, returns new tensor
```

**Example Usage:**
```
# Standard attention with hook points for LoRA
module HookableAttention(d_model, num_heads):
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv @hook(AfterQKVProjection, mode=modify)

      qkv -> split_reshape() -> (q, k, v)
      (q, k) -> qk_norm() -> (q_norm, k_norm) @hook(AfterQKNorm)
      (q_norm, k_norm) -> rope() -> (q_rope, k_rope)

      (q_rope, k_rope, v) -> flash_attention() -> (attn_out, lse) @hook(AfterAttention)

      attn_out -> reshape() -> attn_flat
      attn_flat -> Linear(out_weight) -> out @hook(AfterAttnOutProjection, mode=modify)

# MoE with router debugging hooks
module HookableMoELayer():
  forward:
    graph:
      in -> Linear(gate) -> router_logits @hook(AfterRouterProjection)
      router_logits -> moe_router() -> (weights, indices, aux_loss)
      (in, weights, indices) -> moe_experts() -> out @hook(AfterExpertCompute)

# Feature extraction hook before LM head
model InstrumentedLlama():
  forward:
    graph:
      ...
      xN -> rmsnorm(final_norm) -> xF @hook(BeforeLMHead, mode=observe)
      xF -> Linear(lm_head) -> logits @hook(AfterLMHead)
```

**Runtime Hook Registration (Python pseudo-code):**
```python
# Register a LoRA hook
def lora_qkv_hook(tensor, layer_idx, stream):
    # tensor: [B, T, qkv_dim]
    lora_delta = lora_A[layer_idx] @ lora_B[layer_idx]  # [B, T, qkv_dim]
    tensor += lora_delta  # In-place modification
    return None  # modify mode

model.register_hook(HookPoint.AfterQKVProjection, lora_qkv_hook)

# Register an observation hook for debugging
def debug_hook(tensor, layer_idx, stream):
    print(f"Layer {layer_idx}: mean={tensor.mean()}, std={tensor.std()}")
    return None

model.register_hook(HookPoint.AfterAttention, debug_hook, mode='observe')
```

---

#### 7.3.2 @adapter

**Description:**
Declares that an operation supports adapter injection (like LoRA, IA³, or other PEFT methods). This is a higher-level annotation that combines hook registration with adapter-specific metadata.

**Syntax:**
```
@adapter(type, params...)
```

**Adapter Types:**

| Type      | Description                                  | Parameters                            |
| --------- | -------------------------------------------- | ------------------------------------- |
| `lora`    | Low-Rank Adaptation                          | `rank`, `alpha`, `dropout`, `targets` |
| `ia3`     | Infused Adapter by Inhibiting and Amplifying | `targets`                             |
| `prefix`  | Prefix Tuning                                | `length`, `hidden_dim`                |
| `prompt`  | Prompt Tuning                                | `length`                              |
| `adapter` | Bottleneck Adapter                           | `bottleneck_dim`, `activation`        |

**Specification:**
```
annotation @adapter:
  """
  Declare adapter injection point with configuration.

  Higher-level than @hook: specifies adapter type and parameters.
  Compiler generates appropriate hook registration and weight allocation.
  """

  params:
    type: enum(lora, ia3, prefix, prompt, adapter)

    # LoRA-specific
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: [string] = []      # Which parts of fused projection (e.g., [q, k, v])

    # IA³-specific
    # (uses targets only)

    # Prefix/Prompt-specific
    length: int = 10
    hidden_dim: int? = None

    # Bottleneck adapter-specific
    bottleneck_dim: int = 64
    activation: enum(relu, gelu, silu) = relu

  applies_to: [tensor_output, linear_operation]

  compiler_behavior:
    - Generate adapter weight parameters
    - Register appropriate forward/backward hooks
    - Handle adapter weight initialization
```

**Example Usage:**
```
# LoRA on attention projections
module LoRAAttention(d_model, num_heads, lora_rank=16, lora_alpha=32):
  forward:
    graph:
      # LoRA on QKV (all three: q, k, v)
      in -> Linear(qkv_weight) -> qkv @adapter(lora, rank=lora_rank, alpha=lora_alpha, targets=[q, k, v])

      qkv -> attention_ops() -> attn_out

      # LoRA on output projection
      attn_out -> Linear(out_weight) -> out @adapter(lora, rank=lora_rank, alpha=lora_alpha)

# LoRA only on Q and V (common configuration)
module QVLoRAAttention():
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv @adapter(lora, rank=16, targets=[q, v])
      # K is not adapted

# LoRA on MLP
module LoRAMLP(d_model, d_ff, lora_rank=16):
  forward:
    graph:
      in -> Linear(up_weight) -> gate_up @adapter(lora, rank=lora_rank, targets=[gate, up])
      gate_up -> split_swiglu() -> hidden
      hidden -> Linear(down_weight) -> out @adapter(lora, rank=lora_rank)

# IA³ adaptation (learned scaling vectors)
module IA3Attention():
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv @adapter(ia3, targets=[k, v])
      qkv -> attention_ops() -> attn_out
      attn_out -> Linear(out_weight) -> out @adapter(ia3)

# Full model with adapter configuration
model LoRALlama(
  d_model, n_layers, ...,
  lora_rank=16,
  lora_alpha=32,
  lora_targets=["q", "k", "v", "o", "up", "down"]
):
  params:
    # Base weights (frozen during LoRA training)
    blocks[{i}].attention.qkv_weight: [...] @frozen
    blocks[{i}].attention.out_weight: [...] @frozen
    blocks[{i}].mlp.up_weight: [...] @frozen
    blocks[{i}].mlp.down_weight: [...] @frozen

    # LoRA weights (trainable)
    blocks[{i}].lora.qkv_A: [lora_rank, d_model] @trainable
    blocks[{i}].lora.qkv_B: [qkv_dim, lora_rank] @trainable
    # ... etc
```

---

#### 7.3.3 @probe

**Description:**
Inserts a debugging probe that captures tensor statistics or values during execution. Unlike hooks, probes are purely observational and have minimal overhead when disabled.

**Syntax:**
```
@probe(name)
@probe(name, capture=what)
```

**Capture Modes:**

| Mode         | Description                     | Overhead |
| ------------ | ------------------------------- | -------- |
| `stats`      | Mean, std, min, max, norm       | Low      |
| `histogram`  | Value distribution              | Medium   |
| `sample`     | Random subset of values         | Low      |
| `full`       | Complete tensor copy            | High     |
| `grad_stats` | Gradient statistics in backward | Low      |
| `grad_full`  | Full gradient capture           | High     |

**Specification:**
```
annotation @probe:
  """
  Insert debugging probe for tensor analysis.

  Probes capture tensor information for debugging, profiling,
  and training diagnostics. Can be globally enabled/disabled.
  """

  params:
    name: string                 # Probe identifier
    capture: enum(stats, histogram, sample, full, grad_stats, grad_full) = stats
    condition: string? = None    # Only capture when condition is true
    frequency: int = 1           # Capture every N iterations

  applies_to: [tensor_output]

  runtime_behavior:
    - When probes disabled: zero overhead (compiled out)
    - When enabled: async capture to avoid blocking compute
    - Results available via probe API
```

**Example Usage:**
```
# Monitor attention patterns
module ProbedAttention():
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv @probe("qkv_output", capture=stats)
      qkv -> attention() -> attn_out @probe("attention_output", capture=stats)
      attn_out -> Linear(out_weight) -> out @probe("attn_proj_output", capture=grad_stats)

# Conditional probing (only on NaN/Inf)
module DebugAttention():
  forward:
    graph:
      in -> attention() -> out @probe("attention", capture=full, condition="has_nan_or_inf(out)")

# Probe every 100 iterations
model ProbedModel():
  forward:
    graph:
      ...
      xN -> lm_head() -> logits @probe("logits", capture=histogram, frequency=100)
```

**Runtime Probe Access (Python pseudo-code):**
```python
# Enable probes
model.enable_probes()

# Run forward
logits = model(input_ids)

# Access probe data
probe_data = model.get_probe("attention_output")
print(f"Attention mean: {probe_data.mean}, std: {probe_data.std}")

# Disable for production
model.disable_probes()
```

---

### 7.4 Precision Annotations

Precision annotations control numerical precision for computation, storage, and communication. Essential for mixed-precision training and FP8/FP4 quantization.

---

#### 7.4.1 @dtype

**Description:**
Specifies the data type of a tensor output. The compiler will insert conversion operations if necessary.

**Syntax:**
```
@dtype(type)
```

**Supported Types:**

| Type       | Bits | Range     | Use Case                     |
| ---------- | ---- | --------- | ---------------------------- |
| `fp32`     | 32   | Full      | Master weights, accumulation |
| `fp16`     | 16   | ±65504    | Legacy mixed precision       |
| `bf16`     | 16   | ±3.4e38   | Default for training         |
| `fp8_e4m3` | 8    | ±448      | Forward activations          |
| `fp8_e5m2` | 8    | ±57344    | Backward gradients           |
| `fp4_e2m1` | 4    | ±6        | Blackwell+ inference         |
| `int8`     | 8    | -128..127 | Quantized inference          |
| `int4`     | 4    | -8..7     | Extreme quantization         |

**Specification:**
```
annotation @dtype:
  """
  Specify tensor data type.

  Compiler inserts dtype conversion if input type differs.
  Binding annotation: output will have exactly this dtype.
  """

  params:
    type: dtype

  applies_to: [tensor_output, parameter]

  compiler_behavior:
    - If input dtype matches: no-op
    - If input dtype differs: insert conversion kernel
    - For parameters: affects storage format
```

**Example Usage:**
```
# Force output to bf16 after FP8 matmul
module FP8Linear():
  forward:
    graph:
      # Matmul in FP8, output in BF16
      (in, weight) -> matmul() -> out @dtype(bf16)

# Store weights in FP8 for memory efficiency
module CompressedLinear():
  params:
    weight: [out_dim, in_dim] @dtype(fp8_e4m3)

  forward:
    graph:
      # Weight is FP8, dequantized during matmul
      (in, weight) -> matmul() -> out

# Ensure accumulation in FP32
module SafeReduction():
  forward:
    graph:
      in -> reduce_sum() -> out @dtype(fp32)
```

---

#### 7.4.2 @precision

**Description:**
Fine-grained control over numerical precision for an operation, specifying compute dtype, accumulation dtype, and output dtype independently.

**Syntax:**
```
@precision(compute=dtype, accumulate=dtype, output=dtype)
@precision(policy=name)
```

**Precision Policies:**

| Policy       | Compute  | Accumulate | Output   | Use Case                |
| ------------ | -------- | ---------- | -------- | ----------------------- |
| `fp32`       | fp32     | fp32       | fp32     | Maximum precision       |
| `bf16`       | bf16     | fp32       | bf16     | Standard training       |
| `fp16`       | fp16     | fp32       | fp16     | Legacy mixed precision  |
| `fp8_hybrid` | fp8_e4m3 | fp32       | bf16     | Forward activations     |
| `fp8_pure`   | fp8_e4m3 | fp32       | fp8_e4m3 | Aggressive quantization |
| `fp4`        | fp4_e2m1 | fp32       | bf16     | Blackwell inference     |

**Specification:**
```
annotation @precision:
  """
  Fine-grained precision control for an operation.

  Allows independent specification of:
  - compute: dtype for the actual computation
  - accumulate: dtype for intermediate accumulation (matmul inner products)
  - output: dtype for the result tensor

  Also supports named policies for common configurations.
  """

  params:
    compute: dtype? = None       # Computation dtype
    accumulate: dtype = fp32     # Accumulation dtype (always fp32 for correctness)
    output: dtype? = None        # Output dtype
    policy: string? = None       # Named policy (overrides individual settings)

  applies_to: [matmul, reduction, normalization]

  compiler_behavior:
    - Insert quantization before op if compute dtype < input dtype
    - Use specified accumulation in kernel
    - Insert conversion after op if output dtype differs
```

**Example Usage:**
```
# FP8 forward, BF16 output
module FP8Attention():
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv @precision(compute=fp8_e4m3, output=bf16)
      qkv -> attention() -> out @precision(policy=bf16)

# Different precision for forward vs backward (via recipe)
module HybridPrecisionLinear():
  forward:
    graph:
      in -> matmul(weight) -> out @precision(compute=fp8_e4m3, output=bf16)

  backward:
    graph:
      # Use E5M2 for gradients (larger range)
      d_out -> matmul(weight) -> d_in @precision(compute=fp8_e5m2, output=bf16)
      (in, d_out) -> matmul() -> d_weight @precision(compute=fp8_e5m2, output=fp32)

# Ensure loss computation in full precision
module CrossEntropyLoss():
  forward:
    graph:
      (logits, labels) -> cross_entropy() -> loss @precision(policy=fp32)

# FP4 inference on Blackwell
module FP4Linear():
  params:
    weight: [out_dim, in_dim] @dtype(fp4_e2m1)

  forward:
    graph:
      in -> matmul(weight) -> out @precision(compute=fp4_e2m1, output=bf16)
```

---

#### 7.4.3 @scaling

**Description:**
Controls quantization scaling strategy for FP8/FP4 operations. Specifies how scale factors are computed and applied.

Hardware Specifics: For extreme quantization formats (e.g., fp4_e2m1), the compiler enforces hardware-mandated 2D block structures when strategy=per_block is used.

**Syntax:**
```
@scaling(strategy, params...)
```

**Scaling Strategies:**

| Strategy      | Description                       | Use Case        |
| ------------- | --------------------------------- | --------------- |
| `per_tensor`  | Single scale for entire tensor    | Simple, fast    |
| `per_channel` | Scale per output channel          | Better accuracy |
| `per_token`   | Scale per sequence position       | Attention       |
| `per_block`   | 2D block scaling (e.g., 32x32)    | FP4, Blackwell  |
| `delayed`     | Use historical max, update async  | Training        |
| `dynamic`     | Compute scale from current tensor | Inference       |

**Specification:**
```
annotation @scaling:
  """
  Quantization scaling strategy for FP8/FP4 operations.

  Controls how scale factors are computed, stored, and applied.
  Critical for maintaining accuracy with low-precision formats.
  """

  params:
    strategy: enum(per_tensor, per_channel, per_token, per_block, delayed, dynamic)

    # Delayed scaling options
    history_len: int = 16        # Amax history length
    update_interval: int = 1     # Steps between scale updates

    # Block scaling options
    block_size: (int, int) = (32, 32)  # 2D block dimensions

    # Dynamic scaling options
    margin: float = 0.0          # Headroom margin (0 = use full range)

  applies_to: [matmul, quantize]

  compiler_behavior:
    - Allocate scale factor storage based on strategy
    - Insert scale computation kernels
    - Apply scaling in quantize/dequantize
```

**Example Usage:**
```
# Delayed scaling for training (standard FP8 recipe)
module FP8TrainingLinear():
  forward:
    graph:
      in -> quantize() -> in_q @scaling(delayed, history_len=16)
      (in_q, weight) -> matmul() -> out

# Per-channel scaling for better accuracy
module PerChannelFP8Linear():
  forward:
    graph:
      in -> matmul(weight) -> out @scaling(per_channel)

# Block scaling for FP4 (Blackwell)
module FP4BlockLinear():
  params:
    weight: [out_dim, in_dim] @dtype(fp4_e2m1) @scaling(per_block, block_size=(32, 32))

  forward:
    graph:
      in -> matmul(weight) -> out

# Dynamic scaling for inference
module DynamicFP8Linear():
  forward:
    graph:
      in -> matmul(weight) -> out @scaling(dynamic, margin=0.1)
```

---

#### 7.4.4 Explicit Cast Semantics

When dtypes differ between tensors in an operation, the compiler follows these rules:

**Cast Insertion Rules:**

| Scenario | Compiler Behavior |
|----------|------------------|
| Input dtype ≠ operation's compute dtype | Insert `quantize` before operation |
| Output dtype ≠ operation's output dtype | Insert `dequantize`/`cast` after operation |
| Two inputs with different dtypes | Promote to wider type, or error if incompatible |

**Cast Operations in IR:**

Casts are explicit nodes in the lowered IR, making debugging and profiling tractable:

```
# High-level DSL
in -> matmul(weight) -> out @precision(compute=fp8_e4m3, output=bf16)

# Lowered IR (conceptual)
in_bf16 -> quantize(fp8_e4m3, scale=s1) -> in_fp8
weight_bf16 -> quantize(fp8_e4m3, scale=s2) -> weight_fp8
(in_fp8, weight_fp8) -> matmul_fp8(out_scale=s3) -> out_fp8
out_fp8 -> dequantize(bf16, scale=s3) -> out_bf16
```

**Dtype Compatibility Matrix:**

| From → To | Implicit? | Notes |
|-----------|-----------|-------|
| bf16 → fp32 | Yes | Widening, lossless |
| fp32 → bf16 | Yes | Narrowing, automatic rounding |
| bf16 → fp8_e4m3 | No | Requires explicit `@precision` or `@scaling` |
| fp8_e4m3 → bf16 | Yes | Dequantization is automatic |
| bf16 → fp4_e2m1 | No | Requires explicit annotation + 2D block scaling |
| int8 → bf16 | No | Requires explicit dequant with scale |

**Precision Recipe Concept:**

A **Precision Recipe** bundles dtype decisions for an entire compilation:

```
# Recipe defined at compile time
recipe FP8HybridRecipe:
  forward_activation_dtype: fp8_e4m3
  backward_gradient_dtype: fp8_e5m2
  accumulation_dtype: fp32
  output_dtype: bf16
  weight_storage_dtype: bf16
  scaling_strategy: delayed
  amax_history_len: 16

# Usage at model level
model Llama2():
  @recipe(FP8HybridRecipe)    # Applies to entire model

  forward:
    ...
```

**Recipe Override:**
Instance-level `@precision` annotations override recipe defaults for specific operations.

---

### 7.5 Fusion Annotations

Fusion annotations guide the compiler's kernel fusion decisions. They can suggest fusion opportunities or prevent fusion for debugging.

---

#### 7.5.1 @fuse

**Description:**
Hints to the compiler that a sequence of operations should be fused into a single kernel for better performance.

**Syntax:**
```
@fuse(pattern_name)
@fuse(ops=[op1, op2, ...])
```

**Common Fusion Patterns:**

| Pattern             | Operations                 | Benefit                   |
| ------------------- | -------------------------- | ------------------------- |
| `residual_norm`     | Add + RMSNorm              | 1 kernel instead of 2     |
| `norm_linear`       | RMSNorm + Linear           | Reduce memory traffic     |
| `gated_mlp`         | Split + SwiGLU             | Avoid intermediate tensor |
| `attention_softmax` | Scale + Softmax + Dropout  | Flash attention style     |
| `bias_activation`   | Add(bias) + Activation     | Common pattern            |
| `linear_bias_act`   | Matmul + Bias + Activation | Fully fused               |

**Specification:**
```
annotation @fuse:
  """
  Suggest kernel fusion to compiler.

  Advisory annotation: compiler may ignore if fusion is not profitable
  or not implemented. Use @fuse_required for mandatory fusion.
  """

  params:
    pattern: string? = None      # Named fusion pattern
    ops: [string]? = None        # Explicit operation list
    scope: enum(local, global) = local  # Fusion scope

  applies_to: [operation_sequence]

  compiler_behavior:
    - Check if fusion pattern is supported
    - Estimate fusion benefit (memory vs compute)
    - Generate fused kernel if beneficial
```

**Example Usage:**
```
# Fused residual + normalization
module FusedResidualBlock():
  forward:
    graph:
      (residual, attn_out) -> add() -> rmsnorm(weight) -> out @fuse(residual_norm)

# Explicit operation list
module FusedMLP():
  forward:
    graph:
      in -> Linear(weight) -> add(bias) -> silu() -> out @fuse(ops=[matmul, add, silu])

# Fuse attention score computation
module FusedAttentionScores():
  forward:
    graph:
      (q, k) -> batched_matmul() -> scale(1/sqrt(d)) -> softmax() -> scores
        @fuse(attention_softmax)

# Auto-detect fusion opportunities
module AutoFuseBlock():
  forward:
    graph:
      # Compiler will automatically detect and fuse where beneficial
      in -> attention() -> mlp() -> out @fuse(scope=global)
```

---

#### 7.5.2 @nofuse

**Description:**
Prevents the compiler from fusing operations. Useful for debugging, profiling individual operations, or when fusion causes numerical issues.

**Syntax:**
```
@nofuse
@nofuse(reason=string)
```

**Specification:**
```
annotation @nofuse:
  """
  Prevent kernel fusion.

  Binding annotation: compiler must not fuse this operation with neighbors.
  Useful for debugging and profiling.
  """

  params:
    reason: string? = None       # Documentation for why fusion is disabled

  applies_to: [operation]

  compiler_behavior:
    - Mark operation as fusion barrier
    - Insert sync point if needed
```

**Example Usage:**
```
# Disable fusion for debugging NaN
module DebugBlock():
  forward:
    graph:
      in -> rmsnorm(weight) -> normed @nofuse(reason="checking for NaN source")
      normed -> Linear(w) -> out @nofuse

# Profile individual operations
module ProfiledAttention():
  forward:
    graph:
      in -> Linear(qkv_weight) -> qkv @nofuse(reason="profiling QKV projection")
      qkv -> rope() -> (q, k) @nofuse
      (q, k, v) -> flash_attention() -> out @nofuse(reason="profiling attention kernel")
```

---

#### 7.5.3 @kernel

**Description:**
Specifies an explicit kernel implementation to use for an operation, bypassing automatic kernel selection.

**Syntax:**
```
@kernel(name)
@kernel(name, params...)
```

**Specification:**
```
annotation @kernel:
  """
  Specify explicit kernel implementation.

  Bypasses automatic kernel selection. Useful for:
  - Using optimized vendor kernels
  - Testing alternative implementations
  - Debugging kernel issues
  """

  params:
    name: string                    # Kernel name/identifier
    autotune: bool = true           # Use CuBLASLt autotuning
    workspace: int? = None          # Workspace size override
    fallback: bool = true           # Allow fallback if kernel unavailable

  applies_to: [operation]

  compiler_behavior:
    - Use specified kernel instead of auto-selection
    - Error if kernel unavailable and fallback=false
```

**Example Usage:**
```
# Use cuDNN attention instead of custom
module CuDNNAttention():
  forward:
    graph:
      (q, k, v) -> flash_attention() -> out @kernel("cudnn_fmha_v2")

# Use specific CUTLASS GEMM
module CutlassLinear():
  forward:
    graph:
      in -> matmul(weight) -> out @kernel("cutlass_gemm_sm90_fp8")

# Test custom kernel with fallback
module CustomKernelTest():
  forward:
    graph:
      in -> my_custom_op() -> out @kernel("my_experimental_kernel", fallback=true)
```

---

### 7.6 Sharding Annotations

Sharding annotations control how tensors are distributed across multiple devices for tensor parallelism, expert parallelism, and pipeline parallelism.

#### 7.6.0 Distributed Execution Model

This section defines the consistent execution model for distributed training.

**Parallelism Topology:**

The DSL assumes a hierarchical parallelism topology:

```
# Default topology (can be configured at compile time)
topology:
  dp_size: 8          # Data parallel (outer)
  tp_size: 8          # Tensor parallel
  pp_size: 4          # Pipeline parallel
  ep_size: 8          # Expert parallel (for MoE)
  # Total GPUs = dp_size * tp_size * pp_size = 256

# Communication groups
groups:
  "dp": devices within same TP/PP position across DP dimension
  "tp": devices within same DP/PP position across TP dimension
  "pp": devices within same DP/TP position across PP dimension
  "ep": devices within same DP position across EP dimension
```

**Implicit vs Explicit Communication:**

| Sharding Annotation | Implicit Communication |
|---------------------|----------------------|
| `@shard(column)` + `@shard(row)` pair | All-reduce between column and row |
| `@shard(row)` alone | All-reduce after operation |
| `@shard(sequence)` | All-gather before, reduce-scatter after |
| `@shard(expert)` | All-to-all at MoE boundaries |
| `@partition(stage)` boundary | Send/recv between stages |

**When to use `@comm` explicitly:**
- Override inferred communication pattern
- Add extra communication (e.g., gradient synchronization)
- Control async behavior precisely
- Debug communication by making it visible

**Checkpoint and Pipeline Interaction:**

```
# Checkpoint boundaries must align with pipeline stages
model PipelineModel(pp_size=4):
  forward:
    graph:
      # VALID: checkpoint at stage boundary
      x0 -> blocks[0:8]() -> x8 @partition(0) @checkpoint

      # INVALID: checkpoint spans stages
      # x8 -> blocks[8:12]() @partition(1) -> x12 @checkpoint  # E025
      # x12 -> blocks[12:16]() @partition(2) -> x16

      # VALID: checkpoint per stage
      x8 -> blocks[8:16]() -> x16 @partition(1) @checkpoint
```

**Hook Execution in Distributed Context:**

Hooks execute on each rank independently:

| Hook Location | Execution |
|---------------|-----------|
| Before sharded op | Each rank executes on local shard |
| After all-reduce | Each rank has full tensor (replicated) |
| At pipeline boundary | Only sending/receiving rank executes |
| MoE expert hook | Each rank executes for local experts |

```
# Hook receives local shard
x -> Linear(weight) -> y @shard(column) @hook(AfterQKV, mode=observe)
# Hook on rank 0 sees y[:, :, 0:d//tp]
# Hook on rank 1 sees y[:, :, d//tp:2*d//tp]
# etc.
```

**Distributed Lowering Pass:**

The compiler's distributed lowering pass:

1. Analyzes sharding annotations on all tensors
2. Infers required communication from sharding mismatches
3. Inserts explicit collective ops in IR
4. Generates single-device code per rank with communication calls
5. Optimizes communication (overlap with compute, bucketing)

After lowering, the IR contains explicit `all_reduce`, `all_gather`, `send`, `recv` ops.

---

#### 7.6.1 @shard

**Description:**
Specifies how a parameter or activation tensor should be partitioned across devices. 

**Syntax:**
```
@shard(strategy)
@shard(strategy, params...)
```

**Sharding Strategies:**

| Strategy     | Description                       | Typical Use                |
| ------------ | --------------------------------- | -------------------------- |
| `column`     | Shard along columns (output dim)  | QKV, MLP up projections    |
| `row`        | Shard along rows (input dim)      | Attention output, MLP down |
| `replicated` | Full copy on each device          | Embeddings, layer norms    |
| `expert`     | Distribute experts across devices | MoE expert parallel        |
| `sequence`   | Shard along sequence dimension    | Sequence parallel          |
| `batch`      | Shard along batch dimension       | Data parallel              |
| `head`       | Shard along head dimension        | Attention head parallel    |

**Specification:**
```
annotation @shard:
  """
  Tensor sharding strategy for distributed execution.

  Defines how parameters and activations are partitioned across
  devices in tensor parallel, expert parallel, or pipeline parallel.
  """

  params:
    strategy: enum(column, row, replicated, expert, sequence, batch, head)

    # Parallelism sizes
    tp_size: int? = None         # Tensor parallel size
    ep_size: int? = None         # Expert parallel size
    pp_size: int? = None         # Pipeline parallel size

    # Communication options
    async_comm: bool = true      # Overlap communication with compute
    reduce_scatter: bool = true  # Use reduce-scatter instead of all-reduce

  applies_to: [parameter, tensor_output]

  runtime_behavior:
    - Allocate local shard of specified size
    - Insert communication ops (all-gather, reduce-scatter, etc.)
    - Handle shard-aware indexing
```

**Example Usage:**
```
# Standard tensor parallel attention
module TPAttention(d_model, num_heads, tp_size=8):
  let:
    local_heads = num_heads // tp_size
    local_head_dim = d_model // num_heads
    local_qkv_dim = local_heads * local_head_dim * 3

  params:
    # QKV: column parallel (each rank has subset of heads)
    qkv_weight: [local_qkv_dim, d_model] @shard(column, tp_size=tp_size)

    # Output: row parallel (each rank has partial output, need all-reduce)
    out_weight: [d_model, local_heads * local_head_dim] @shard(row, tp_size=tp_size)

  forward:
    graph:
      # No communication needed for QKV (column parallel)
      in -> Linear(qkv_weight) -> qkv_local

      qkv_local -> local_attention() -> attn_out_local

      # All-reduce after row-parallel output projection
      attn_out_local -> Linear(out_weight) -> out_partial
      out_partial -> all_reduce() -> out

# Expert parallel MoE
module EPMoELayer(d_model, num_experts, ep_size=8):
  let:
    local_experts = num_experts // ep_size

  params:
    gate: [num_experts, d_model] @shard(replicated)  # All ranks have full router
    expert_weights: [local_experts, ...] @shard(expert, ep_size=ep_size)

  forward:
    graph:
      # Route tokens to experts (all-to-all communication)
      in -> router(gate) -> (weights, indices)
      (in, weights, indices) -> expert_all_to_all() -> permuted

      # Local expert computation
      permuted -> local_experts(expert_weights) -> expert_out

      # All-to-all back to original positions
      expert_out -> expert_all_to_all_backward() -> out

# Sequence parallel (reduces activation memory)
module SPAttention(d_model, num_heads, sp_size=8):
  params:
    qkv_weight: [qkv_dim, d_model] @shard(column)
    out_weight: [d_model, d_model] @shard(row)

  forward:
    graph:
      # Input is sequence-sharded: [B, T/sp_size, d_model]
      in -> Linear(qkv_weight) -> qkv_local

      # All-gather sequence for attention
      qkv_local -> all_gather(dim=1) -> qkv_full
      qkv_full -> attention() -> attn_full

      # Reduce-scatter back to sequence-sharded
      attn_full -> reduce_scatter(dim=1) -> attn_local
      attn_local -> Linear(out_weight) -> out
```

---

#### 7.6.2 @partition

**Description:**
Defines pipeline parallelism stage boundaries. Operations within the same partition execute on the same device.

**Syntax:**
```
@partition(stage)
@partition(stage, params...)
```

**Specification:**
```
annotation @partition:
  """
  Pipeline parallelism stage assignment.

  Groups operations into pipeline stages that execute on different devices.
  Compiler inserts send/recv operations at stage boundaries.
  """

  params:
    stage: int                   # Pipeline stage index (0-based)
    micro_batch: bool = true     # Enable micro-batching
    recompute: bool = false      # Recompute activations within stage

  applies_to: [operation, block, layer]

  compiler_behavior:
    - Assign operations to specified stage
    - Insert send/recv at stage boundaries
    - Handle micro-batch scheduling
```

**Example Usage:**
```
# 4-stage pipeline parallel model
model PipelineLlama(n_layers=32, pp_size=4):
  let:
    layers_per_stage = n_layers // pp_size

  forward:
    graph:
      # Stage 0: embedding + first 8 layers
      token_ids -> embedding() -> x0 @partition(0)
      x0 -> blocks[0:8]() -> x8 @partition(0)

      # Stage 1: layers 8-16
      x8 -> blocks[8:16]() -> x16 @partition(1)

      # Stage 2: layers 16-24
      x16 -> blocks[16:24]() -> x24 @partition(2)

      # Stage 3: layers 24-32 + LM head
      x24 -> blocks[24:32]() -> x32 @partition(3)
      x32 -> rmsnorm() -> lm_head() -> logits @partition(3)
```

---

#### 7.6.3 @comm

**Description:**
Specifies communication operations explicitly or configures communication behavior.

**Syntax:**
```
@comm(op)
@comm(op, params...)
```

**Communication Operations:**

| Operation        | Description                  |
| ---------------- | ---------------------------- |
| `all_reduce`     | Sum across all ranks         |
| `all_gather`     | Gather shards from all ranks |
| `reduce_scatter` | Reduce then scatter result   |
| `all_to_all`     | Transpose across ranks       |
| `send`           | Point-to-point send          |
| `recv`           | Point-to-point receive       |
| `broadcast`      | Broadcast from one rank      |

**Specification:**
```
annotation @comm:
  """
  Explicit communication operation specification.

  Normally communication is inferred from sharding, but this
  annotation allows explicit control when needed.
  """

  params:
    op: enum(all_reduce, all_gather, reduce_scatter, all_to_all, send, recv, broadcast)
    group: string = "default"    # Communication group
    async: bool = true           # Async communication
    bucket_size: int? = None     # Gradient bucketing size

  applies_to: [tensor_output]

  runtime_behavior:
    - Insert specified communication operation
    - Use specified group (TP, DP, PP, etc.)
    - Handle async execution and synchronization
```

**Example Usage:**
```
# Explicit all-reduce after row-parallel matmul
module ExplicitTPLinear():
  forward:
    graph:
      in -> matmul(weight) -> partial @shard(row)
      partial -> out @comm(all_reduce, group="tp")

# Async all-reduce with compute overlap
module OverlappedTPLinear():
  forward:
    graph:
      in -> matmul(weight) -> partial
      partial -> out @comm(all_reduce, async=true)
      # Next operation can start while all-reduce in progress

# Gradient bucketing for efficiency
model BucketedGradSync():
  backward:
    graph:
      ... -> d_weight @comm(all_reduce, group="dp", bucket_size=25_000_000)
```

---

### 7.7 Debug and Profiling Annotations

Annotations for debugging, profiling, and development workflows.

---

#### 7.7.1 @assert

**Description:**
Inserts runtime assertions to catch errors during development.

**Syntax:**
```
@assert(condition)
@assert(condition, message=string)
```

**Specification:**
```
annotation @assert:
  """
  Runtime assertion for debugging.

  Checks condition and raises error with message if false.
  Can be globally disabled for production.
  """

  params:
    condition: string            # Condition expression
    message: string? = None      # Error message
    enabled: bool = true         # Can be disabled

  applies_to: [tensor_output]

  runtime_behavior:
    - If assertions enabled: check condition
    - If condition false: raise error with message
    - If assertions disabled: no-op
```

**Example Usage:**
```
# Check for NaN/Inf
module SafeAttention():
  forward:
    graph:
      (q, k, v) -> attention() -> out @assert(!has_nan(out), message="NaN in attention output")

# Check shape constraints
module ShapeCheckedLinear():
  forward:
    graph:
      in -> Linear(weight) -> out @assert(out.shape[-1] == d_model, message="Output dim mismatch")

# Check value ranges
module BoundedActivation():
  forward:
    graph:
      in -> sigmoid() -> out @assert(all(out >= 0) && all(out <= 1), message="Sigmoid out of range")
```

---

#### 7.7.2 @profile

**Description:**
Marks an operation for detailed profiling and timing.

**Syntax:**
```
@profile(name)
@profile(name, params...)
```

**Specification:**
```
annotation @profile:
  """
  Enable detailed profiling for an operation.

  Records timing, memory usage, and other metrics.
  Results available through profiling API.
  """

  params:
    name: string                 # Profile region name
    memory: bool = false         # Track memory allocations
    cuda_events: bool = true     # Use CUDA events for timing
    nvtx: bool = true            # Emit NVTX markers for Nsight

  applies_to: [operation, block, layer]

  runtime_behavior:
    - Insert timing start/end markers
    - Record metrics to profiling buffer
    - Emit NVTX markers if enabled
```

**Example Usage:**
```
# Profile attention vs MLP time
block ProfiledBlock():
  forward:
    graph:
      in -> attention() -> attn_out @profile("attention", memory=true)
      attn_out -> mlp() -> out @profile("mlp", memory=true)

# Profile entire forward pass
model ProfiledModel():
  forward:
    graph:
      token_ids -> forward_pass() -> logits @profile("forward", nvtx=true)
```

---

#### 7.7.3 @trace

**Description:**
Enables execution tracing for debugging control flow and operation ordering.

**Syntax:**
```
@trace
@trace(level=detail_level)
```

**Specification:**
```
annotation @trace:
  """
  Enable execution tracing.

  Logs operation execution with inputs/outputs for debugging.
  """

  params:
    level: enum(ops, tensors, values) = ops
    # ops: log operation names
    # tensors: log tensor shapes
    # values: log actual values (expensive)

  applies_to: [operation, block, graph]
```

**Example Usage:**
```
# Trace operation execution
module TracedAttention():
  forward:
    graph:
      in -> qkv_proj() -> attention() -> out_proj() -> out @trace(level=tensors)

# Output:
# [TRACE] qkv_proj: in=[4, 512, 4096] -> out=[4, 512, 12288]
# [TRACE] attention: qkv=[4, 512, 12288] -> out=[4, 512, 4096]
# [TRACE] out_proj: in=[4, 512, 4096] -> out=[4, 512, 4096]
```

---

### 7.8 Training Annotations

Annotations specific to training behavior.

---

#### 7.8.1 @frozen

**Description:**
Marks a parameter as frozen (not updated during training).

**Syntax:**
```
@frozen
```

**Specification:**
```
annotation @frozen:
  """
  Mark parameter as frozen (no gradient updates).

  Parameter will not accumulate gradients and is excluded from optimizer.
  Essential for transfer learning and adapter training.
  """

  applies_to: [parameter]

  compiler_behavior:
    - Exclude from gradient computation
    - Exclude from optimizer state allocation
    - May enable additional optimizations (weight caching, etc.)
```

**Example Usage:**
```
# Freeze base model for LoRA
model LoRALlama():
  params:
    embedding: [vocab_size, d_model] @frozen
    blocks[{i}].attention.qkv_weight: [...] @frozen
    blocks[{i}].attention.out_weight: [...] @frozen
    blocks[{i}].mlp.up_weight: [...] @frozen
    blocks[{i}].mlp.down_weight: [...] @frozen

    # LoRA weights are trainable
    blocks[{i}].lora_A: [rank, d_model] @trainable
    blocks[{i}].lora_B: [d_model, rank] @trainable
```

---

#### 7.8.2 @trainable

**Description:**
Explicitly marks a parameter as trainable (receives gradient updates).

**Syntax:**
```
@trainable
@trainable(params...)
```

**Specification:**
```
annotation @trainable:
  """
  Mark parameter as trainable.

  Parameter will accumulate gradients and be updated by optimizer.
  Default for parameters; use explicitly to override @frozen inheritance.
  """

  params:
    lr_scale: float = 1.0        # Learning rate multiplier
    wd_scale: float = 1.0        # Weight decay multiplier
    grad_clip: float? = None     # Per-parameter gradient clipping

  applies_to: [parameter]
```

**Example Usage:**
```
# Different learning rates for different parameters
model MultiLRModel():
  params:
    embedding: [...] @trainable(lr_scale=0.1)     # Lower LR for embeddings
    blocks[{i}].weights: [...] @trainable          # Default LR
    lm_head: [...] @trainable(lr_scale=2.0)        # Higher LR for head

# Disable weight decay for certain params
model NoWDNorms():
  params:
    blocks[{i}].ln1_weight: [...] @trainable(wd_scale=0.0)  # No weight decay on norms
    blocks[{i}].ln2_weight: [...] @trainable(wd_scale=0.0)
```

---

#### 7.8.3 @gradient

**Description:**
Controls gradient computation and accumulation behavior.

**Syntax:**
```
@gradient(mode)
@gradient(mode, params...)
```

**Specification:**
```
annotation @gradient:
  """
  Control gradient behavior for an operation or parameter.
  """

  params:
    mode: enum(compute, skip, accumulate, checkpoint)
    scale: float = 1.0           # Gradient scaling factor
    clip: float? = None          # Gradient clipping threshold

  applies_to: [operation, parameter, tensor_output]
```

**Example Usage:**
```
# Skip gradient for auxiliary loss
module MoEWithAuxLoss():
  forward:
    graph:
      in -> router() -> (weights, indices, aux_loss @gradient(skip))
      # aux_loss contributes to total loss but doesn't backprop through router

# Gradient scaling for stability
module ScaledGradient():
  backward:
    graph:
      d_out -> Linear_backward() -> d_in @gradient(scale=0.1)
```

---

## 8. Weight Mapping

### 8.1 HuggingFace Config Mapping

```
model Llama2(...):
  hf_config:
    architecture: "LlamaForCausalLM"
    config_class: "LlamaConfig"

    # Map DSL params to HF config fields
    param_mapping:
      d_model: hidden_size
      n_layers: num_hidden_layers
      num_heads: num_attention_heads
      num_kv_heads: num_key_value_heads
      d_ff: intermediate_size
      vocab_size: vocab_size
      max_seq: max_position_embeddings
      eps: rms_norm_eps
      rope_theta: rope_theta
```

### 8.2 Weight Mapping Syntax

```
model Llama2(...):
  hf_mapping:
    # Direct mapping
    embedding: "model.embed_tokens.weight"
    final_norm: "model.norm.weight"
    lm_head: "lm_head.weight"

    # Per-layer mapping with {layer} placeholder
    blocks[{layer}].ln1_weight: "model.layers.{layer}.input_layernorm.weight"
    blocks[{layer}].ln2_weight: "model.layers.{layer}.post_attention_layernorm.weight"

    # Fused weight mapping (multiple HF tensors -> one internal tensor)
    blocks[{layer}].attention.qkv_weight: fuse(
      "model.layers.{layer}.self_attn.q_proj.weight",  # rows [0, q_dim)
      "model.layers.{layer}.self_attn.k_proj.weight",  # rows [q_dim, q_dim+kv_dim)
      "model.layers.{layer}.self_attn.v_proj.weight",  # rows [q_dim+kv_dim, qkv_dim)
      dim=0
    )

    # Similarly for MLP gate+up fusion
    blocks[{layer}].mlp.up_weight: fuse(
      "model.layers.{layer}.mlp.gate_proj.weight",
      "model.layers.{layer}.mlp.up_proj.weight",
      dim=0
    )

    blocks[{layer}].mlp.down_weight: "model.layers.{layer}.mlp.down_proj.weight"
    blocks[{layer}].attention.out_weight: "model.layers.{layer}.self_attn.o_proj.weight"
```

### 8.3 Export Mapping

For saving back to HuggingFace format:

```
model Llama2(...):
  hf_export:
    # Split fused tensors back
    blocks[{layer}].attention.qkv_weight -> split(
      "model.layers.{layer}.self_attn.q_proj.weight": [0, q_dim],
      "model.layers.{layer}.self_attn.k_proj.weight": [q_dim, q_dim+kv_dim],
      "model.layers.{layer}.self_attn.v_proj.weight": [q_dim+kv_dim, qkv_dim],
      dim=0
    )
```

### 8.4 Optional Weights

```
hf_mapping:
  # Optional bias (may not exist in checkpoint)
  blocks[{layer}].attention.qkv_bias?: "model.layers.{layer}.self_attn.qkv_proj.bias"
```

### 8.5 Weight Transformations

```
hf_mapping:
  # Apply transformation during load
  blocks[{layer}].mamba.A_log: transform(
    "model.layers.{layer}.mamba.A_log",
    fn: negate_exp  # A = -exp(A_log)
  )
```

---

## 9. Compilation Model

### 9.1 Overview: Runtime Interpretation Architecture

The Module DSL uses **runtime interpretation** rather than C++ code generation. This design decision was made after analyzing the Surogate codebase architecture:

**Key Insight**: Surogate's CUDA kernels accept `Tensor&` arguments (not template-dependent types), enabling dynamic dispatch at runtime. The template-based model architecture organizes data structures, but kernel invocations themselves are runtime-bindable.

**Architecture:**
```
┌─────────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Parse (Lark)   │ --> │   Resolve   │ --> │   Lower     │ --> │ Runtime Execute │
│  Python AST     │     │   (Types)   │     │ (Graph IR)  │     │  (CUDA Kernels) │
└─────────────────┘     └─────────────┘     └─────────────┘     └─────────────────┘
```

### 9.2 Parser Implementation (Python/Lark)

The DSL parser is implemented in Python using the **Lark** parsing library.

#### 9.2.1 Why Lark?

After evaluating parsing libraries, Lark was selected over alternatives (e.g., DHParser) for the following reasons:

| Criteria | Lark | DHParser |
|----------|------|----------|
| **Parser Type** | LALR(1), Earley, CYK | PEG with packrat |
| **Performance** | O(n) with LALR | O(n) with memoization |
| **Grammar Format** | EBNF-like | EBNF |
| **AST Construction** | Excellent Transformer API | Manual visitor pattern |
| **Community** | Large, active (5k+ GitHub stars) | Niche/academic |
| **Indentation** | Built-in support (`%declare`, `%ignore`) | Manual handling |
| **Error Messages** | Good, customizable | Basic |
| **Ambiguity Handling** | Explicit with Earley | Left-recursion problematic |

**Decision**: Lark provides the best combination of performance (LALR for deterministic grammars), developer experience (Transformer API), and community support.

#### 9.2.2 Parser Structure

```python
from lark import Lark, Transformer, v_args

# Grammar definition (see Appendix A for full EBNF)
GRAMMAR = r'''
    start: (import_decl | module_decl | block_decl | model_decl | primitive_decl)*

    // Declarations
    module_decl: "module" NAME "(" [param_list] ")" ["extends" NAME] ":" _NL _INDENT module_body _DEDENT
    block_decl: "block" NAME "(" [param_list] ")" ["extends" NAME] ":" _NL _INDENT block_body _DEDENT
    model_decl: "model" NAME "(" [param_list] ")" ":" _NL _INDENT model_body _DEDENT

    // Graph expressions
    graph_stmt: source ("->" operation)* "->" dest [annotation]*
    operation: NAME "(" [arg_list] ")"
    annotation: "@" NAME ["(" [annotation_args] ")"]

    // Types and shapes
    tensor_type: "[" shape_dims "]" [":" dtype]
    shape_dims: dim ("," dim)*
    dim: NAME | INT | "*"

    // Standard tokens
    %import common.CNAME -> NAME
    %import common.INT
    %import common.FLOAT
    %import common.WS
    %declare _INDENT _DEDENT
    %ignore WS
'''

class ModuleDSLTransformer(Transformer):
    """Transforms parse tree into typed AST nodes."""

    @v_args(inline=True)
    def module_decl(self, name, params, extends, body):
        return ModuleNode(
            name=str(name),
            params=params or [],
            extends=str(extends) if extends else None,
            **body
        )

    @v_args(inline=True)
    def graph_stmt(self, source, *operations_and_dest):
        operations = operations_and_dest[:-1]
        dest = operations_and_dest[-1]
        return GraphStatement(source, list(operations), dest)

    @v_args(inline=True)
    def tensor_type(self, shape_dims, dtype=None):
        return TensorType(dims=shape_dims, dtype=dtype or "auto")

    # ... additional transformer methods

# Parser instantiation with indentation support
parser = Lark(
    GRAMMAR,
    parser='lalr',
    postlex=IndentationPostLexer(),  # Custom postlexer for indent/dedent
    propagate_positions=True,  # Track source locations for errors
)
```

#### 9.2.3 AST Node Types

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class ModuleNode:
    name: str
    params: List['ParamDecl']
    extends: Optional[str]
    param_decls: List['TensorDecl']
    forward: 'ForwardBlock'
    backward: Optional['BackwardBlock']
    annotations: List['Annotation']

@dataclass
class GraphStatement:
    source: 'TensorRef'
    operations: List['Operation']
    dest: 'TensorRef'
    annotations: List['Annotation'] = None

@dataclass
class Operation:
    name: str
    args: Dict[str, Any]

@dataclass
class TensorType:
    dims: List['Dim']  # Symbolic or concrete
    dtype: str

@dataclass
class Annotation:
    name: str
    params: Dict[str, Any]
```

### 9.3 Resolution Phase

- Import resolution
- Type inference and checking
- Symbolic dimension resolution
- Shape validation
- Inheritance resolution

### 9.4 Lowering Phase

- Graph canonicalization
- Fusion application
- Memory planning (lifetime analysis)
- Backward graph validation
- Saved tensor analysis

#### 9.4.1 Automatic Backward Derivation 
While the DSL prioritizes explicit backward passes, the compiler provides a Standard Backward Utility. If a backward: block is omitted for a chain of simple element-wise primitives (e.g., Add -> Mul -> SiLU), the compiler will automatically derive the gradient graph using internal primitive rules.

#### 9.4.2 Saved Tensor Analysis
The compiler validates that all tensors accessed via saved.name in the backward pass are either explicitly in the save: list or covered by an instance-level @memory(save) annotation.

### 9.5 Runtime Execution

Instead of generating C++ code, the DSL produces a **Graph IR** that is interpreted at runtime by a **Graph Executor**. The executor dispatches operations to existing Surogate CUDA kernels.

#### 9.5.1 Graph Executor Architecture

```cpp
class GraphExecutor {
public:
    GraphExecutor(const GraphIR& graph, const Recipe& recipe);

    // Execute forward pass
    void forward(
        DynamicWeights& weights,
        DynamicActivations& acts,
        DynamicInputs& inputs,
        cudaStream_t stream
    );

    // Execute backward pass
    void backward(
        DynamicWeights& weights,
        DynamicGradients& grads,
        DynamicActivations& saved,
        cudaStream_t stream
    );

private:
    std::vector<ScheduledOp> ops_;
    const Recipe& recipe_;

    void dispatch(const ScheduledOp& op, TensorMap& tensors, cudaStream_t stream);
};
```

#### 9.5.2 Dynamic Tensor Management

The runtime uses `TensorMap` (a `std::unordered_map<std::string, Tensor>`) instead of typed structs:

```cpp
using TensorMap = std::unordered_map<std::string, Tensor>;

struct DynamicWeights {
    TensorMap tensors;
    Tensor& get(const std::string& name) { return tensors.at(name); }
};

struct DynamicActivations {
    TensorMap tensors;
    void allocate(const std::string& name, const Shape& shape, DataType dtype);
};
```

#### 9.5.3 Operation Dispatch

Each DSL primitive maps to a kernel dispatch function:

```cpp
void GraphExecutor::dispatch(const ScheduledOp& op, TensorMap& t, cudaStream_t stream) {
    switch (op.kernel_type) {
        case KernelType::MATMUL:
            // Recipe controls precision (BF16, FP8, FP4)
            recipe_.forward_matmul(
                t[op.inputs[0]],   // input
                t[op.inputs[1]],   // weight
                t[op.outputs[0]],  // output
                stream
            );
            break;

        case KernelType::RMSNORM:
            rmsnorm_forward(
                t[op.outputs[0]],  // output
                t[op.inputs[0]],   // input
                t[op.inputs[1]],   // weight
                op.attrs.get<float>("eps"),
                stream
            );
            break;

        case KernelType::FLASH_ATTENTION:
            flash_attention_forward(
                t[op.outputs[0]],  // output
                t[op.outputs[1]],  // lse (for backward)
                t[op.inputs[0]],   // q
                t[op.inputs[1]],   // k
                t[op.inputs[2]],   // v
                op.attrs.get<float>("scale"),
                op.attrs.get<bool>("causal"),
                stream
            );
            break;

        case KernelType::SWIGLU:
            swiglu_forward(
                t[op.outputs[0]],
                t[op.inputs[0]],  // gate
                t[op.inputs[1]],  // up
                stream
            );
            break;

        // ... other kernel types
    }
}
```

#### 9.5.4 Hook Integration

The executor invokes hooks at designated points, enabling LoRA injection:

```cpp
void GraphExecutor::forward_with_hooks(
    DynamicWeights& weights,
    DynamicActivations& acts,
    ForwardHook hook,
    cudaStream_t stream
) {
    for (size_t i = 0; i < ops_.size(); ++i) {
        const auto& op = ops_[i];

        // Dispatch the kernel
        dispatch(op, acts.tensors, stream);

        // Invoke hook if this op has a hook point
        if (op.hook_point != HookPoint::NONE) {
            hook(op.layer_idx, stream, op.hook_point, &acts.tensors[op.outputs[0]]);
        }
    }
}
```

### 9.6 Compilation Outputs

The DSL compiler produces runtime artifacts (not C++ code):

1. **Graph IR**: Serialized computation graph (forward and backward)
2. **Schedule IR**: Lowered execution plan with buffer assignments
3. **Weight Schema**: JSON mapping for HuggingFace weight import/export
4. **Config Schema**: HuggingFace config ↔ ModelConfig translation rules
5. **Metadata**: Shape information, dtype requirements, hook points

### 9.7 Intermediate Representation (IR)

The compiler uses two internal IRs to represent the program:

#### 9.7.1 Graph IR (Functional)

The Graph IR represents the computation as a typed DAG without scheduling concerns.

**Conceptual Schema:**
```protobuf
message GraphIR {
  string name = 1;
  repeated TensorDecl inputs = 2;
  repeated TensorDecl outputs = 3;
  repeated Node nodes = 4;
  repeated Edge edges = 5;
}

message Node {
  string id = 1;
  string op_type = 2;              // "matmul", "rmsnorm", "flash_attention", etc.
  map<string, Value> attributes = 3;
  repeated Annotation annotations = 4;
}

message Edge {
  string source_node = 1;
  string source_output = 2;
  string dest_node = 3;
  string dest_input = 4;
  TensorType dtype = 5;
  repeated int64 shape = 6;
}

message TensorDecl {
  string name = 1;
  TensorType dtype = 2;
  repeated int64 shape = 3;        // Symbolic dims as negative integers
  MemoryMode memory_mode = 4;      // SAVE, RECOMPUTE, TEMPORARY, etc.
}

message Annotation {
  string type = 1;                 // "memory", "hook", "shard", etc.
  map<string, Value> params = 2;
}
```

**Graph IR Properties:**
- Nodes are pure functions (no side effects except hooks)
- Edges carry typed tensors with known shapes
- No explicit scheduling or buffer allocation
- Supports symbolic shapes (resolved later)

#### 9.7.2 Schedule IR (Imperative)

The Schedule IR represents a concrete execution plan with buffers, lifetimes, and communication.

**Conceptual Schema:**
```protobuf
message ScheduleIR {
  string name = 1;
  repeated BufferDecl buffers = 2;
  repeated ScheduledOp ops = 3;
  repeated SyncPoint sync_points = 4;
  ActivationLayout activation_layout = 5;
}

message BufferDecl {
  string id = 1;
  int64 size_bytes = 2;
  BufferKind kind = 3;             // ACTIVATION, GRADIENT, WEIGHT, WORKSPACE
  int64 lifetime_start = 4;        // First op that uses this buffer
  int64 lifetime_end = 5;          // Last op that uses this buffer
  repeated string aliases = 6;     // Other buffers that share this memory
}

message ScheduledOp {
  int64 order = 1;                 // Execution order
  string kernel = 2;               // Kernel name to invoke
  repeated BufferRef inputs = 3;
  repeated BufferRef outputs = 4;
  StreamAssignment stream = 5;     // CUDA stream
  optional RecomputeSegment recompute = 6;
}

message BufferRef {
  string buffer_id = 1;
  int64 offset = 2;
  int64 size = 3;
}

message SyncPoint {
  int64 after_op = 1;
  SyncKind kind = 2;               // STREAM_SYNC, NCCL_WAIT, etc.
  string comm_group = 3;
}

message ActivationLayout {
  // Compiler-generated struct for saved activations
  repeated FieldDecl fields = 1;
  int64 total_size = 2;
}
```

**Schedule IR Properties:**
- Explicit operation ordering
- Concrete buffer assignments with aliasing
- Communication ops are explicit nodes
- Recompute segments marked explicitly
- Sync points for multi-stream execution

#### 9.7.3 Lowering: Graph IR → Schedule IR

The lowering pass transforms functional Graph IR into imperative Schedule IR:

1. **Canonicalization**: Break chains into explicit ops with named intermediates
2. **Memory Planning**: Analyze lifetimes, compute buffer aliasing
3. **Recompute Planning**: Identify recompute frontiers, insert replay segments
4. **Communication Insertion**: Add collectives based on sharding
5. **Stream Assignment**: Assign ops to CUDA streams for overlap
6. **Buffer Allocation**: Assign concrete offsets in activation struct

**Example Lowering:**
```
# Graph IR (conceptual)
Node: qkv_proj {op: matmul, inputs: [in, qkv_weight], outputs: [qkv]}
Node: rope {op: rope, inputs: [q, k, freqs], outputs: [q_rope, k_rope]}
Node: attention {op: flash_attention, inputs: [q_rope, k_rope, v], outputs: [attn_out, lse]}

# Schedule IR (conceptual)
Op[0]: matmul(in, qkv_weight) -> buf[0]   # qkv
Op[1]: split(buf[0]) -> buf[1], buf[2], buf[3]  # q, k, v
Op[2]: rope(buf[1], buf[2], freqs) -> buf[4], buf[5]  # q_rope, k_rope
Op[3]: flash_attention(buf[4], buf[5], buf[3]) -> buf[6], buf[7]  # attn_out, lse

Buffers:
  buf[0]: 48KB, lifetime=[0,1], aliases=[]
  buf[1]: 16KB, lifetime=[1,2], aliases=[buf[0]:0]  # View into buf[0]
  buf[2]: 16KB, lifetime=[1,2], aliases=[buf[0]:16KB]
  buf[3]: 16KB, lifetime=[1,3], aliases=[buf[0]:32KB]
  ...
```

### 9.8 Verification and Validation

The compiler includes verification passes for correctness:

#### 9.8.1 Autograd Oracle Check

For development and testing, the compiler can verify user-defined backward graphs:

```
# Enable oracle check mode
compile --verify-backward model.module

# Compiler behavior:
# 1. Generate small random inputs
# 2. Run forward to compute outputs
# 3. Run user backward to compute gradients
# 4. Run numerical gradient check (finite differences)
# 5. Compare and report max relative error
```

**Configuration:**
```
verify_backward:
  epsilon: 1e-4              # Finite difference step
  tolerance: 1e-3            # Max allowed relative error
  shapes: {B: 2, T: 8, C: 64}  # Test shapes (small for speed)
```

#### 9.8.2 Shape and Dtype Checks

The compiler validates:

1. **Shape consistency**: All ops receive correctly-shaped inputs
2. **Dtype compatibility**: No implicit narrowing conversions
3. **Gradient completeness**: All trainable params have gradients
4. **Save coverage**: All `saved.x` references are satisfied

#### 9.8.3 Gradient Invariants

Primitives can declare invariants that the compiler checks:

```
primitive softmax:
  ...
  backward:
    in: (d_out, out)         # Uses output, not input
    ...

  invariants:
    - backward_uses: [out]   # Validate 'out' is in save list
```

---

## 10. Standard Library

### 10.1 Module Organization

```
std/
├── primitives/          # Built-in primitives
│   ├── matmul.module
│   ├── rmsnorm.module
│   ├── attention.module
│   └── ...
├── modules/             # Common building blocks
│   ├── linear.module
│   ├── mlp.module
│   ├── attention/
│   │   ├── mha.module
│   │   ├── gqa.module
│   │   └── mqa.module
│   └── moe/
│       ├── router.module
│       └── expert.module
├── blocks/              # Transformer blocks
│   ├── dense.module
│   ├── parallel.module
│   ├── moe.module
│   └── mamba.module
└── models/              # Reference architectures
    ├── llama.module
    ├── qwen.module
    ├── mistral.module
    └── mixtral.module
```

### 10.2 Standard Modules

#### std.modules.Linear
```
module Linear(in_dim: int, out_dim: int, bias: bool = false):
  params:
    weight: [out_dim, in_dim]
    bias: [out_dim] if bias

  forward:
    in: [*, in_dim]
    out: [*, out_dim]
    graph:
      in -> matmul(weight, transpose=TN) -> y
      if bias:
        (y, bias) -> add() -> out
      else:
        y -> out
    save: [in]

  backward:
    d_out: [*, out_dim]
    d_in: [*, in_dim]
    graph:
      (d_out, weight) -> matmul(transpose=NN) -> d_in
      (saved.in, d_out) -> matmul(transpose=NT, accumulate=true) -> d_weight
      if bias:
        d_out -> reduce_sum(dims=[0]) -> d_bias
```

#### std.modules.SwiGLU_MLP
```
module SwiGLU_MLP(d_model: int, d_ff: int):
  params:
    up_weight: [2 * d_ff, d_model]
    down_weight: [d_model, d_ff]

  forward:
    in: [B, T, d_model]
    out: [B, T, d_model]
    graph:
      in -> Linear(up_weight) -> gate_up
      gate_up -> split([d_ff, d_ff], dim=-1) -> (gate, up)
      (gate, up) -> swiglu() -> hidden
      hidden -> Linear(down_weight) -> out
    save: [in]
    recompute: [gate_up, gate, up, hidden]

  backward:
    d_out: [B, T, d_model]
    d_in: [B, T, d_model]
    graph:
      recompute:
        saved.in -> Linear(up_weight) -> gate_up
        gate_up -> split([d_ff, d_ff], dim=-1) -> (gate, up)
        (gate, up) -> swiglu() -> hidden

      (d_out, down_weight) -> matmul(transpose=NN) -> d_hidden
      (hidden, d_out) -> matmul(transpose=NT, accumulate=true) -> d_down_weight

      (d_hidden, gate, up) -> swiglu_backward() -> (d_gate, d_up)
      (d_gate, d_up) -> concat(dim=-1) -> d_gate_up

      (d_gate_up, up_weight) -> matmul(transpose=NN) -> d_in
      (saved.in, d_gate_up) -> matmul(transpose=NT, accumulate=true) -> d_up_weight
```

### 10.3 Standard Blocks

#### std.blocks.DenseTransformerBlock
```
block DenseTransformerBlock(
  d_model: int,
  num_heads: int,
  num_kv_heads: int,
  d_ff: int,
  max_seq: int,
  eps: float = 1e-6,
  use_qk_norm: bool = false
):
  params:
    ln1: RMSNormParams(d_model)
    attention: CausalSelfAttention(d_model, num_heads, num_kv_heads, max_seq, use_qk_norm)
    ln2: RMSNormParams(d_model)
    mlp: SwiGLU_MLP(d_model, d_ff)

  forward:
    inputs:
      x: [B, T, d_model]           # Previous layer's MLP output
      residual: [B, T, d_model]    # Running residual
    outputs:
      out: [B, T, d_model]         # This layer's MLP output
      residual_out: [B, T, d_model] # Updated residual

    graph:
      (residual, x) -> fused_residual_rmsnorm(ln1.weight, eps) -> (residual_mid, ln1_out)
      ln1_out -> attention -> attn_out
        @hook(AfterAttnOutProjection)
        @adapter(lora)

      (residual_mid, attn_out) -> fused_residual_rmsnorm(ln2.weight, eps) -> (residual_out, ln2_out)
      ln2_out -> mlp -> out
        @hook(AfterMLPDownProjection)
        @adapter(lora)

    save: [residual, ln1_out, attn_out, residual_mid, ln2_out]

  backward:
    inputs:
      d_out: [B, T, d_model]
      d_residual: [B, T, d_model]
    outputs:
      d_x: [B, T, d_model]
      d_residual_out: [B, T, d_model]

    graph:
      (d_out, saved.ln2_out, mlp) -> mlp.backward() -> (d_ln2_out, d_mlp_params)

      (d_ln2_out, d_residual, saved.residual_mid, ln2.weight)
        -> fused_residual_rmsnorm_backward() -> (d_residual_mid, d_attn_out, d_ln2_weight)

      (d_attn_out, saved.ln1_out, attention) -> attention.backward() -> (d_ln1_out, d_attn_params)

      (d_ln1_out, d_residual_mid, saved.residual, ln1.weight)
        -> fused_residual_rmsnorm_backward() -> (d_residual_out, d_x, d_ln1_weight)
```

---

## 11. Examples

### 11.1 Complete Llama-2 Definition

```
"""
Llama-2 architecture implementation.
"""

import std.primitives
import std.modules.{Linear, SwiGLU_MLP}
import std.modules.attention.GQA as CausalSelfAttention

model Llama2(
  vocab_size: int = 32000,
  d_model: int = 4096,
  n_layers: int = 32,
  num_heads: int = 32,
  num_kv_heads: int = 32,
  d_ff: int = 11008,
  max_seq: int = 4096,
  eps: float = 1e-5,
  rope_theta: float = 10000.0
):
  """
  Llama-2 language model.

  Supports 7B, 13B, 70B configurations via parameter changes.
  """

  let:
    d_head = d_model // num_heads

  params:
    embedding: [vocab_size, d_model]
    blocks: [n_layers] × DenseTransformerBlock(
      d_model, num_heads, num_kv_heads, d_ff, max_seq, eps
    )
    final_norm: [d_model]
    lm_head: tied_to(embedding)

  forward:
    in: [B, T]              # int32 token IDs
    out: [B, T, vocab_size] # float logits

    graph:
      in -> embedding(embedding) -> x0

      # Initialize residual to zeros for first layer
      zeros([B, T, d_model]) -> residual0

      # Stack all transformer blocks
      (x0, residual0) -> StackedBlocks(blocks, n_layers) -> (xN, residualN)

      # Final norm on residual + last output
      (residualN, xN) -> fused_residual_rmsnorm(final_norm, eps) -> (_, xF)

      # LM head (tied weights)
      xF -> Linear(lm_head) -> out

    save: [x0, per_layer_saves..., xF]

  backward:
    d_out: [B, T, vocab_size]

    graph:
      # LM head backward
      (d_out, lm_head) -> matmul(transpose=NN) -> d_xF
      (saved.xF, d_out) -> matmul(transpose=NT, accumulate=true) -> d_lm_head

      # Final norm backward
      (d_xF, saved.residualN, final_norm) -> rmsnorm_backward() -> (d_residualN, d_xN, d_final_norm)

      # Backward through all blocks (reverse order)
      (d_xN, d_residualN) -> StackedBlocksBackward(blocks, n_layers) -> (d_x0, d_residual0)

      # Embedding backward
      (d_x0, saved.token_ids) -> embedding_backward() -> d_embedding

  # HuggingFace compatibility
  hf_config:
    architecture: "LlamaForCausalLM"
    config_class: "LlamaConfig"
    param_mapping:
      d_model: hidden_size
      n_layers: num_hidden_layers
      num_heads: num_attention_heads
      num_kv_heads: num_key_value_heads
      d_ff: intermediate_size
      vocab_size: vocab_size
      max_seq: max_position_embeddings
      eps: rms_norm_eps
      rope_theta: rope_theta

  hf_mapping:
    embedding: "model.embed_tokens.weight"
    final_norm: "model.norm.weight"
    lm_head: "lm_head.weight"

    blocks[{i}].ln1.weight: "model.layers.{i}.input_layernorm.weight"
    blocks[{i}].attention.qkv_weight: fuse(
      "model.layers.{i}.self_attn.q_proj.weight",
      "model.layers.{i}.self_attn.k_proj.weight",
      "model.layers.{i}.self_attn.v_proj.weight",
      dim=0
    )
    blocks[{i}].attention.out_weight: "model.layers.{i}.self_attn.o_proj.weight"
    blocks[{i}].ln2.weight: "model.layers.{i}.post_attention_layernorm.weight"
    blocks[{i}].mlp.up_weight: fuse(
      "model.layers.{i}.mlp.gate_proj.weight",
      "model.layers.{i}.mlp.up_proj.weight",
      dim=0
    )
    blocks[{i}].mlp.down_weight: "model.layers.{i}.mlp.down_proj.weight"
```

### 11.2 MoE Block Example

```
block MoETransformerBlock(
  d_model: int,
  num_heads: int,
  num_kv_heads: int,
  d_ff: int,
  num_experts: int,
  top_k: int,
  max_seq: int,
  eps: float = 1e-6
):
  """Mixture-of-Experts transformer block (Mixtral/Qwen3-MoE style)."""

  params:
    ln1: RMSNormParams(d_model)
    attention: CausalSelfAttention(d_model, num_heads, num_kv_heads, max_seq)
    ln2: RMSNormParams(d_model)
    router: [num_experts, d_model]
    experts: [num_experts] × ExpertMLP(d_model, d_ff)

  forward:
    inputs:
      x: [B, T, d_model]
      residual: [B, T, d_model]
    outputs:
      out: [B, T, d_model]
      residual_out: [B, T, d_model]

    graph:
      # Attention sublayer (same as dense)
      (residual, x) -> fused_residual_rmsnorm(ln1.weight, eps) -> (residual_mid, ln1_out)
      ln1_out -> attention -> attn_out

      # MoE sublayer
      (residual_mid, attn_out) -> fused_residual_rmsnorm(ln2.weight, eps) -> (residual_out, ln2_out)

      # Routing
      ln2_out -> view([B*T, d_model]) -> flat_input
      (flat_input, router) -> moe_router(top_k, normalize=true) -> (weights, indices, aux_loss)
        @hook(AfterRouterProjection)

      # Permute to expert order
      (flat_input, indices) -> moe_permute() -> permuted_input

      # Expert computation (grouped GEMM)
      (permuted_input, experts, indices) -> moe_experts() -> expert_outputs

      # Unpermute and combine
      (expert_outputs, weights, indices) -> moe_unpermute() -> combined
      combined -> view([B, T, d_model]) -> out

    save: [residual, ln1_out, attn_out, residual_mid, ln2_out, flat_input, weights, indices, permuted_input, expert_outputs]

  backward:
    # Similar structure, reverse order
    ...
```

### 11.3 Mamba Block Example

```
block MambaBlock(
  d_model: int,
  d_state: int = 16,
  d_conv: int = 4,
  expand: int = 2,
  n_groups: int = 1,
  eps: float = 1e-6
):
  """Mamba SSM block for hybrid architectures."""

  let:
    d_inner = d_model * expand
    d_conv_dim = d_inner + 2 * n_groups * d_state

  params:
    ln: RMSNormParams(d_model)
    in_proj: [d_inner + d_conv_dim + n_groups, d_model]
    conv1d_weight: [d_conv_dim, 1, d_conv]
    conv1d_bias: [d_conv_dim]?
    A_log: [d_inner, d_state]
    D: [d_inner]
    dt_bias: [d_inner]
    norm_weight: [d_inner]
    out_proj: [d_model, d_inner]

  forward:
    inputs:
      x: [B, T, d_model]
      residual: [B, T, d_model]
      ssm_state: [B, d_inner, d_state]?   # Recurrent state
      conv_state: [B, d_conv_dim, d_conv]? # Conv state
    outputs:
      out: [B, T, d_model]
      residual_out: [B, T, d_model]
      ssm_state_out: [B, d_inner, d_state]
      conv_state_out: [B, d_conv_dim, d_conv]

    graph:
      (residual, x) -> fused_residual_rmsnorm(ln.weight, eps) -> (residual_out, ln_out)

      # Input projection
      ln_out -> Linear(in_proj) -> proj
      proj -> mamba_split_proj(d_inner, d_conv_dim) -> (gate, conv_in, delta)

      # Causal conv1d
      (conv_in, conv1d_weight, conv1d_bias, conv_state)
        -> mamba_conv1d(d_conv, silu=true) -> (conv_out, conv_state_out)

      # Split conv output
      conv_out -> mamba_split_conv(d_inner, n_groups, d_state) -> (u, B_ssm, C_ssm)

      # Selective scan
      A_log -> exp_neg() -> A
      (u, delta, A, B_ssm, C_ssm, D, dt_bias, ssm_state)
        -> mamba_selective_scan() -> (scan_out, ssm_state_out)

      # Gate and normalize
      (gate, scan_out) -> silu_mul() -> gated
      gated -> group_rmsnorm(norm_weight, n_groups, eps) -> normed

      # Output projection
      normed -> Linear(out_proj) -> out

    save: [ln_out, proj, gate, conv_in, conv_out, u, B_ssm, C_ssm, delta, scan_out, gated]

  backward:
    # Mamba backward is complex - explicit graph required
    ...
```

### 11.4 Constraint and Versioning Example

```
import std.primitives.matmul.v1

module SafeLinear(in_dim: int, out_dim: int):
  let:
    constraint:
      in_dim > 0, "Input dimension must be positive"
      out_dim % 8 == 0, "FP8 requires output dim multiple of 8"
  
  params:
    weight: [out_dim, in_dim] @dtype(fp8_e4m3)
    
  forward:
    in: [B, in_dim]
    out: [B, out_dim]
    graph:
      (in, weight) -> matmul() -> out @precision(accumulate=fp32)
```

---

## Appendix

### A. Grammar (EBNF)

```ebnf
(* Top-level *)
program        = { import_decl | module_decl | block_decl | model_decl | primitive_decl | recipe_decl } ;

(* Imports *)
import_decl    = "import" module_path [ "as" identifier ]
               | "from" module_path "import" import_list ;
module_path    = identifier { "." identifier } [ ".v" integer ] ;  (* Versioned imports *)
import_list    = import_item { "," import_item } ;
import_item    = identifier [ "as" identifier ] ;

(* Module declaration *)
module_decl    = ["abstract"] "module" identifier "(" param_list ")"
                 [ "extends" identifier ] ":" module_body ;
module_body    = [ doc_string ]
                 [ let_section ]
                 [ params_section ]
                 forward_section
                 [ backward_section ] ;   (* Optional - can be auto-derived *)

(* Block declaration *)
block_decl     = "block" identifier "(" param_list ")" ":" block_body ;
block_body     = [ doc_string ]
                 [ let_section ]
                 params_section
                 ( pattern_section | ( forward_section [ backward_section ] ) ) ;

(* Model declaration *)
model_decl     = "model" identifier "(" param_list ")" ":" model_body ;
model_body     = [ doc_string ]
                 [ let_section ]
                 params_section
                 forward_section
                 [ backward_section ]
                 [ hf_config_section ]
                 [ hf_mapping_section ] ;

(* Primitive declaration *)
primitive_decl = "primitive" identifier ":" primitive_body ;
primitive_body = [ doc_string ]
                 [ params_section ]
                 primitive_forward
                 primitive_backward
                 [ save_section ]
                 [ impl_section ]
                 [ invariants_section ] ;

(* Primitive-specific forward/backward (simpler than module) *)
primitive_forward  = "forward:" INDENT io_signature DEDENT ;
primitive_backward = "backward:" INDENT backward_spec DEDENT ;

io_signature   = "in:" io_type_spec
                 "out:" io_type_spec ;

io_type_spec   = type_expr                            (* Single tensor *)
               | "(" named_tensor { "," named_tensor } ")" ;  (* Named tuple *)

named_tensor   = identifier ":" type_expr ;

backward_spec  = "in:" io_type_spec
                 "out:" io_type_spec
               | backward_expr_list ;   (* Shorthand: d_A = expr, d_B = expr *)

backward_expr_list = { identifier "=" expression } ;

impl_section   = "impl:" INDENT
                   [ "forward:" kernel_ref ]
                   [ "backward:" kernel_ref ]
                 DEDENT ;

kernel_ref     = module_path | "pointer_arithmetic" | "metadata_only" ;

invariants_section = "invariants:" INDENT { invariant_stmt } DEDENT ;
invariant_stmt = "-" identifier ":" tensor_list ;

(* Recipe declaration *)
recipe_decl    = "recipe" identifier ":" INDENT recipe_body DEDENT ;
recipe_body    = { identifier ":" expression } ;

(* Sections *)
let_section    = "let:" INDENT { let_binding | constraint_section } DEDENT ;
let_binding    = identifier "=" expression ;
constraint_section = "constraint:" INDENT { constraint_stmt } DEDENT ;
constraint_stmt = const_expr "," string ;   (* condition, error message *)

params_section = "params:" INDENT { param_decl } DEDENT ;
param_decl     = identifier ":" type_expr [ "if" const_expr ] { annotation } ;

forward_section = "forward:" INDENT forward_body DEDENT ;
forward_body   = [ "inputs:" input_list ]
                 [ "in:" type_expr ]
                 [ "outputs:" output_list ]
                 [ "out:" type_expr ]
                 "graph:" graph_body
                 [ save_section ]
                 [ recompute_section ] ;

input_list     = identifier ":" type_expr { "," identifier ":" type_expr } ;
output_list    = identifier ":" type_expr { "," identifier ":" type_expr } ;

save_section   = "save:" tensor_list ;
recompute_section = "recompute:" tensor_list ;

backward_section = "backward:" INDENT backward_body DEDENT ;
backward_body  = [ "d_" identifier ":" type_expr { "," "d_" identifier ":" type_expr } ]  (* inputs *)
                 [ "d_" identifier ":" type_expr { "," "d_" identifier ":" type_expr } ]  (* outputs *)
                 "graph:" graph_body ;

pattern_section = "pattern:" pattern_type INDENT pattern_body DEDENT ;
pattern_type   = "sequential_residual" | "parallel_residual" | identifier ;
pattern_body   = { pattern_stmt } ;
pattern_stmt   = "sublayers:" sublayer_list ;
sublayer_list  = { "-" "(" identifier { "," identifier } ")" } ;

(* Graph *)
graph_body     = INDENT { graph_statement } DEDENT ;
graph_statement = data_flow | conditional | recompute_block ;
data_flow      = source "->" operation { "->" operation } "->" destination { annotation } ;
source         = identifier | "(" identifier_list ")" | indexing_expr ;
destination    = identifier | "(" identifier_list ")" ;
indexing_expr  = identifier "[" slice_spec { "," slice_spec } "]" ;
slice_spec     = const_expr | const_expr ":" const_expr | ":" | "..." ;
operation      = identifier [ "(" arg_list ")" ] ;
conditional    = "if" const_expr ":" graph_body [ "else:" graph_body ] ;  (* const_expr guard *)
recompute_block = "recompute:" graph_body ;

(* Types *)
type_expr      = tensor_type | tuple_type | optional_type | array_type | identifier ;
tensor_type    = "[" shape_list [ "," dtype ] "]" ;
tuple_type     = "(" type_expr { "," type_expr } ")" ;
optional_type  = type_expr "?" ;
array_type     = "[" const_expr "]" ( "×" | "x" ) type_expr   (* Both × and x allowed *)
               | "Array" "<" const_expr "," type_expr ">" ;   (* Alternative syntax *)
shape_list     = shape_dim { "," shape_dim } ;
shape_dim      = "*" | const_expr ;

dtype          = "bf16" | "fp32" | "fp16" | "fp8_e4m3" | "fp8_e5m2"
               | "fp4_e2m1" | "int8" | "int32" ;

(* Expressions - two classes *)
expression     = const_expr | runtime_expr ;

(* ConstExpr: compile-time determinable *)
const_expr     = const_or_expr ;
const_or_expr  = const_and_expr { "or" const_and_expr } ;
const_and_expr = const_not_expr { "and" const_not_expr } ;
const_not_expr = [ "not" ] const_comparison ;
const_comparison = const_arith { comp_op const_arith } ;
const_arith    = const_term { ("+" | "-") const_term } ;
const_term     = const_factor { ("*" | "/" | "//" | "%") const_factor } ;
const_factor   = [ "-" ] const_power ;
const_power    = const_atom [ "**" const_factor ] ;
const_atom     = identifier           (* Must be module param or let-bound *)
               | literal
               | "(" const_expr ")"
               | const_call_expr ;
const_call_expr = pure_intrinsic "(" [ const_arg_list ] ")" ;
pure_intrinsic = "sqrt" | "ceil_div" | "min" | "max" | "abs" ;
const_arg_list = const_expr { "," const_expr } ;

(* RuntimeExpr: tensor values *)
runtime_expr   = identifier ;  (* Tensor names in graph context *)

comp_op        = "==" | "!=" | "<" | ">" | "<=" | ">=" ;

(* Annotations *)
annotation     = "@" identifier [ "(" arg_list ")" ] ;
arg_list       = arg { "," arg } ;
arg            = [ identifier "=" ] const_expr ;  (* Annotation args are ConstExpr *)

(* Literals *)
literal        = integer | float | string | boolean | "None" ;
integer        = digit { digit } ;
float          = digit { digit } "." digit { digit } [ ("e" | "E") ["+" | "-"] digit { digit } ] ;
string         = '"' { char | escape } '"' ;
boolean        = "true" | "false" ;
doc_string     = '"""' { any_char } '"""' ;

(* Common *)
param_list     = [ param { "," param } ] ;
param          = identifier ":" type_expr [ "=" const_expr ] ;
identifier_list = identifier { "," identifier } ;
tensor_list    = "[" identifier_list "]"
               | "[" identifier_pattern { "," identifier_pattern } "]" ;  (* Wildcard support *)
identifier_pattern = identifier | identifier "*" | "*" identifier | "*" ;  (* e.g., ln*_out, *_rstd *)

(* HuggingFace sections *)
hf_config_section = "hf_config:" INDENT hf_config_body DEDENT ;
hf_config_body = { identifier ":" ( string | hf_param_mapping ) } ;
hf_param_mapping = "param_mapping:" INDENT { identifier ":" identifier } DEDENT ;

hf_mapping_section = "hf_mapping:" INDENT { hf_weight_mapping } DEDENT ;
hf_weight_mapping = identifier_pattern ":" hf_weight_spec ;
hf_weight_spec = string                                  (* Direct mapping *)
               | "fuse" "(" string { "," string } [ "," "dim" "=" integer ] ")"
               | "transform" "(" string "," "fn" ":" identifier ")"
               | "tied_to" "(" identifier ")" ;
```

### B. Reserved Identifiers

The following identifiers are reserved and cannot be used as user-defined names:

```
# Type names
Tensor, int, float, bool, string

# Dtype names
bf16, fp32, fp16, fp8_e4m3, fp8_e5m2, fp4_e2m1, int8, int32

# Built-in functions
zeros, ones, full, arange, tied_to, fuse, split, transform

# Graph keywords
saved, recompute

# Annotation names
memory, hook, adapter, dtype, precision, fuse, nofuse, shard, checkpoint
```

### C. Compilation Errors

| Code | Description                            |
| ---- | -------------------------------------- |
| E001 | Syntax error                           |
| E002 | Undefined identifier                   |
| E003 | Type mismatch                          |
| E004 | Shape mismatch                         |
| E005 | Missing required gradient              |
| E006 | Saved tensor not available in backward |
| E007 | Circular dependency in graph           |
| E008 | Invalid annotation                     |
| E009 | Duplicate parameter name               |
| E010 | Invalid weight mapping                 |
| E011 | Incompatible inheritance               |
| E012 | Missing required parameter             |
| E013 | Invalid fusion pattern                 |
| E014 | Unsupported primitive                  |
| E015 | Invalid dtype for operation            |
| E016 | `if` guard must be ConstExpr           |
| E017 | Tensor redefinition in same scope (SSA violation) |
| E018 | Gradient shape mismatch                |
| E019 | Circular gradient dependency           |
| E020 | Gradient dtype incompatible            |
| E021 | Recompute tensor not derivable from saved tensors |
| E022 | Circular recompute dependency          |
| E023 | Import name conflict                   |
| E024 | Import alias shadows existing name     |
| E025 | Checkpoint spans pipeline stages       |
| E026 | Invalid constraint expression          |
| E027 | Constraint violation                   |

### D. Warnings

| Code | Description                            |
| ---- | -------------------------------------- |
| W001 | User definition shadows primitive      |
| W002 | Local definition shadows import        |
| W003 | Suboptimal auto-derived backward       |
| W004 | Unused saved tensor                    |
| W005 | Implicit dtype narrowing               |

### D. Version History

| Version | Date       | Changes                     |
| ------- | ---------- | --------------------------- |
| 0.1.0   | 2026-01-19 | Initial draft specification |

---

## References

1. Surogate BlockExecutor implementation: `csrc/src/modules/composite/block_executor.h`
2. Surogate WeightMapping system: `csrc/src/modules/weights/weight_mapping.h`
3. Surogate ModelConfig: `csrc/src/modules/model_config.h`