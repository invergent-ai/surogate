# DSL-Driven Recompute System

## Overview

Replace hardcoded C++ recompute logic with a declarative system where the Python DSL
defines which activations can be recomputed and their dependencies. The C++ runtime
interprets these hints and executes recompute dynamically.

## Current State

### Problems

1. **Hardcoded recompute segments** in `graph_executor_backward.cpp`:
   - `recompute_attention_segment()` - knows exactly what ops to run
   - `recompute_ffn_segment()` - same
   - Adding new architectures (MoE, Mamba) requires new C++ functions

2. **LoRA vs full finetune (non-LoRA) special-casing** is imperative:
   ```cpp
   const bool skip_attention_recompute = !rs.is_lora_only_mode();
   ```

3. **Fragmented hints** that aren't actually used:
   - `Activation(..., recompute=True)` -> stored but ignored
   - `ActivationMemoryHint::RECOMPUTE` -> exists but C++ doesn't read it
   - `TensorSlotRegistry::can_recompute()` -> available but unused

## Proposed Design

### 1. Python DSL Extensions (DONE)

```python
# surogate/dsl/decorators.py

class Activation:
    def __init__(
        self,
        tensor_type: TensorAnnotation,
        *,
        # Existing
        save: bool = False,
        recompute: bool = False,
        shares_with: str | None = None,

        # New: explicit recompute dependencies (slots, params, inputs)
        recompute_from: list[str] | None = None,

        # New: recompute operation and attributes
        recompute_op: str | None = None,  # Dispatch key: DSL op name or recompute-specific kernel
        recompute_attrs: dict[str, Any] | None = None,

        # New: policy for full finetune vs LoRA-only recompute
        # "always" = recompute in all modes
        # "lora_only" = skip in full finetune (non-bit-exact ops)
        recompute_policy: Literal["always", "lora_only", "never"] = "always",

        # New: allow fused ops to emit multiple outputs
        recompute_group: str | None = None,          # ops with same group are merged
        recompute_outputs: list[str] | None = None,  # explicit output list for fused ops

        # New: declare LoRA targets for this recompute op
        lora_targets: list[str] | None = None,  # e.g., ["q", "k", "v"], ["o"], ["up", "gate"]
    ):
        ...
```

**Resolution rules for `recompute_from`:**
- `@input:<name>` refers to a block input (e.g., `@input:residual`, `@input:position_ids`).
- `@param:<name>` refers to a block parameter (e.g., `@param:qkv_weight`, `@param:qkv_bias`).
- `@global:<name>` refers to non-block activations (e.g., `@global:freq_cis`).
- Unprefixed names are activation slots and may use aliases.
- Optional dependencies can be prefixed with `?` (e.g., `?@param:qkv_bias`).

`recompute_attrs` should carry op-specific details (transpose, rotary_dim, activation type,
matmul op id, etc.) so the runtime can call the exact same kernel path as forward.
Use DSL op names for forward ops (`matmul`, `flash_attention`, `qkv_qk_norm_rope`) and
recompute-only kernels for apply_saved variants (e.g., `fused_residual_rmsnorm_apply_saved`).

### 2. Block Definition Example

```python
@block
class DenseTransformerBlock:
    """Standard transformer block with DSL-driven recompute."""

    # === Checkpoints (always saved, small footprint) ===
    ln1_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True)
    ln2_rstd = Activation(Tensor["B", "T"], dtype="fp32", save=True)
    lse = Activation(
        Tensor["B", "Hq", "T"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_policy="lora_only",
        recompute_group="attn_fwd",
        description="Attention log-sum-exp (overwritten in LoRA recompute)",
    )
    q_rstd = Activation(
        Tensor["B", "T", "Hq"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_policy="lora_only",
        recompute_group="qk_norm_rope",
        when="use_qk_norm",
    )
    k_rstd = Activation(
        Tensor["B", "T", "Hkv"],
        dtype="fp32",
        save=True,
        recompute=True,
        recompute_policy="lora_only",
        recompute_group="qk_norm_rope",
        when="use_qk_norm",
    )

    # === Recomputable activations ===
    res_ffn = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_ffn"],
        recompute=True,
        recompute_group="ln1_fused",
        recompute_outputs=["res_ffn", "ln1"],
        recompute_from=["@input:residual", "@input:x", "ln1_rstd", "@param:ln1_weight"],
        recompute_op="fused_residual_rmsnorm_apply_saved",
        recompute_policy="always",
        description="Residual stream after LN1 fused residual",
    )

    ln1 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln1_flat"],
        recompute=True,
        recompute_group="ln1_fused",
        description="LN1 output",
    )

    qkv = Activation(
        Tensor["B", "T", "QKV"],
        aliases=["qkv_flat", "qkv_biased"],
        recompute=True,
        recompute_from=["ln1", "@param:qkv_weight", "?@param:qkv_bias"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "qkv", "transpose": "NT"},
        recompute_policy="lora_only",
        lora_targets=["q", "k", "v"],
        description="QKV projection output",
    )

    qkv_rope = Activation(
        Tensor["B", "T", "QKV"],
        recompute=True,
        recompute_group="qk_norm_rope",
        recompute_outputs=["qkv_rope", "q_rstd", "k_rstd"],
        recompute_from=[
            "qkv",
            "@global:freq_cis",
            "@input:position_ids",
            "?@param:q_norm_weight",
            "?@param:k_norm_weight",
        ],
        recompute_op="qkv_qk_norm_rope",
        recompute_attrs={"use_qk_norm": "use_qk_norm", "rope_fused": "auto", "rotary_dim": "D"},
        recompute_policy="lora_only",
        description="QKV after RoPE (and optional QK-norm)",
    )

    att = Activation(
        Tensor["B", "T", "AttnDim"],
        aliases=["att_flat", "attn"],
        recompute=True,
        recompute_group="attn_fwd",
        recompute_outputs=["att", "lse"],
        recompute_from=["qkv_rope"],
        recompute_op="flash_attention",
        recompute_attrs={"attn_impl": "cudnn"},
        recompute_policy="lora_only",  # cuDNN is not bit-exact in full finetune
        description="Attention output (softmax @ V)",
    )

    att_out = Activation(
        Tensor["B", "T", "C"],
        aliases=["att_out_flat"],
        recompute=True,
        recompute_from=["att", "@param:out_weight"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "attn_out", "transpose": "NT"},
        recompute_policy="lora_only",
        lora_targets=["o"],
        description="Output projection",
    )

    # Fused residual + LN2 (multi-output op)
    res_att = Activation(
        Tensor["B", "T", "C"],
        aliases=["residual_att"],
        recompute=True,
        recompute_group="ln2_fused",
        recompute_outputs=["res_att", "ln2"],
        recompute_from=["res_ffn", "att_out", "ln2_rstd", "@param:ln2_weight"],
        recompute_op="fused_residual_rmsnorm_apply_saved",
        recompute_policy="always",
        description="Residual + attention output",
    )

    ln2 = Activation(
        Tensor["B", "T", "C"],
        aliases=["ln2_flat"],
        recompute=True,
        recompute_group="ln2_fused",
        description="LN2 output",
    )

    mlp_up = Activation(
        Tensor["B", "T", "MUp"],
        aliases=["mlp_up_flat"],
        recompute=True,
        recompute_from=["ln2", "@param:mlp_up_weight"],
        recompute_op="matmul",
        recompute_attrs={"matmul_op": "mlp_up", "transpose": "NT"},
        recompute_policy="lora_only",
        lora_targets=["up", "gate"],
        description="MLP up projection (gate + up)",
    )

    swiglu = Activation(
        Tensor["B", "T", "M"],
        aliases=["swiglu_flat"],
        recompute=True,
        recompute_from=["mlp_up"],
        recompute_op="swiglu",
        recompute_attrs={"activation": "swiglu"},
        recompute_policy="lora_only",
        description="SwiGLU activation",
    )

    # === Gradient slots ===
    d_ln1 = Gradient(Tensor["B", "T", "C"], gradient_of="ln1")
    d_qkv = Gradient(Tensor["B", "T", "QKV"], gradient_of="qkv")
    d_att = Gradient(Tensor["B", "T", "AttnDim"], gradient_of="att")
    d_ln2 = Gradient(Tensor["B", "T", "C"], gradient_of="ln2")
    d_mlp_up = Gradient(Tensor["B", "T", "MUp"], gradient_of="mlp_up")
    d_swiglu = Gradient(Tensor["B", "T", "M"], gradient_of="swiglu")
    d_res_att = Gradient(Tensor["B", "T", "C"], gradient_of="res_att")
    d_res_ffn = Gradient(Tensor["B", "T", "C"], gradient_of="res_ffn")
```

Notes:
- `*_flat` aliases are views; recompute does not need explicit view ops. The runtime
  can infer shapes and create views using the alias map and slot shapes.
- `flash_attention` writes `lse` as an output (not an input); in LoRA mode it can
  overwrite the shared `lse` buffer.
- `qk_norm + rope` must follow the same fused/unfused path as forward. Use
  `recompute_attrs` plus the forward plan to select kernels.

### 3. IR Extensions (DONE)

Add recompute metadata to the DSL spec and IR layers:

```python
# surogate/dsl/specs.py (ActivationSlotSpec)
recompute_in_backward: bool = False
recompute_from: list[str] = field(default_factory=list)
recompute_op: str | None = None
recompute_attrs: dict[str, Any] = field(default_factory=dict)
recompute_policy: str = "always"
recompute_group: str | None = None
recompute_outputs: list[str] = field(default_factory=list)
lora_targets: list[str] = field(default_factory=list)
condition: Callable[[Any], bool] | None = None
```

```python
# surogate/dsl/py_compiler.py (ActivationSlotIR)
recompute_in_backward: bool = False
recompute_from: list[str] = field(default_factory=list)
recompute_op: str | None = None
recompute_attrs: dict[str, Any] = field(default_factory=dict)
recompute_policy: str = "always"
recompute_group: str | None = None
recompute_outputs: list[str] = field(default_factory=list)
lora_targets: list[str] = field(default_factory=list)
condition: str | None = None  # serialize `when` (e.g., "use_qk_norm")
```

Also update:
- `_activation_slot_ir_to_dict` and `_compile_activation_slot` to carry the new fields.
- C++ IR in `csrc/src/dsl/ir.h` (ActivationSlotIR) + JSON loader in `csrc/src/dsl/ir.cpp`.
- `TensorSlotRegistry` to store the new recompute metadata.
- Only string `when` conditions can be serialized; callable `when` should be evaluated in Python
  (include or drop the slot) so C++ doesn't need to interpret lambdas.

### 4. C++ Runtime Changes (DONE)

#### 4.1 RecomputePlan Structure (DONE)

```cpp
// csrc/src/dsl/recompute_plan.h

namespace dsl {

enum class RecomputePolicy : uint8_t {
    Always,
    LoraOnly,
    Never,
};

struct RecomputeOp {
    std::string op_type;                 // Dispatch key (DSL op name or recompute kernel)
    std::vector<std::string> inputs;     // Dependencies (slots/params/inputs)
    std::vector<std::string> outputs;    // One or more outputs (fused ops)
    AttrMap attrs;                       // Op-specific attributes
    RecomputePolicy policy = RecomputePolicy::Always;
    std::vector<std::string> lora_targets;
};

struct LayerRecomputePlan {
    std::vector<RecomputeOp> topo_ops;   // Topologically ordered by dependencies
    std::unordered_map<std::string, size_t> producer_index;  // output -> op index
};

class RecomputePlan {
public:
    void init_from_layout(const ActivationLayoutIR& layout, const PretrainedConfig& cfg);

    void execute_layer(int layer_idx, DslRunState& rs, DslWeightStore& weights,
                       bool lora_only_mode, cudaStream_t stream);

    bool can_recompute(const std::string& name) const;
    const std::vector<std::string>& get_dependencies(const std::string& name) const;

private:
    LayerRecomputePlan mPlan;
    void execute_op(const RecomputeOp& op, int layer_idx, DslRunState& rs,
                    DslWeightStore& weights, cudaStream_t stream);
};

}  // namespace dsl
```

Plan construction details:
- Merge ops by `recompute_group` or identical `(op_type, inputs, attrs)` signatures.
- Expand aliases using the layout alias map (`ActivationLayoutIR::build_alias_map()`).
- Apply `condition` (e.g., `use_qk_norm`) using config at plan-build time.
- Drop `recompute_outputs` entries for slots that are not present due to `condition` or `when`.
- Topologically sort by activation dependencies (params/inputs are leaves).

#### 4.2 Operation Dispatch (DONE)

Use the exact same forward kernels and plans to preserve numerical behavior:

```cpp
void RecomputePlan::execute_op(const RecomputeOp& op, int layer_idx,
                               DslRunState& rs, DslWeightStore& weights,
                               cudaStream_t stream) {
    // Resolve inputs from slots/params/inputs/global buffers (support optional deps)
    auto in = resolve_inputs(op.inputs, layer_idx, rs, weights);
    auto out = resolve_outputs(op.outputs, layer_idx, rs);

    if (op.op_type == "rmsnorm_apply_saved") {
        rmsnorm_apply_saved(out[0], in["residual"], in["ln_weight"], in["rstd"], B, T, C, stream);

    } else if (op.op_type == "fused_residual_rmsnorm_apply_saved") {
        // Example shown for LN2; LN1 uses residual + x and outputs res_ffn + ln1.
        fused_residual_rmsnorm_apply_saved(out["res_att"], out["ln2"],
                                           in["residual"], in["att_out"],
                                           in["ln2_weight"], in["ln2_rstd"],
                                           B * T, C, stream);

    } else if (op.op_type == "matmul") {
        // Use MatmulForwardPlan + recipe to mirror forward (FP8/FP4/cache/bias)
        run_matmul_with_plan(op.attrs, in, out[0], rs, weights, layer_idx, stream);
        apply_lora_if_needed(op, in, out[0], rs, layer_idx, stream);

    } else if (op.op_type == "qkv_qk_norm_rope") {
        run_qk_norm_rope_with_plan(op.attrs, in, out, rs, layer_idx, stream);

    } else if (op.op_type == "flash_attention") {
        attention_forward_cudnn(out["att"], out["lse"], in["qkv_rope"], rs.scratch().cudnn_workspace,
                                rs.CudnnHandle, B, T, Hq, Hkv, Hs, stream);

    } else if (op.op_type == "swiglu") {
        swiglu_forward(out[0], in["mlp_up"], nullptr, B, T, D, stream);
    }
}
```

Key constraints:
- Matmuls must reuse `MatmulForwardPlan` and recipe logic (FP8/FP4, cached weights,
  bias presence, delayed quantizer index). The current logic in
  `graph_executor_backward.cpp` should be reused, not duplicated.
- RoPE/QK-norm must match the forward fused/unfused path and rotary_dim.
- Fused residual recompute must consume the correct inputs (`@input:x` + `@input:residual` for LN1,
  `res_ffn` + `att_out` for LN2) and write `res_ffn`/`res_att`. `res_ffn` is owned by the residual
  manager, not `simplified_acts`.
- `@input:*` resolution must map to per-layer block inputs (layer 0 uses encoded/initial residual,
  later layers use previous-layer outputs like `mlp_down` + `res_att`).
- Residual offload: if residuals are offloaded, fetch them before recompute.
- Mark recomputed layers so `resolve_recomputed_block_tensor()` can find fresh values.

#### 4.3 Integration with Backward Pass (DONE)

```cpp
// Replace hardcoded recompute_block() with plan-based execution
void GraphExecutor::recompute_block(int layer_idx, long B, long T) {
    if (!mOptions.recompute_enabled()) return;

    if (!mRecomputePlan) {
        throw std::runtime_error("DSL recompute plan missing; hardcoded path removed");
    }
    mRecomputePlan->execute_layer(
        layer_idx, mRunState, mWeights,
        mRunState.is_lora_only_mode(),
        mRunState.MainStream
    );
}
```

### 5. LoRA Integration (DONE)

`recompute_policy` controls full finetune vs LoRA-only behavior:

```python
att = Activation(
    Tensor["B", "T", "AttnDim"],
    recompute=True,
    recompute_from=["qkv_rope"],
    recompute_op="flash_attention",
    recompute_policy="lora_only",  # skip in full finetune due to non-determinism
)
```

C++ runtime gating:
```cpp
if (!lora_only_mode && op.policy == RecomputePolicy::LoraOnly) {
    // Activation was saved during forward, skip recompute in full finetune
    return;
}
```

### 6. LoRA Contribution Re-application (DONE)

LoRA contributions must be re-applied after recomputing base outputs:

```python
qkv = Activation(
    Tensor["B", "T", "QKV"],
    recompute=True,
    recompute_from=["ln1", "@param:qkv_weight"],
    recompute_op="matmul",
    lora_targets=["q", "k", "v"],
)
```

Runtime:
```cpp
if (op.op_type == "matmul" && lora_enabled && !op.lora_targets.empty()) {
    apply_lora_contribution(output, op.lora_targets, input, layer_idx, ...);
}
```

Use the existing LoRA helper paths (dropout seeds, rank, scaling) to preserve
numerical behavior.

### 7. Memory Allocation Integration (PENDING)

DSL hints also drive buffer sharing decisions. Sharing must respect the same
mode gating as recompute execution:

```cpp
// In DslRunState::allocate_simplified_activations(layout, ...)
    for (const auto& slot : layout.slots) {
        if (!slot.condition || eval_condition(slot.condition, cfg)) {
            const bool recompute_enabled = mRecomputeLevel >= RecomputeLevel::Enabled;
            const bool lora_only = mLoraOnlyMode;
            const bool can_recompute_now = slot.recompute_in_backward && recompute_enabled &&
                (lora_only || slot.recompute_policy != "lora_only");

            if (can_recompute_now && slot.scope == "block" && slot.shares_with.empty()) {
                // Shared buffer for all layers (safe because we will recompute)
                acts.set(slot.name, mSharedBuffers[slot.name]);
            } else if (slot.save_for_backward || slot.recompute_in_backward) {
                // If recompute is skipped in this mode, keep per-layer storage
                acts.set(slot.name, allocate_per_layer(slot));
            }
        }
    }
```

## Migration Plan

### Initial Targets (Priority)
1. **Qwen3 (core runtime)**: `surogate/core/model/models/qwen3.py`
2. **Qwen3 MoE (DSL model)**: `surogate/dsl/models/qwen3_moe.py`
3. Expand to Mamba and other architectures only after Qwen3 + Qwen3MoE are stable.

### Phase 1: Infrastructure (Direct Switch)
1. [DONE] Add `recompute_from`, `recompute_op`, `recompute_attrs`, `recompute_policy`,
   `recompute_group`, `recompute_outputs`, `lora_targets` to `Activation`.
2. [DONE] Extend `ActivationSlotSpec` and `ActivationSlotIR` in Python + JSON serialization.
3. [DONE] Extend C++ `ActivationSlotIR` and JSON loader.
4. [DONE] Implement `RecomputePlan` and wire it directly into the backward path.
5. [DONE] Remove legacy recompute usage immediately (no fallback path).

### Phase 2: Direct Implementation
1. [PENDING] Add DSL recompute declarations to `DenseTransformerBlock` and `Qwen3MoEBlock`.
2. [DONE] `RecomputePlan` parses DSL and generates a topo-ordered execution plan.
3. [DONE] Enable DSL-driven recompute unconditionally.
4. [PENDING] Validate results for **Qwen3** (core) and **Qwen3MoE** (DSL) in LoRA-only and full finetune.

### Phase 3: Expand Coverage
1. [PENDING] Add `MoETransformerBlock` recompute declarations.
2. [PENDING] Add `MambaBlock` recompute declarations.
3. [DONE] Remove hardcoded `recompute_*_segment()` functions.

### Phase 4: Advanced Features
1. [PENDING] Auto-generate recompute plan from the forward graph (no manual declarations).
2. [PENDING] Mixed checkpoint/recompute strategies per model.

## Testing Strategy

1. **Unit tests**: Each `recompute_op` type produces correct output.
2. **Integration tests**: Full forward/backward matches non-recompute baseline.
3. **Regression tests**: LoRA + recompute gradient norms match expected values.
4. **Mode tests**: LoRA-only vs full finetune path gating for non-bit-exact ops.
5. **Conditional tests**: `use_qk_norm` on/off and optional bias handling.
6. **Offload tests**: residual offload + recompute (layer 0 and layer > 0).

## Benefits Summary

| Aspect                 | Before                                | After                                   |
| ---------------------- | ------------------------------------- | --------------------------------------- |
| Add MoE recompute      | Write C++ `recompute_moe_segment()`   | Annotate `MoEBlock` activations         |
| Add Mamba recompute    | Write C++ `recompute_mamba_segment()` | Annotate `MambaBlock` activations       |
| Debug recompute issues | Trace C++ execution                   | Inspect recompute plan from Python      |
| LoRA vs full finetune  | Hardcoded `is_lora_only_mode()`       | Declarative `recompute_policy`          |
| Buffer sharing         | Hardcoded in `DslRunState`            | Driven by DSL recompute hints + policy  |
| Test coverage          | Integration tests only                | Unit test each recompute op             |
