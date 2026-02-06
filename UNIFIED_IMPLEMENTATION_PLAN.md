# Unified Implementation Plan: Weight Mapping + QLoRA Architecture

## Overview

This document outlines the implementation order for both proposals:
1. **Weight Mapping Proposal** (DSL-driven HF weight loading)
2. **QLoRA Architecture Proposal** (Architecture-agnostic quantization)

## Dependency Analysis

```
┌─────────────────────────────────────┐
│  Weight Mapping (DSL)               │
│  - Defines how to load from HF     │
│  - @hf_mapping decorator           │
│  - Hierarchical IR                 │
│  - DslWeightLoader (C++)           │
└────────────────┬────────────────────┘
                 │
                 │ Provides weights
                 ▼
┌─────────────────────────────────────┐
│  Generic QLoRA                      │
│  - IQuantizer interface            │
│  - Quantize/dequantize any tensor  │
│  - GenericWeightManager            │
│  - Group-based offloading          │
└─────────────────────────────────────┘
```

**Key Insight**: Weight mapping happens BEFORE quantization
- Weight mapping loads tensors from HF → BF16 in memory
- QLoRA quantizes those BF16 tensors → NF4/FP8/FP4
- Therefore: **Weight mapping is a prerequisite for QLoRA**

## Implementation Phases

### Phase 1: Foundation (Both Systems) ⚡ PARALLEL

Can be done independently in parallel:

#### 1A. Weight Mapping DSL Enhancements
**Goal**: Add `@hf_mapping` decorator and metadata to Python DSL

**Files to modify:**
- `surogate/dsl/core.py`: Add `@hf_mapping` decorator
- `surogate/dsl/modules/__init__.py`: Add `quantizable`, `offload_group` to `Param()`
- `surogate/dsl/py_compiler.py`: Extract `@hf_mapping` and export to IR

**Deliverables:**
```python
# Can now write:
@module
class Attention(Module):
    qkv_weight = Param(Tensor[...], quantizable=True, offload_group=-1)

    @hf_mapping
    def weight_mapping(layer=-1, prefix="model.layers"):
        return {
            "qkv_weight": [
                f"{prefix}.{{layer}}.qkv_weight",  # Try fused
                fuse("q_proj", "k_proj", "v_proj", dim=0),  # Fallback
            ]
        }
```

**Tests:**
- Unit tests for `@hf_mapping` decorator
- Compiler test: verify IR contains hierarchical weight mapping
- Validate `quantizable` and `offload_group` metadata

**Estimated effort**: 2-3 days

---

#### 1B. Generic QLoRA Primitives
**Goal**: Define interfaces without implementation

**Files to create:**
- `csrc/src/runtime/qlora/generic_quantizer.h`: `IQuantizer` interface
- `csrc/src/runtime/qlora/quantized_tensor.h`: `QuantizedTensor` struct
- `csrc/src/runtime/qlora/offload_manager.h`: `OffloadManager` interface

**Deliverables:**
```cpp
// Generic quantizer interface (no impl yet)
class IQuantizer {
    virtual QuantizedTensor quantize(const Tensor& input, ...) = 0;
    virtual void dequantize(const QuantizedTensor& q, Tensor& output, ...) = 0;
};

// Offload manager interface (no impl yet)
class OffloadManager {
    void register_tensor(QuantizedTensor& tensor, int group_id);
    void load_group(int group_id, cudaStream_t stream);
    void unload_group(int group_id, cudaStream_t stream);
};
```

**Tests:**
- Compile-only tests (verify interfaces compile)
- Mock implementations for testing

**Estimated effort**: 1-2 days

---

### Phase 2: Weight Mapping Implementation 🔵 SEQUENTIAL

**Dependency**: Requires Phase 1A

**Goal**: Implement C++ weight loader that interprets DSL mapping IR

#### 2A. DslWeightLoader (C++)
**Files to create:**
- `csrc/src/runtime/dsl/dsl_weight_loader.h`
- `csrc/src/runtime/dsl/dsl_weight_loader.cpp`

**Key functionality:**
```cpp
class DslWeightLoader {
    // Load weights following IR mapping specs
    bool load_from_spec(const json& spec, Tensor& target, int layer, int expert);

    // Handle different mapping kinds
    bool load_direct(const json& spec, Tensor& target);
    bool try_multiple(const json& options, Tensor& target);
    bool load_fused(const json& spec, Tensor& target);
};
```

**Integration points:**
- Uses existing `SafeTensorsReader`
- Produces `Tensor` (BF16, not quantized yet)
- No QLoRA dependency yet

**Tests:**
- Unit tests: Load single weight with "Direct" mapping
- Unit tests: Load fused weights (Q+K+V → QKV)
- Unit tests: "TryMultiple" fallback logic
- Integration test: Load full Llama-3.2 model

**Estimated effort**: 3-4 days

---

#### 2B. Update Existing Modules
**Files to update:**
- `surogate/dsl/modules/attention.py`: Add `@hf_mapping`
- `surogate/dsl/modules/mlp.py`: Add `@hf_mapping`
- `surogate/dsl/modules/rmsnorm.py`: Add `@hf_mapping`
- `surogate/dsl/blocks/dense.py`: Compose module mappings
- `surogate/dsl/models/llama.py`: Use new system

**Deliverable**: Llama-3.2 loads via new system

**Tests:**
- Integration test: Load Llama-3.2-1B weights
- Verify loaded weights match old system

**Estimated effort**: 2-3 days

---

### Phase 3: Generic QLoRA Implementation 🟢 SEQUENTIAL

**Dependency**: Requires Phase 1B

**Goal**: Implement generic quantization without architectural knowledge

#### 3A. Concrete Quantizers
**Files to create:**
- `csrc/src/runtime/qlora/bnb_quantizer.cpp` (implements `IQuantizer`)
- `csrc/src/runtime/qlora/fp8_quantizer.cpp`
- `csrc/src/runtime/qlora/fp4_quantizer.cpp`

**Key changes from current system:**
- No `BnBBlockWeights` structure
- Just `QuantizedTensor quantize(const Tensor& input)`
- Works on ANY tensor (no attention/MLP knowledge)

**Refactoring:**
```cpp
// OLD (architecture-aware)
void quantize_block(BnBBlockWeights& block, const SafeTensorsReader& reader);

// NEW (generic)
QuantizedTensor quantize(const Tensor& input, std::string_view name);
```

**Tests:**
- Unit test: Quantize random tensor, dequantize, check error
- Unit test: Verify NF4 codebook correctness
- Benchmark: Compare speed with old system

**Estimated effort**: 3-4 days

---

#### 3B. Offload Manager
**Files to create:**
- `csrc/src/runtime/qlora/offload_manager.cpp`

**Key functionality:**
- Group-based offloading (not expert-specific)
- LRU eviction policy
- Async prefetching

**Integration points:**
- Uses `TensorAllocator` for CPU/GPU memory
- Works with `QuantizedTensor` (not experts)

**Tests:**
- Unit test: Register tensors to groups
- Unit test: Load/unload group to/from CPU
- Unit test: LRU eviction when max groups exceeded
- Performance test: Prefetch overlap with compute

**Estimated effort**: 2-3 days

---

#### 3C. GenericWeightManager
**Files to create:**
- `csrc/src/runtime/qlora/generic_weight_manager.h`
- `csrc/src/runtime/qlora/generic_weight_manager.cpp`

**Key functionality:**
```cpp
class GenericWeightManager {
    // Flat storage (name → quantized tensor)
    std::unordered_map<std::string, QuantizedTensor> mQuantizedWeights;
    std::unordered_map<std::string, Tensor> mDequantBuffers;

    // Generic API
    Tensor& get(std::string_view name, cudaStream_t stream);
    void prefetch_group(int group_id, cudaStream_t stream);
};
```

**Tests:**
- Unit test: Quantize weight, retrieve it (lazy dequant)
- Unit test: Prefetch group, verify dequant happened
- Integration test: Load Llama weights, quantize, access

**Estimated effort**: 2-3 days

---

### Phase 4: Integration 🟣 SEQUENTIAL

**Dependency**: Requires Phase 2 + Phase 3

**Goal**: Connect weight mapping with QLoRA

#### 4A. Weight Loading + Quantization Pipeline
**Files to modify:**
- `csrc/src/runtime/dsl/dsl_model.cpp`

**New flow:**
```cpp
// 1. Load weights via DslWeightLoader (BF16)
DslWeightLoader loader(reader, weight_mapping_ir);
Tensor bf16_weight;
loader.load_from_spec(spec, bf16_weight, layer_idx, expert_idx);

// 2. Quantize via GenericWeightManager
GenericWeightManager weight_mgr(config);
weight_mgr.quantize_and_store("blocks[0].qkv_weight", bf16_weight, group_id);

// 3. Access via name (lazy dequant)
Tensor& qkv = weight_mgr.get("blocks[0].qkv_weight", stream);
```

**Key integration point:**
- `GenericWeightManager::import_and_quantize()` uses `DslWeightLoader`
- Loop over all parameters in IR, load + quantize

**Tests:**
- Integration test: Load + quantize Llama-3.2
- Verify memory usage matches old system
- Verify accuracy (compare outputs)

**Estimated effort**: 2-3 days

---

#### 4B. Update Runtime to Use New System
**Files to modify:**
- `csrc/src/runtime/dsl/dsl_model.cpp`: Use `GenericWeightManager`
- `csrc/src/runtime/dsl/graph_executor.cpp`: Access weights by name

**Changes:**
```cpp
// OLD
auto& block = bnb_provider->get_block(layer_idx);
matmul(input, block.qkv_weight, output);

// NEW
Tensor& qkv = weight_mgr.get("blocks[" + std::to_string(layer_idx) + "].qkv_weight");
matmul(input, qkv, output);
```

**Tests:**
- End-to-end test: Train Llama-3.2 with new system
- Compare loss curve with old system (should match)

**Estimated effort**: 3-4 days

---

### Phase 5: Advanced Modules 🔴 SEQUENTIAL

**Dependency**: Requires Phase 4

**Goal**: Implement complex architectures (Mamba, MoE, Hybrid)

#### 5A. Mamba Module
**Files to update:**
- `surogate/dsl/modules/mamba.py`: Add `@hf_mapping` with multiple fallbacks
- Update IR to include Mamba mappings

**Key challenge**: Multiple HF naming conventions
```python
@hf_mapping
def weight_mapping(layer=-1, prefix="backbone.layers"):
    return {
        "in_proj_weight": [
            f"{prefix}.{{layer}}.mixer.in_proj.weight",  # Nemotron-H
            "model.layers.{layer}.in_proj.weight",       # Pure Mamba
        ],
    }
```

**Tests:**
- Load Nemotron-H weights (backbone.layers.{layer}.mixer.*)
- Load hypothetical pure Mamba model (model.layers.{layer}.*)

**Estimated effort**: 1-2 days

---

#### 5B. MoE with Expert Offloading
**Files to update:**
- `surogate/dsl/modules/moe.py`: Add `@hf_mapping` + `offload_group` per expert
- Update compiler to assign group IDs

**Key feature**: Each expert = one offload group
```python
@module
class Expert(Module):
    gate_weight = Param(Tensor[...], offload_group="{expert}")  # Group ID from expert index
```

**Tests:**
- Load Qwen3-MoE weights
- Enable offloading, verify experts move to CPU
- Prefetch selected experts, verify performance

**Estimated effort**: 2-3 days

---

#### 5C. Nemotron-H (Hybrid)
**Files to update:**
- `surogate/dsl/blocks/nemotron_h.py`: Use optional modules
- Update weight loader to skip missing weights for inactive modules

**Key feature**: All mappings defined, only load what exists
```python
@block
class NemotronHBlock(Block):
    norm = Param(RMSNorm)                        # All layers
    attention = Param(Attention, optional=True)  # Only * and E layers
    mamba = Param(Mamba, optional=True)          # Only M layers
    mlp = Param(MLP, optional=True)              # Only * and - layers
    moe = Param(MoE, optional=True)              # Only E layers
```

**Tests:**
- Load Nemotron-H weights
- Verify layer 0 (Mamba) only loads Mamba weights
- Verify layer 1 (MoE) only loads Attention + MoE weights

**Estimated effort**: 2-3 days

---

### Phase 6: Optimization & Cleanup 🟡 PARALLEL

Can be done independently:

#### 6A. Performance Optimization
- [ ] Buffer reuse for same-shape tensors
- [ ] Async quantization during loading
- [ ] Prefetch next layer during compute
- [ ] Benchmark and profile

**Estimated effort**: 2-3 days

---

#### 6B. Deprecation & Migration
- [ ] Remove old `BnBBlockWeights` structures
- [ ] Remove `bnb_weights.cpp` old code
- [ ] Remove hybrid pattern logic from QLoRA
- [ ] Update all models to new system
- [ ] Update documentation


## Critical Path

```
START
  ↓
Phase 1A: Weight Mapping DSL (2-3 days)
  ↓
Phase 2A: DslWeightLoader (3-4 days)
  ↓
Phase 2B: Update Llama (2-3 days)
  ↓
[WAIT for Phase 3C: GenericWeightManager]
  ↓
Phase 4A: Integration (2-3 days)
  ↓
Phase 4B: Update Runtime (3-4 days)
  ↓
Phase 5C: Nemotron-H (2-3 days)
  ↓
Phase 6B: Deprecation (2-3 days)
  ↓
DONE
```

**Parallel track (can overlap with main path):**
```
START
  ↓
Phase 1B: QLoRA Interfaces (1-2 days)
  ↓
Phase 3A: Concrete Quantizers (3-4 days)
  ↓
Phase 3B: Offload Manager (2-3 days)
  ↓
Phase 3C: GenericWeightManager (2-3 days)
  ↓
[MERGE with main path at Phase 4A]
```

## Milestones

### Milestone 1: Weight Mapping Working (Phase 2B complete)
- ✅ Can load Llama-3.2 weights via new system
- ✅ Weights loaded match old system
- ⚠️ No quantization yet (BF16 only)

**Demo**: Load Llama-3.2, print weight names and shapes

---

### Milestone 2: QLoRA Working (Phase 3C complete)
- ✅ Can quantize any tensor (NF4/FP8/FP4)
- ✅ Generic offload manager works
- ⚠️ Not integrated with weight loading yet

**Demo**: Quantize random tensors, verify dequant accuracy

---

### Milestone 3: Integration Complete (Phase 4B complete)
- ✅ Load + quantize + train Llama-3.2
- ✅ Loss matches old system
- ✅ Memory usage matches old system

**Demo**: Train Llama-3.2 with QLoRA, compare with baseline

---

### Milestone 4: All Architectures Supported (Phase 5C complete)
- ✅ Llama, Qwen3, Qwen3-MoE, Nemotron-H all work
- ✅ Expert offloading works
- ✅ Hybrid models work

**Demo**: Train Nemotron-H with QLoRA + expert offloading

---

### Milestone 5: Production Ready (Phase 6B complete)
- ✅ Old code removed
- ✅ Documentation updated
- ✅ All tests passing
- ✅ Performance benchmarks documented

**Demo**: Full release with migration guide

---

## Risk Mitigation

### Risk 1: Weight loader performance
**Mitigation**: Benchmark after Phase 2A, optimize before Phase 4

### Risk 2: QLoRA accuracy regression
**Mitigation**: Extensive testing in Phase 3A (compare with old kernels)

### Risk 3: Integration complexity
**Mitigation**: Phase 4A is dedicated integration phase with tests

### Risk 4: Hybrid models edge cases
**Mitigation**: Comprehensive tests in Phase 5C for all block type combinations

---

## Recommended Approach

1. Start both tracks simultaneously (Phase 1A + 1B)
2. Let them develop independently
3. Merge at Phase 4A

**Start in Parallel**:
1. Assign one person to weight mapping track (Phases 1A → 2)
2. Assign another to QLoRA track (Phases 1B → 3)
3. Both meet at Phase 4 for integration
4. Continue together through Phases 5-6

---

## Next Steps

1. **Start Phase 1A and 1B** simultaneously
