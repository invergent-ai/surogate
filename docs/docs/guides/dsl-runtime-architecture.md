---
title: DSL Runtime Architecture
---

# DSL Runtime Architecture for Model Definition

## Overview

This document describes a **runtime-interpreted DSL (Domain-Specific Language)** for defining new transformer model architectures in Surogate without code generation. The DSL compiles to `BlockSpec` definitions that are executed by the existing `BlockExecutor` infrastructure.

## Core Concept

Instead of writing C++ code or generating C++ from DSL definitions, we interpret DSL model specifications **at runtime** and translate them into `BlockSpec` objects that drive the existing execution engine.

```
┌──────────────────┐
│  my_model.ndsl   │  (YAML file - user writes)
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ RuntimeModelSpec         │  (Python parser)
│ .from_file()             │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ .to_block_spec()         │  (Python compiler)
│ Returns Python dict      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ RuntimeBlockBuilder      │  (C++ binding)
│ .build_from_dsl()        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ BlockSpec (C++ object)   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ BlockExecutor::forward() │  (Existing C++ code)
│ Runs CUDA kernels        │
└──────────────────────────┘
```

### The Role of Python vs C++

The system uses a **two-stage architecture**:

1. **Python Frontend** (Compiler):
   - Parses `.ndsl` YAML files
   - Compiles high-level DSL operations into `BlockSpec` format
   - Manages model registry and configuration
   - **No code generation** - just data transformation

2. **C++ Backend** (Executor):
   - Receives compiled `BlockSpec` from Python
   - Executes operations using existing CUDA kernels
   - Zero overhead - same performance as hand-written code

This is similar to how PyTorch works: Python defines the model, C++/CUDA executes it.

## Why Runtime Interpretation?

### Advantages

1. **No Code Generation**: Changes to model architectures don't require recompilation
2. **Rapid Iteration**: Modify `.ndsl` file and immediately retrain
3. **Python-Friendly**: DSL parsing and manipulation in Python
4. **Reuses Infrastructure**: Leverages existing `BlockExecutor`, primitives, and recipes
5. **Dynamic Composition**: Can compose blocks at runtime based on configuration
6. **Easy Validation**: Validate DSL specs before training starts
7. **Debugging**: Can inspect and print `BlockSpec` at runtime

### What Gets Interpreted

- ✅ Block structure (order of operations)
- ✅ Residual connections (parallel vs sequential)
- ✅ Conditional operations (QKNorm, etc.)
- ✅ Model hyperparameters
- ✅ Weight mappings for HuggingFace compatibility

### What Stays Compiled (C++)

- ✅ All CUDA kernels (RMSNorm, Flash Attention, matmuls, etc.)
- ✅ `BlockExecutor` (runtime dispatcher)
- ✅ Weight manager
- ✅ Recipe system (BF16, FP8, FP4)
- ✅ Training loop infrastructure

## DSL Syntax

### Basic Structure

A model definition in the DSL consists of:

1. **Neurons**: Reusable building blocks (primitives or composites)
2. **Models**: Complete model architectures composed from neurons
3. **Configuration**: Hyperparameters and architectural choices
4. **HuggingFace Mapping**: Weight layout for checkpoint compatibility

### Example: LLaMA-style Model

```yaml
# models/llama2.ndsl
name: Llama2
type: model

# Block structure (interpreted at runtime)
architecture:
  variant: dense  # Options: dense, parallel, moe

  block:
    # Attention sublayer
    - op: RMSNorm
      name: ln1
      config: {eps: 1e-6}

    - op: CausalSelfAttention
      config:
        use_qk_norm: false
        use_qkv_bias: false
      expands_to:
        - QKV          # QKV projection
        - RoPE         # Rotary position embedding
        - Attention    # Flash Attention
        - AttnOut      # Output projection

    - op: ResidualAdd
      fused_with_next: ln2  # Creates ResidualLN2 op

    # MLP sublayer
    - op: RMSNorm
      name: ln2
      config: {eps: 1e-6}
      fused_with_residual: true

    - op: SwiGLU_FFN
      expands_to:
        - MLPUp        # Gate + Up projection (fused)
        - SwiGLU       # SwiGLU activation
        - MLPDown      # Down projection

    - op: ResidualAdd
      output: residual

# Model hyperparameters
config:
  vocab_size: 32000
  hidden_size: 4096
  num_layers: 32
  num_query_heads: 32
  num_kv_heads: 32
  intermediate_size: 11008
  max_position_embeddings: 2048
  rms_norm_eps: 1e-6
  rope_theta: 10000.0

# HuggingFace compatibility
hf_architecture: "LlamaForCausalLM"
hf_mapping:
  embedding.weight: "model.embed_tokens.weight"
  blocks[{i}].ln1.weight: "model.layers.{i}.input_layernorm.weight"
  blocks[{i}].attention.qkv_weight:
    concat:
      - "model.layers.{i}.self_attn.q_proj.weight"
      - "model.layers.{i}.self_attn.k_proj.weight"
      - "model.layers.{i}.self_attn.v_proj.weight"
  blocks[{i}].attention.out_weight: "model.layers.{i}.self_attn.o_proj.weight"
  blocks[{i}].ln2.weight: "model.layers.{i}.post_attention_layernorm.weight"
  blocks[{i}].mlp.up_weight:
    concat:
      - "model.layers.{i}.mlp.gate_proj.weight"
      - "model.layers.{i}.mlp.up_proj.weight"
  blocks[{i}].mlp.down_weight: "model.layers.{i}.mlp.down_proj.weight"
  final_norm.weight: "model.norm.weight"
  lm_head.weight: "lm_head.weight"
```

### Example: Parallel Residual (GPT-NeoX style)

```yaml
# models/gpt_neox.ndsl
name: GPT_NeoX
type: model

architecture:
  variant: parallel  # Both attention and MLP read from same residual

  block:
    # Single pre-norm for both paths
    - op: RMSNorm
      name: ln1
      config: {eps: 1e-5}

    # Attention path
    - op: CausalSelfAttention
      branch: attention

    # MLP path (parallel to attention)
    - op: RMSNorm
      name: ln2
      config: {eps: 1e-5}
      reads_from: residual  # Same input as ln1

    - op: SwiGLU_FFN
      branch: mlp

    # Combine all three: residual + attention + mlp
    - op: ResidualAdd
      inputs: [residual, attention, mlp]
      output: residual

config:
  vocab_size: 50432
  hidden_size: 6144
  num_layers: 44
  num_query_heads: 64
  num_kv_heads: 64
  intermediate_size: 24576
  max_position_embeddings: 2048
  rms_norm_eps: 1e-5
  rope_theta: 10000.0

hf_architecture: "GPTNeoXForCausalLM"
# ... mappings
```

### Example: MoE Architecture (Qwen3-MoE style)

```yaml
# models/qwen3_moe.ndsl
name: Qwen3MoE
type: model

architecture:
  variant: moe

  block:
    # Attention sublayer (same as dense)
    - op: RMSNorm
      name: ln1
      config: {eps: 1e-6}

    - op: CausalSelfAttention
      config:
        use_qk_norm: true
        use_qkv_bias: true

    - op: ResidualAdd
      fused_with_next: ln2

    # MoE sublayer (replaces FFN)
    - op: RMSNorm
      name: ln2
      config: {eps: 1e-6}
      fused_with_residual: true

    - op: MoE_FFN
      config:
        num_experts: 64
        top_k: 8
        norm_topk_prob: true
      expands_to:
        - Router         # Expert routing
        - Experts        # Grouped expert computation
        - Combine        # Weighted combination

    - op: ResidualAdd
      output: residual

config:
  vocab_size: 151936
  hidden_size: 2048
  num_layers: 24
  num_query_heads: 16
  num_kv_heads: 16
  intermediate_size: 5632
  max_position_embeddings: 32768
  rms_norm_eps: 1e-6
  rope_theta: 1000000.0

  # MoE-specific config
  num_experts: 64
  num_experts_per_tok: 8
  moe_intermediate_size: 1408
  decoder_sparse_step: 1
  norm_topk_prob: true
  router_aux_loss_coef: 0.001

hf_architecture: "Qwen3MoeForCausalLM"
# ... mappings
```

## Architecture Components

The DSL system has three main components that work together:

### Component Roles

| Component | Language | Purpose | Output |
|-----------|----------|---------|--------|
| **Parser** | Python | Parse `.ndsl` YAML files | `RuntimeModelSpec` object |
| **Compiler** | Python | Translate DSL → BlockSpec | Python dict |
| **Executor** | C++ | Run BlockSpec operations | CUDA kernel calls |

### 1. Python-Side Parser

**Purpose:** Parse `.ndsl` files (YAML format) and convert them into Python objects.

```python
# surogate/dsl/runtime_model.py

class RuntimeModelSpec:
    """Runtime representation of a DSL-defined model"""

    def __init__(self, spec_dict: Dict[str, Any]):
        self.name = spec_dict['name']
        self.architecture = spec_dict['architecture']
        self.config = spec_dict['config']
        self.hf_mapping = spec_dict['hf_mapping']
        self._parse_architecture()

    @classmethod
    def from_file(cls, filepath: str) -> 'RuntimeModelSpec':
        """Load DSL spec from .ndsl file"""
        import yaml
        with open(filepath, 'r') as f:
            spec_dict = yaml.safe_load(f)
        return cls(spec_dict)
```

**Example:**
```python
# Load YAML file
spec = RuntimeModelSpec.from_file("models/llama2.ndsl")

# Access parsed data
print(spec.name)  # "Llama2"
print(spec.config['hidden_size'])  # 4096
```

### 1b. Python-Side Compiler

**Purpose:** Compile high-level DSL operations into low-level `BlockSpec` format.

```python
class RuntimeModelSpec:
    def to_block_spec(self, layer_idx: int) -> Dict[str, Any]:
        """Convert DSL block definition to BlockSpec dict"""
        variant = self.architecture['variant']
        block_ops = self._compile_block_ops(self.architecture['block'])

        return {
            'variant': variant,
            'forward_ops': block_ops,
            'use_qk_norm': self._has_qk_norm(),
            'ln2_on_residual_att': variant == 'dense'
        }

    def to_model_config(self) -> 'ModelConfig':
        """Generate ModelConfig from DSL definition"""
        cfg = ModelConfig()
        cfg.VocabSize = self.config['vocab_size']
        cfg.HiddenSize = self.config['hidden_size']
        cfg.NumLayers = self.config['num_layers']
        # ... populate all fields
        return cfg

    def _compile_block_ops(self, block_def: List[Dict]) -> List[str]:
        """Compile high-level block definition to BlockOp sequence"""
        ops = []

        for step in block_def:
            op_type = step['op']

            if op_type == 'RMSNorm':
                # Context-dependent: LN1, LN2, or fused?
                if step.get('fused_with_residual'):
                    ops.append('ResidualLN2')
                elif step['name'] == 'ln1':
                    ops.append('LN1')
                else:
                    ops.append('LN2')

            elif op_type == 'CausalSelfAttention':
                # Expand to constituent ops
                expands = step.get('expands_to', ['QKV', 'RoPE', 'Attention', 'AttnOut'])
                if step['config'].get('use_qk_norm'):
                    expands.insert(1, 'QKNorm')  # After QKV
                ops.extend(expands)

            elif op_type == 'SwiGLU_FFN':
                expands = step.get('expands_to', ['MLPUp', 'SwiGLU', 'MLPDown'])
                ops.extend(expands)

            elif op_type == 'MoE_FFN':
                expands = step.get('expands_to', ['Router', 'Experts', 'Combine'])
                ops.extend(expands)

            elif op_type == 'ResidualAdd':
                # Handled implicitly by variant
                if step.get('fused_with_next'):
                    pass  # Next op will be fused
                elif self.architecture['variant'] == 'parallel':
                    ops.append('ResidualAdd')

        return ops
```

**Example:**
```python
# Input: DSL operations
block = [
    {'op': 'RMSNorm', 'name': 'ln1'},
    {'op': 'CausalSelfAttention', 'config': {'use_qk_norm': False}},
    {'op': 'SwiGLU_FFN'}
]

# Output: Compiled BlockOp sequence
spec.to_block_spec(layer_idx=0)
# Returns:
# {
#     'variant': 'dense',
#     'forward_ops': ['LN1', 'QKV', 'RoPE', 'Attention', 'AttnOut',
#                     'ResidualLN2', 'MLPUp', 'SwiGLU', 'MLPDown'],
#     'use_qk_norm': False
# }
```

### 2. Runtime Model Registry

**Purpose:** Manage available DSL-defined models globally (similar to HuggingFace model hub).

```python
# surogate/dsl/registry.py

class RuntimeModelRegistry:
    """Global registry for DSL-defined models"""

    _models: Dict[str, RuntimeModelSpec] = {}

    @classmethod
    def register(cls, name: str, spec_file: str):
        """Register a DSL model from file"""
        spec = RuntimeModelSpec.from_file(spec_file)
        cls._models[name] = spec
        logger.info(f"Registered DSL model: {name}")

    @classmethod
    def register_builtin_models(cls):
        """Register built-in DSL models"""
        builtin_dir = Path(__file__).parent / "models"
        for ndsl_file in builtin_dir.glob("*.ndsl"):
            spec = RuntimeModelSpec.from_file(str(ndsl_file))
            cls._models[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> RuntimeModelSpec:
        """Get registered model spec"""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not registered. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered DSL models"""
        return list(cls._models.keys())

    @classmethod
    def create_model_config(cls, name: str) -> 'ModelConfig':
        """Create ModelConfig for registered model"""
        spec = cls.get(name)
        return spec.to_model_config()

    @classmethod
    def create_block_spec(cls, name: str, layer_idx: int) -> Dict[str, Any]:
        """Create BlockSpec dict for a specific layer"""
        spec = cls.get(name)
        return spec.to_block_spec(layer_idx)
```

**Example:**
```python
# Register a custom model
RuntimeModelRegistry.register("MyModel", "models/my_model.ndsl")

# Later, retrieve it
spec = RuntimeModelRegistry.get("MyModel")
config = RuntimeModelRegistry.create_model_config("MyModel")
```

### 3. C++ Integration (Extended BlockBuilder)

**Purpose:** Convert Python dict (from compiler) into actual C++ `BlockSpec` objects that `BlockExecutor` can run.

```cpp
// csrc/src/modules/composite/runtime_block_builder.h

namespace modules {

/**
 * Runtime block builder that constructs BlockSpecs from DSL definitions.
 * Extends the static BlockBuilder with data-driven spec generation.
 */
class RuntimeBlockBuilder {
public:
    /**
     * Build BlockSpec from runtime DSL specification.
     *
     * @param dsl_spec Serialized DSL spec (parsed in Python, passed as dict)
     * @param layer_idx Current layer index
     * @return BlockSpec for execution
     */
    static BlockSpec build_from_dsl(
        const py::dict& dsl_spec,
        int layer_idx)
    {
        BlockSpec spec;

        // Parse variant
        std::string variant = dsl_spec["variant"].cast<std::string>();
        spec.variant = parse_variant(variant);

        // Parse forward ops
        py::list ops = dsl_spec["forward_ops"];
        for (auto op_handle : ops) {
            std::string op_str = op_handle.cast<std::string>();
            spec.forward_ops.push_back(parse_block_op(op_str));
        }

        // Parse flags
        spec.use_qk_norm = dsl_spec["use_qk_norm"].cast<bool>();
        spec.ln2_on_residual_att = dsl_spec["ln2_on_residual_att"].cast<bool>();

        return spec;
    }

private:
    static BlockVariant parse_variant(const std::string& variant) {
        if (variant == "dense") return BlockVariant::Dense;
        if (variant == "parallel") return BlockVariant::Parallel;
        if (variant == "moe") return BlockVariant::MoE;
        throw std::runtime_error("Unknown variant: " + variant);
    }

    static BlockOp parse_block_op(const std::string& op_str) {
        static const std::unordered_map<std::string, BlockOp> op_map = {
            {"LN1", BlockOp::LN1},
            {"QKV", BlockOp::QKV},
            {"QKNorm", BlockOp::QKNorm},
            {"RoPE", BlockOp::RoPE},
            {"Attention", BlockOp::Attention},
            {"AttnOut", BlockOp::AttnOut},
            {"ResidualAdd", BlockOp::ResidualAdd},
            {"ResidualLN2", BlockOp::ResidualLN2},
            {"LN2", BlockOp::LN2},
            {"MLPUp", BlockOp::MLPUp},
            {"SwiGLU", BlockOp::SwiGLU},
            {"MLPDown", BlockOp::MLPDown},
            {"Router", BlockOp::Router},
            {"Experts", BlockOp::Experts},
            {"Combine", BlockOp::Combine},
        };

        auto it = op_map.find(op_str);
        if (it == op_map.end()) {
            throw std::runtime_error("Unknown BlockOp: " + op_str);
        }
        return it->second;
    }
};

} // namespace modules
```

**Example:**
```cpp
// Input: Python dict from compiler
py::dict dsl_spec;
dsl_spec["variant"] = "dense";
dsl_spec["forward_ops"] = py::list({"LN1", "QKV", "RoPE", "Attention", ...});
dsl_spec["use_qk_norm"] = false;

// Output: C++ BlockSpec object
BlockSpec spec = RuntimeBlockBuilder::build_from_dsl(dsl_spec, layer_idx=0);

// Now BlockExecutor can execute it
BlockExecutor::forward(spec, recipe, rs, weights, ...);
```

### 4. Python Bindings

**Purpose:** Expose C++ `BlockSpec` types to Python so they can be passed back and forth.

```cpp
// csrc/src/binding/py_dsl.cpp

void bind_dsl_runtime(py::module_& m) {
    // Expose RuntimeBlockBuilder to Python
    py::class_<modules::RuntimeBlockBuilder>(m, "RuntimeBlockBuilder")
        .def_static("build_from_dsl", &modules::RuntimeBlockBuilder::build_from_dsl,
                    py::arg("dsl_spec"), py::arg("layer_idx"),
                    "Build BlockSpec from DSL specification");

    // Expose BlockSpec to Python for inspection
    py::class_<modules::BlockSpec>(m, "BlockSpec")
        .def_readonly("variant", &modules::BlockSpec::variant)
        .def_readonly("use_qk_norm", &modules::BlockSpec::use_qk_norm)
        .def_readonly("ln2_on_residual_att", &modules::BlockSpec::ln2_on_residual_att)
        .def_readonly("forward_ops", &modules::BlockSpec::forward_ops);

    py::enum_<modules::BlockVariant>(m, "BlockVariant")
        .value("Dense", modules::BlockVariant::Dense)
        .value("Parallel", modules::BlockVariant::Parallel)
        .value("MoE", modules::BlockVariant::MoE);

    py::enum_<modules::BlockOp>(m, "BlockOp")
        .value("LN1", modules::BlockOp::LN1)
        .value("QKV", modules::BlockOp::QKV)
        .value("QKNorm", modules::BlockOp::QKNorm)
        .value("RoPE", modules::BlockOp::RoPE)
        .value("Attention", modules::BlockOp::Attention)
        .value("AttnOut", modules::BlockOp::AttnOut)
        .value("ResidualAdd", modules::BlockOp::ResidualAdd)
        .value("ResidualLN2", modules::BlockOp::ResidualLN2)
        .value("LN2", modules::BlockOp::LN2)
        .value("MLPUp", modules::BlockOp::MLPUp)
        .value("SwiGLU", modules::BlockOp::SwiGLU)
        .value("MLPDown", modules::BlockOp::MLPDown)
        .value("Router", modules::BlockOp::Router)
        .value("Experts", BlockOp::Experts)
        .value("Combine", modules::BlockOp::Combine);
}
```

## Why Python for the Compiler?

### Three Key Reasons

1. **Easy YAML Parsing**
   - Python has native YAML support (`yaml.safe_load()`)
   - No need to write custom parsers in C++
   - Can leverage existing Python libraries

2. **Flexible String Manipulation & Compilation**
   - Easy to transform DSL syntax into BlockSpec format
   - Simple dict/list operations for building operation sequences
   - Quick to iterate on compiler logic without recompilation

3. **User-Facing API**
   - Training scripts are already in Python
   - Users define models, configs, and training in Python
   - Natural integration with existing workflow

### What Python Does NOT Do

- ❌ Does NOT execute models (that's C++/CUDA)
- ❌ Does NOT run CUDA kernels
- ❌ Does NOT compute gradients
- ❌ Does NOT allocate GPU memory

### What Python DOES Do

- ✅ Parses YAML files
- ✅ Compiles DSL → BlockSpec dict
- ✅ Manages model registry
- ✅ Provides user-facing API

**The key insight:** Python is the **"compiler frontend"** (parses and compiles DSL), while C++ is the **"execution backend"** (runs the compiled BlockSpec). This is the same pattern as PyTorch: Python defines models, C++/CUDA executes them.

## Usage Examples

### Defining a New Model

```yaml
# models/my_custom_model.ndsl
name: MyCustomModel
type: model

architecture:
  variant: dense

  block:
    - op: RMSNorm
      name: ln1
      config: {eps: 1e-5}

    - op: CausalSelfAttention
      config:
        use_qk_norm: true   # Enable QK normalization
        use_qkv_bias: false

    - op: ResidualAdd
      fused_with_next: ln2

    - op: RMSNorm
      name: ln2
      config: {eps: 1e-5}
      fused_with_residual: true

    - op: SwiGLU_FFN

    - op: ResidualAdd

config:
  vocab_size: 50000
  hidden_size: 2048
  num_layers: 24
  num_query_heads: 32
  num_kv_heads: 8    # GQA with 8 KV heads
  intermediate_size: 8192
  max_position_embeddings: 4096
  rms_norm_eps: 1e-5
  rope_theta: 10000.0

hf_architecture: "MyCustomModelForCausalLM"
hf_mapping:
  # ... define weight mappings
```

### Training with DSL Model

```python
# train.py
from surogate.dsl import RuntimeModelRegistry
from surogate import Trainer

# Register custom model
RuntimeModelRegistry.register("MyCustomModel", "models/my_custom_model.ndsl")

# Create config from DSL
config = RuntimeModelRegistry.create_model_config("MyCustomModel")

# Train as usual - uses BlockExecutor with DSL-generated BlockSpec
trainer = Trainer(
    model_config=config,
    training_args=training_args,
    use_modular_blocks=True  # Enable BlockExecutor path
)
trainer.train()
```

### CLI Integration

```bash
# List available DSL models
surogate list-models --dsl

# Validate DSL model definition
surogate validate models/my_custom_model.ndsl

# Train with DSL model
surogate sft config.yaml --model MyCustomModel

# Or train from DSL file directly
surogate sft config.yaml --model-dsl models/my_custom_model.ndsl
```

### Inspecting Generated BlockSpec

```python
from surogate.dsl import RuntimeModelRegistry

# Load model
spec = RuntimeModelRegistry.get("Llama2")

# Generate BlockSpec for layer 0
block_spec_dict = spec.to_block_spec(layer_idx=0)
print(f"Variant: {block_spec_dict['variant']}")
print(f"Forward ops: {block_spec_dict['forward_ops']}")

# Output:
# Variant: dense
# Forward ops: ['LN1', 'QKV', 'RoPE', 'Attention', 'AttnOut', 'ResidualLN2', 'MLPUp', 'SwiGLU', 'MLPDown']
```

## Implementation Roadmap

### Phase 1: Core Parser (Week 1-2)
- [ ] YAML parser for basic DSL syntax
- [ ] `RuntimeModelSpec` class
- [ ] `RuntimeModelRegistry` for model registration
- [ ] Basic BlockSpec generation (dense variant only)
- [ ] Unit tests for parser

### Phase 2: C++ Integration (Week 2-3)
- [ ] `RuntimeBlockBuilder` C++ class
- [ ] Python bindings for BlockSpec inspection
- [ ] Integration with existing `BlockExecutor`
- [ ] Model factory support for DSL models

### Phase 3: Full Feature Support (Week 3-4)
- [ ] Parallel residual variant support
- [ ] MoE variant support
- [ ] Conditional operations (QKNorm, sliding window)
- [ ] HuggingFace weight mapping
- [ ] Per-layer customization

### Phase 4: Tooling & Documentation (Week 4-5)
- [ ] CLI commands (`list-models`, `validate`)
- [ ] DSL validation and error messages
- [ ] Built-in model library (Llama2, GPT-NeoX, Qwen3-MoE)
- [ ] User guide and examples
- [ ] Migration guide for existing models

### Phase 5: Advanced Features (Week 5-6)
- [ ] Layer-specific overrides (heterogeneous architectures)
- [ ] Custom operation definitions
- [ ] DSL composition (import/extend)
- [ ] Visual BlockSpec debugger
- [ ] Performance profiling per operation

## Design Decisions & Trade-offs

### Why YAML?

**Pros:**
- Readable and writable by humans
- Native Python support
- Supports comments
- Hierarchical structure

**Cons:**
- Not as expressive as a custom language
- No syntax highlighting (yet)

**Alternative:** Custom DSL syntax (like your original proposal) with a proper parser. Could be added later with YAML as intermediate representation.

### Why Compile to BlockSpec?

**Pros:**
- Reuses existing, battle-tested execution engine
- Zero overhead vs hand-written BlockSpec
- All optimizations (FP8, FP4, recipes) work automatically
- Gradual migration path

**Cons:**
- Limited to operations supported by BlockExecutor
- Can't define entirely new primitive operations without C++ changes

**Mitigation:** New primitives can be added to BlockExecutor as needed. The DSL focuses on **composition**, not defining new low-level operations.

### Runtime vs Compile-Time

**Runtime Interpretation:**
- ✅ No recompilation needed
- ✅ Easy to debug and iterate
- ✅ Can load models dynamically
- ❌ Small overhead from Python-C++ calls (negligible)

**Compile-Time (Code Generation):**
- ✅ Absolutely zero overhead
- ✅ Can validate at compile time
- ❌ Requires recompilation
- ❌ Slower development iteration

**Decision:** Start with runtime interpretation. Can add optional code generation later for maximum performance.

## Limitations & Future Work

### Current Limitations

1. **Fixed Operation Set**: Limited to operations in `BlockOp` enum
2. **No Custom Kernels**: Can't define new CUDA kernels from DSL
3. **Block-Level Only**: Can't customize within operations (e.g., attention mechanism details)
4. **Single Block Template**: All layers use same block structure (can add layer overrides later)

### Future Extensions

1. **Custom Operations**: Allow DSL to register new BlockOps backed by user CUDA kernels
2. **Nested Composition**: Define sub-blocks and compose them
3. **Conditional Graphs**: Runtime branching based on input (for vision-language models)
4. **Optimization Passes**: DSL compiler can fuse/reorder ops automatically
5. **Cross-Architecture**: Same DSL for GPUs, TPUs, other accelerators
6. **Visual Editor**: Drag-and-drop model architecture builder

## Conclusion

The runtime DSL interpreter provides a **declarative, composable way to define new model architectures** without sacrificing performance or requiring code generation. It:

- **Leverages existing infrastructure**: BlockExecutor, recipes, primitives
- **Enables rapid iteration**: No recompilation needed
- **Maintains performance**: Compiles to same BlockSpec as hand-written code
- **Improves accessibility**: Define models without C++ knowledge
- **Supports all features**: FP8, FP4, MoE, LoRA, all recipes

The key insight is that **BlockSpec is already a runtime IR** for transformer blocks. The DSL simply provides a higher-level, more readable way to generate BlockSpecs instead of writing them manually in C++.

This approach makes Surogate more accessible to researchers who want to experiment with novel architectures while keeping the performance-critical execution path in highly optimized C++/CUDA code.
