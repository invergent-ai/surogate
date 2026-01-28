# Golden data generators for DSL ops

This folder contains scripts that generate tiny, deterministic golden cases for DSL compiled ops and composed modules.

## Primitive Operations

Run generators for primitive operations (matmul, swiglu, flash_attention, etc.):

```bash
.venv/bin/python tests/ops/generate_goldens.py --list
.venv/bin/python tests/ops/generate_goldens.py --op matmul_swiglu
.venv/bin/python tests/ops/generate_goldens.py --all
```

## Composed Modules

Module-level golden tests verify correctness of composed operations (multiple primitives chained together):

```bash
# Generate module golden tests
.venv/bin/python tests/ops/generate_goldens.py --op swiglu_mlp
.venv/bin/python tests/ops/generate_goldens.py --op gqa_attention

# Run module tests
./csrc/build/unit-tests "[dsl][modules][goldens]"
```

### Available Module Tests

- **swiglu_mlp**: Complete SwiGLU MLP module (matmul → swiglu → matmul)
- **gqa_attention**: GQA attention module (qkv_proj → rope → flash_attn → out_proj)

## Golden File Format

Output files are JSON under `tests/ops/goldens/`.
Each JSON has:
- `op`: operation or module name
- `case`: case identifier
- `inputs`: input tensors (weights, activations, position_ids, etc.)
- `outputs`: expected output tensors (including intermediate activations)
- `grads`: (optional) expected gradients for backward pass
- `attrs`: operation attributes
- `meta`: metadata (dimensions, notes)

### Module Golden Files

Module golden files include intermediate activations to allow verification of composition:
- **Inputs**: All weights and input data
- **Outputs**: Final outputs + intermediate activations (e.g., `up`, `swiglu` for MLP)
- **Grads**: Gradients for all learnable parameters

## Adding New Tests

### Primitive Operations

Add a generator function in `generate_goldens.py`:

```python
def gen_my_op() -> List[GoldenCase]:
    # Compute reference outputs using PyTorch/NumPy
    # Return GoldenCase with inputs/outputs/grads
    ...

# Register in OP_GENERATORS
OP_GENERATORS = {
    ...
    "my_op": gen_my_op,
}
```

### Composed Modules

Add a module generator that composes multiple primitives:

```python
def gen_my_module() -> List[GoldenCase]:
    """Generate golden for composed module."""
    # Initialize weights and inputs
    # Run forward pass through PyTorch reference
    # Save intermediate activations
    # Compute gradients via autograd
    return [GoldenCase(op="my_module", case="small_case_1", payload={...})]
```

Add a C++ test in [test_dsl_module_goldens.cpp](../../csrc/src/testing/utilities/test_dsl_module_goldens.cpp):

```cpp
TEST_CASE("dsl module goldens: my_module", "[dsl][modules][goldens]") {
    // Load golden file
    // Validate structure and shapes
    // TODO: Add compilation and execution when input injection is implemented
}
```

## Testing Strategy

1. **Primitive ops** ([test_dsl_goldens.cpp](../../csrc/src/testing/utilities/test_dsl_goldens.cpp)): Test individual kernels in isolation
2. **Module composition** ([test_dsl_module_goldens.cpp](../../csrc/src/testing/utilities/test_dsl_module_goldens.cpp)): Verify correctness of composed operations
3. **Integration tests**: Full model forward/backward (see `csrc/src/testing/training/`)

Module tests currently validate:
- Golden file structure and tensor shapes
- Presence of required inputs/outputs/gradients
- Dimension consistency across operations

**Future work**: Full execution requires implementing input injection mechanism for DSL graphs.
