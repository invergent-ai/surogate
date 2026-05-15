# Python API

Surogate’s primary Python API is exposed via the `surogate._surogate` extension module (nanobind).

## Importing

```python
from surogate._surogate import (
    DataLoader,
    OptimizerConfig,
    PretrainedConfig,
    RuntimeOptions,
    SurogateTrainer,
)
```

## Trainer lifecycle (high level)

Typical usage:

1. Build a `PretrainedConfig` (model architecture) and `RuntimeOptions` (runtime/memory/precision).
2. Construct a `SurogateTrainer`.
3. Feed token batches via `step()` and run optimizer updates via `update_with_config()`.
4. Save checkpoints with `save_checkpoint()`.

### Shapes and dtypes

`SurogateTrainer.step()` expects NumPy arrays:

- `inputs`: `int32` shaped `[batch_size * world_size, seq_length]`
- `targets`: `int32` shaped `[batch_size * world_size, seq_length]`

## Useful entry points

- `SurogateTrainer.from_pretrained(...)`: create from pretrained weights/config (see docstring in stubs)
- `SurogateTrainer.import_weights(path)`: import weights from a Hugging Face safetensors file
- `SurogateTrainer.export_model(path)`: export full model weights
- `SurogateTrainer.export_adapter(path, base_model_path=...)`: export LoRA adapters (PEFT-compatible)
- `SurogateTrainer.get_allocator_info(gpu_id=0)`: allocator stats (useful for memory debugging)

## Where to look next

The most complete API surface is documented in the generated typing stubs:

- `surogate/_surogate.pyi`

---

## See also

- [CLI reference](cli.md)
- [Config reference](config.md)
- [Back to docs index](../index.mdx)
