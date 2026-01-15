# Training Metrics & Monitoring

Surogate provides comprehensive training metrics that can be logged to file (JSON), stdout, and external backends like Weights & Biases (wandb) or Aim.

## Metrics Overview

| Category    | Metrics                                       | Description                     |
| ----------- | --------------------------------------------- | ------------------------------- |
| Training    | loss, norm, lr, epoch, tps                    | Core training progress          |
| Evaluation  | loss, tps                                     | Validation performance          |
| GPU         | power, temperature, utilization, memory, PCIe | Hardware monitoring             |
| MoE         | aux_loss, z_loss, utilization, imbalance      | Router health (MoE models only) |
| Performance | SOL, FLOPs                                    | Speed-of-Light estimates        |
| Memory      | allocations by segment                        | Memory usage breakdown          |

## Training Metrics

Logged every training step via `log_step()`:

| Metric              | Type  | Description                                                  |
| ------------------- | ----- | ------------------------------------------------------------ |
| `loss`              | float | Cross-entropy loss for the current step                      |
| `norm`              | float | Gradient norm (L2) after clipping                            |
| `lr`                | float | Current learning rate                                        |
| `epoch`             | float | Fractional epoch progress (e.g., 1.25 = 25% through epoch 2) |
| `step_tokens`       | int   | Number of tokens processed this step                         |
| `duration_ms`       | int   | Step duration in milliseconds                                |
| `tokens_per_second` | float | Throughput (computed: `step_tokens / duration_ms * 1000`)    |

### Console Output

**Dense models:**
```
:: step     100 [ 12.5%] \ loss 2.3456 | norm  0.1234 | 125.3k tps |   156 ms | sol 78.2 | eta 02h15m
```

**MoE models (includes aux_loss and load_imbalance inline):**
```
:: step     100 [ 12.5%] \ loss 2.3456 | norm  0.1234 | 125.3k tps |   156 ms | aux 0.0234 | imbal 1.45 | sol 78.2 | eta 02h15m
```

Legend:
- `\ / ` - Loss trend indicator (decreasing/increasing)
- `!` before norm - Gradient spike detected (>5x moving average)
- `aux` - MoE auxiliary load balancing loss (MoE models only)
- `imbal` - MoE load imbalance ratio (MoE models only)
- `sol` - Speed-of-light percentage (actual vs theoretical peak)
- `eta` - Estimated time remaining

### JSON Log Format

**Dense models:**
```json
{
  "log": "step",
  "time": "2026-01-15T10:30:00.123456",
  "step": 100,
  "epoch": 1.25,
  "step_tokens": 16384,
  "duration_ms": 156,
  "norm": 0.1234,
  "loss": 2.3456,
  "lr": 0.0002
}
```

**MoE models (includes MoE metrics inline):**
```json
{
  "log": "step",
  "time": "2026-01-15T10:30:00.123456",
  "step": 100,
  "epoch": 1.25,
  "step_tokens": 16384,
  "duration_ms": 156,
  "norm": 0.1234,
  "loss": 2.3456,
  "lr": 0.0002,
  "moe_aux_loss": 0.023400,
  "moe_z_loss": 0.001200,
  "moe_load_imbalance": 1.4500
}
```

## Evaluation Metrics

Logged during validation via `log_eval()`:

| Metric              | Type  | Description                                                        |
| ------------------- | ----- | ------------------------------------------------------------------ |
| `loss`              | float | Evaluation loss                                                    |
| `eval_tokens`       | int   | Number of tokens evaluated                                         |
| `duration_ms`       | int   | Evaluation duration                                                |
| `tokens_per_second` | float | Evaluation throughput                                              |
| `gap`               | float | Difference between eval loss and average train loss (console only) |

### Console Output

```
>> eval                      loss 2.1234 | gap +0.0123 | 145.2k tps |   234 ms
```

### JSON Log Format

```json
{
  "log": "eval",
  "time": "2026-01-15T10:35:00.123456",
  "step": 500,
  "epoch": 1.5,
  "eval_tokens": 32768,
  "duration_ms": 234,
  "loss": 2.1234
}
```

## MoE Metrics

For Mixture-of-Experts models (e.g., Qwen3-MoE), additional metrics track router health:

| Metric               | Type  | Range | Description                                                  |
| -------------------- | ----- | ----- | ------------------------------------------------------------ |
| `aux_loss`           | float | ≥0    | Load balancing auxiliary loss (sum across layers)            |
| `z_loss`             | float | ≥0    | Router z-loss for logit regularization (sum across layers)   |
| `expert_utilization` | float | 0-1   | Fraction of experts receiving tokens (average across layers) |
| `load_imbalance`     | float | ≥1    | Ratio of max to mean token counts per expert (1.0 = perfect) |

### Interpreting MoE Metrics

| Metric               | Healthy Range | Warning Signs                         |
| -------------------- | ------------- | ------------------------------------- |
| `aux_loss`           | 0.001-0.1     | >1.0 indicates severe load imbalance  |
| `z_loss`             | 0.0001-0.01   | >0.1 may indicate routing collapse    |
| `expert_utilization` | 0.8-1.0       | <0.5 means many experts are unused    |
| `load_imbalance`     | 1.0-2.0       | >3.0 indicates poor load distribution |

### JSON Log Format

```json
{
  "log": "moe",
  "time": "2026-01-15T10:30:00.123456",
  "step": 100,
  "aux_loss": 0.0234,
  "z_loss": 0.0012,
  "expert_utilization": 0.9375,
  "load_imbalance": 1.45
}
```

### Tracking in External Backends

**Weights & Biases:**
- `moe/aux_loss`
- `moe/z_loss`
- `moe/expert_utilization`
- `moe/load_imbalance`

**Aim:**
- Same metric names tracked via `run.track()`

## GPU Metrics

Hardware telemetry logged via `log_gpu_state()`:

| Metric          | Unit    | Description                      |
| --------------- | ------- | -------------------------------- |
| `clock`         | MHz     | Current GPU clock speed          |
| `max_clock`     | MHz     | Maximum GPU clock speed          |
| `power`         | mW      | Current power draw               |
| `power_limit`   | mW      | Configured power limit           |
| `temperature`   | °C      | GPU temperature                  |
| `temp_slowdown` | °C      | Temperature throttling threshold |
| `gpu_util`      | %       | GPU compute utilization          |
| `mem_util`      | %       | GPU memory utilization           |
| `dram_free`     | bytes   | Free GPU memory                  |
| `pcie_rx`       | bytes/s | PCIe receive bandwidth           |
| `pcie_tx`       | bytes/s | PCIe transmit bandwidth          |
| `fan`           | %       | Fan speed (0 if passive cooling) |
| `throttle`      | string  | Throttling reason (if any)       |

### JSON Log Format

```json
{
  "log": "gpu",
  "time": "2026-01-15T10:30:00.123456",
  "step": 100,
  "id": 0,
  "clock": 1800,
  "max_clock": 2100,
  "fan": 65,
  "power": 350000,
  "power_limit": 400000,
  "temperature": 72,
  "temp_slowdown": 83,
  "gpu_util": 98,
  "mem_util": 85,
  "throttle": "",
  "dram_free": 8589934592,
  "pcie_rx": 1073741824,
  "pcie_tx": 536870912
}
```

### Tracking in External Backends

**Weights & Biases:**
- `gpu/clock`, `gpu/power`, `gpu/temperature`, etc.
- `dram_free` converted to MiB
- `pcie_rx`/`pcie_tx` converted to MiB/s

**Aim:**
- `gpu/{gpu_id}/{metric_name}` (e.g., `gpu/0/power`)

## Speed-of-Light (SOL) Metrics

Theoretical peak performance estimates via `log_sol_estimate()`:

| Metric      | Type    | Description                                     |
| ----------- | ------- | ----------------------------------------------- |
| `blocks`    | FLOPs   | Floating-point operations in transformer blocks |
| `lm_head`   | FLOPs   | Floating-point operations in LM head            |
| `attention` | FLOPs   | Floating-point operations in attention          |
| `tps`       | int     | Theoretical peak tokens/second                  |
| `tf32_peak` | TFLOP/s | GPU peak TF32 throughput                        |
| `bf16_peak` | TFLOP/s | GPU peak BF16 throughput                        |
| `fp16_peak` | TFLOP/s | GPU peak FP16 throughput                        |
| `fp8_peak`  | TFLOP/s | GPU peak FP8 throughput                         |

### Console Output

```
[Speed of Light]
  Peak BF16:    989.5 TFLOP/s
  Blocks:       123.4 GFLOP   in bf16
  LM-Head:       45.6 GFLOP   in bf16
  Attention:     78.9 GFLOP   in bf16
  SOL:          156000 tok/s
```

### Per-Step SOL Percentage

Each training step shows SOL percentage in the console:
```
:: step     100 [ 12.5%] \ loss 2.3456 | norm  0.1234 | 125.3k tps |   156 ms | sol 78.2
```

This represents `(theoretical_time / actual_time) * 100`, where higher is better (80%+ is excellent).

## Memory Metrics

Memory allocation tracking via `log_allocator()`:

| Field      | Type   | Description            |
| ---------- | ------ | ---------------------- |
| `name`     | string | Allocator segment name |
| `device`   | bytes  | GPU device memory      |
| `managed`  | bytes  | CUDA managed memory    |
| `pinned`   | bytes  | Pinned host memory     |
| `pageable` | bytes  | Pageable host memory   |

### Common Segment Names

| Segment       | Description                          |
| ------------- | ------------------------------------ |
| `weights`     | Model weights                        |
| `gradients`   | Gradient buffers                     |
| `optimizer`   | Optimizer state (momentum, variance) |
| `activations` | Forward pass activations             |
| `lora`        | LoRA adapter weights                 |
| `qlora`       | QLoRA quantized weights and scales   |

### JSON Log Format

```json
{
  "log": "allocator",
  "time": "2026-01-15T10:30:00.123456",
  "step": 0,
  "stats": [
    {"name": "weights", "device": 4294967296, "managed": 0, "pinned": 0, "pageable": 0},
    {"name": "gradients", "device": 1073741824, "managed": 0, "pinned": 0, "pageable": 0},
    {"name": "optimizer", "device": 2147483648, "managed": 0, "pinned": 8589934592, "pageable": 0}
  ]
}
```

### Tracking in External Backends

**Weights & Biases:**
- Pie chart visualization of GPU allocations by segment

**Aim:**
- `allocator/{segment_name}_mib` for each segment

## Dataset Metrics

Dataset information logged at startup via `log_dataset()`:

| Field         | Type   | Description               |
| ------------- | ------ | ------------------------- |
| `split`       | string | "train" or "eval"         |
| `files`       | int    | Number of data files      |
| `tokens`      | int    | Total tokens in split     |
| `file_index`  | int    | Current file index        |
| `chunk_index` | int    | Current chunk index       |
| `seed`        | int    | Random seed for shuffling |

## GPU Model Information

Hardware info logged at startup via `log_gpu_model()`:

| Field           | Type   | Description                                    |
| --------------- | ------ | ---------------------------------------------- |
| `name`          | string | GPU model name (e.g., "NVIDIA H100 80GB HBM3") |
| `sm_count`      | int    | Number of streaming multiprocessors            |
| `major`/`minor` | int    | CUDA compute capability                        |
| `memory`        | bytes  | Total GPU memory                               |
| `free`          | bytes  | Free GPU memory at startup                     |
| `l2_size`       | bytes  | L2 cache size                                  |
| `uuid`          | string | GPU UUID                                       |
| `ecc`           | bool   | ECC memory enabled                             |
| `cuda_driver`   | int    | CUDA driver version                            |
| `cuda_runtime`  | int    | CUDA runtime version                           |

## Configuration

### Enabling Metric Backends

```yaml
# config.yaml
report_to: wandb              # Single backend
report_to: [wandb, aim]       # Multiple backends

# Weights & Biases options
wandb_project: my-project
wandb_name: my-run

# Aim options
aim_experiment: my-experiment
aim_repo: /path/to/aim/repo
aim_name: my-run
```

### Log File Location

```yaml
log_file: ./output/training.json  # Default: {output_dir}/training.json
```

### Verbosity Levels

Control console output detail:

| Level | Description                                             |
| ----- | ------------------------------------------------------- |
| -1    | Minimal (eval only)                                     |
| 0     | Normal (default)                                        |
| 1     | Verbose (includes GPU telemetry, per-file dataset info) |

## Programmatic Access

### Python API

```python
from surogate import _surogate

# Create logger
logger = _surogate.TrainingRunLogger(
    "/path/to/log.json",
    callback=my_callback,  # Optional: receives JSON lines
    verbosity=_surogate.LogVerbosity.DEFAULT
)

# Log metrics
logger.log_step(step, epoch, tokens, duration_ms, norm, loss, lr)
logger.log_eval(step, epoch, tokens, duration_ms, loss)
logger.log_moe_stats(step, aux_loss, z_loss, utilization, imbalance)

# Get MoE stats from trainer
moe_stats = trainer.get_moe_stats()
# Returns: {'aux_loss': float, 'z_loss': float,
#           'expert_utilization': float, 'load_imbalance': float, 'valid': bool}
```

### Reading Log Files

The log file is a JSON array that can be parsed directly:

```python
import json

with open("training.json") as f:
    logs = json.load(f)

# Filter by log type
train_steps = [l for l in logs if l.get("log") == "step"]
eval_steps = [l for l in logs if l.get("log") == "eval"]
moe_logs = [l for l in logs if l.get("log") == "moe"]
```

## Best Practices

1. **Monitor loss trend**: The `\ /` indicators help spot divergence early
2. **Watch gradient norms**: The `!` indicator flags potential instability
3. **Track SOL percentage**: Sustained <60% SOL may indicate a bottleneck
4. **For MoE models**: Keep `expert_utilization` > 0.7 and `load_imbalance` < 2.5
5. **Use external backends**: wandb/Aim provide better visualization for long runs
