# Multi-Node Training

Surogate supports multi-node distributed training using [Ray](https://www.ray.io/) for cluster management and NCCL for high-speed GPU communication. This guide explains how to set up and run training across multiple machines.

## Overview

In multi-node training:
- **Ray** manages the cluster and coordinates work across nodes
- **NCCL** handles GPU-to-GPU communication (uses InfiniBand when available, falls back to TCP)
- Each node runs all its local GPUs using the threaded backend
- Data is automatically sharded across nodes for efficient parallel processing

## Setup

### 1. Install Ray

First, install Surogate with Ray support:

```bash
pip install surogate[distributed]
```

### 2. Start the Ray Cluster

On your **head node** (the main machine), start Ray:

```bash
ray start --head --port=6379
```

Ray will output the head node's IP address. Note this down—you'll need it for connecting worker nodes.

On each **worker node**, connect to the head node:

```bash
ray start --address='<head-node-ip>:6379'
```

Replace `<head-node-ip>` with the actual IP address from the head node output.

### 3. Configure Your Training

Add a `distributed` section to your training configuration YAML:

```yaml
# Your existing config...
model: meta-llama/Llama-3.1-8B
per_device_train_batch_size: 2
sequence_len: 2048

# Multi-node configuration
distributed:
  ray_address: "auto"          # Connect to existing Ray cluster
  num_nodes: 2                 # Total number of nodes
  gpus_per_node: 8             # GPUs per node (0 = use config.gpus)
  worker_output_dir: null      # Optional: shared directory for tokenized data
```

**Configuration options:**
- `ray_address`: How to connect to Ray
  - `"auto"`: Connect to existing cluster (recommended)
  - `"local"`: Start local Ray instance (single-node testing)
  - `"ray://host:port"`: Connect to specific head node
- `num_nodes`: Total number of nodes to use
- `gpus_per_node`: GPUs per node (leave at 0 to use `gpus` from main config)
- `worker_output_dir`: Optional directory for worker-local data (defaults to `/tmp/surogate-{run_name}`)

### 4. Run Training

Start training from the head node:

```bash
surogate sft config.yaml
```

Surogate automatically detects the distributed configuration and:
1. Spawns one Ray worker per node
2. Each node downloads the model independently
3. Each node tokenizes its own data shard (1/num_nodes of the dataset)
4. All nodes synchronize before initializing NCCL
5. Training proceeds with gradients synchronized across nodes

## How It Works

### Data Sharding

When using distributed training, each node processes a different portion of your dataset:
- Training data is automatically split: node 0 gets samples 0, N, 2N, ...; node 1 gets samples 1, N+1, 2N+1, ...
- Each node tokenizes only its shard, reducing memory pressure and enabling parallel processing
- Validation data is replicated on all nodes for consistent metrics

### Model Synchronization

- All nodes start with identical model weights
- During training, each node computes gradients on its local data
- NCCL synchronizes gradients across nodes after each training step
- The optimizer updates are applied identically on all nodes, keeping weights in sync

### Network Communication

Surogate automatically configures NCCL for optimal performance:
- Uses InfiniBand/RoCE when available for maximum bandwidth
- Falls back to TCP sockets over standard network interfaces
- Auto-detects the correct network interface on each node

## Monitoring

During training, Surogate logs progress from the head node:

```
Starting distributed training with 2 nodes...
  Nodes: 2
  GPUs per node: 8
  Total GPUs: 16
  Tokens per step: 524288
  Max steps: 1000

Step 0/1000 | Loss: 2.3456 | Norm: 1.23 | LR: 1.00e-05 | 2.34s | 223891 tok/s
Step 1/1000 | Loss: 2.2891 | Norm: 1.18 | LR: 2.00e-05 | 2.31s | 226926 tok/s
...
```

The logged metrics are averaged across all nodes.

## Checkpointing and Export

- Only node 0 saves checkpoints and exports the final model
- All nodes participate in checkpoint synchronization (NCCL barriers)
- Checkpoints are saved to the `output_dir` on the head node
- LoRA adapters are exported in PEFT-compatible format

## Troubleshooting

### Connection Issues

If nodes can't communicate:
1. Check firewall rules allow traffic on the required ports
2. Verify all nodes can reach each other: `ping <node-ip>`
3. Check NCCL logs in `/tmp/nccl_debug_node_*.log` (if debug is enabled)

### Performance Tips

- Use InfiniBand or high-bandwidth networking for best performance
- Ensure all nodes have identical GPU configurations
- Use a shared filesystem (NFS, Lustre) for `worker_output_dir` to enable data caching across runs
- Increase `gradient_accumulation_steps` to reduce communication frequency

### Ray Cluster Management

View cluster status:
```bash
ray status
```

Stop all Ray processes:
```bash
ray stop
```

## Example: Training Llama 3.1 8B on 2 Nodes

```yaml
# config.yaml
model_name: meta-llama/Llama-3.1-8B
dataset: timdettmers/openassistant-guanaco

per_device_train_batch_size: 4
sequence_len: 2048
gradient_accumulation_steps: 4
num_epochs: 3

learning_rate: 2e-5
warmup_steps: 100

output_dir: ./output/llama-3.1-8b-guanaco

distributed:
  ray_address: "auto"
  num_nodes: 2
  gpus_per_node: 8
```

Start Ray on both nodes, then run:
```bash
surogate sft config.yaml
```

This configuration trains on 16 GPUs total (2 nodes × 8 GPUs) with an effective batch size of 128 sequences per step.

