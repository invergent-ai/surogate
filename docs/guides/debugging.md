# Debugging training issues

## Training memory breakdown

You can visualize the memory breakdown of your model training setting the `debug_memory_breakdown` flag to `true`. This will print a detailed breakdown of memory usage by different components such as model parameters, optimizer states, gradients, and activations and propose optimization suggestions:

```shell
================================================================================
                              MEMORY BREAKDOWN
================================================================================

[Allocator Segments]
--------------------------------------------------------------------------------
Segment                                   Device      Managed       Pinned
--------------------------------------------------------------------------------
FP8_Embeddings                             296.8          0.0          0.0
FP8_DequantBuf                              15.0          0.0          0.0
FP8_Weights                                420.2          0.0          0.0
Modular_LoRA_RunState                        6.1          0.0          0.0
Modular_LoRA_Grads                          77.0          0.0          0.0
Modular_LoRA_Weights                        57.8          0.0          0.0
Free                                      4665.1          0.0          0.0
Reserved                                   382.4          0.0          0.0
Other                                     1885.2          0.0          0.0
--------------------------------------------------------------------------------

Memory by Category:
--------------------------------------------------------------------------------
Category                              Size (MiB)   Top Tensors
--------------------------------------------------------------------------------
Other                                     1405.1   mlp_up_w(336M), attn_qkv_w(224M), gate_up_fp8(168M)
Model Weights                              608.5   embeddings(297M), embedding(297M), fp8_mlp_up_weight(6M)
Activations (per-layer)                    250.2   att(112M), ln2(56M), ln1(56M)
LoRA Adapters                              140.8   lora_slice(6M)
Gradients                                   50.0   d_mlp_up_w(12M), d_qkv_w(8M), d_mlp_down_w(6M)
Workspace/Temp                              32.0   cublas_ws(32M)
QLoRA Dequant Buffers                       15.0   dequant_gate_up(6M), dequant_qkv(4M), dequant_down(3M)
QLoRA Quantized Weights                      0.1
--------------------------------------------------------------------------------
TOTAL DEVICE MEMORY                       2501.7

QLoRA-Specific Memory:
  Quantized base weights:          717.0 MiB
  Memory savings ratio:            0.4x vs FP16

Stack Allocations (high-water mark):
--------------------------------------------------------------------------------
  output_simulate                                74.2 MiB
--------------------------------------------------------------------------------
  STACK TOTAL                                    74.2 MiB

Top 20 Largest Tensors:
--------------------------------------------------------------------------------
  mlp_up_w                                      336.0 MiB
  embeddings                                    296.8 MiB
  embedding                                     296.8 MiB
  attn_qkv_w                                    224.0 MiB
  gate_up_fp8                                   168.0 MiB
  mlp_down_w                                    168.0 MiB
  attn_out_w                                    112.0 MiB
  qkv_fp8                                       112.0 MiB
  att                                           112.0 MiB
  down_fp8                                       84.0 MiB
  stack                                          74.2 MiB
  ln2                                            56.1 MiB
  ln1                                            56.1 MiB
  out_fp8                                        56.0 MiB
  cublas_ws                                      32.0 MiB
  d_mlp_up_w                                     12.0 MiB
  d_qkv_w                                         8.0 MiB
  qkv_shared                                      8.0 MiB
  d_mlp_down_w                                    6.0 MiB
  lora_slice                                      6.0 MiB

Model Configuration:
  Hidden size (C):                1024
  Intermediate size (D):          3072
  Num layers (L):                   28
  Batch size (B):                    1
  Sequence length (T):            1024

Theoretical Activation Memory (BF16, no sharing):
  Per layer:                      36.0 MiB
  All 28 layers:                1008.0 MiB

CUDA Memory Analysis:
--------------------------------------------------------------------------------
  CUDA total used (nvidia-smi):   3140.4 MiB
  Tracked by TensorAllocator:     2501.7 MiB
  Untracked CUDA overhead:         638.7 MiB (20.3%)

  Untracked memory breakdown (estimated):
    - CUDA context:             ~200-400 MiB
    - cuDNN handles/workspace:  ~200-500 MiB
    - cuBLAS handles:           ~50-100 MiB
    - Memory fragmentation:     variable
--------------------------------------------------------------------------------

Optimization Suggestions:
  (No major optimization opportunities detected)

================================================================================
```

### Training timing breakdown

You can also enable the `debug_timing_breakdown` flag to get a detailed timing breakdown of different components during training, which can help identify bottlenecks in the training process.

### Debug tokenization issues

If you suspect your dataset is being labeled/masked incorrectly (for example, wrong tokens are ignored), run the tokenizer step with `--debug`:

```bash
surogate tokenize config.yaml --debug
```

This prints tokens alongside their labels so you can confirm that the right spans are being ignored.

---

## See also

- [Memory](memory.md)
- [Config reference](../reference/config.md)
- [Back to docs index](../index.md)
