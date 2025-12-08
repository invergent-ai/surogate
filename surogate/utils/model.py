import math
import re
from typing import Optional

from transformers import PretrainedConfig

def estimate_model_parameters(config: PretrainedConfig) -> int:
    # Dense components
    embed_params = config.vocab_size * config.hidden_size

    # Attention parameters (always dense)
    attn_params_per_layer = (
            config.hidden_size * config.hidden_size * 3 +  # Q, K, V
            config.hidden_size * config.hidden_size         # Output projection
    )

    # Normalization parameters
    norm_params_per_layer = config.hidden_size * 2  # Pre and post attention

    # FFN parameters
    if getattr(config, 'use_moe', False):
        # MoE: gating + experts
        gate_params = config.hidden_size
        expert_params = (
                                config.hidden_size * config.intermediate_size * 3  # Gate, up, down
                        ) * config.num_experts
        ffn_params_per_layer = gate_params + expert_params
    else:
        # Dense FFN
        ffn_params_per_layer = config.hidden_size * config.intermediate_size * 3

    # Total per layer
    params_per_layer = attn_params_per_layer + ffn_params_per_layer + norm_params_per_layer

    # Final components
    final_norm = config.hidden_size
    lm_head = 0 if getattr(config, 'tie_word_embeddings', True) else config.vocab_size * config.hidden_size

    num_layers = getattr(config, 'num_layers', None) or getattr(config, 'num_hidden_layers', None) or 0
    return embed_params + (params_per_layer * num_layers) + final_norm + lm_head

def recommend_training_params(
    model_size_b: float,          # e.g., 4, 7, 8, 13, 70 (Billion params)
    dataset_size: int,            # Number of training samples
    quantization: str = "bf16",   # "4bit", "8bit", "bf16", "fp16"
    seq_len: int = 4096,          # Context length
    num_gpus: int = 1,            # Total GPUs available
    vram_per_gpu_gb: int = 24,    # Memory per GPU (e.g., 24, 32, 40, 80)
    offload_to_nvme: bool = False,
    nvme_path: str = "/tmp/deepspeed_offload",
    conservative: bool = True,    # Use conservative estimates (recommended)
):
    """
    Recommend training hyperparameters and DeepSpeed config based on hardware constraints.

    Returns a dict with:
        - per_device_train_batch_size
        - gradient_accumulation_steps
        - gradient_checkpointing
        - deepspeed config
        - warnings list
        - memory_breakdown dict (for debugging)
    """
    warnings = []
    zero3_allgather_peak = 0  # Initialize for later use

    # --- Estimate FULL training memory footprint ---
    bytes_per_param = {
        "4bit": 0.5,
        "8bit": 1.0,
        "bf16": 2.0,
        "fp16": 2.0,
    }.get(quantization, 2.0)

    # Weight memory
    weight_mem_gb = model_size_b * bytes_per_param

    # Gradient memory (same dtype as weights for mixed precision)
    grad_mem_gb = model_size_b * bytes_per_param

    # Optimizer states (Adam: 2 states per param, typically fp32)
    # For 4-bit QLoRA, optimizer only covers LoRA params (~1-5% of model)
    if quantization == "4bit":
        lora_fraction = 0.02  # ~2% of params are trainable in QLoRA
        optimizer_mem_gb = model_size_b * 8 * lora_fraction
    else:
        optimizer_mem_gb = model_size_b * 8  # 2 states × 4 bytes each

    total_training_mem_gb = weight_mem_gb + grad_mem_gb + optimizer_mem_gb

    # --- Select DeepSpeed Stage ---
    ds_stage = 2

    if quantization == "4bit":
        # QLoRA is incompatible with ZeRO-3 weight sharding
        ds_stage = 2
        # ZeRO-2 shards gradients + optimizer, but NOT weights
        per_gpu_mem = weight_mem_gb + (grad_mem_gb + optimizer_mem_gb) / num_gpus

        if weight_mem_gb > vram_per_gpu_gb * 0.8:
            warnings.append(
                f"CRITICAL: 4-bit model weights ({weight_mem_gb:.1f}GB) may not fit in "
                f"{vram_per_gpu_gb}GB VRAM. ZeRO-2 cannot shard weights. Consider a smaller model."
            )
    else:
        # Calculate memory per GPU for both ZeRO stages
        # ZeRO-2: shards grads + optimizer, NOT weights (each GPU has full weights)
        zero2_per_gpu = weight_mem_gb + (grad_mem_gb + optimizer_mem_gb) / num_gpus

        # ZeRO-3: shards everything BUT has hidden costs:
        # - All-gather temporarily reconstructs full layer weights during forward/backward
        # - Communication buffers for all-gather/reduce-scatter
        # - Peak memory is much higher than steady-state
        zero3_sharded = total_training_mem_gb / num_gpus

        # CRITICAL: ZeRO-3 peak memory multiplier
        # During forward pass, full weights of current layer must be gathered
        # Largest layer is typically ~10-15% of total params for transformers
        largest_layer_fraction = 0.12
        zero3_allgather_peak = weight_mem_gb * largest_layer_fraction

        # Communication buffers (allgather + reduce_scatter)
        comm_buffer_gb = 0.4  # ~400MB for bucket sizes

        zero3_per_gpu = zero3_sharded + zero3_allgather_peak + comm_buffer_gb

        # Use ZeRO-3 if ZeRO-2 would exceed 50% of VRAM
        if zero2_per_gpu > (vram_per_gpu_gb * 0.50):
            ds_stage = 3
            per_gpu_mem = zero3_per_gpu
        else:
            ds_stage = 2
            per_gpu_mem = zero2_per_gpu

    # --- Activation memory estimation (with gradient checkpointing) ---
    # More realistic estimate based on empirical data:
    # - Hidden states: batch * seq * hidden_dim * 2 bytes
    # - Attention: batch * heads * seq * seq * 2 bytes (this dominates!)
    # - With grad checkpointing: still need to store boundary activations

    # Estimate hidden_dim from model size (rough heuristic)
    hidden_dim = int(1024 * math.sqrt(model_size_b))  # ~2048 for 4B, ~4096 for 16B
    num_heads = max(16, hidden_dim // 128)

    # Activation memory per sample (bytes), with gradient checkpointing
    # Key insight: attention scores scale as O(seq^2)
    attn_mem_per_sample = num_heads * (seq_len ** 2) * 2 / 1e9  # GB
    hidden_mem_per_sample = seq_len * hidden_dim * 2 * 4 / 1e9  # GB (few layers kept)

    act_per_sample = attn_mem_per_sample + hidden_mem_per_sample

    # Apply conservative multiplier for misc buffers, temp tensors
    if conservative:
        act_per_sample *= 1.5

    # Minimum activation memory
    act_per_sample = max(act_per_sample, 0.5)

    # --- Determine if offloading is needed ---
    # More conservative buffer for PyTorch overhead, NCCL, fragmentation
    safe_buffer = 5.0 if conservative else 3.0

    available_for_batch = vram_per_gpu_gb - per_gpu_mem - safe_buffer

    use_offload = False
    offload_device = "none"

    if available_for_batch < 3.0:
        use_offload = True
        offload_device = "nvme" if offload_to_nvme else "cpu"
        warnings.append(
            f"Enabling {offload_device.upper()} offload. "
            f"Model+buffers exceed available GPU memory by {math.fabs(available_for_batch):.1f} GB"
        )
        # After offloading optimizer states, recalculate available memory
        if ds_stage == 3:
            # Offload both params and optimizer - but still need gather buffer
            available_for_batch = vram_per_gpu_gb - safe_buffer - zero3_allgather_peak - 2.0
        else:
            # Offload optimizer only (weights still on GPU)
            available_for_batch = vram_per_gpu_gb - weight_mem_gb - safe_buffer

    if available_for_batch <= 0:
        warnings.append(
            f"CRITICAL: Negative available VRAM ({available_for_batch:.1f}GB). "
            f"Model may not fit even with offloading. Try reducing seq_len or use smaller model."
        )
        available_for_batch = 1.0  # Force minimum

    # --- Per device batch size ---
    max_possible_batch = int(available_for_batch / act_per_sample)

    # Snap to power of 2, cap at 4 for memory safety with ZeRO-3
    max_batch_cap = 4 if ds_stage == 3 else 8
    per_device_batch = 1
    for b in [max_batch_cap, 4, 2, 1]:
        if max_possible_batch >= b:
            per_device_batch = b
            break

    # --- Target Global Batch Size ---
    # Heuristic based on dataset size for stable training
    if dataset_size < 500:
        target_global_batch = 8
    elif dataset_size < 1000:
        target_global_batch = 16
    elif dataset_size < 10000:
        target_global_batch = 32
    elif dataset_size < 100000:
        target_global_batch = 64
    else:
        target_global_batch = 128

    # --- Gradient Accumulation Steps ---
    total_physical_batch = per_device_batch * num_gpus
    grad_accum = max(1, math.ceil(target_global_batch / total_physical_batch))

    actual_gbs = total_physical_batch * grad_accum

    # --- Build DeepSpeed Config ---
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": 1.0,
        "zero_allow_untested_optimizer": True,
        "fp16": {
            "enabled": quantization in ["fp16", "4bit", "8bit"],
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": quantization == "bf16"
        },
        "zero_optimization": {
            "stage": ds_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 1e7,   # 10MB - smaller to reduce peak memory
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 1e7,      # 10MB
            "contiguous_gradients": True,
            "round_robin_gradients": True,
        }
    }

    # --- Add offload settings if needed ---
    if use_offload:
        offload_settings = {"device": offload_device}
        if offload_device == "nvme":
            offload_settings["nvme_path"] = nvme_path
            offload_settings["buffer_count"] = 4
            offload_settings["buffer_size"] = 1e8  # 100MB

        ds_config["zero_optimization"]["offload_optimizer"] = offload_settings

        # Only offload params in Stage 3 (Stage 2 doesn't shard params)
        if ds_stage == 3:
            ds_config["zero_optimization"]["offload_param"] = offload_settings

    # --- Stage 3 specific settings (conservative) ---
    if ds_stage == 3:
        ds_config["zero_optimization"].update({
            "stage3_prefetch_bucket_size": 1e6,          # 1MB - very conservative
            "stage3_param_persistence_threshold": 0,     # offload all params
            "stage3_max_live_parameters": 5e7,           # 50M params max in GPU memory
            "stage3_max_reuse_distance": 5e8,
            "stage3_gather_16bit_weights_on_model_save": True,
        })

    # --- Memory breakdown for debugging ---
    memory_breakdown = {
        "weight_mem_gb": round(weight_mem_gb, 2),
        "grad_mem_gb": round(grad_mem_gb, 2),
        "optimizer_mem_gb": round(optimizer_mem_gb, 2),
        "total_training_mem_gb": round(total_training_mem_gb, 2),
        "per_gpu_mem_gb": round(per_gpu_mem, 2),
        "activation_per_sample_gb": round(act_per_sample, 2),
        "available_for_batch_gb": round(available_for_batch, 2),
        "max_possible_batch": max_possible_batch,
        "zero3_allgather_peak_gb": round(zero3_allgather_peak, 2) if ds_stage == 3 else 0,
    }

    return {
        "per_device_train_batch_size": per_device_batch,
        "gradient_accumulation_steps": grad_accum,
        "actual_global_batch_size": actual_gbs,
        "gradient_checkpointing": True,
        "deepspeed": ds_config,
        "deepspeed_stage": ds_stage,
        "use_offload": use_offload,
        "offload_device": offload_device,
        "warnings": warnings,
        "memory_breakdown": memory_breakdown,
    }

def get_model_name(model_id_or_path: str) -> Optional[str]:
    assert isinstance(model_id_or_path, str), f'model_id_or_path: {model_id_or_path}'
    # compat hf hub
    model_id_or_path = model_id_or_path.rstrip('/')
    match_ = re.search('/models--.+?--(.+?)/snapshots/', model_id_or_path)
    if match_ is not None:
        return match_.group(1)

    model_name = model_id_or_path.rsplit('/', 1)[-1]
    # compat modelscope snapshot_download
    model_name = model_name.replace('___', '.')
    return model_name

