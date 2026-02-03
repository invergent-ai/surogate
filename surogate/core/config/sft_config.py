from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal

from namer import generate as generate_unique_name

from surogate import _surogate
from surogate.core.config.ct_config import ChatTemplateConfig
from surogate.core.config.model_config import ModelConfig
from surogate.core.config.train_dataset_config import TrainDatasetConfig
from surogate.utils.dict import DictDefault
from surogate.utils.fs import to_abspath
from surogate.utils.logger import get_logger
from surogate.utils.model import estimate_model_parameters

@dataclass
class DistributedConfig:
    """
    Configuration for multi-node distributed training via Ray.

    When distributed training is enabled (num_nodes > 1), Ray handles cluster
    management and node coordination, while NCCL handles GPU-to-GPU communication.

    Args:
        ray_address: Ray cluster address. Options:
            - "auto": Connect to an existing Ray cluster
            - "local": Start a local Ray instance
            - "ray://host:port": Connect to a specific Ray head node
        num_nodes: Total number of nodes to use for training.
        gpus_per_node: Number of GPUs per node. If 0, uses config.gpus.
        worker_output_dir: Base directory for worker-local tokenized data.
            Each worker will create a subdirectory node-{rank}/ under this path.
            If None, uses /tmp/surogate-{run_name}/ on each worker.
            This path must be accessible by all worker nodes.
    """
    ray_address: str = "auto"
    num_nodes: int = 1
    gpus_per_node: int = 0  # 0 = use config.gpus
    worker_output_dir: Optional[str] = None  # None = use /tmp/surogate-{run_name}/

@dataclass
class SFTConfig(ModelConfig, TrainDatasetConfig, ChatTemplateConfig):
    """
    SFTConfig class is a dataclass that holds configuration parameters for Supervised Fine-Tuning (SFT)

    Args:
        run_name (Optional[str], defaults to auto-generated):
            A descriptor for the run.
        apply_recommended_values (Optional[bool]):
            Whether to apply recommended configuration values. Default is True.
        num_epochs (Optional[int], default to 1):
            Total number of training epochs to perform.
        output_dir (Optional[str], defaults to 'output'):
            The output directory where the model predictions and checkpoints will be written.
        checkpoint_dir (Optional[str], defaults to None):
            Directory to save checkpoints during training. If None, defaults to `output_dir`.
        resume_from_checkpoint (Optional[bool], defaults to None):
            Continue from checkpoint. Uses the latest checkpoint.
        save_steps (`int` or Optional[float], defaults to 50):
            Number of steps between saving checkpoints..
        save_total_limit (Optional[int], defaults to 5):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`.

        recompute (bool, defaults to True):
            Enable activation recomputation to trade compute for memory:
            - False: Save all activations. Maximum memory, fastest training.
                     Guarantees bit-exact gradients.
            - True: Recompute intermediates from checkpoints. Saves ~17% VRAM.
                    Recommended for most training scenarios.
        recompute_lora (Optional[bool], defaults to None):
            Allow recomputation of FFT-only activations in LoRA-only mode to reduce VRAM.
            If None, auto-enables for QLoRA and disables otherwise.
        offload_residual (Optional[bool], defaults to False):
            Offload the residuals (of the ffn block; the only remaining part of the block that is not recomputed) to pinned host memory.
            Combined with recompute, the total activation memory consumption becomes independent of the network depth.
            This saves GPU memory at the cost of increased data transfer overhead.
        offload_master (Optional[bool], defaults to False):
            Store master weights in pinned host memory.
        offload_quants (Optional[bool], defaults to False):
            Store quantized weights in pinned host memory. Requires --persistent-quants.
        persistent_quants (Optional[bool], defaults to False):
            Allows avoiding re-quantization of weights; this increases memory, however, when combined with --offload-quants, the additional memory is placed on the host.
            In a PCIe setting where any GPU-to-GPU communication has to pass through host memory anway, this can actually lead to significant speed-ups, especially if combined with the --memcpy-all-gather option.
            Requires shard-weights.
        offload_optimizer (Optional[bool], defaults to False):
            Store optimizer state in pinned host memory.
            This will slow down the optimizer step drastically (memory-bound operation), but if enough gradient accumulation steps are performed, the overall contribution of the optimizer step will be negligible.
        offload_grads (Optional[bool], defaults to False):
            Offload gradients to pinned host memory.
        use_zero_copy (Optional[bool], defaults to False):
            Use ZeroCopy memory access, instead of double-buffered cudaMemcpy, for offloaded optimizer states. On consumer cards, DMA appears to be much slower, whereas on professional cards it is faster.
        use_write_combined (Optional[bool], defaults to False):
            Use write-combined memory for offloaded tensors. In some situations, this may improve PCie throughput.
        zero_level (Optional[int], defaults to 1):
            ZeRO redundancy optimization level:
            1: Sharded optimizer states (default)
            2: Sharded gradients + optimizer states
            3: Sharded weights + gradients + optimizer states
            You can also configure weights and gradients individually, using the shard-weights and shard-gradients flags. When training in fp8, for example, it makes sense to enable weight sharding before gradient sharding, as weights need only half the amount of bandwidth.
        shard_weights (Optional[bool], defaults to False):
            Whether to shard model weights across data-parallel processes. Enables more effective use of offloading and reduces memory consumption.
        shard_gradients (Optional[bool], defaults to False):
            Whether to shard gradients across data-parallel processes. Enables more effective use of offloading and reduces memory consumption.
        use_all_to_all_reduce (Optional[bool], defaults to False):
             Use all-to-all-based reduce algorithm (combine with --memcpy-send-recv).
        memcpy-all-gather (Optional[bool], defaults to False):
            Use memcpy for all-gather operations (threads backend only). Memcpy generally gets better bandwidth utilization on PCIe, and does not consume SM resources.
        memcpy-send-recv (Optional[bool], defaults to False):
            Use memcpy for send/receive operations (threads backend only).
        init_projections_to_zero (Optional[bool], defaults to False):
            Initialize projection weights (FFN down and attention out) to zero. Only used when training from scratch.
        from_scratch (Optional[bool], defaults to False):
            Whether to train the model from scratch (random initialization) rather than fine-tuning a pre-trained model.
        lmhead_chunks (Optional[int], defaults to 1):
            Split LM-head computation into N chunks, so that the required size of the logit tensor is reduced by a factor of N.
        attn_bwd_chunks (Optional[int], defaults to 1):
            Split attention backward pass into N chunks to save workspace memory.
        gradient_dtype (Optional[str], defaults to None):
            Which dtype to use for (activation) gradients / backward matmul policy. Defaults to matmul-dtype. Note: recipes may override backward dtype.
        master_dtype (Optional[str], defaults to None):
            Master weight dtype used for optimizer updates (e.g. FP32 for more stable full fine-tuning). Defaults to model-dtype.
        recipe (Optional[Literal['bf16', 'fp8_hybrid', 'nvfp4']], defaults to 'bf16'):
            Mixed precision training recipe to use: bf16 (default), fp8-hybrid, nvfp4
        use_fused_rope (Optional[bool], defaults to False):
            Use fused RoPE kernel with on-the-fly cos/sin computation (saves memory, reduces bandwidth)
        fp8_amax_history (Optional[int], defaults to 16):
            FP8 delayed scaling amax history length (default: 16, for fp8-hybrid recipe)
        fp4_backend (Optional[Literal['cutlass', 'cudnn']], defaults to 'cutlass'):
            FP4 matmul backend: cutlass (default) or cudnn (for nvfp4 recipe)
        skip_quant_first_layers (Optional[int], defaults to 0):
            Skip quantization for the first N transformer layers (embedding layers kept in BF16)
        skip_quant_last_layers (Optional[int], defaults to 0):
            Skip quantization for the last N transformer layers (lm_head layers kept in BF16)

        gpus (Optional[int], defaults to first GPU):
            Number of GPUs to use for training. Default is the first available GPU. Use 0 for all available GPUs.
        use_cuda_graphs (Optional[bool], defaults to True):
            Enable or disable CUDA graphs for performance.
        use_dsl_ir (Optional[bool], defaults to False):
            Enable DSL IR backend. DSL training always runs full-step execution; use_cuda_graphs controls capture.

        optimizer (Optional[Literal['adamw_8bit', 'normuon']], defaults to 'adamw_8bit'):
            Optimizer type to use for training. Supports:
            - 'adamw_8bit': 8-bit blockwise quantized AdamW (default)
            - 'normuon': NorMuon optimizer with orthogonalized momentum for 2D weights
              (uses AdamW for embeddings, norms, and lm_head; NorMuon for attention/MLP weights)
        learning_rate (Optional[float], defaults to 1e-4):
            The initial learning rate for the optimizer.
        lr_scheduler_type (Optional[Literal['linear', 'cosine', 'wsd']]*, defaults to `"linear"`):
           Learning rate schedule function: Cosine or Linear
        cooldown_steps (Optional[int], defaults to 0):
            Number of steps used for a linear cooldown from `learning_rate` to `final_lr_fraction * learning_rate`.
        final_lr_fraction (Optional[float], defaults to 0.0):
            Final learning rate as a fraction of the initial learning rate.
        gradient_accumulation_steps (Optional[int], defaults to 4):
           Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
           Warning: When using gradient accumulation, one step is counted as one step with backward pass.
           Effective batch size = batch-size × grad-accumulation × num-gpus.
        max_grad_norm (Optional[float], defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        per_device_train_batch_size (Optional[int], defaults to 2):
            Batch size per device during training.
        weight_decay (Optional[float], defaults to 0.1):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            optimizer.
        max_steps (Optional[int], defaults to -1):
            Total number of training steps. -1 derives from epochs and dataset size.
        adamw_beta1: (Optional[float], defaults to 0.9):
            The beta1 parameter for the AdamW optimizer.
        adamw_beta2: (Optional[float], defaults to 0.999):
            The beta2 parameter for the AdamW optimizer.
        adamw_epsilon: (Optional[float], defaults to 1e-8):
            The epsilon parameter for the AdamW optimizer.
        normuon_momentum: (Optional[float], defaults to 0.95):
            The momentum (beta1) parameter for NorMuon optimizer.
        normuon_beta2: (Optional[float], defaults to 0.95):
            The beta2 parameter for NorMuon variance estimation EMA.
        normuon_cautious_wd: (Optional[bool], defaults to True):
            Whether to use cautious (sign-aware) weight decay in NorMuon.

        eval_steps (Optional[int], defaults to 100):
             Run evaluation every N optimizer steps. Evaluation runs on the full eval dataset.
        report_to (Optional[Literal['wandb', 'aim']], *optional*, defaults to None):
            Report the results and logs to. Supported platforms are `"wandb"`, `"aim"`.
        warmup_ratio (Optional[float], defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        warmup_steps (Optional[int], defaults to 0):
            Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
        log_file (Optional[str], defaults to None):
            "Where to save the training log.

        lora (Optional[bool], defaults to True):
            Whether to use LoRA adapters for training.
        lora_rank (Optional[int], defaults to 8):
            Rank for LoRA adapters.
        lora_alpha (Optional[int], defaults to 32):
            Alpha value for LoRA adapters.
        lora_dropout (Optional[float], defaults to 0.05):
            Dropout rate for LoRA adapters.
        lora_dype(Optional[Literal['bf16','fp32']], defaults to 'fp32):
            Dropout rate for LoRA adapters.
        lora_target_modules (Optional[str], default to 'all'):
            List of comma-separated module names to apply LoRA adapters to.
        train_router (Optional[bool], defaults to False):
            Train the MoE router gate weights during LoRA fine-tuning.
            When enabled, the router weights are unfrozen and included in the adapter.
        router_aux_loss_coef (Optional[float], defaults to None):
            MoE auxiliary (load balancing) loss coefficient. None uses the model config default.
        router_z_loss_coef (Optional[float], defaults to None):
            MoE z-loss (router logit regularization) coefficient. None uses the model config default.
        qlora_fp4: (Optional[bool], defaults to False):
            Enable NVFP4 QLoRA mode (base weights quantized to FP4 E2M1). Requires Blackwell GPU (SM100+)
        qlora_fp8: (Optional[bool], defaults to False):
            Enable FP8 QLoRA mode (base weights quantized to FP8 with per-block scales)
        qlora_bnb: (Optional[bool], defaults to False):
            Enable BitsAndBytes NF4 QLoRA mode (base weights quantized to NF4 with per-block absmax).
            Works on any CUDA GPU (no SM89+ or SM100+ requirement).
        qlora_block_size: (Optional[int], defaults to 128):
            Block size for FP8 QLoRA quantization. Valid values are 64, 128, 256.
        qlora_bnb_block_size: (Optional[int], defaults to 64):
            Block size for BnB NF4 QLoRA quantization. Valid values are 64, 128, 256, 512.
        qlora_bnb_double_quant: (Optional[bool], defaults to True):
            Enable double quantization for BnB (quantize absmax values to INT8 for extra memory savings).
        qlora_four_over_six: (Optional[bool], defaults to True):
            Enable Four Over Six (4/6) adaptive block scaling for NVFP4 QLoRA quantization.
            Evaluates both max=4 and max=6 scaling per block and selects lower error option.
        qlora_selective_expert_dequant: (Optional[bool], defaults to True):
            Enable selective expert dequantization for MoE models. When enabled, only the experts
            selected by the router are dequantized, reducing memory usage from O(num_experts)
            to O(top_k) for dequantization buffers. Significant memory savings for models with
            many experts (e.g., 128 experts with top_k=8 saves ~93% of dequant buffer memory).

        use_chat_template (Optional[bool], defaults to True):
            Whether to use chat template for training.
        merge_adapter: (Optional[bool], defaults to False):
            Whether to merge LoRA adapters into the base model after training.
            When enabled, saves both the adapter (in output_dir/adapter/) and merged model (in output_dir/merged/).

        debug_time_breakdown (Optional[bool], defaults to False):
            Whether to enable detailed training timing breakdown for debugging.
        debug_memory_breakdown (Optional[bool], defaults to False):
            Print detailed memory breakdown after model allocation (useful for QLoRA optimization).
            
        wandb_project (Optional[str], defaults to "Surogate"):
            WandB project name for logging.
        wandb_name (Optional[str], defaults to run_name):
            WandB run name for logging.
        aim_experiment: (Optional[str], defaults to "Surogate"):
            Aim experiment name for logging.
        aim_repo: (Optional[str], defaults to None):
            Aim repository path for logging.
        aim_name: (Optional[str], defaults to run_name):
            Aim run name for logging.
    """
    run_name: Optional[str] = None
    apply_recommended_values: Optional[bool] = False
    num_epochs: Optional[int] = 3
    output_dir: Optional[str] = 'output'
    checkpoint_dir: Optional[str] = None  # Defaults to output_dir if not specified
    resume_from_checkpoint: Optional[bool] = True
    save_steps: Optional[int] = 50
    save_total_limit: Optional[int] = 5

    recompute: Optional[bool] = True
    recompute_lora: Optional[bool] = None

    offload_residual: Optional[bool] = False
    offload_master: Optional[bool] = False
    offload_quants: Optional[bool] = False
    persistent_quants: Optional[bool] = False
    offload_optimizer: Optional[bool] = False
    offload_grads: Optional[bool] = False
    use_zero_copy: Optional[bool] = False
    use_write_combined: Optional[bool] = False
    zero_level: Optional[int] = 1
    shard_weights: Optional[bool] = False
    shard_gradients: Optional[bool] = False
    use_all_to_all_reduce: Optional[bool] = False
    memcpy_all_gather: Optional[bool] = False
    memcpy_send_recv: Optional[bool] = False
    init_projections_to_zero: Optional[bool] = False
    from_scratch: Optional[bool] = False
    lmhead_chunks: Optional[int] = 1
    attn_bwd_chunks: Optional[int] = 1
    gradient_dtype: Optional[str] = None
    master_dtype: Optional[str] = None
    recipe: Optional[Literal['bf16', 'fp8_hybrid', 'nvfp4']] = 'bf16'
    use_fused_rope: Optional[bool] = False
    fp8_amax_history: Optional[int] = 16
    fp4_backend: Optional[Literal['cutlass', 'cudnn']] = 'cutlass'
    skip_quant_first_layers: Optional[int] = 0
    skip_quant_last_layers: Optional[int] = 0

    gpus: Optional[int] = 1
    use_cuda_graphs: Optional[bool] = True
    use_dsl_ir: Optional[bool] = True

    optimizer: Optional[Literal['adamw_8bit', 'normuon']] = 'adamw_8bit'
    learning_rate: Optional[float] = 2e-4
    lr_scheduler_type: Optional[Literal['linear', 'cosine', 'wsd']] = 'linear'
    cooldown_steps: Optional[int] = 0
    final_lr_fraction: Optional[float] = 0.0
    gradient_accumulation_steps: Optional[int] = 4
    max_grad_norm: Optional[float] = 0.0
    weight_decay: Optional[float] = 0.1
    max_steps: Optional[int] = -1
    adamw_beta1: Optional[float] = 0.9
    adamw_beta2: Optional[float] = 0.999
    adamw_epsilon: Optional[float] = 1e-8
    # NorMuon optimizer parameters (used when optimizer='normuon')
    # NorMuon uses a hybrid approach: AdamW for embeddings/norms/lm_head,
    # orthogonalized momentum for 2D weight matrices
    normuon_momentum: Optional[float] = 0.95
    normuon_beta2: Optional[float] = 0.95
    normuon_cautious_wd: Optional[bool] = True
    eval_steps: Optional[int] = 100
    per_device_train_batch_size: Optional[int] = 2
    report_to: Optional[List[Literal['wandb', 'aim']]] = None
    warmup_ratio: Optional[float] = 0
    warmup_steps: Optional[int] = 0
    log_file: Optional[str] = None

    lora: Optional[bool] = True
    lora_rank: Optional[int] = 16
    lora_alpha: Optional[int] = 32
    lora_dropout: Optional[float] = 0.05
    lora_dtype: Optional[Literal['bf16','fp32']] = 'fp32'
    lora_target_modules: Optional[List[str]] = None
    train_router: Optional[bool] = False
    router_aux_loss_coef: Optional[float] = None
    router_z_loss_coef: Optional[float] = None
    qlora_fp4: Optional[bool] = False
    qlora_fp8: Optional[bool] = False
    qlora_bnb: Optional[bool] = False
    qlora_block_size: Optional[int] = 128
    qlora_bnb_block_size: Optional[int] = 64
    qlora_bnb_double_quant: Optional[bool] = True
    qlora_four_over_six: Optional[bool] = True
    qlora_selective_expert_dequant: Optional[bool] = False
    qlora_offload_experts: Optional[bool] = False

    merge_adapter: Optional[bool] = False
    use_chat_template: Optional[bool] = True
    debug_time_breakdown: Optional[bool] = False
    debug_memory_breakdown: Optional[bool] = False
    log_gpu_util: Optional[int] = 100
    
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    aim_experiment: Optional[str] = None
    aim_repo: Optional[str] = None
    aim_name: Optional[str] = None

    # Multi-node distributed training config (optional)
    distributed: Optional[DistributedConfig] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.run_name = cfg['run_name'] or self.generate_run_name()
        self.apply_recommended_values = cfg.get('apply_recommended_values', self.apply_recommended_values)
        self.num_epochs = cfg.get('num_epochs', self.num_epochs)
        self.output_dir = cfg.get('output_dir', self.output_dir)
        self.resume_from_checkpoint = cfg.get('resume_from_checkpoint', self.resume_from_checkpoint)
        self.save_steps = cfg.get('save_steps', self.save_steps)
        self.save_total_limit = cfg.get('save_total_limit', self.save_total_limit)

        # Parse recompute setting (accepts bool or legacy string values)
        recompute_raw = cfg.get('recompute', self.recompute)
        if isinstance(recompute_raw, bool):
            self.recompute = recompute_raw
        elif isinstance(recompute_raw, str):
            if recompute_raw.lower() in ('false', 'none', '0'):
                self.recompute = False
            else:
                raise ValueError(f"recompute must be true or false, got '{recompute_raw}'")
        else:
            self.recompute = bool(recompute_raw)
        self.recompute_lora = cfg.get('recompute_lora', self.recompute_lora)

        self.offload_residual = cfg.get('offload_residual', self.offload_residual)
        self.offload_master = cfg.get('offload_master', self.offload_master)
        self.offload_quants = cfg.get('offload_quants', self.offload_quants)
        self.persistent_quants = cfg.get('persistent_quants', self.persistent_quants)
        self.offload_optimizer = cfg.get('offload_optimizer', self.offload_optimizer)
        self.offload_grads = cfg.get('offload_grads', self.offload_grads)
        self.use_zero_copy = cfg.get('use_zero_copy', self.use_zero_copy)
        self.use_write_combined = cfg.get('use_write_combined', self.use_write_combined)
        self.zero_level = cfg.get('zero_level', self.zero_level)
        self.shard_weights = cfg.get('shard_weights', self.shard_weights)
        self.shard_gradients = cfg.get('shard_gradients', self.shard_gradients)
        self.use_all_to_all_reduce = cfg.get('use_all_to_all_reduce', self.use_all_to_all_reduce)
        self.memcpy_all_gather = cfg.get('memcpy_all_gather', self.memcpy_all_gather)
        self.memcpy_send_recv = cfg.get('memcpy_send_recv', self.memcpy_send_recv)
        self.init_projections_to_zero = cfg.get('init_projections_to_zero', self.init_projections_to_zero)
        self.from_scratch = cfg.get('from_scratch', self.from_scratch)
        self.lmhead_chunks = cfg.get('lmhead_chunks', self.lmhead_chunks)
        self.attn_bwd_chunks = cfg.get('attn_bwd_chunks', self.attn_bwd_chunks)
        self.gradient_dtype = cfg.get('gradient_dtype', self.gradient_dtype)
        self.master_dtype = cfg.get('master_dtype', self.master_dtype)
        self.recipe = cfg.get('recipe', self.recipe)
        self.use_fused_rope = cfg.get('use_fused_rope', self.use_fused_rope)
        self.fp8_amax_history = cfg.get('fp8_amax_history', self.fp8_amax_history)
        self.fp4_backend = cfg.get('fp4_backend', self.fp4_backend)
        self.skip_quant_first_layers = cfg['skip_quant_first_layers'] if 'skip_quant_first_layers' in cfg else self.skip_quant_first_layers
        self.skip_quant_last_layers = cfg['skip_quant_last_layers'] if 'skip_quant_last_layers' in cfg else self.skip_quant_last_layers

        self.gpus = cfg.get('gpus', self.gpus)
        self.use_cuda_graphs = cfg.get('use_cuda_graphs', self.use_cuda_graphs)
        self.use_dsl_ir = cfg.get('use_dsl_ir', self.use_dsl_ir)

        self.optimizer = cfg.get('optimizer', self.optimizer)
        self.learning_rate = float(cfg.get('learning_rate', self.learning_rate))
        self.lr_scheduler_type = cfg.get('lr_scheduler_type', self.lr_scheduler_type)
        self.cooldown_steps = cfg['cooldown_steps'] if 'cooldown_steps' in cfg else self.cooldown_steps
        self.final_lr_fraction = float(cfg['final_lr_fraction']) if 'final_lr_fraction' in cfg else self.final_lr_fraction
        self.gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', self.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg['max_grad_norm']) if 'max_grad_norm' in cfg else self.max_grad_norm
        self.weight_decay = float(cfg['weight_decay']) if 'weight_decay' in cfg else self.weight_decay
        self.max_steps = cfg.get('max_steps', self.max_steps)
        self.adamw_beta1 = float(cfg.get('adamw_beta1', self.adamw_beta1))
        self.adamw_beta2 = float(cfg.get('adamw_beta2', self.adamw_beta2))
        self.adamw_epsilon = float(cfg.get('adamw_epsilon', self.adamw_epsilon))
        self.normuon_momentum = float(cfg.get('normuon_momentum', self.normuon_momentum))
        self.normuon_beta2 = float(cfg.get('normuon_beta2', self.normuon_beta2))
        self.normuon_cautious_wd = cfg.get('normuon_cautious_wd', self.normuon_cautious_wd)
        self.eval_steps = cfg.get('eval_steps', self.eval_steps)
        self.per_device_train_batch_size = cfg.get('per_device_train_batch_size', self.per_device_train_batch_size)
        self.report_to = cfg.get('report_to', self.report_to)
        self.warmup_ratio = float(cfg['warmup_ratio']) if 'warmup_ratio' in cfg else self.warmup_ratio
        self.warmup_steps = cfg['warmup_steps'] if 'warmup_steps' in cfg else self.warmup_steps
        self.log_file = cfg.get('log_file', self.log_file)

        self.lora = cfg.get('lora', self.lora)
        self.lora_rank = cfg.get('lora_rank', self.lora_rank)
        self.lora_alpha = cfg.get('lora_alpha', self.lora_alpha)
        self.lora_dropout = cfg['lora_dropout'] if 'lora_dropout' in cfg else self.lora_dropout
        self.lora_dtype = cfg.get('lora_dtype', self.lora_dtype)
        self.lora_target_modules = cfg.get('lora_target_modules', ['all'])
        self.train_router = cfg.get('train_router', self.train_router)
        self.router_aux_loss_coef = cfg.get('router_aux_loss_coef', self.router_aux_loss_coef)
        self.router_z_loss_coef = cfg.get('router_z_loss_coef', self.router_z_loss_coef)
        self.qlora_fp4 = cfg.get('qlora_fp4', self.qlora_fp4)
        self.qlora_fp8 = cfg.get('qlora_fp8', self.qlora_fp8)
        self.qlora_bnb = cfg.get('qlora_bnb', self.qlora_bnb)
        self.qlora_block_size = cfg.get('qlora_block_size', self.qlora_block_size)
        self.qlora_bnb_block_size = cfg.get('qlora_bnb_block_size', self.qlora_bnb_block_size)
        self.qlora_bnb_double_quant = cfg.get('qlora_bnb_double_quant', self.qlora_bnb_double_quant)
        self.qlora_four_over_six = cfg.get('qlora_four_over_six', self.qlora_four_over_six)
        self.qlora_selective_expert_dequant = cfg.get('qlora_selective_expert_dequant', self.qlora_selective_expert_dequant)
        self.qlora_offload_experts = cfg.get('qlora_offload_experts', self.qlora_offload_experts)

        self.merge_adapter = cfg.get('merge_adapter', self.merge_adapter)
        self.use_chat_template = cfg.get('use_chat_template', self.use_chat_template)
        self.debug_time_breakdown = cfg.get('debug_time_breakdown', self.debug_time_breakdown)
        self.debug_memory_breakdown = cfg.get('debug_memory_breakdown', self.debug_memory_breakdown)
        
        self.wandb_project = cfg.get('wandb_project', self.wandb_project)
        self.wandb_name = cfg.get('wandb_name', self.wandb_name or self.run_name)
        self.aim_experiment = cfg.get('aim_experiment', self.aim_experiment)
        self.aim_repo = cfg.get('aim_repo', self.aim_repo)
        self.aim_name = cfg.get('aim_name', self.aim_name or self.run_name)
        
        # Validate recompute is boolean
        if not isinstance(self.recompute, bool):
            raise ValueError(f"recompute must be true or false, got '{self.recompute}'")

        # Parse distributed config
        distributed_cfg = cfg.get('distributed', None)
        if distributed_cfg:
            if isinstance(distributed_cfg, dict):
                self.distributed = DistributedConfig(
                    ray_address=distributed_cfg.get('ray_address', 'auto'),
                    num_nodes=distributed_cfg.get('num_nodes', 1),
                    gpus_per_node=distributed_cfg.get('gpus_per_node', 0),
                    worker_output_dir=distributed_cfg.get('worker_output_dir', None),
                )
            elif isinstance(distributed_cfg, DistributedConfig):
                self.distributed = distributed_cfg
        else:
            self.distributed = None

    def __post_init__(self):
        logger = get_logger()
        
        ModelConfig.__post_init__(self)
        TrainDatasetConfig.__post_init__(self)
        ChatTemplateConfig.__post_init__(self)
        
        self.prompt_template = self.model_template.chat_template
        if self.use_chat_template is None:
            self.use_chat_template = True

        if len(self.validation_datasets) > 0 and self.validation_split_ratio > 0:
            # Don't split training data if validation datasets are provided or dataset streaming is enabled
            self.validation_split_ratio = 0.0

        if self.sequence_len is None:
            self.sequence_len = self.model_info.max_model_len

        if self.sample_packing and self.sequence_len == self.model_info.max_model_len:
            logger.warning(
                f"Setting sequence_len to model's max_model_len {self.model_info.max_model_len}.")

        if self.learning_rate is None:
            logger.info(f"Learning rate is not set. Setting learning rate to {self.learning_rate}.")
            self.learning_rate = 2e-4

        if self.learning_rate < 1e-7:
            logger.warning(
                f"Your learning rate {self.learning_rate} is set to a very low value. Consider increasing it to avoid vanishing gradients!")
        elif self.learning_rate > 1:
            logger.warning(
                f"Your learning rate {self.learning_rate} is set to a very high value. Consider decreasing it to avoid exploding gradients!")

        self._validate_chunking_config()
        if self.offload_optimizer and not self.use_zero_copy:
            logger.warning(
                "offload_optimizer is enabled but use_zero_copy is false; "
                "optimizer state will remain on device. Set use_zero_copy=true to offload.")
        if self.offload_grads and self.use_dsl_ir:
            num_shards = None
            if self.distributed:
                gpus_per_node = self.distributed.gpus_per_node or self.gpus
                if gpus_per_node and gpus_per_node > 0:
                    num_shards = self.distributed.num_nodes * gpus_per_node
            elif self.gpus and self.gpus > 0:
                num_shards = self.gpus
            if num_shards and num_shards > 1:
                logger.warning(
                    "offload_grads is enabled with num_shards=%d; "
                    "true ZeRO-2 sharded/offloaded grads for DSL is not implemented yet.",
                    num_shards,
                )

        self.create_runtime_config()
        self.create_lora_config()
        self.create_qlora_config()

        self.ensure_directories()

    def _validate_chunking_config(self):
        """Validate that chunking parameters are compatible with batch size and sequence length."""
        batch_size = self.per_device_train_batch_size
        seq_len = self.sequence_len
        total_tokens = batch_size * seq_len

        if self.attn_bwd_chunks > 1 and batch_size % self.attn_bwd_chunks != 0:
            raise ValueError(
                f"attn_bwd_chunks ({self.attn_bwd_chunks}) must evenly divide "
                f"per_device_train_batch_size ({batch_size}). "
                f"Either increase batch size to a multiple of {self.attn_bwd_chunks} "
                f"or reduce attn_bwd_chunks."
            )

        if self.lmhead_chunks > 1 and total_tokens % self.lmhead_chunks != 0:
            raise ValueError(
                f"lmhead_chunks ({self.lmhead_chunks}) must evenly divide "
                f"per_device_train_batch_size * sequence_len ({batch_size} * {seq_len} = {total_tokens}). "
                f"Either adjust batch size or sequence length, or reduce lmhead_chunks."
            )


    def ensure_directories(self):
        # Skip directory creation if running inside a Ray worker
        # Only the head node should create directories
        from surogate.utils.dist import is_ray_worker
        if is_ray_worker():
            return

        # Convert output_dir to absolute path for consistent hash file location
        self.output_dir = str(Path(self.output_dir).resolve())
        _output_dir = Path(self.output_dir)
        if _output_dir.exists():
            if not _output_dir.is_dir():
                raise ValueError(f"Save path '{_output_dir}' already exists and is not a directory. Aborting.")

            if any(item.is_dir() and item.name.startswith("checkpoint-") for item in _output_dir.iterdir()):
                logger.warning_once(f"Save path '{_output_dir}' contains previously saved checkpoints.")
        else:
            _output_dir.mkdir(parents=True, exist_ok=True)

        # Set checkpoint_dir to output_dir if not specified, then convert to absolute path
        self.checkpoint_dir = str(Path(self.checkpoint_dir or self.output_dir).resolve())
        _checkpoint_dir = Path(self.checkpoint_dir)
        if not _checkpoint_dir.exists():
            _checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.log_file is None:
            date_time = "{:%Y%m%d-%H%M%S}".format(datetime.now())
            self.log_file = f"{self.output_dir}/log-{self.run_name}-{date_time}.json"
            self.log_file = to_abspath(self.log_file)

        if self.log_file:
            log_path = to_abspath(self.log_file)
            log_dir = Path(log_path).parent
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
  
    def create_runtime_config(self):
        shard_gradients = self.shard_gradients
        shard_weights = self.shard_weights

        if self.zero_level >= 2:
            shard_gradients = True
        if self.zero_level >= 3:
            shard_weights = True

        if self.qlora_bnb or self.qlora_fp8 or self.qlora_fp4:
            # QLoRA requires recompute enabled
            if not self.recompute:
                self.recompute = True
            self.use_cuda_graphs = False  # Disable CUDA graphs for QLoRA
            if self.recompute_lora is None:
                self.recompute_lora = True
        elif self.recompute_lora is None:
            self.recompute_lora = False


        if self.lora and self.recompute and self.offload_residual:
            self.use_cuda_graphs = False  # Disable CUDA graphs when offloading residuals with recompute

        self.runtime_config = _surogate.RuntimeOptions(
            recompute="true" if self.recompute else "false",
            offload_residual=self.offload_residual,
            offload_master=self.offload_master,
            offload_quants=self.offload_quants,
            offload_optimizer=self.offload_optimizer,
            offload_grads=self.offload_grads,
            persistent_quants=self.persistent_quants,
            use_cuda_graphs=self.use_cuda_graphs,
            trigger_timing_events=self.debug_time_breakdown,
            shard_weights=shard_weights,
            shard_gradients=shard_gradients,
            use_all_to_all_reduce=self.use_all_to_all_reduce,
            init_projections_to_zero=self.init_projections_to_zero,
            debug_memory_breakdown=self.debug_memory_breakdown,
            lmhead_chunks=self.lmhead_chunks,
            attn_bwd_chunks=self.attn_bwd_chunks,
            matmul_type="",
            gradient_type=self.gradient_dtype or "",
            master_dtype=self.master_dtype or "",
            recipe=self.recipe,
            use_fused_rope=self.use_fused_rope,
            fp8_amax_history=self.fp8_amax_history,
            fp4_backend=self.fp4_backend,
            skip_quant_first_layers=self.skip_quant_first_layers,
            skip_quant_last_layers=self.skip_quant_last_layers,
            use_dsl_ir=self.use_dsl_ir,
        )
        self.runtime_config.use_zero_copy = self.use_zero_copy
        self.runtime_config.use_write_combined = self.use_write_combined
        self.runtime_config.recompute_lora = bool(self.recompute_lora)
        self.runtime_config.selective_expert_dequant = self.qlora_selective_expert_dequant
        self.runtime_config.offload_experts = self.qlora_offload_experts
        # MoE loss coefficients (None means use model config default)
        if self.router_aux_loss_coef is not None:
            self.runtime_config.router_aux_loss_coef = float(self.router_aux_loss_coef)
        if self.router_z_loss_coef is not None:
            self.runtime_config.router_z_loss_coef = float(self.router_z_loss_coef)

    def create_lora_config(self):
        # Only create LoRA config when lora is enabled
        # When lora=False, lora_config must be None so C++ uses full fine-tuning path
        if not self.lora:
            self.lora_config = None
            return
        self.lora_config = _surogate.LoRAAdapterConfig(
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            dtype=self.lora_dtype,
            target_modules=self.lora_target_modules,
            use_rslora=False,
            train_router=self.train_router
        )

    def create_qlora_config(self):
        self.qlora_config = None

        # QLoRA is only supported together with LoRA adapters in Surogate:
        # base weights remain frozen and only LoRA params are trained.
        if (self.qlora_fp4 or self.qlora_fp8 or self.qlora_bnb) and not self.lora:
            raise ValueError(
                "QLoRA options (qlora_bnb/qlora_fp8/qlora_fp4) require `lora: true` "
                "(base weights are frozen; only LoRA adapters are trained). "
                "Either enable LoRA or disable QLoRA."
            )

        if self.qlora_fp4:
            self.qlora_config = _surogate.QLoRAConfig.nvfp4()
            self.qlora_config.enable_four_over_six = self.qlora_four_over_six
        elif self.qlora_fp8:
            self.qlora_config = _surogate.QLoRAConfig.fp8(block_size=self.qlora_block_size)
            self.qlora_config.enable_four_over_six = self.qlora_four_over_six
        elif self.qlora_bnb:
            self.qlora_config = _surogate.QLoRAConfig.bnb(
                block_size=self.qlora_bnb_block_size,
                double_quant=self.qlora_bnb_double_quant
            )

        # Populate MoE fields from model config for QLoRA quantization
        if self.qlora_config is not None and self.model_info.is_moe_model:
            from surogate.core.model.hf_config import HfConfigFactory
            config = self.model_info.config
            num_experts = HfConfigFactory.get_config_attr(config, 'num_experts') or 0
            num_experts_per_tok = HfConfigFactory.get_config_attr(config, 'num_experts_per_tok') or 8
            moe_intermediate_size = HfConfigFactory.get_config_attr(config, 'moe_intermediate_size') or 0
            if num_experts > 0:
                self.qlora_config.num_experts = num_experts
                self.qlora_config.num_experts_per_tok = num_experts_per_tok
                self.qlora_config.moe_intermediate_size = moe_intermediate_size

    def generate_run_name(self):
        return generate_unique_name(category='science')

    def estimate_training_memory(self, model):
        params = estimate_model_parameters(model.config)
        dtype = getattr(model.config, 'torch_dtype', None)
        if dtype is None:
            # Fall back to checking the first parameter's dtype
            dtype = next(model.parameters()).dtype

        import torch
        dtype_to_bytes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float8_e4m3fn: 1,
            torch.float8_e5m2: 1,
            torch.int8: 1,
        }
        bytes_per_param = dtype_to_bytes.get(dtype, 4)  # Default to 4 if unknown
        model_memory = params * bytes_per_param / 1e9

        # Optimizer memory depends on training precision, not storage precision
        # For mixed precision training, optimizer states are typically in fp32
        optimizer_memory = params * 8 / 1e9  # Adam needs ~8 bytes per parameter

        # Activation memory uses the training dtype
        activation_memory = (
                self.per_device_train_batch_size *
                self.sequence_len *
                model.config.hidden_size *
                bytes_per_param / 1e9
        )

        total_memory_needed = model_memory + optimizer_memory + activation_memory

        logger.info(f"Memory estimates - Model: {model_memory:.2f}GB, "
                    f"Optimizer: {optimizer_memory:.2f}GB, "
                    f"Activations: {activation_memory:.2f}GB, "
                    f"Total: {total_memory_needed:.2f}GB")
