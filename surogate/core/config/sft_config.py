from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal

from namer import generate as generate_unique_name

from surogate import _surogate
from surogate.core.config.ct_config import ChatTemplateConfig
from surogate.core.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.core.config.model_config import ModelConfig
from surogate.core.config.train_dataset_config import TrainDatasetConfig
from surogate.core.datasets.datasets import get_default_process_count
from surogate.utils.dict import DictDefault
from surogate.utils.fs import to_abspath
from surogate.utils.logger import get_logger
from surogate.utils.model import estimate_model_parameters
from surogate.utils.seed import RAND_SEED

logger = get_logger()


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
            Continue from checkpoint. If no number is given, uses the latest checkpoint.
        save_steps (`int` or Optional[float], defaults to 50):
            Number of steps between saving checkpoints..
        save_total_limit (Optional[int], defaults to 5):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`.

        recompute_swiglu (Optional[bool], defaults to True):
            Recompute SwiGLU activation during backward pass to save activation memory.
            As SwiGLU is at the widest part of the model, this will result in substantial memory savings at only moderate compute increases (especially for large models).
            This reduces GPU memory usage but slows down training.
        recompute_rmsnorm (Optional[bool], defaults to True):
            Whether to enable recompute for RMSNorm activations during the backward pass to save memory during training.
            This reduces GPU memory usage but slows down training.
        recompute_ffn (Optional[bool], defaults to True):
            Whether to enable recompute for Feed-Forward Network (FFN) activations during the backward pass to save memory during training.
            Implies --recompute-swiglu.
            This reduces GPU memory usage but slows down training.
        recompute_qkv (Optional[bool], defaults to True):
            Whether to enable recompute for QKV projections during the backward pass to save memory during training.
            This reduces GPU memory usage but slows down training.
        recompute_att (Optional[bool], defaults to True):
            Whether to enable recompute for the attention block to save memory during training.
            Implies --recompute-qkv.
            This reduces GPU memory usage but slows down training.
        recompute_block (Optional[bool], defaults to True):
            Whether to enable recompute for entire Transformer block to save memory during training.
            This reduces GPU memory usage but slows down training.

        offload_residual (Optional[bool], defaults to False):
            Offload the residuals (of the ffn block; the only remaining part of the block that is not recomputed) to pinned host memory.
            Combined with --recompute-block, the total activation memory consumption becomes independent of the network depth.
            This saves GPU memory at the cost of increased data transfer overhead.
        offload_master (Optional[bool], defaults to False):
            Store master weights in pinned host memory.
        offload_quants (Optional[bool], defaults to False):
            Store quantized weights in pinned host memory. Requires --persistent-quants.
        persistent_quants (Optional[bool], defaults to False):
            Allows avoiding re-quantization of weights; this increases memory, however, when combined with --offload-quants, the additional memory is placed on the host.
            In a PCIe setting where any GPU-to-GPU communication has to pass through host memory anway, this can actually lead to significant speed-ups, especially if combined with the --memcpy-all-gather option.
            Requires --shard-weights.
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
            You can also configure weights and gradients individually, using the --shard-weights and --shard-gradients flags. When training in fp8, for example, it makes sense to enable weight sharding before gradient sharding, as weights need only half the amount of bandwidth.
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
        fp8_amax_history (Optional[int], defaults to 1024):
            FP8 delayed scaling amax history length (default: 1024, for fp8-hybrid recipe)
        fp4_backend (Optional[Literal['cutlass', 'cudnn']], defaults to 'cutlass'):
            FP4 matmul backend: cutlass (default) or cudnn (for nvfp4 recipe)
        no_fp4_hadamard (Optional[bool], defaults to False):
            Disable Random Hadamard Transform for NVFP4 recipe.
        no_fp4_stochastic_rounding (Optional[bool], defaults to False):
            Disable stochastic rounding for NVFP4 gradient quantization.
        skip_quant_first_layers (Optional[int], defaults to 0):
            Skip quantization for the first N transformer layers (embedding layers kept in BF16)
        skip_quant_last_layers (Optional[int], defaults to 0):
            Skip quantization for the last N transformer layers (lm_head layers kept in BF16)

        gpus (Optional[int], defaults to first GPU):
            Number of GPUs to use for training. Default is the first available GPU. Use 0 for all available GPUs.
        use_cuda_graphs (Optional[bool], defaults to True):
            Enable or disable CUDA graphs for performance.

        learning_rate (Optional[float], defaults to 1e-4):
            The initial learning rate for [`AdamW`] optimizer.
        lr_scheduler_type (Optional[Literal['linear', 'cosine']]*, defaults to `"linear"`):
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
        per_device_eval_batch_size (Optional[int], defaults to 2):
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

        eval_steps (Optional[int], defaults to 100):
             Run evaluation every N optimizer steps.
        eval_num_steps (Optional[int], defaults to 100):
             Number of evaluation batches to process.
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
        lora_target_modules (Optional[str], default to 'all-linear'):
            List of comma-separated module names to apply LoRA adapters to.
        qlora_fp4: (Optional[bool], defaults to True):
            Enable NVFP4 QLoRA mode (base weights quantized to FP4 E2M1). Requires Blackwell GPU (SM100+)
        qlora_fp8: (Optional[bool], defaults to False):
            Enable FP8 QLoRA mode (base weights quantized to FP8 with per-block scales)
        qlora_block_size: (Optional[int], defaults to 128):
            Block size for QLoRA quantization. Valid values are 64, 128, 256.
        qlora_four_over_six: (Optional[bool], defaults to True):
            Enable Four Over Six (4/6) adaptive block scaling for NVFP4 QLoRA quantization.
            Evaluates both max=4 and max=6 scaling per block and selects lower error option.

        use_chat_template (Optional[bool], defaults to True):
            Whether to use chat template for training.
        merge_adapter: (Optional[bool], defaults to False):
            Whether to merge LoRA adapters into the base model after training.

        debug_time_breakdown (Optional[bool], defaults to False):
            Whether to enable detailed training timing breakdown for debugging.
        debug_memory_breakdown (Optional[bool], defaults to False):
            Print detailed memory breakdown after model allocation (useful for QLoRA optimization).
    """
    run_name: Optional[str] = None
    apply_recommended_values: Optional[bool] = False
    num_epochs: Optional[int] = 3
    output_dir: Optional[str] = 'output'
    checkpoint_dir: Optional[str] = 'output'
    resume_from_checkpoint: Optional[bool] = False
    save_steps: Optional[int] = 50
    save_total_limit: Optional[int] = 5

    recompute_swiglu: Optional[bool] = True
    recompute_rmsnorm: Optional[bool] = True
    recompute_ffn: Optional[bool] = True
    recompute_qkv: Optional[bool] = True
    recompute_att: Optional[bool] = True
    recompute_block: Optional[bool] = True

    offload_residual: Optional[bool] = True
    offload_master: Optional[bool] = False
    offload_quants: Optional[bool] = False
    persistent_quants: Optional[bool] = False
    offload_optimizer: Optional[bool] = True
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
    lmhead_chunks: Optional[int] = 1
    attn_bwd_chunks: Optional[int] = 1
    gradient_dtype: Optional[str] = None
    master_dtype: Optional[str] = None
    recipe: Optional[Literal['bf16', 'fp8_hybrid', 'nvfp4']] = 'bf16'
    use_fused_rope: Optional[bool] = False
    fp8_amax_history: Optional[int] = 1024
    fp4_backend: Optional[Literal['cutlass', 'cudnn']] = 'cutlass'
    no_fp4_hadamard: Optional[bool] = False
    no_fp4_stochastic_rounding: Optional[bool] = False
    skip_quant_first_layers: Optional[int] = 0
    skip_quant_last_layers: Optional[int] = 0

    gpus: Optional[int] = 1
    use_cuda_graphs: Optional[bool] = True

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
    eval_steps: Optional[int] = 100
    per_device_train_batch_size: Optional[int] = 2
    per_device_eval_batch_size: Optional[int] = 2
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
    qlora_fp4: Optional[bool] = False
    qlora_fp8: Optional[bool] = False
    qlora_block_size: Optional[int] = 128
    qlora_four_over_six: Optional[bool] = True

    merge_adapter: Optional[bool] = False
    use_chat_template: Optional[bool] = True
    debug_time_breakdown: Optional[bool] = False
    debug_memory_breakdown: Optional[bool] = False
    log_gpu_util: Optional[int] = 100

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.run_name = cfg['run_name'] or self.generate_run_name()
        self.apply_recommended_values = cfg.get('apply_recommended_values', self.apply_recommended_values)
        self.num_epochs = cfg.get('num_epochs', self.num_epochs)
        self.output_dir = cfg.get('output_dir', self.output_dir)
        self.resume_from_checkpoint = cfg.get('resume_from_checkpoint', self.resume_from_checkpoint)
        self.save_steps = cfg.get('save_steps', self.save_steps)
        self.save_total_limit = cfg.get('save_total_limit', self.save_total_limit)

        self.recompute_swiglu = cfg.get('recompute_swiglu', self.recompute_swiglu)
        self.recompute_rmsnorm = cfg.get('recompute_rmsnorm', self.recompute_rmsnorm)
        self.recompute_ffn = cfg.get('recompute_ffn', self.recompute_ffn)
        self.recompute_qkv = cfg.get('recompute_qkv', self.recompute_qkv)
        self.recompute_att = cfg.get('recompute_att', self.recompute_att)
        self.recompute_block = cfg.get('recompute_block', self.recompute_block)

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
        self.lmhead_chunks = cfg.get('lmhead_chunks', self.lmhead_chunks)
        self.attn_bwd_chunks = cfg.get('attn_bwd_chunks', self.attn_bwd_chunks)
        self.gradient_dtype = cfg.get('gradient_dtype', self.gradient_dtype)
        self.master_dtype = cfg.get('master_dtype', self.master_dtype)
        self.recipe = cfg.get('recipe', self.recipe)
        self.use_fused_rope = cfg.get('use_fused_rope', self.use_fused_rope)
        self.fp8_amax_history = cfg.get('fp8_amax_history', self.fp8_amax_history)
        self.fp4_backend = cfg.get('fp4_backend', self.fp4_backend)
        self.no_fp4_hadamard = cfg.get('no_fp4_hadamard', self.no_fp4_hadamard)
        self.no_fp4_stochastic_rounding = cfg.get('no_fp4_stochastic_rounding', self.no_fp4_stochastic_rounding)
        self.skip_quant_first_layers = cfg.get('skip_quant_first_layers', self.skip_quant_first_layers)
        self.skip_quant_last_layers = cfg.get('skip_quant_last_layers', self.skip_quant_last_layers)

        self.gpus = cfg.get('gpus', self.gpus)
        self.use_cuda_graphs = cfg.get('use_cuda_graphs', self.use_cuda_graphs)

        self.learning_rate = float(cfg.get('learning_rate', self.learning_rate))
        self.lr_scheduler_type = cfg.get('lr_scheduler_type', self.lr_scheduler_type)
        self.cooldown_steps = cfg.get('cooldown_steps', self.cooldown_steps)
        self.final_lr_fraction = float(cfg.get('final_lr_fraction', self.final_lr_fraction))
        self.gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', self.gradient_accumulation_steps)
        self.max_grad_norm = cfg.get('max_grad_norm', self.max_grad_norm)
        self.weight_decay = float(cfg.get('weight_decay', self.weight_decay))
        self.max_steps = cfg.get('max_steps', self.max_steps)
        self.adamw_beta1 = float(cfg.get('adamw_beta1', self.adamw_beta1))
        self.adamw_beta2 = float(cfg.get('adamw_beta2', self.adamw_beta2))
        self.adamw_epsilon = float(cfg.get('adamw_epsilon', self.adamw_epsilon))
        self.eval_steps = cfg.get('eval_steps', self.eval_steps)
        self.per_device_train_batch_size = cfg.get('per_device_train_batch_size', self.per_device_train_batch_size)
        self.per_device_eval_batch_size = cfg.get('per_device_eval_batch_size', self.per_device_eval_batch_size)
        self.report_to = cfg.get('report_to', self.report_to)
        self.warmup_ratio = float(cfg.get('warmup_ratio', self.warmup_ratio))
        self.warmup_steps = cfg.get('warmup_steps', self.warmup_steps)
        self.log_file = cfg.get('log_file', self.log_file)

        self.lora = cfg.get('lora', self.lora)
        self.lora_rank = cfg.get('lora_rank', self.lora_rank)
        self.lora_alpha = cfg.get('lora_alpha', self.lora_alpha)
        self.lora_dropout = cfg.get('lora_dropout', self.lora_dropout)
        self.lora_dtype = cfg.get('lora_dtype', self.lora_dtype)
        self.lora_target_modules = cfg.get('lora_target_modules', ['all-linear'])
        self.qlora_fp4 = cfg.get('qlora_fp4', self.qlora_fp4)
        self.qlora_fp8 = cfg.get('qlora_fp8', self.qlora_fp8)
        self.qlora_block_size = cfg.get('qlora_block_size', self.qlora_block_size)
        self.qlora_four_over_six = cfg.get('qlora_four_over_six', self.qlora_four_over_six)

        self.merge_adapter = cfg.get('merge_adapter', self.merge_adapter)
        self.use_chat_template = cfg.get('use_chat_template', self.use_chat_template)
        self.debug_time_breakdown = cfg.get('debug_time_breakdown', self.debug_time_breakdown)
        self.debug_memory_breakdown = cfg.get('debug_memory_breakdown', self.debug_memory_breakdown)

    def __post_init__(self):
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

        self.create_runtime_config()
        self.create_lora_config()
        self.create_qlora_config()

        self.ensure_directories()


    def ensure_directories(self):
        _output_dir = Path(self.output_dir)
        if _output_dir.exists():
            if not _output_dir.is_dir():
                raise ValueError(f"Save path '{_output_dir}' already exists and is not a directory. Aborting.")

            if any(item.is_dir() and item.name.startswith("checkpoint-") for item in _output_dir.iterdir()):
                logger.warning_once(f"Save path '{_output_dir}' contains previously saved checkpoints.")
        else:
            _output_dir.mkdir(parents=True, exist_ok=True)

        _checkpoint_dir = self.checkpoint_dir or self.output_dir
        _checkpoint_dir = Path(_checkpoint_dir)
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

        # Handle recomputation hierarchy (block -> att/ffn -> individual components)
        recompute_block = self.recompute_block
        recompute_att = self.recompute_att or recompute_block
        recompute_ffn = self.recompute_ffn or recompute_block
        recompute_qkv = self.recompute_qkv or recompute_att
        recompute_swiglu = self.recompute_swiglu or recompute_ffn
        recompute_rmsnorm = self.recompute_rmsnorm or recompute_block

        self.runtime_config = _surogate.RuntimeOptions(
            recompute_swiglu=recompute_swiglu,
            recompute_rmsnorm=recompute_rmsnorm,
            recompute_ffn=recompute_ffn,
            recompute_qkv=recompute_qkv,
            recompute_att=recompute_att,
            recompute_block=recompute_block,
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
            no_fp4_hadamard=self.no_fp4_hadamard,
            no_fp4_stochastic_rounding=self.no_fp4_stochastic_rounding,
            skip_quant_first_layers=self.skip_quant_first_layers,
            skip_quant_last_layers=self.skip_quant_last_layers,
        )
        self.runtime_config.use_zero_copy = self.use_zero_copy
        self.runtime_config.use_write_combined = self.use_write_combined

    def create_lora_config(self):
        self.lora_config = _surogate.LoRAAdapterConfig(
            rank=self.lora_rank if self.lora else 0,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            dtype=self.lora_dtype,
            target_modules=self.lora_target_modules,
            use_rslora=False
        )

    def create_qlora_config(self):
        self.qlora_config = None
        if self.qlora_fp4:
            self.qlora_config = _surogate.QLoRAConfig.nvfp4()
            self.qlora_config.enable_four_over_six = self.qlora_four_over_six
        elif self.qlora_fp8:
            self.qlora_config = _surogate.QLoRAConfig.fp8(block_size=self.qlora_block_size)

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
