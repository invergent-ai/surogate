from __future__ import annotations
import collections.abc
import nanobind
import enum
import typing
import numpy as np
import numpy.typing as npt

__all__: list[str] = ['DataLoader', 'GPUInfo', 'GPUUtilInfo', 'LoRAAdapterConfig', 'LogVerbosity', 'OptimizerConfig', 'OptimizerType', 'PretrainedConfig', 'QLoRAConfig', 'QLoRAQuantStrategy', 'RuntimeOptions', 'SurogateTrainer', 'SystemInfo', 'TrainingRunLogger', 'clean_old_checkpoints', 'find_latest_checkpoint', 'get_all_checkpoints', 'get_checkpoint_path', 'get_num_gpus']

class OptimizerType(enum.Enum):
    """
    Optimizer algorithm types.

    Values:
    - ADAMW_8BIT: 8-bit AdamW with blockwise quantization.
    - NORMUON: NorMuon hybrid optimizer (orthogonalized momentum for 2D weights, AdamW for others).
    """
    ADAMW_8BIT: typing.ClassVar[OptimizerType]  # value = OptimizerType.ADAMW_8BIT
    NORMUON: typing.ClassVar[OptimizerType]  # value = OptimizerType.NORMUON

class OptimizerConfig:
    """
    Optimizer configuration.

    Contains all hyperparameters for supported optimizers.
    Parameters for unused optimizers are ignored.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, *, optimizer: str = 'adamw_8bit', learning_rate: float = 2e-4, weight_decay: float = 0.1, grad_clip: float = 0.0, adamw_beta1: float = 0.9, adamw_beta2: float = 0.999, adamw_epsilon: float = 1e-8, normuon_momentum: float = 0.95, normuon_beta2: float = 0.95, normuon_lr: float = 0.02, normuon_cautious_wd: bool = True) -> None:
        """
        Create an optimizer configuration.

        Parameters:
        - optimizer: Type of optimizer ('adamw_8bit' or 'normuon').
        - learning_rate: Base learning rate.
        - weight_decay: Weight decay coefficient.
        - grad_clip: Gradient clipping threshold (0 = disabled).
        - adamw_beta1/beta2/epsilon: AdamW hyperparameters.
        - normuon_momentum/beta2/lr/cautious_wd: NorMuon hyperparameters.
        """
    def __repr__(self) -> str:
        """
        Return a debug string representation.
        """
    @staticmethod
    def adamw_8bit(lr: float = 2e-4, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.1, grad_clip: float = 0.0) -> OptimizerConfig:
        """
        Create AdamW 8-bit configuration.
        """
    @staticmethod
    def normuon(lr: float = 0.02, momentum: float = 0.95, beta2: float = 0.95, weight_decay: float = 0.01, grad_clip: float = 0.0, cautious_wd: bool = True) -> OptimizerConfig:
        """
        Create NorMuon configuration.

        NorMuon uses orthogonalized momentum for 2D weight matrices and AdamW for other parameters.
        """
    @property
    def learning_rate(self) -> float:
        """Base learning rate."""
    @learning_rate.setter
    def learning_rate(self, arg: float) -> None:
        """Base learning rate."""
    @property
    def weight_decay(self) -> float:
        """Weight decay coefficient."""
    @weight_decay.setter
    def weight_decay(self, arg: float) -> None:
        """Weight decay coefficient."""
    @property
    def grad_clip(self) -> float:
        """Gradient clipping threshold."""
    @grad_clip.setter
    def grad_clip(self, arg: float) -> None:
        """Gradient clipping threshold."""
    @property
    def adamw_beta1(self) -> float:
        """AdamW beta1."""
    @adamw_beta1.setter
    def adamw_beta1(self, arg: float) -> None:
        """AdamW beta1."""
    @property
    def adamw_beta2(self) -> float:
        """AdamW beta2."""
    @adamw_beta2.setter
    def adamw_beta2(self, arg: float) -> None:
        """AdamW beta2."""
    @property
    def adamw_epsilon(self) -> float:
        """AdamW epsilon."""
    @adamw_epsilon.setter
    def adamw_epsilon(self, arg: float) -> None:
        """AdamW epsilon."""
    @property
    def normuon_momentum(self) -> float:
        """NorMuon momentum (beta1)."""
    @normuon_momentum.setter
    def normuon_momentum(self, arg: float) -> None:
        """NorMuon momentum (beta1)."""
    @property
    def normuon_beta2(self) -> float:
        """NorMuon variance EMA (beta2)."""
    @normuon_beta2.setter
    def normuon_beta2(self, arg: float) -> None:
        """NorMuon variance EMA (beta2)."""
    @property
    def normuon_lr(self) -> float:
        """NorMuon learning rate."""
    @normuon_lr.setter
    def normuon_lr(self, arg: float) -> None:
        """NorMuon learning rate."""
    @property
    def normuon_cautious_wd(self) -> bool:
        """Use cautious weight decay."""
    @normuon_cautious_wd.setter
    def normuon_cautious_wd(self, arg: bool) -> None:
        """Use cautious weight decay."""
    @property
    def type(self) -> str:
        """Optimizer type as string."""
    @type.setter
    def type(self, arg: str) -> None:
        """Optimizer type as string."""
class DataLoader:
    """
    Streaming token dataset loader.
    
    Loads fixed-size chunks from a list of token files and fills preallocated arrays.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, file_list: collections.abc.Sequence[str], chunk_size: int, seed: int = 42) -> None:
        """
        Create a DataLoader.
        
        Parameters:
        - file_list: List of dataset file paths.
        - chunk_size: Chunk size in tokens.
        - seed: RNG seed controlling shuffling/order.
        """
    def advance_epoch(self) -> None:
        """
        Advance to the next epoch and reshuffle chunk order.
        """
    def epoch(self) -> int:
        """
        Return the current epoch number (0-based).
        """
    def has_next(self, chunks: int = 1) -> bool:
        """
        Return True if at least `chunks` more chunks are available in the current epoch.
        """
    def load_batch(self, inputs: npt.NDArray[np.int32], targets: npt.NDArray[np.int32]) -> None:
        """
        Fill `inputs` and `targets` with the next batch.
        
        Parameters:
        - inputs: Preallocated int32 array [batch, seq_len].
        - targets: Preallocated int32 array [batch, seq_len].
        """
    def progress(self) -> float:
        """
        Return progress within the current epoch (percent).
        """
    def set_state(self, seed: int, epoch: int, file_index: int, chunk_index: int) -> None:
        """
        Set the internal iteration state.
        
        Parameters:
        - seed: RNG seed.
        - epoch: Epoch number.
        - file_index: Current file index.
        - chunk_index: Current chunk index within the file.
        """
    @property
    def num_chunks(self) -> int:
        """
        Total number of chunks across all files.
        """
    @property
    def num_files(self) -> int:
        """
        Number of files in `file_list`.
        """
    @property
    def num_tokens(self) -> int:
        """
        Total number of tokens across all files.
        """
    @property
    def seed(self) -> int:
        """
        Current RNG seed.
        """
    @property
    def seq_len(self) -> int:
        """
        Sequence length produced by this loader.
        """
    @property
    def vocab_size(self) -> int:
        """
        Vocabulary size declared by the dataset.
        """
class GPUInfo:
    """
    Information about a single GPU device.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self) -> str:
        """
        Return a debug string representation.
        """
    @property
    def compute_capability_major(self) -> int:
        """
        Compute capability major version.
        """
    @compute_capability_major.setter
    def compute_capability_major(self, arg: int) -> None:
        """
        Compute capability major version.
        """
    @property
    def compute_capability_minor(self) -> int:
        """
        Compute capability minor version.
        """
    @compute_capability_minor.setter
    def compute_capability_minor(self, arg: int) -> None:
        """
        Compute capability minor version.
        """
    @property
    def device_id(self) -> int:
        """
        Device ID (0-indexed).
        """
    @device_id.setter
    def device_id(self, arg: int) -> None:
        """
        Device ID (0-indexed).
        """
    @property
    def name(self) -> str:
        """
        Device name.
        """
    @name.setter
    def name(self, arg: str) -> None:
        """
        Device name.
        """
    @property
    def total_memory(self) -> int:
        """
        Total device memory (bytes).
        """
    @total_memory.setter
    def total_memory(self, arg: int) -> None:
        """
        Total device memory (bytes).
        """
class GPUUtilInfo:
    """
    Snapshot of GPU utilization/telemetry.
    
    All fields are read/write and represent the most recently sampled values.
    Units are implementation-defined; typically MHz for clocks, W for power, C for temperatures, and bytes for memory counters.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self) -> str:
        """
        Return a debug string representation.
        """
    @property
    def clock(self) -> int:
        """
        Current GPU clock (typically MHz).
        """
    @clock.setter
    def clock(self, arg: int) -> None:
        """
        Current GPU clock (typically MHz).
        """
    @property
    def fan(self) -> int:
        """
        Fan speed (typically percent).
        """
    @fan.setter
    def fan(self, arg: int) -> None:
        """
        Fan speed (typically percent).
        """
    @property
    def gpu_utilization(self) -> float:
        """
        GPU utilization (typically percent).
        """
    @gpu_utilization.setter
    def gpu_utilization(self, arg: float) -> None:
        """
        GPU utilization (typically percent).
        """
    @property
    def max_clock(self) -> int:
        """
        Maximum GPU clock (typically MHz).
        """
    @max_clock.setter
    def max_clock(self, arg: int) -> None:
        """
        Maximum GPU clock (typically MHz).
        """
    @property
    def mem_free(self) -> int:
        """
        Free device memory (bytes).
        """
    @mem_free.setter
    def mem_free(self, arg: int) -> None:
        """
        Free device memory (bytes).
        """
    @property
    def mem_reserved(self) -> int:
        """
        Reserved device memory (bytes).
        """
    @mem_reserved.setter
    def mem_reserved(self, arg: int) -> None:
        """
        Reserved device memory (bytes).
        """
    @property
    def mem_total(self) -> int:
        """
        Total device memory (bytes).
        """
    @mem_total.setter
    def mem_total(self, arg: int) -> None:
        """
        Total device memory (bytes).
        """
    @property
    def mem_utilization(self) -> float:
        """
        Memory utilization (typically percent).
        """
    @mem_utilization.setter
    def mem_utilization(self, arg: float) -> None:
        """
        Memory utilization (typically percent).
        """
    @property
    def pcie_rx(self) -> int:
        """
        PCIe receive throughput (implementation-defined units).
        """
    @pcie_rx.setter
    def pcie_rx(self, arg: int) -> None:
        """
        PCIe receive throughput (implementation-defined units).
        """
    @property
    def pcie_tx(self) -> int:
        """
        PCIe transmit throughput (implementation-defined units).
        """
    @pcie_tx.setter
    def pcie_tx(self, arg: int) -> None:
        """
        PCIe transmit throughput (implementation-defined units).
        """
    @property
    def power(self) -> int:
        """
        Current GPU power draw (typically W).
        """
    @power.setter
    def power(self, arg: int) -> None:
        """
        Current GPU power draw (typically W).
        """
    @property
    def power_limit(self) -> int:
        """
        Configured power limit (typically W).
        """
    @power_limit.setter
    def power_limit(self, arg: int) -> None:
        """
        Configured power limit (typically W).
        """
    @property
    def temp_slowdown(self) -> int:
        """
        Thermal slowdown threshold (typically Celsius).
        """
    @temp_slowdown.setter
    def temp_slowdown(self, arg: int) -> None:
        """
        Thermal slowdown threshold (typically Celsius).
        """
    @property
    def temperature(self) -> int:
        """
        GPU temperature (typically Celsius).
        """
    @temperature.setter
    def temperature(self, arg: int) -> None:
        """
        GPU temperature (typically Celsius).
        """
    @property
    def throttle_reason(self) -> str:
        """
        Vendor-specific throttling reason bitmask/string code.
        """
    @throttle_reason.setter
    def throttle_reason(self, arg: str) -> None:
        """
        Vendor-specific throttling reason bitmask/string code.
        """
class LoRAAdapterConfig:
    """
    LoRA (Low-Rank Adaptation) adapter configuration.
    
    Controls which modules receive LoRA adapters and with which rank/scaling/dtype.
    
    Backwards-compatibility: `LoRAConfig` is an alias of this class.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, *, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0, target_modules: collections.abc.Sequence[str] = ['q_proj', 'k_proj', 'v_proj', 'o_proj'], dtype: str = 'bf16', use_rslora: bool = False) -> None:
        """
        Create a LoRA configuration.
        
        Parameters:
        - rank: LoRA rank.
        - alpha: LoRA alpha (scaling numerator).
        - dropout: LoRA dropout probability.
        - target_modules: Module name suffixes to apply LoRA to.
        - dtype: Adapter dtype.
        - use_rslora: Enable RS-LoRA scaling variant.
        """
    def __repr__(self) -> str:
        """
        Return a debug string representation.
        """
    def applies_to(self, module_name: str) -> bool:
        """
        Return True if LoRA should be applied to `module_name`.
        """
    @property
    def alpha(self) -> float:
        """
        LoRA alpha.
        """
    @alpha.setter
    def alpha(self, arg: float) -> None:
        """
        LoRA alpha.
        """
    @property
    def dropout(self) -> float:
        """
        LoRA dropout probability.
        """
    @dropout.setter
    def dropout(self, arg: float) -> None:
        """
        LoRA dropout probability.
        """
    @property
    def dtype(self) -> str:
        """
        Adapter dtype as a string.
        """
    @dtype.setter
    def dtype(self, arg: str) -> None:
        """
        Adapter dtype as a string.
        """
    @property
    def rank(self) -> int:
        """
        LoRA rank.
        """
    @rank.setter
    def rank(self, arg: int) -> None:
        """
        LoRA rank.
        """
    @property
    def scaling(self) -> float:
        """
        Computed scaling factor (= alpha / rank, RS-LoRA aware).
        """
    @property
    def target_modules(self) -> list[str]:
        """
        List of module suffixes the adapter should apply to.
        """
    @target_modules.setter
    def target_modules(self, arg: collections.abc.Sequence[str]) -> None:
        """
        List of module suffixes the adapter should apply to.
        """
    @property
    def use_rslora(self) -> bool:
        """
        Whether to use RS-LoRA variant.
        """
    @use_rslora.setter
    def use_rslora(self, arg: bool) -> None:
        """
        Whether to use RS-LoRA variant.
        """
class LogVerbosity(enum.Enum):
    """
    Logger verbosity level.
    """
    DEFAULT: typing.ClassVar[LogVerbosity]  # value = LogVerbosity.DEFAULT
    QUIET: typing.ClassVar[LogVerbosity]  # value = LogVerbosity.QUIET
    SILENT: typing.ClassVar[LogVerbosity]  # value = LogVerbosity.SILENT
    VERBOSE: typing.ClassVar[LogVerbosity]  # value = LogVerbosity.VERBOSE
class PretrainedConfig:
    """
    Model configuration used to build/initialize a transformer.
    
    Notes:
    - Some defaults depend on `architecture`.
    - `dtype` controls the model's compute/storage type where applicable.
    
    Backwards-compatibility: `LLamaConfig` is an alias of this class.
    """
    from_name: typing.ClassVar[nanobind.nb_func]  # value = <nanobind.nb_func object>
    from_pretrained: typing.ClassVar[nanobind.nb_func]  # value = <nanobind.nb_func object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, *, architecture: str, bos_token_id: int | None = None, eos_token_id: int | None = None, hidden_size: int, intermediate_size: int, vocab_size: int | None = None, num_attention_heads: int, num_key_value_heads: int, num_hidden_layers: int, max_position_embeddings: int | None = None, rope_theta: float | None = None, rms_norm_eps: float, tie_word_embeddings: bool, use_qkv_bias: bool | None = None, dtype: str = 'bf16') -> None:
        """
        Create a model configuration.
        
        Parameters:
        - architecture: Model family identifier (currently: qwen2).
        - bos_token_id/eos_token_id: Token IDs; if None, architecture defaults are used.
        - hidden_size/intermediate_size: Transformer dimensions.
        - vocab_size: Vocabulary size; if None, architecture default is used.
        - num_attention_heads/num_key_value_heads: Attention head counts.
        - num_hidden_layers: Number of transformer blocks.
        - max_position_embeddings: Max sequence length; if None, architecture default is used.
        - rope_theta: RoPE base; if None, architecture default is used.
        - rms_norm_eps: Epsilon for RMSNorm.
        - tie_word_embeddings: Whether input/output embeddings are tied.
        - use_qkv_bias: Whether QKV projections use bias; if None, architecture default is used.
        - dtype: Tensor dtype string (e.g. 'bf16', 'fp16', 'fp32').
        """
    @property
    def architecture(self) -> ...:
        """
        Architecture identifier (enum-backed).
        """
    @architecture.setter
    def architecture(self, arg: ...) -> None:
        """
        Architecture identifier (enum-backed).
        """
    @property
    def bos_token_id(self) -> int:
        """
        Beginning-of-sequence token id.
        """
    @bos_token_id.setter
    def bos_token_id(self, arg: int) -> None:
        """
        Beginning-of-sequence token id.
        """
    @property
    def dtype(self) -> str:
        """
        Model dtype as a string (e.g. 'bf16', 'fp16', 'fp32').
        """
    @dtype.setter
    def dtype(self, arg: str) -> None:
        """
        Model dtype as a string (e.g. 'bf16', 'fp16', 'fp32').
        """
    @property
    def eos_token_id(self) -> int:
        """
        End-of-sequence token id.
        """
    @eos_token_id.setter
    def eos_token_id(self, arg: int) -> None:
        """
        End-of-sequence token id.
        """
    @property
    def head_size(self) -> int:
        """
        Attention head size (= hidden_size / num_attention_heads).
        """
    @property
    def hidden_size(self) -> int:
        """
        Transformer hidden size.
        """
    @hidden_size.setter
    def hidden_size(self, arg: int) -> None:
        """
        Transformer hidden size.
        """
    @property
    def intermediate_size(self) -> int:
        """
        FFN intermediate size.
        """
    @intermediate_size.setter
    def intermediate_size(self, arg: int) -> None:
        """
        FFN intermediate size.
        """
    @property
    def max_position_embeddings(self) -> int:
        """
        Maximum supported sequence length.
        """
    @max_position_embeddings.setter
    def max_position_embeddings(self, arg: int) -> None:
        """
        Maximum supported sequence length.
        """
    @property
    def model_name(self) -> ...:
        """
        Canonical model name derived from the configuration.
        """
    @property
    def num_attention_heads(self) -> int:
        """
        Number of query attention heads.
        """
    @num_attention_heads.setter
    def num_attention_heads(self, arg: int) -> None:
        """
        Number of query attention heads.
        """
    @property
    def num_hidden_layers(self) -> int:
        """
        Number of transformer layers/blocks.
        """
    @num_hidden_layers.setter
    def num_hidden_layers(self, arg: int) -> None:
        """
        Number of transformer layers/blocks.
        """
    @property
    def num_key_value_heads(self) -> int:
        """
        Number of key/value attention heads (for GQA/MQA).
        """
    @num_key_value_heads.setter
    def num_key_value_heads(self, arg: int) -> None:
        """
        Number of key/value attention heads (for GQA/MQA).
        """
    @property
    def qkv_channels(self) -> int:
        """
        Total QKV channel count used internally.
        """
    @property
    def rms_norm_eps(self) -> float:
        """
        Epsilon used in RMSNorm.
        """
    @rms_norm_eps.setter
    def rms_norm_eps(self, arg: float) -> None:
        """
        Epsilon used in RMSNorm.
        """
    @property
    def rope_theta(self) -> float:
        """
        RoPE base parameter (theta).
        """
    @rope_theta.setter
    def rope_theta(self, arg: float) -> None:
        """
        RoPE base parameter (theta).
        """
    @property
    def tie_word_embeddings(self) -> bool:
        """
        Whether input/output embeddings are tied.
        """
    @tie_word_embeddings.setter
    def tie_word_embeddings(self, arg: bool) -> None:
        """
        Whether input/output embeddings are tied.
        """
    @property
    def use_qkv_bias(self) -> bool:
        """
        Whether QKV projections use bias.
        """
    @use_qkv_bias.setter
    def use_qkv_bias(self, arg: bool) -> None:
        """
        Whether QKV projections use bias.
        """
    @property
    def vocab_size(self) -> int:
        """
        Vocabulary size.
        """
    @vocab_size.setter
    def vocab_size(self, arg: int) -> None:
        """
        Vocabulary size.
        """
class QLoRAConfig:
    """
    QLoRA (Quantized LoRA) configuration for memory-efficient adapter training.
    
    Configures quantization of base model weights. The base model is stored in a
    quantized format (FP8 or FP4) while LoRA adapters remain in full precision.
    
    Use QLoRAConfig.fp8() or QLoRAConfig.nvfp4() factory methods to create configs.
    """
    fp8: typing.ClassVar[nanobind.nb_func]  # value = <nanobind.nb_func object>
    none: typing.ClassVar[nanobind.nb_func]  # value = <nanobind.nb_func object>
    nvfp4: typing.ClassVar[nanobind.nb_func]  # value = <nanobind.nb_func object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, *, enabled: bool = False, strategy: str = 'none', block_size: int = 128, base_dtype: str = '', adapter_dtype: str = 'bf16') -> None:
        """
        Create a QLoRA configuration.
        
        Parameters:
        - enabled: Whether QLoRA is enabled.
        - strategy: Quantization strategy ('none', 'fp8', 'nvfp4').
        - block_size: Block size for per-block quantization (FP8: 64/128/256, FP4: 16).
        - base_dtype: Storage dtype for quantized base weights.
        - adapter_dtype: Dtype for LoRA adapter weights (not quantized).
        """
    def __repr__(self) -> str:
        """
        Return a debug string representation.
        """
    @property
    def adapter_dtype(self) -> str:
        """
        Dtype for LoRA adapter weights.
        """
    @adapter_dtype.setter
    def adapter_dtype(self, arg: str) -> None:
        """
        Dtype for LoRA adapter weights.
        """
    @property
    def base_dtype(self) -> str:
        """
        Storage dtype for quantized base weights.
        """
    @base_dtype.setter
    def base_dtype(self, arg: str) -> None:
        """
        Storage dtype for quantized base weights.
        """
    @property
    def block_size(self) -> int:
        """
        Block size for per-block quantization.
        """
    @block_size.setter
    def block_size(self, arg: int) -> None:
        """
        Block size for per-block quantization.
        """
    @property
    def enabled(self) -> bool:
        """
        Whether QLoRA is enabled.
        """
    @enabled.setter
    def enabled(self, arg: bool) -> None:
        """
        Whether QLoRA is enabled.
        """
    @property
    def is_fp4(self) -> bool:
        """
        Whether using FP4 quantization.
        """
    @property
    def is_fp8(self) -> bool:
        """
        Whether using FP8 quantization.
        """
    @property
    def is_quantized(self) -> bool:
        """
        Whether quantization is active (enabled and strategy != None).
        """
    @property
    def is_moe(self) -> bool:
        """
        Whether this is an MoE model (num_experts > 0).
        """
    @property
    def strategy(self) -> str:
        """
        Quantization strategy as a string.
        """
    @strategy.setter
    def strategy(self, arg: str) -> None:
        """
        Quantization strategy as a string.
        """
    @property
    def num_experts(self) -> int:
        """
        Number of experts for MoE models (0 = dense model, >0 = MoE model).
        """
    @num_experts.setter
    def num_experts(self, arg: int) -> None:
        """
        Number of experts for MoE models (0 = dense model, >0 = MoE model).
        """
    @property
    def num_experts_per_tok(self) -> int:
        """
        Number of experts selected per token (top-k routing).
        """
    @num_experts_per_tok.setter
    def num_experts_per_tok(self, arg: int) -> None:
        """
        Number of experts selected per token (top-k routing).
        """
    @property
    def moe_intermediate_size(self) -> int:
        """
        Per-expert MLP intermediate size (0 = use regular intermediate_size).
        """
    @moe_intermediate_size.setter
    def moe_intermediate_size(self, arg: int) -> None:
        """
        Per-expert MLP intermediate size (0 = use regular intermediate_size).
        """
class QLoRAQuantStrategy(enum.Enum):
    """
    Quantization strategy for QLoRA base weights.
    """
    FP8: typing.ClassVar[QLoRAQuantStrategy]  # value = QLoRAQuantStrategy.FP8
    NONE: typing.ClassVar[QLoRAQuantStrategy]  # value = QLoRAQuantStrategy.NONE
    NVFP4: typing.ClassVar[QLoRAQuantStrategy]  # value = QLoRAQuantStrategy.NVFP4
class RuntimeOptions:
    """
    Execution/training options controlling recomputation, offloading, sharding, and dtypes.
    
    Many flags trade compute for memory (recompute) or host/device transfers (offload).
    
    Backwards-compatibility: `LLamaOptions` is an alias of this class.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, *, recompute_swiglu: bool = False, recompute_rmsnorm: bool = False, recompute_ffn: bool = False, recompute_qkv: bool = False, recompute_att: bool = False, recompute_block: bool = False, offload_residual: bool = False, use_cuda_graphs: bool = True, trigger_timing_events: bool = False, offload_master: bool = False, offload_quants: bool = False, offload_optimizer: bool = False, offload_grads: bool = False, use_zero_copy: bool = False, use_write_combined: bool = False, shard_weights: bool = False, persistent_quants: bool = False, shard_gradients: bool = False, use_all_to_all_reduce: bool = False, init_projections_to_zero: bool = False, lmhead_chunks: int = 1, attn_bwd_chunks: int = 1, matmul_type: str = '', gradient_type: str = '', master_dtype: str = '', recipe: str = 'bf16', matmul_backend: str = '', use_fused_rope: bool = False, fp8_amax_history: int = 1024, fp4_backend: str = 'cutlass', no_fp4_stochastic_rounding: bool = False, skip_quant_first_layers: int = 0, skip_quant_last_layers: int = 0) -> None:
        """
        Create runtime/training options.
        
        Parameters:
        - recompute_*: Enable recomputation for submodules to reduce activation memory.
        - offload_*: Offload specific buffers/states; may reduce VRAM at performance cost.
        - use_cuda_graphs: Enable CUDA graphs where supported.
        - trigger_timing_events: Log additional timing information.
        - shard_*: Enable sharding of weights/gradients across GPUs.
        - use_all_to_all_reduce: Use all-to-all based reduction (if supported by backend).
        - *_type/master_dtype: Dtype strings (empty means default/auto for optional fields).
        - recipe: Training recipe (bf16, fp8-hybrid, nvfp4).
        - matmul_backend: Matmul backend (auto, cublaslt, cutlass).
        - use_fused_rope: Use fused RoPE kernel with on-the-fly cos/sin computation.
        - fp8_amax_history: FP8 delayed scaling amax history length (for fp8-hybrid recipe).
        - fp4_backend: FP4 matmul backend (cudnn, cutlass).
        - no_fp4_stochastic_rounding: Disable stochastic rounding for NVFP4 gradients.
        - skip_quant_first_layers: Skip quantization for first N layers.
        - skip_quant_last_layers: Skip quantization for last N layers.
        """
    def set_recipe(self, recipe_name: str) -> None:
        """
        Set training recipe by name (bf16, fp8-hybrid, nvfp4).
        """
    @property
    def attn_bwd_chunks(self) -> int:
        """
        Split attention backward into this many chunks.
        """
    @attn_bwd_chunks.setter
    def attn_bwd_chunks(self, arg: int) -> None:
        """
        Split attention backward into this many chunks.
        """
    @property
    def fp4_enabled(self) -> bool:
        """
        Whether FP4 training is enabled.
        """
    @property
    def fp8_enabled(self) -> bool:
        """
        Whether FP8 forward pass is enabled.
        """
    @property
    def gradient_type(self) -> ETensorDType:
        """
        Optional override dtype for gradient computations (empty/None means default).
        """
    @gradient_type.setter
    def gradient_type(self, arg: str) -> None:
        """
        Optional override dtype for gradient computations (empty/None means default).
        """
    @property
    def init_projections_to_zero(self) -> bool:
        """
        Initialize certain projections to zero (for experiments).
        """
    @init_projections_to_zero.setter
    def init_projections_to_zero(self, arg: bool) -> None:
        """
        Initialize certain projections to zero (for experiments).
        """
    @property
    def lmhead_chunks(self) -> int:
        """
        Split LM head computation into this many chunks.
        """
    @lmhead_chunks.setter
    def lmhead_chunks(self, arg: int) -> None:
        """
        Split LM head computation into this many chunks.
        """
    @property
    def master_dtype(self) -> typing.Any:
        """
        Optional override dtype for master weights (empty/None means default).
        """
    @master_dtype.setter
    def master_dtype(self, arg: str) -> None:
        """
        Optional override dtype for master weights (empty/None means default).
        """
    @property
    def matmul_backend(self) -> str:
        """
        Matmul backend (auto, cublaslt, cutlass).
        """
    @matmul_backend.setter
    def matmul_backend(self, arg: str) -> None:
        """
        Matmul backend (auto, cublaslt, cutlass).
        """
    @property
    def matmul_type(self) -> ETensorDType:
        """
        Optional override dtype for matmul kernels (empty/None means default).
        """
    @matmul_type.setter
    def matmul_type(self, arg: str) -> None:
        """
        Optional override dtype for matmul kernels (empty/None means default).
        """
    @property
    def offload_grads(self) -> bool:
        """
        Offload gradients.
        """
    @offload_grads.setter
    def offload_grads(self, arg: bool) -> None:
        """
        Offload gradients.
        """
    @property
    def offload_master(self) -> bool:
        """
        Offload FP32 master weights (optimizer state).
        """
    @offload_master.setter
    def offload_master(self, arg: bool) -> None:
        """
        Offload FP32 master weights (optimizer state).
        """
    @property
    def offload_optimizer(self) -> bool:
        """
        Offload optimizer state (momentum and variance buffers).
        """
    @offload_optimizer.setter
    def offload_optimizer(self, arg: bool) -> None:
        """
        Offload optimizer state (momentum and variance buffers).
        """
    @property
    def offload_quants(self) -> bool:
        """
        Offload quantized weights (if applicable).
        """
    @offload_quants.setter
    def offload_quants(self, arg: bool) -> None:
        """
        Offload quantized weights (if applicable).
        """
    @property
    def offload_residual(self) -> bool:
        """
        Offload residual stream buffers.
        """
    @offload_residual.setter
    def offload_residual(self, arg: bool) -> None:
        """
        Offload residual stream buffers.
        """
    @property
    def persistent_quants(self) -> bool:
        """
        Keep quant buffers persistent across steps.
        """
    @persistent_quants.setter
    def persistent_quants(self, arg: bool) -> None:
        """
        Keep quant buffers persistent across steps.
        """
    @property
    def recipe_name(self) -> str:
        """
        Current training recipe name.
        """
    @property
    def recompute_att(self) -> bool:
        """
        Recompute attention in backward.
        """
    @recompute_att.setter
    def recompute_att(self, arg: bool) -> None:
        """
        Recompute attention in backward.
        """
    @property
    def recompute_block(self) -> bool:
        """
        Recompute the whole block (coarse-grained).
        """
    @recompute_block.setter
    def recompute_block(self, arg: bool) -> None:
        """
        Recompute the whole block (coarse-grained).
        """
    @property
    def recompute_ffn(self) -> bool:
        """
        Recompute FFN in backward.
        """
    @recompute_ffn.setter
    def recompute_ffn(self, arg: bool) -> None:
        """
        Recompute FFN in backward.
        """
    @property
    def recompute_qkv(self) -> bool:
        """
        Recompute QKV projections in backward.
        """
    @recompute_qkv.setter
    def recompute_qkv(self, arg: bool) -> None:
        """
        Recompute QKV projections in backward.
        """
    @property
    def recompute_rms_norm(self) -> bool:
        """
        Recompute RMSNorm in backward.
        """
    @recompute_rms_norm.setter
    def recompute_rms_norm(self, arg: bool) -> None:
        """
        Recompute RMSNorm in backward.
        """
    @property
    def recompute_swiglu(self) -> bool:
        """
        Recompute SwiGLU activations in backward.
        """
    @recompute_swiglu.setter
    def recompute_swiglu(self, arg: bool) -> None:
        """
        Recompute SwiGLU activations in backward.
        """
    @property
    def shard_gradients(self) -> bool:
        """
        Shard gradients across GPUs.
        """
    @shard_gradients.setter
    def shard_gradients(self, arg: bool) -> None:
        """
        Shard gradients across GPUs.
        """
    @property
    def shard_weights(self) -> bool:
        """
        Shard model weights across GPUs.
        """
    @shard_weights.setter
    def shard_weights(self, arg: bool) -> None:
        """
        Shard model weights across GPUs.
        """
    @property
    def trigger_timing_events(self) -> bool:
        """
        Log additional timing information.
        """
    @trigger_timing_events.setter
    def trigger_timing_events(self, arg: bool) -> None:
        """
        Log additional timing information.
        """
    @property
    def use_all_to_all_reduce(self) -> bool:
        """
        Use all-to-all reduce strategy when reducing gradients.
        """
    @use_all_to_all_reduce.setter
    def use_all_to_all_reduce(self, arg: bool) -> None:
        """
        Use all-to-all reduce strategy when reducing gradients.
        """
    @property
    def use_cuda_graphs(self) -> bool:
        """
        Enable CUDA graphs for steady-state execution.
        """
    @use_cuda_graphs.setter
    def use_cuda_graphs(self, arg: bool) -> None:
        """
        Enable CUDA graphs for steady-state execution.
        """
    @property
    def use_fused_rope(self) -> bool:
        """
        Use fused RoPE kernel with on-the-fly cos/sin computation.
        """
    @use_fused_rope.setter
    def use_fused_rope(self, arg: bool) -> None:
        """
        Use fused RoPE kernel with on-the-fly cos/sin computation.
        """
    @property
    def use_write_combined(self) -> bool:
        """
        Use write-combined host memory (pinned).
        """
    @use_write_combined.setter
    def use_write_combined(self, arg: bool) -> None:
        """
        Use write-combined host memory (pinned).
        """
    @property
    def use_zero_copy(self) -> bool:
        """
        Use zero-copy buffers where supported.
        """
    @use_zero_copy.setter
    def use_zero_copy(self, arg: bool) -> None:
        """
        Use zero-copy buffers where supported.
        """
class SurogateTrainer:
    """
    Multi-GPU trainer wrapper.
    
    Provides training/evaluation steps and checkpoint/weight import/export.
    Some operations may run asynchronously (see method docs).
    """
    from_pretrained: typing.ClassVar[nanobind.nb_func]  # value = <nanobind.nb_func object>
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, ngpu: int, config: PretrainedConfig, options: RuntimeOptions, batch_size: int, seq_len: int, grad_accum: int, memcpy_all_gather: bool = True, memcpy_send_recv: bool = True, lora_config: surogate._surogate.LoRAAdapterConfig | None = None, qlora_config: surogate._surogate.QLoRAConfig | None = None) -> None:
        """
        Create a trainer instance.
        
        Parameters:
        - ngpu: Number of GPUs to use.
        - config: Model configuration.
        - options: Runtime/training options.
        - batch_size: Per-GPU batch size (effective batch is batch_size * world_size).
        - seq_len: Sequence length.
        - grad_accum: Gradient accumulation steps.
        - memcpy_all_gather/memcpy_send_recv: Enable memcpy-based collectives where supported.
        - lora_config: Optional LoRA configuration for adapter training (freezes base model).
        - qlora_config: Optional QLoRA configuration for quantized base weights (FP8/FP4).
        """
    def export_adapter(self, path: str, base_model_path: str = '') -> None:
        """
        Export LoRA adapter weights to a directory (PEFT-compatible format).
        
        Only works if the model was created with a LoRA configuration.
        Creates adapter_model.safetensors and adapter_config.json.
        
        Parameters:
        - path: Output directory path.
        - base_model_path: Optional path/name of base model for adapter_config.json.
        """
    def export_model(self, path: str) -> None:
        """
        Export model weights and config to a directory.
        
        Parameters:
        - path: Output directory path.
        """
    def get_allocator_info(self, gpu_id: int = 0) -> dict:
        """
        Get current memory allocator statistics.
        
        Parameters:
        - gpu_id: Which GPU to query.
        
        Returns: dict[str, dict] with per-segment counters; stack entries include {'stack': bytes}.
        """
    def get_gpu_info(self) -> list[GPUUtilInfo]:
        """
        Return current GPU utilization info for all GPUs (implementation-defined structure).
        """
    def get_gradients(self, gpu_id: int) -> dict:
        """
        Return gradient shards for debugging.
        
        Parameters:
        - gpu_id: Which GPU's shard to return.
        
        Returns: dict[str, ndarray] mapping parameter name -> gradient view.
        Note: blocking; intended for debugging only.
        """
    def get_lora_gradients(self, gpu_id: int) -> dict:
        """
        Return LoRA adapter gradients for debugging.
        
        Only works if the trainer was constructed with a LoRA configuration.
        
        Parameters:
        - gpu_id: Which GPU's gradients to return.
        
        Returns: dict[str, ndarray] mapping adapter parameter name -> gradient view.
        Note: blocking; intended for debugging only.
        """
    def import_weights(self, path: str) -> None:
        """
        Import weights from a HuggingFace model file.
        
        Parameters:
        - path: Path to model.safetensors or model.safetensors.index.json.
        """
    def init_weights(self) -> None:
        """
        Initialize weights from scratch (random init).
        """
    def load_checkpoint(self, path: str, step: int) -> None:
        """
        Load a checkpoint.
        
        Parameters:
        - path: Checkpoint directory.
        - step: Step number to load.
        """
    def save_checkpoint(self, path: str, step: int) -> None:
        """
        Save a checkpoint.
        
        Parameters:
        - path: Checkpoint directory.
        - step: Step number to save.
        """
    def step(self, inputs: npt.NDArray[np.int32], targets: npt.NDArray[np.int32]) -> None:
        """
        Perform one training step (forward + backward).
        
        This call is asynchronous; the loss becomes available on the next `update()`.
        
        Parameters:
        - inputs: int32 token ids shaped [batch_size * world_size, seq_length].
        - targets: int32 token ids shaped [batch_size * world_size, seq_length].
        """
    def update_with_config(self, config: OptimizerConfig, step: int) -> dict:
        """
        Run the optimizer step with full configuration and return metrics.

        This call blocks until the optimizer step is complete.
        Supports both AdamW 8-bit and NorMuon optimizers based on config.type.

        Parameters:
        - config: OptimizerConfig with all hyperparameters.
        - step: Global step index.

        Returns: dict with keys {loss: float, norm: float}.
        """
    def validate(self, inputs: npt.NDArray[np.int32], targets: npt.NDArray[np.int32]) -> float:
        """
        Compute validation loss for one batch (forward only).
        
        Parameters:
        - inputs: int32 token ids shaped [batch_size * world_size, seq_length].
        - targets: int32 token ids shaped [batch_size * world_size, seq_length].
        
        Returns: loss (float).
        """
    @property
    def batch_size(self) -> int:
        """
        Per-GPU batch size configured for this trainer.
        """
    @property
    def seq_length(self) -> int:
        """
        Sequence length configured for this trainer.
        """
    @property
    def world_size(self) -> int:
        """
        Number of participating GPUs.
        """
class SystemInfo:
    """
    System information utility class.

    Provides methods to query CUDA, NCCL, cuDNN versions and GPU information.
    """
    @staticmethod
    def get_cuda_driver_version() -> int:
        """
        Get CUDA driver version.

        Returns: Integer version (e.g., 12010 for CUDA 12.1).
        """
    @staticmethod
    def get_cuda_runtime_version() -> int:
        """
        Get CUDA runtime version.

        Returns: Integer version (e.g., 12010 for CUDA 12.1).
        """
    @staticmethod
    def get_nccl_version() -> int:
        """
        Get NCCL version.

        Returns: Integer version.
        """
    @staticmethod
    def get_cudnn_version() -> int:
        """
        Get cuDNN version.

        Returns: Integer version.
        """
    @staticmethod
    def get_gpu_info() -> list[GPUInfo]:
        """
        Get information about all available GPUs.

        Returns: List of GPUInfo objects, one per device.
        """
class TrainingRunLogger:
    """
    Training run logger.
    
    Writes a structured log to `file_name` and optionally forwards messages to a Python callback.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, file_name: str, callback: object | None = None, verbosity: LogVerbosity = LogVerbosity.DEFAULT) -> None:
        """
        Create a logger.

        Parameters:
        - file_name: Output log file path.
        - callback: Optional Python callable that receives log strings.
        - verbosity: LogVerbosity level.
        """
    def log_allocator(self, stats: dict) -> None:
        """
        Log memory allocator statistics.
        
        Parameters:
        - stats: dict[str, dict]. For segments: keys {device, managed, pinned, pageable}. For stacks: key {'stack': bytes}.
        """
    def log_cmd(self, args: collections.abc.Sequence[str]) -> None:
        """
        Log command line arguments.
        
        Parameters:
        - args: List of argv-style strings.
        """
    def log_dataset(self, train_loader: DataLoader, eval_loader: DataLoader) -> None:
        """
        Log dataset information.
        
        Parameters:
        - train_loader: Training DataLoader.
        - eval_loader: Evaluation DataLoader.
        """
    def log_eval(self, step: int, epoch: float, eval_tokens: int, duration_ms: int, loss: float) -> None:
        """
        Log an evaluation step.
        
        Parameters:
        - step: Global step index.
        - epoch: Current epoch.
        - eval_tokens: Tokens processed.
        - duration_ms: Eval wall time.
        - loss: Eval loss.
        """
    def log_gpu_state(self, step: int, gpu_id: int, gpu_util: GPUUtilInfo) -> None:
        """
        Log GPU utilization state.
        
        Parameters:
        - step: Global step.
        - gpu_id: GPU index.
        - gpu_util: GPUUtilInfo snapshot.
        """
    def log_options(self, options: dict) -> None:
        """
        Log training options.
        
        Parameters:
        - options: dict[str, (bool|int|float|str)]. Unsupported value types raise.
        """
    def log_step(self, step: int, epoch: float, step_tokens: int, duration_ms: int, norm: float, loss: float, lr: float) -> None:
        """
        Log a training step.
        
        Parameters:
        - step: Global step index.
        - epoch: Current epoch.
        - step_tokens: Tokens processed in this step.
        - duration_ms: Step wall time.
        - norm: Gradient norm.
        - loss: Training loss.
        - lr: Learning rate.
        """
    def set_expected_time_per_token(self, trainer: SurogateTrainer) -> None:
        """
        Log a compute/throughput estimate based on the current model/trainer configuration.
        
        Parameters:
        - trainer: SurogateTrainer instance to read config/options/shape from.
        """
clean_old_checkpoints: nanobind.nb_func  # value = <nanobind.nb_func object>
find_latest_checkpoint: nanobind.nb_func  # value = <nanobind.nb_func object>
get_all_checkpoints: nanobind.nb_func  # value = <nanobind.nb_func object>
get_checkpoint_path: nanobind.nb_func  # value = <nanobind.nb_func object>
get_num_gpus: nanobind.nb_func  # value = <nanobind.nb_func object>
