import inspect
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Literal

import torch
from namer import generate as generate_unique_name
from transformers import TrainingArguments
from transformers.training_args import OptimizerNames

from surogate.core.config.ct_config import ChatTemplateConfig
from surogate.core.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.core.config.model_config import ModelConfig
from surogate.core.config.ray_config import RayConfig
from surogate.core.datasets.datasets import get_default_process_count
from surogate.train.deepspeed import DEEPSPEED_CONFIGS
from surogate.utils.aim import AimCallback
from surogate.utils.dict import DictDefault
from surogate.utils.dist import get_dist_setting, get_device_count, is_mp, is_dist, set_device, is_master
from surogate.utils.fs import to_abspath
from surogate.utils.logger import get_logger
from surogate.utils.model import estimate_model_parameters
from surogate.utils.seed import RAND_SEED

logger = get_logger()

def is_flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class SFTConfig(ModelConfig, RayConfig, ChatTemplateConfig):
    """
    SFTConfig class is a dataclass that holds configuration parameters for Supervised Fine-Tuning (SFT)

    Args:
        run_name (Optional[str]): Name of the training run. Defaults to None.
        apply_recommended_values (Optional[bool]): Whether to apply recommended configuration values. Default is True.
        num_train_epochs (Optional[int]): Number of training epochs. Defaults to 3.
        save_path (Optional[str]): Directory to save the output. Defaults to './output'.
        resume_from_checkpoint (Optional[str]): Path to a checkpoint to resume training from. Loads model weights, optimizer state, random seed, and resumes training from the last step. Defaults to None.
        seed (Optional[int]): Random seed for reproducibility. Default is 1234.
        datasets (Optional[List[DatasetConfig]]): List of datasets for training. Default is None.
        validation_datasets (Optional[List[DatasetConfig]]): List of datasets for validation during training. Default is None.
        validation_split_ratio (Optional[float]): Ratio of training data to use for validation if no validation_datasets are provided. Default is 0.1.
        stream_datasets (Optional[bool]): Whether to stream datasets during training to save memory. Default is False.
        sequence_len (Optional[int]): Maximum token length after tokenizer.encode for a single data sample (to prevent OOM during training). Samples exceeding this limit are truncated to this length. Default is None, meaning it’s set to the model’s maximum supported sequence length..
        gradient_checkpointing (Optional[bool]): Whether to enable gradient checkpointing to save memory during training. This significantly reduces GPU memory usage but slows down training. Default is True.
        learning_rate (Optional[float]): Learning rate for training. Default is 1e-4.
        checkpoint_steps (Optional[int]): Number of steps between saving checkpoints. Default is 50.
        max_checkpoints_to_keep (Optional[int]): Maximum number of checkpoints to keep. Older checkpoints are deleted. Default is 5.
        report_to (Optional[Literal['wandb']]): Report training metrics to the provided platform. Default is None.
        max_steps (Optional[int]): Total number of training steps to perform. If set to -1, training continues for the specified number of epochs. Default is -1.
        warmup_ratio (Optional[float]): The ratio of total training steps used for learning rate warmup. Default is 0.05.
        weight_decay (Optional[float]): Weight decay (L2 regularization) coefficient. Default is 0.1.
        gradient_clip_norm (Optional[float]): Maximum norm for gradient clipping. Default is 1.0.
        per_device_train_batch_size (Optional[int]): Batch size per device during training. Default is 1.
        gradient_accumulation_steps (Optional[int]): Number of gradient accumulation steps. Default is None, meaning it's automatically calculated so that total_batch_size >= 16
        deepspeed (Optional[str]): DeepSpeed configuration: 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'. Default is None.
        sample_packing (Optional[bool]): Whether to enable sample packing to fit more data samples into a single sequence. Requires Flash Attantion. Packing reduces the number of samples in the dataset; please adjust the gradient accumulation steps and learning rate accordingly. Default is True.
        sequence_parallel_size (int): Size for sequence parallelism. Default is 1.

        lora_rank (Optional[int]): Rank for LoRA adapters. Default is 8.
        lora_alpha (Optional[int]): Alpha value for LoRA adapters. Default is 32.
        lora_dropout (Optional[float]): Dropout rate for LoRA adapters. Default is 0.05.
        lora_target_modules (Optional[List[str]]): List of module names to apply LoRA adapters to. Default is  all linear modules.
        qlora: Optional[bool]: Whether to use QLoRA for training. Default is False.
        merge_adapter: Optional[bool]: Whether to merge LoRA adapters into the base model after training. Default is False.

        usr_ray: Optional[bool]: Whether to use Ray for distributed training. Default is False.
    """
    run_name: Optional[str] = None
    apply_recommended_values: Optional[bool] = True
    num_train_epochs: Optional[int] = None
    save_path: Optional[str] = None

    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False

    seed: Optional[int] = None

    datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_split_ratio: Optional[float] = None
    stream_datasets: Optional[bool] = None
    dataloader_num_workers: Optional[int] = None

    sequence_len: Optional[int] = None

    gradient_checkpointing: Optional[bool] = None

    learning_rate: Optional[float] = None
    checkpoint_steps: Optional[int] = None
    max_checkpoints_to_keep: Optional[int] = None
    eval_steps: Optional[int] = None
    report_to: Optional[List[Literal['wandb', 'aim']]] = None
    max_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    weight_decay: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    deepspeed: Optional[str] = None
    sample_packing: Optional[bool] = None
    sequence_parallel_size: int = 1

    qlora: Optional[bool] = False
    qlora_fast: Optional[bool] = False
    merge_adapter: Optional[bool] = False
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None
    lora_bias: Literal['none', 'all'] = 'none'
    lora_fast: Optional[bool] = False

    modules_to_save: List[str] = field(default_factory=list)

    use_chat_template: Optional[bool] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.apply_recommended_values = cfg.get('apply_recommended_values', True)
        self.num_train_epochs = cfg['num_train_epochs'] or 3
        self.save_path = cfg['save_path'] or './output'
        self.resume_from_checkpoint = cfg['resume_from_checkpoint']
        self.resume_only_model = cfg.get('resume_only_model', False)
        self.seed = cfg['seed'] or RAND_SEED
        self.datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('datasets', [])]
        self.validation_datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('validation_datasets', [])]
        self.validation_split_ratio = cfg['validation_split_ratio'] or 0.1
        self.stream_datasets = cfg.get('stream_datasets', False)
        self.dataloader_num_workers = cfg.get('dataloader_num_workers', get_default_process_count())
        self.sequence_len = cfg['sequence_len']
        self.gradient_checkpointing = cfg.get('gradient_checkpointing', True)
        self.learning_rate = float(cfg.get('learning_rate', 1e-4))
        self.checkpoint_steps = cfg.get('checkpoint_steps', 50)
        self.max_checkpoints_to_keep = cfg.get('max_checkpoints_to_keep', 5)
        self.eval_steps = cfg.get('eval_steps', 100)
        self.report_to = cfg['report_to'] or []
        self.max_steps = cfg['max_steps'] or -1
        self.warmup_ratio = cfg['warmup_ratio'] or 0.05
        self.weight_decay = cfg['weight_decay'] or 0.01
        self.gradient_clip_norm = cfg['gradient_clip_norm'] or 1.0
        self.per_device_train_batch_size = cfg['per_device_train_batch_size'] or 1
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps'] or 1
        self.deepspeed = cfg['deepspeed']
        self.sample_packing = cfg.get('sample_packing', True)
        self.sequence_parallel_size = cfg.get('sequence_parallel_size', 1)
        self.qlora = cfg.get('qlora', False)
        self.qlora_fast = cfg.get('qlora_fast', False)
        self.merge_adapter = cfg.get('merge_adapter', False)
        self.lora_rank = cfg['lora_rank'] or 8
        self.lora_alpha = cfg['lora_alpha'] or 32
        self.lora_dropout = cfg['lora_dropout'] or 0
        self.lora_bias = cfg['lora_bias'] or 'none'
        self.lora_target_modules = cfg.get('lora_target_modules', ['all-linear'])
        self.lora_fast = cfg.get('lora_fast', False)
        self.run_name = cfg['run_name'] or self.generate_run_name()
        self.modules_to_save = cfg.get('modules_to_save', [])

    def __post_init__(self):
        _save_path = Path(self.save_path)
        if _save_path.exists():
            if not _save_path.is_dir():
                raise ValueError(f"Save path '{_save_path}' already exists and is not a directory. Aborting.")

            if any(item.is_dir() and item.name.startswith("checkpoint-") for item in _save_path.iterdir()):
                logger.warning_once(f"Save path '{_save_path}' contains previously saved checkpoints.")
        else:
            if is_master():
                _save_path.mkdir(parents=True, exist_ok=True)


        if self.resume_from_checkpoint:
            self.resume_from_checkpoint = to_abspath(self.resume_from_checkpoint, True)

        self.rank, self.local_rank, self.global_world_size, self.local_world_size = get_dist_setting()

        ModelConfig.__post_init__(self)
        ChatTemplateConfig.__post_init__(self)

        self._init_mixed_precision()

        if self.qlora:
            self.quant_method = 'bnb_4bit'
        if self.qlora_fast:
            self.quant_method = 'falqon'

        self.prompt_template = self.model_template.chat_template
        if self.use_chat_template is None:
            self.use_chat_template = True

        if len(self.validation_datasets) > 0 or self.stream_datasets and self.validation_split_ratio > 0:
            # Don't split training data if validation datasets are provided or dataset streaming is enabled
            self.validation_split_ratio = 0.0

        if self.max_model_len is None:
            self.max_model_len = self.model_info.max_model_len

        if self.sample_packing and self.sequence_len is None:
            logger.info(f"Sample packing is enabled but sequence_len is not set. Setting sequence_len to model's max_model_len {self.max_model_len}.")
            self.sequence_len = self.max_model_len

        self.metric_for_best_model = 'loss'

        if self.learning_rate is None:
            logger.info(f"Learning rate is not set. Setting learning rate to {self.learning_rate}.")
            self.learning_rate = 1e-4

        self.eval_strategy = 'steps' if self.eval_steps > 0 else 'no'

        self._init_deepspeed()
        self._init_device()
        self.accelerator_config = {'dispatch_batches': False}

        self.trainer_args = self.to_trainer_args()
        self.trainer_args.remove_unused_columns = False
        self.trainer_args.output_dir = self.save_path
        self.trainer_args.run_name = self.run_name
        self.trainer_args.logging_dir = None

        if 'aim' in self.report_to:
            from transformers.integrations import INTEGRATION_TO_CALLBACK
            INTEGRATION_TO_CALLBACK['aim'] = AimCallback(
                experiment=self.save_path
            )

    def to_trainer_args(self):
        args_dict = asdict(self)
        parameters = inspect.signature(TrainingArguments).parameters
        for k in list(args_dict.keys()):
            if k not in parameters:
                args_dict.pop(k)

        args_dict['logging_steps'] = 1
        args_dict['fp16'] = self.fp16
        args_dict['bf16'] = self.bf16
        args_dict['optim'] = OptimizerNames.ADAMW_8BIT
        args_dict['use_liger_kernel'] = self.sample_packing and False

        return TrainingArguments(**args_dict)

    def _init_mixed_precision(self):
        if self.torch_dtype in {torch.float16, torch.float32}:
            self.fp16, self.bf16 = True, False
        elif self.torch_dtype == torch.bfloat16:
            self.fp16, self.bf16 = False, True
        else:
            raise ValueError(f'args.torch_dtype: {self.torch_dtype}')

    def _init_deepspeed(self):
        if self.deepspeed:
            if is_mp() and not self.use_ray:
                raise ValueError('DeepSpeed is not compatible with `device_map`. '
                                 f'n_gpu: {get_device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')

            self.deepspeed_config = DEEPSPEED_CONFIGS[self.deepspeed]
            logger.info(f'Using deepspeed: {self.deepspeed}')


    def _init_device(self):
        if is_dist():
            set_device()

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
