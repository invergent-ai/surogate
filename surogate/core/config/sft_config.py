import inspect
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Literal, Union

import torch
from namer import generate as generate_unique_name
from transformers import TrainingArguments, SchedulerType
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


@dataclass
class SFTConfig(ModelConfig, RayConfig, ChatTemplateConfig):
    """
    SFTConfig class is a dataclass that holds configuration parameters for Supervised Fine-Tuning (SFT)

    Args:
        run_name (Optional[str], defaults to auto-generated):
            A descriptor for the run.
        apply_recommended_values (Optional[bool]):
            Whether to apply recommended configuration values. Default is True.
        num_epochs (Optional[int], default to 3):
            Total number of training epochs to perform (if not an integer, will perform the decimal part percents
            of the last epoch before stopping training).
        output_dir (Optional[str], defaults to './output'):
            The output directory where the model predictions and checkpoints will be written.
        resume_from_checkpoint (Optional[str], defaults to None):
            Path to a checkpoint to resume training from. Loads model weights, optimizer state, random seed,
            and resumes training from the last step.
        seed (Optional[int], defaults to 1234):
            Random seed for reproducibility.
        datasets (Optional[List[DatasetConfig]]):
            List of datasets for training. Default is None.
        validation_datasets (Optional[List[DatasetConfig]]):
            List of datasets for validation during training. Default is None.
        validation_split_ratio (Optional[float]):
            Ratio of training data to use for validation if no validation_datasets are provided. Default is 0.1.
        stream_datasets (Optional[bool], defaults to False):
            Whether to stream datasets during training to save memory.
        dataloader_num_workers (`int`, *optional*, defaults to 0):
            Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the
            main process.
        sample_packing (Optional[bool], defaults to True):
            Whether to enable sample packing to fit more data samples into a single sequence. Requires Flash Attantion.
            Packing reduces the number of samples in the dataset; please adjust the gradient accumulation steps and
            learning rate accordingly.
        sequence_len (Optional[int], defaults to None):
            Maximum token length after tokenizer.encode for a single data sample (to prevent OOM during training).
            Samples exceeding this limit are truncated to this length.
            Default is None, meaning it’s set to the model’s maximum supported sequence length..
        gradient_checkpointing (Optional[bool], defaults to True):
            Whether to enable gradient checkpointing to save memory during training. This significantly reduces
            GPU memory usage but slows down training.
        learning_rate (Optional[float], defaults to 1e-4):
            The initial learning rate for [`AdamW`] optimizer.
        lr_scheduler_type (`str` or [`SchedulerType`], *optional*, defaults to `"linear"`):
            The scheduler type to use. See the documentation of [`SchedulerType`] for all possible values.
        save_steps (`int` or `float`, *optional*, defaults to 50):
            Number of steps between saving checkpoints..
        save_total_limit (`int`, *optional*, defaults to 5):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            `output_dir`. When `load_best_model_at_end` is enabled, the "best" checkpoint according to
            `metric_for_best_model` will always be retained in addition to the most recent ones. For example, for
            `save_total_limit=5` and `load_best_model_at_end`, the four last checkpoints will always be retained
            alongside the best model. When `save_total_limit=1` and `load_best_model_at_end`, it is possible that two
            checkpoints are saved: the last one and the best one (if they are different).
        eval_steps (`int` or `float`, *optional*):
            Number of update steps between two evaluations if `eval_strategy="steps"`.
            Will default to the same value as `logging_steps` if not set.  Should be an integer or a float
            in range `[0,1)`. If smaller than 1, will be interpreted as ratio of total training steps.
        report_to (`str` or `list[str]`, *optional*, defaults to `"all"`):
            The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,
            `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"neptune"`,
            `"swanlab"`, `"tensorboard"`, `"trackio"`, `"aim"` and `"wandb"`. Use `"all"` to report to all integrations
            installed, `"none"` for no integrations.
        max_steps (`int`, *optional*, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`.
            For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until
            `max_steps` is reached. If set to -1, training continues for the specified number of epochs.
        warmup_ratio (`float`, *optional*, defaults to 0.0):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        warmup_steps (`int`, *optional*, defaults to 0):
            Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.
        weight_decay (`float`, *optional*, defaults to 0.1):
            The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in [`AdamW`]
            optimizer.
        max_grad_norm (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        per_device_eval_batch_size (`int`, *optional*, defaults to 1):
            Batch size per device during training. Default is 1.
        gradient_accumulation_steps (`int`, *optional*, defaults to None):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
            Warning: When using gradient accumulation, one step is counted as one step with backward pass.
            Default is None, meaning it's automatically calculated so that total_batch_size >= 16
        optim (`str` or [`training_args.OptimizerNames`], *optional*, defaults to `"adamw_8bit"`):
            The optimizer to use, such as "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", "adamw_anyprecision",
            "adafactor". See `OptimizerNames` for a full list of optimizers.
        deepspeed (Optional[str], defaults to None):
            DeepSpeed configuration: 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'.
        sequence_parallel_size (`int`, *optional*, defaults to 1):
            Size for sequence parallelism.
        lora_rank (Optional[int], defaults to 8):
            Rank for LoRA adapters.
        lora_alpha (Optional[int], defaults to 32):
            Alpha value for LoRA adapters.
        lora_dropout (Optional[float], defaults to 0.05):
            Dropout rate for LoRA adapters.
        lora_target_modules (Optional[List[str]], default to ['all-linear']):
            List of module names to apply LoRA adapters to.
        qlora: (Optional[bool], defaults to True):
            Whether to use QLoRA for training.
        merge_adapter: (Optional[bool], defaults to False):
            Whether to merge LoRA adapters into the base model after training.
        use_ray: (Optional[bool], defaults to False):
            Whether to use Ray for distributed training.
    """
    run_name: Optional[str] = None
    apply_recommended_values: Optional[bool] = True
    num_epochs: Optional[int] = None
    output_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    resume_only_model: bool = False
    seed: Optional[int] = None
    datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_split_ratio: Optional[float] = None
    stream_datasets: Optional[bool] = None
    dataloader_num_workers: Optional[int] = None
    sample_packing: Optional[bool] = None
    sequence_len: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    learning_rate: Optional[float] = None
    lr_scheduler_type: Optional[str] = None
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    eval_steps: Optional[int] = None
    report_to: Optional[List[Literal['wandb', 'aim']]] = None
    max_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    warmup_steps: Optional[int] = None
    weight_decay: Optional[float] = None
    max_grad_norm: Optional[float] = None
    per_device_eval_batch_size: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    optim: Optional[str] = None
    deepspeed_level: Optional[str] = None
    deepspeed: Optional[dict] = None
    sequence_parallel_size: int = 1

    qlora: Optional[bool] = False
    merge_adapter: Optional[bool] = False
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None
    lora_bias: Literal['none', 'all'] = 'none'
    lora_fast: Optional[bool] = False

    modules_to_save: List[str] = field(default_factory=list)

    use_chat_template: Optional[bool] = None

    # internal fields
    accelerator_config: Optional[Union[dict, str]] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.run_name = cfg['run_name'] or self.generate_run_name()
        self.apply_recommended_values = cfg.get('apply_recommended_values', False)
        self.num_epochs = cfg['num_epochs'] or 3
        self.output_dir = cfg['output_dir'] or './output'
        self.resume_from_checkpoint = cfg['resume_from_checkpoint']
        self.resume_only_model = cfg.get('resume_only_model', False)
        self.seed = cfg['seed'] or RAND_SEED
        self.datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('datasets', [])]
        self.validation_datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('validation_datasets', [])]
        self.validation_split_ratio = cfg['validation_split_ratio'] or 0.1
        self.stream_datasets = cfg.get('stream_datasets', False)
        self.dataloader_num_workers = cfg.get('dataloader_num_workers', get_default_process_count())
        self.sample_packing = cfg.get('sample_packing', True)
        self.sequence_len = cfg['sequence_len']
        self.gradient_checkpointing = cfg.get('gradient_checkpointing', False)
        self.learning_rate = float(cfg.get('learning_rate', 1e-4))
        self.lr_scheduler_type = cfg['lr_scheduler_type'] or SchedulerType.LINEAR
        self.save_steps = cfg.get('save_steps', 50)
        self.save_total_limit = cfg.get('save_total_limit', 5)
        self.eval_steps = cfg.get('eval_steps', 100)
        self.report_to = cfg['report_to'] or []
        self.max_steps = cfg['max_steps'] or -1
        self.warmup_ratio = cfg['warmup_ratio'] or 0.05
        self.warmup_steps = cfg['warmup_steps'] or 0
        self.weight_decay = cfg['weight_decay'] or 0.01
        self.max_grad_norm = cfg['gradient_clip_norm'] or 1.0
        self.per_device_eval_batch_size = cfg['per_device_eval_batch_size'] or 1
        self.per_device_train_batch_size = cfg['per_device_train_batch_size'] or 1
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps'] or 1
        self.optim = cfg['optim'] or OptimizerNames.ADAMW_8BIT
        self.deepspeed_level = cfg['deepspeed_level']
        self.sequence_parallel_size = cfg.get('sequence_parallel_size', 1)
        self.qlora = cfg.get('qlora', False)
        self.merge_adapter = cfg.get('merge_adapter', False)
        self.lora_rank = cfg['lora_rank'] or 8
        self.lora_alpha = cfg['lora_alpha'] or 32
        self.lora_dropout = cfg['lora_dropout'] or 0
        self.lora_bias = cfg['lora_bias'] or 'none'
        self.lora_target_modules = cfg.get('lora_target_modules', ['all-linear'])
        self.lora_fast = cfg.get('lora_fast', False)
        self.modules_to_save = cfg.get('modules_to_save', [])

    def __post_init__(self):
        _save_path = Path(self.output_dir)
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

        if self.qlora:
            self.quant_method = 'bnb_4bit'

        ModelConfig.__post_init__(self)
        ChatTemplateConfig.__post_init__(self)

        self._init_mixed_precision()

        self.prompt_template = self.model_template.chat_template
        if self.use_chat_template is None:
            self.use_chat_template = True

        if len(self.validation_datasets) > 0 or self.stream_datasets and self.validation_split_ratio > 0:
            # Don't split training data if validation datasets are provided or dataset streaming is enabled
            self.validation_split_ratio = 0.0

        if self.max_model_len is None:
            self.max_model_len = self.model_info.max_model_len

        if self.sample_packing and self.sequence_len is None:
            logger.info(
                f"Sample packing is enabled but sequence_len is not set. Setting sequence_len to model's max_model_len {self.max_model_len}.")
            self.sequence_len = self.max_model_len

        self.metric_for_best_model = 'loss'

        if self.learning_rate is None:
            logger.info(f"Learning rate is not set. Setting learning rate to {self.learning_rate}.")
            self.learning_rate = 1e-4

        self.eval_strategy = 'steps' if self.eval_steps > 0 else 'no'

        self._init_deepspeed()
        self._init_device()
        self.accelerator_config = {'dispatch_batches': False}

        if self.learning_rate < 1e-7:
            logger.warning(
                f"Your learning rate {self.learning_rate} is set to a very low value. Consider increasing it to avoid vanishing gradients!")
        elif self.learning_rate > 1:
            logger.warning(
                f"Your learning rate {self.learning_rate} is set to a very high value. Consider decreasing it to avoid exploding gradients!")

        self.trainer_args = self.to_trainer_args()

        if 'aim' in self.report_to:
            from transformers.integrations import INTEGRATION_TO_CALLBACK
            INTEGRATION_TO_CALLBACK['aim'] = AimCallback(
                experiment=self.output_dir
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
        args_dict['use_liger_kernel'] = False
        args_dict['dataloader_pin_memory'] = True
        args_dict['include_tokens_per_second'] = True
        args_dict['eval_accumulation_steps'] = 2
        args_dict['torch_empty_cache_steps'] = 250
        args_dict['disable_tqdm'] = True
        args_dict['remove_unused_columns'] = False
        args_dict['logging_dir'] = None

        return TrainingArguments(**args_dict)

    def _init_mixed_precision(self):
        if self.model_info.quant_method == 'fp8':
            self.fp16, self.bf16 = False, False
        elif self.torch_dtype in {torch.float16, torch.float32}:
            self.fp16, self.bf16 = True, False
        elif self.torch_dtype == torch.bfloat16:
            self.fp16, self.bf16 = False, True
        else:
            raise ValueError(f'args.torch_dtype: {self.torch_dtype}')

    def _init_deepspeed(self):
        if self.deepspeed_level:
            if self.qlora and self.deepspeed_level == 'zero3':
                raise ValueError('DeepSpeed ZeRO-3 is not compatible with QLoRA.')

            if is_mp() and not self.use_ray:
                raise ValueError('DeepSpeed is not compatible with `device_map`. '
                                 f'n_gpu: {get_device_count()}, '
                                 f'local_world_size: {self.local_world_size}.')

            self.deepspeed = DEEPSPEED_CONFIGS[self.deepspeed_level]
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
