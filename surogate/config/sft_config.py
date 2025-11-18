from dataclasses import dataclass
from typing import Optional, List, Literal

from surogate.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.config.model_config import ModelConfig
from surogate.utils.dict import DictDefault
from surogate.utils.seed import RAND_SEED


@dataclass
class SFTConfig(ModelConfig):
    """
    SFTConfig class is a dataclass that holds configuration parameters for Supervised Fine-Tuning (SFT)

    Args:
        num_train_epochs (Optional[int]): Number of training epochs. Defaults to 3.
        save_path (Optional[str]): Directory to save the output. Defaults to './output'.
        resume_from_checkpoint (Optional[str]): Path to a checkpoint to resume training from. Loads model weights, optimizer state, random seed, and resumes training from the last step. Defaults to None.
        seed (Optional[int]): Random seed for reproducibility. Default is 1234.
        datasets (Optional[List[DatasetConfig]]): List of datasets for training. Default is None.
        validation_datasets (Optional[List[DatasetConfig]]): List of datasets for validation during training. Default is None.
        validation_split_ratio (Optional[float]): Ratio of training data to use for validation if no validation_datasets are provided. Default is 0.0.
        sequence_len (Optional[int]): Maximum token length after tokenizer.encode for a single data sample (to prevent OOM during training). Samples exceeding this limit are truncated to this length. Default is None, meaning it’s set to the model’s maximum supported sequence length..
        gradient_checkpointing (Optional[bool]): Whether to enable gradient checkpointing to save memory during training. This significantly reduces GPU memory usage but slows down training. Default is True.
        learning_rate (Optional[float]): Learning rate for training. Default is 1e-4.
        checkpoint_steps (Optional[int]): Number of steps between saving checkpoints. Default is 50.
        max_checkpoints_to_keep (Optional[int]): Maximum number of checkpoints to keep. Older checkpoints are deleted. Default is 5.
        report_to (Optional[Literal['wandb']]): Report training metrics to the provided platform. Default is None.
        max_steps (Optional[int]): Total number of training steps to perform. If set to -1, training continues for the specified number of epochs. Default is -1.
        warmup_ratio (Optional[float]): The ratio of total training steps used for learning rate warmup. Default is 0.0.
        weight_decay (Optional[float]): Weight decay (L2 regularization) coefficient. Default is 0.1.
        gradient_clip_norm (Optional[float]): Maximum norm for gradient clipping. Default is 1.0.
        per_device_train_batch_size (Optional[int]): Batch size per device during training. Default is 1.
        gradient_accumulation_steps (Optional[int]): Number of gradient accumulation steps. Default is None, meaning it's automatically calculated so that total_batch_size >= 16
        deepspeed (Optional[str]): DeepSpeed configuration: 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'. Default is None.

        lora_rank (Optional[int]): Rank for LoRA adapters. Default is 8.
        lora_alpha (Optional[int]): Alpha value for LoRA adapters. Default is 32.
        lora_dropout (Optional[float]): Dropout rate for LoRA adapters. Default is 0.05.
        lora_target_modules (Optional[List[str]]): List of module names to apply LoRA adapters to. Default is  all linear modules.
    """
    num_train_epochs: Optional[int] = None
    save_path: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    seed: Optional[int] = None
    datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_split_ratio: Optional[float] = None
    sequence_len: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    learning_rate: Optional[float] = None
    checkpoint_steps: Optional[int] = None
    max_checkpoints_to_keep: Optional[int] = None
    report_to: Optional[Literal['wandb']] = None
    max_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    weight_decay: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    deepspeed: Optional[str] = None

    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None

    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.num_train_epochs = cfg['num_train_epochs'] or 3
        self.save_path = cfg['save_path'] or './output'
        self.resume_from_checkpoint = cfg['resume_from_checkpoint']
        self.seed = cfg['seed'] or RAND_SEED
        self.datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('datasets', [])]
        self.validation_datasets = [create_dataset_config(ds_cfg) for ds_cfg in cfg.get('validation_datasets', [])]
        self.validation_split_ratio = cfg['validation_split_ratio'] or 0.0
        self.sequence_len = cfg['sequence_len']
        self.gradient_checkpointing = cfg.get('gradient_checkpointing', True)
        self.learning_rate = cfg.get('learning_rate', 1e-4)
        self.checkpoint_steps = cfg.get('checkpoint_steps', 50)
        self.max_checkpoints_to_keep = cfg.get('max_checkpoints_to_keep', 5)
        self.report_to = cfg['report_to']
        self.max_steps = cfg['max_steps'] or -1
        self.warmup_ratio = cfg['warmup_ratio'] or 0.0
        self.weight_decay = cfg['weight_decay'] or 0.1
        self.gradient_clip_norm = cfg['gradient_clip_norm'] or 1.0
        self.per_device_train_batch_size = cfg['per_device_train_batch_size'] or 1
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps']
        self.deepspeed = cfg['deepspeed']
        self.lora_rank = cfg['lora_rank'] or 8
        self.lora_alpha = cfg['lora_alpha'] or 32
        self.lora_dropout = cfg['lora_dropout'] or 0.05
        self.lora_target_modules = cfg.get('lora_target_modules', ['all-linear'])

        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()

        if not self.datasets:
            raise ValueError(f'At least one dataset must be provided for SFT.')
