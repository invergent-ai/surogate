from dataclasses import dataclass
from typing import Optional, List, Literal

from swift.llm import get_model_info_meta
from swift.llm.model.constant import MLLMModelType, LLMModelType
from swift.utils import is_mp
from transformers import Seq2SeqTrainingArguments, IntervalStrategy, SchedulerType
from transformers.training_args import OptimizerNames

from surogate.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.config.model_config import ModelConfig
from surogate.config.ray_config import RayConfig
from surogate.datasets.datasets import get_default_process_count
from surogate.utils.aim import AimCallback
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.seed import RAND_SEED

logger = get_logger()

def is_flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class SFTConfig(ModelConfig, RayConfig):
    """
    SFTConfig class is a dataclass that holds configuration parameters for Supervised Fine-Tuning (SFT)

    Args:
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

        lora_rank (Optional[int]): Rank for LoRA adapters. Default is 8.
        lora_alpha (Optional[int]): Alpha value for LoRA adapters. Default is 32.
        lora_dropout (Optional[float]): Dropout rate for LoRA adapters. Default is 0.05.
        lora_target_modules (Optional[List[str]]): List of module names to apply LoRA adapters to. Default is  all linear modules.

        usr_ray: Optional[bool]: Whether to use Ray for distributed training. Default is False.
    """
    num_train_epochs: Optional[int] = None
    save_path: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    seed: Optional[int] = None
    datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_datasets: Optional[List[SurogateDatasetConfig]] = None
    validation_split_ratio: Optional[float] = None
    stream_datasets: Optional[bool] = None
    sequence_len: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    learning_rate: Optional[float] = None
    checkpoint_steps: Optional[int] = None
    max_checkpoints_to_keep: Optional[int] = None
    report_to: Optional[List[Literal['wandb', 'aim']]] = None
    max_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    weight_decay: Optional[float] = None
    gradient_clip_norm: Optional[float] = None
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    deepspeed: Optional[str] = None
    sample_packing: Optional[bool] = None

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
        self.validation_split_ratio = cfg['validation_split_ratio'] or 0.1
        self.stream_datasets = cfg.get('stream_datasets', False)
        self.sequence_len = cfg['sequence_len']
        self.gradient_checkpointing = cfg.get('gradient_checkpointing', True)
        self.learning_rate = cfg.get('learning_rate', 1e-4)
        self.checkpoint_steps = cfg.get('checkpoint_steps', 50)
        self.max_checkpoints_to_keep = cfg.get('max_checkpoints_to_keep', 5)
        self.report_to = cfg['report_to'] or []
        self.max_steps = cfg['max_steps'] or -1
        self.warmup_ratio = cfg['warmup_ratio'] or 0.05
        self.weight_decay = cfg['weight_decay'] or 0.1
        self.gradient_clip_norm = cfg['gradient_clip_norm'] or 1.0
        self.per_device_train_batch_size = cfg['per_device_train_batch_size'] or 1
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps']
        self.deepspeed = cfg['deepspeed']
        self.sample_packing = cfg.get('sample_packing', True)
        self.lora_rank = cfg['lora_rank'] or 8
        self.lora_alpha = cfg['lora_alpha'] or 32
        self.lora_dropout = cfg['lora_dropout'] or 0.05
        self.lora_target_modules = cfg.get('lora_target_modules', ['all-linear'])
        self.__post_init__()

    def __post_init__(self):
        if 'aim' in self.report_to:
            from transformers.integrations import INTEGRATION_TO_CALLBACK
            INTEGRATION_TO_CALLBACK['aim'] = AimCallback(
                experiment=self.save_path
            )

    def to_hf_training_args(self) -> Seq2SeqTrainingArguments:
        eval_strategy = IntervalStrategy.STEPS if self.validation_datasets or self.validation_split_ratio > 0.0 else IntervalStrategy.NO
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.save_path,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            use_liger_kernel=self.should_apply_liger_kernel(),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_grad_norm=self.gradient_clip_norm,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            lr_scheduler_type=SchedulerType.COSINE,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=0,
            logging_steps=1,
            save_strategy=IntervalStrategy.STEPS,
            remove_unused_columns=False,
            save_steps=self.checkpoint_steps,
            save_total_limit=self.max_checkpoints_to_keep,
            seed=self.seed,
            bf16=True,
            fp16=False,
            metric_for_best_model='loss',
            greater_is_better=False,
            eval_strategy=eval_strategy,
            eval_steps=self.checkpoint_steps,
            accelerator_config={'dispatch_batches': False},
            dataloader_num_workers=get_default_process_count(),
            optim=OptimizerNames.ADAMW_TORCH_FUSED,
            report_to=self.report_to,
            push_to_hub=False,
            run_name=self.save_path,
            logging_dir=self.logging_dir,
            data_seed=self.seed,
            ddp_timeout=18000000,
            deepspeed=self.deepspeed,
        )
        
        training_args.packing = self.sample_packing

        return training_args

    def should_apply_liger_kernel(self) -> bool:
        if is_mp():
            # liger_kernel does not support device_map. Use DDP/DeepSpeed for multi-GPU training.
            return False

        model_info, model_meta = get_model_info_meta(self.model, use_hf=True)
        return model_info.model_type in [
            MLLMModelType.llama4, LLMModelType.llama3, LLMModelType.llama3_1, LLMModelType.llama3_2,
            MLLMModelType.llama3_2_vision,

            LLMModelType.mistral, LLMModelType.mixtral,

            LLMModelType.gemma, LLMModelType.gemma2, LLMModelType.gemma3_text, MLLMModelType.gemma3_vision,

            MLLMModelType.paligemma,

            LLMModelType.qwen2, LLMModelType.qwen2_5, LLMModelType.qwq, MLLMModelType.qwen2_vl,
            MLLMModelType.qwen2_5_vl,
            LLMModelType.qwen3, LLMModelType.qwen3_moe,

            LLMModelType.phi3,

            LLMModelType.glm4,

            MLLMModelType.internvl3
        ]
