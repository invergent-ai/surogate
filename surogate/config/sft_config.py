from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Literal

from surogate.datasets.progress import create_hfhub_tqdm
from swift.llm import get_model_info_meta
from swift.llm.model.constant import MLLMModelType, LLMModelType
from namer import generate as generate_unique_name
from surogate.config.dataset_config import SurogateDatasetConfig, create_dataset_config
from surogate.config.model_config import ModelConfig
from surogate.config.ray_config import RayConfig
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

    qlora: Optional[bool] = False
    merge_adapter: Optional[bool] = False
    lora_rank: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_dropout: Optional[float] = None
    lora_target_modules: Optional[List[str]] = None


    def __init__(self, cfg: DictDefault):
        super().__init__(cfg)
        self.apply_recommended_values = cfg.get('apply_recommended_values', True)
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
        self.learning_rate = float(cfg.get('learning_rate', 1e-4))
        self.checkpoint_steps = cfg.get('checkpoint_steps', 50)
        self.max_checkpoints_to_keep = cfg.get('max_checkpoints_to_keep', 5)
        self.eval_steps = cfg.get('eval_steps', 100)
        self.report_to = cfg['report_to'] or []
        self.max_steps = cfg['max_steps'] or -1
        self.warmup_ratio = cfg['warmup_ratio'] or 0.05
        self.weight_decay = cfg['weight_decay'] or 0.1
        self.gradient_clip_norm = cfg['gradient_clip_norm'] or 1.0
        self.per_device_train_batch_size = cfg['per_device_train_batch_size'] or 1
        self.gradient_accumulation_steps = cfg['gradient_accumulation_steps']
        self.deepspeed = cfg['deepspeed']
        self.sample_packing = cfg.get('sample_packing', True)
        self.qlora = cfg.get('qlora', False)
        self.merge_adapter = cfg.get('merge_adapter', False)
        self.lora_rank = cfg['lora_rank'] or 8
        self.lora_alpha = cfg['lora_alpha'] or 32
        self.lora_dropout = cfg['lora_dropout'] or 0.05
        self.lora_target_modules = cfg.get('lora_target_modules', ['all-linear'])
        self.run_name = cfg['run_name'] or self.generate_run_name()
        self.__post_init__()

    def __post_init__(self):
        _save_path = Path(self.save_path)
        if _save_path.exists():
            if not _save_path.is_dir():
                raise ValueError(f"Save path {_save_path} already exists and is not a directory. Aborting.")
            if any(_save_path.iterdir()):
                raise ValueError(f"Save path {_save_path} is not empty. Aborting.")
        else:
            _save_path.mkdir(parents=True, exist_ok=True)

        if 'aim' in self.report_to:
            from transformers.integrations import INTEGRATION_TO_CALLBACK
            INTEGRATION_TO_CALLBACK['aim'] = AimCallback(
                experiment=self.save_path
            )

    def should_apply_liger_kernel(self) -> bool:
        model_info, model_meta = get_model_info_meta(self.model, use_hf=True, tqdm_class=create_hfhub_tqdm('Downloading model - '))
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

    def generate_run_name(self):
        return generate_unique_name(category='science')

