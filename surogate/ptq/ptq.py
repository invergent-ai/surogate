import copy
import glob
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import init_empty_weights, infer_auto_device_map
from accelerate.utils import get_max_memory
from modelopt.torch.export import get_model_type, export_hf_checkpoint
from modelopt.torch.quantization import need_calibration
from modelopt.torch.utils import get_max_batch_size, get_dataset_dataloader, create_forward_loop
from modelopt.torch.utils.image_processor import MllamaImageProcessor
from modelopt.torch.utils.memory_monitor import launch_memory_monitor
from swift import get_logger
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, \
    WhisperProcessor
from surogate.utils.config import load_config
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from modelopt.torch.quantization.utils import is_quantized

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

logger = get_logger()

QUANT_CFG_CHOICES: dict[str, dict[str, Any]] = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
    "fp8_pb_wo": mtq.FP8_2D_BLOCKWISE_WEIGHT_ONLY_CFG,
    "fp8_pc_pt": mtq.FP8_PER_CHANNEL_PER_TOKEN_CFG,
    "w4a8_nvfp4_fp8": mtq.W4A8_NVFP4_FP8_CFG,
    "w4a8_mxfp4_fp8": mtq.W4A8_MXFP4_FP8_CFG,
    "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
}

KV_QUANT_CFG_CHOICES = {
    "none": "none",
    "fp8": "FP8_KV_CFG",
    "fp8_affine": "FP8_AFFINE_KV_CFG",
    "nvfp4": "NVFP4_KV_CFG",
    "nvfp4_affine": "NVFP4_AFFINE_KV_CFG",
    "nvfp4_rotate": "NVFP4_KV_ROTATE_CFG",
}

mto.enable_huggingface_checkpointing()

RAND_SEED = 1234


class SurogatePtq:
    def __init__(self, **kwargs):
        self.args = kwargs
        self.config = load_config(self.args['config'])

    def run(self):
        if not torch.cuda.is_available():
            raise OSError("GPU is required for PTQ.")

        random.seed(RAND_SEED)
        np.random.seed(RAND_SEED)

        # launch a memory monitor to read the currently used GPU memory.
        launch_memory_monitor()

        # Force eager execution for all model types.
        torch.compiler.set_stance("force_eager")

        assert (
                self.config['scheme']
                in [
                    "int8_wo",
                    "int4_awq",
                    "fp8",
                    "nvfp4",
                    "nvfp4_awq",
                    "w4a8_awq",
                    "fp8_pb_wo",
                    "w4a8_mxfp4_fp8",
                    "nvfp4_mlp_only",
                ]
                or self.config['kv_cache_scheme'] in KV_QUANT_CFG_CHOICES
        ), f"Quantization format {self.config['scheme']} not supported for HF export path"

        calibration_only = False
        model = get_model(self.config['model'])

        model_is_already_quantized = is_quantized(model)
        model_type = get_model_type(model)

        device = model.device
        if hasattr(model, "model"):
            device = model.model.device

        processor = None
        tokenizer = None

        full_model = model

        if self.config.dataset is None:
            self.config.dataset = ["cnn_dailymail", "nemotron-post-training-dataset-v2"]
            logger.warn(
                "No dataset specified. Defaulting to cnn_dailymail and nemotron-post-training-dataset-v2."
            )

        # Adjust calib_size to match dataset length by extending or truncating as needed
        self.config.calib_size = (self.config.calib_size + [self.config.calib_size[-1]] * len(self.config.dataset))[
            : len(self.config.dataset)
        ]
        tokenizer = get_tokenizer(self.config['checkpoint_path'], trust_remote_code=True)
        default_padding_side = tokenizer.padding_side
        # Left padding usually provides better calibration result.
        tokenizer.padding_side = "left"

        if self.config['scheme'] in QUANT_CFG_CHOICES:
            if "awq" in self.config['scheme']:
                print(
                    "\n####\nAWQ calibration could take longer than other calibration methods. "
                    "Consider reducing calib_size to reduce calibration time.\n####\n"
                )
            if self.config['batch_size'] == 0:
                # Calibration/sparsification will actually take much more memory than regular inference
                # due to intermediate tensors for fake quantization. Setting sample_memory_usage_ratio
                # to 2 to avoid OOM for AWQ/SmoothQuant fake quantization as it will take more memory than inference.
                sample_memory_usage_ratio = 2 if "awq" in self.config['scheme'] or "sq" in self.config[
                    'scheme'] else 1.1
                self.config['batch_size'] = get_max_batch_size(
                    model,
                    max_sample_length=self.config['calib_seq'],
                    sample_memory_usage_ratio=sample_memory_usage_ratio,
                    sample_input_single_batch=None,
                    enable_grad=False,
                )
                self.config['batch_size'] = min(self.config['batch_size'], sum(self.config['calib_size']))

            print(f"Use calib batch_size {self.config['batch_size']}")

            assert tokenizer is not None and isinstance(
                tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
            ), "The PreTrainedTokenizer must be set"

            calib_dataloader = get_dataset_dataloader(
                dataset_name=args.dataset,
                tokenizer=tokenizer,
                batch_size=self.config['batch_size'],
                num_samples=self.config['calib_size'],
                device="cuda",
                include_labels=False,
            )

            quant_cfg = build_quant_cfg(
                self.config['scheme'],
                self.config['kv_cache_scheme'],
                self.config['awq_block_size'],
                model_type,
                QUANT_CFG_CHOICES,
                KV_QUANT_CFG_CHOICES,
            )

            if not model_is_already_quantized or calibration_only:
                # Only run single sample for preview
                input_ids = next(iter(calib_dataloader))["input_ids"][0:1]
                generated_ids_before_ptq = full_model.generate(input_ids, max_new_tokens=100)

                if model_type == "gptoss" and self.config['scheme'] == "nvfp4_mlp_only":
                    print("Applying nvfp4 quantization (MoE only) for gpt-oss")

                # quantize the model
                model = self.quantize_model(model, quant_cfg, calib_dataloader, calibration_only)

                mtq.print_quant_summary(full_model)

                # Run some samples
                torch.cuda.empty_cache()

                generated_ids_after_ptq = None

                if model_type != "llama4":
                    # Our fake quantizer may not be fully compatible with torch.compile.
                    generated_ids_after_ptq = full_model.generate(input_ids, max_new_tokens=100)
                else:
                    logger.warn(
                        "Llama4 Maverick generation after quantization has a bug. Skipping generation sample."
                    )

                def input_decode(input_ids):
                    if processor is not None and isinstance(processor, MllamaImageProcessor):
                        return processor.tokenizer.batch_decode(input_ids)
                    elif processor is not None and isinstance(processor, WhisperProcessor):
                        return first_text
                    elif tokenizer is not None:
                        return tokenizer.batch_decode(input_ids)
                    else:
                        raise ValueError("The processor or tokenizer must be set")

                def output_decode(generated_ids, input_shape):
                    if is_enc_dec(model_type):
                        if processor is not None and isinstance(processor, WhisperProcessor):
                            return processor.tokenizer.batch_decode(
                                generated_ids, skip_special_tokens=True
                            )[0]
                        elif tokenizer is not None:
                            return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    elif processor is not None and isinstance(processor, MllamaImageProcessor):
                        return processor.tokenizer.batch_decode(generated_ids[:, input_shape:])
                    elif tokenizer is not None:
                        return tokenizer.batch_decode(generated_ids[:, input_shape:])
                    else:
                        raise ValueError("The processor or tokenizer must be set")

                if generated_ids_after_ptq is not None:
                    print("--------")
                    print(f"example test input: {input_decode(input_ids)}")
                    print("--------")
                    print(
                        f"example outputs before ptq: {output_decode(generated_ids_before_ptq, input_ids.shape[1])}"
                    )
                    print("--------")
                    print(
                        f"example outputs after ptq: {output_decode(generated_ids_after_ptq, input_ids.shape[1])}"
                    )
            else:
                logger.warn("Skipping quantization: model is already quantized.")

        else:
            assert model_type != "dbrx", f"Does not support export {model_type} without quantizaton"
            print(f"qformat: {self.config['scheme']}. No quantization applied, export {device} model")

        with torch.inference_mode():
            if model_type is None:
                print(f"Unknown model type {type(model).__name__}. Continue exporting...")
                model_type = f"unknown:{type(model).__name__}"

            export_path = self.config['export_path']
            start_time = time.time()

            export_hf_checkpoint(
                full_model,
                export_dir=export_path,
            )

            # Copy custom model files (Python files and JSON configs) if trust_remote_code is used
            copy_custom_model_files(self.config['checkpoint_path'], export_path, True)

            # Restore default padding and export the tokenizer as well.
            if tokenizer is not None:
                tokenizer.padding_side = default_padding_side
                tokenizer.save_pretrained(export_path)

            end_time = time.time()
            print(
                f"Quantized model exported to :{export_path}. Total time used {end_time - start_time}s"
            )

    def quantize_model(self, model, quant_cfg, calib_dataloader=None, calibration_only=False):
        use_calibration = need_calibration(quant_cfg)
        if not use_calibration:
            logger.warn("Dynamic quantization. Calibration skipped.")

        calibrate_loop = create_forward_loop(dataloader=calib_dataloader) if use_calibration else None

        logger.info("Starting model quantization...")
        start_time = time.time()

        if calibration_only:
            model = mtq.calibrate(model, quant_cfg["algorithm"], forward_loop=calibrate_loop)
        else:
            model = mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

        end_time = time.time()
        logger.info(f"Quantization done. Total time used: {end_time - start_time}s")
        return model


def get_model(
        model_id,
        device="cuda",
        gpu_mem_percentage=0.8,
):
    logger.info(f"Initializing model from {model_id}")

    device_map = "auto"
    if device == "cpu":
        device_map = "cpu"

    model_kwargs = {"trust_remote_code": True}
    try:
        hf_config = AutoConfig.from_pretrained(model_id, **model_kwargs)
    except Exception as e:
        logger.error(f"Error: Could not load config from {model_id}: {e}")
        raise RuntimeError(f"Failed to load model configuration from {model_id}") from e

    architecture = hf_config.architectures[0]
    if not hasattr(transformers, architecture):
        logger.warn(f"Architecture {architecture} not found in transformers: {transformers.__version__}. "
                    "Falling back to AutoModelForCausalLM.")
        auto_model_module = AutoModelForCausalLM
        from_config = auto_model_module.from_config
    else:
        auto_model_module = getattr(transformers, architecture)
        from_config = auto_model_module._from_config

    with init_empty_weights():
        # When computing the device_map, assuming half precision by default,
        # unless specified by the hf_config.
        torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
        model_kwargs2 = model_kwargs.copy()
        if auto_model_module != AutoModelForCausalLM:
            model_kwargs2.pop("trust_remote_code", None)
        model_kwargs2["torch_dtype"] = torch_dtype
        model_kwargs2.pop("max_memory", None)
        model = from_config(hf_config, **model_kwargs2)

    max_memory = get_max_memory()
    inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

    on_cpu = "cpu" in inferred_device_map.values()
    if on_cpu:
        for _device in max_memory:
            if isinstance(_device, int):
                max_memory[_device] *= gpu_mem_percentage

        print(
            "Model does not fit to the GPU mem. "
            f"We apply the following memory limit for calibration: \n{max_memory}\n"
            "If you hit GPU OOM issue, please adjust `gpu_mem_percentage` or "
            "reduce the calibration `batch_size` manually."
        )
        model_kwargs["max_memory"] = max_memory

    model = auto_model_module.from_pretrained(
        model_id,
        device_map="cuda",
        **model_kwargs,
    )

    model.eval()

    model = model.to("cuda")

    if device == "cuda" and not is_model_on_gpu(model):
        print("Warning: Some parameters are not on a GPU. Calibration can be slow or hit OOM")

    return model


def is_model_on_gpu(model) -> bool:
    """Returns if the model is fully loaded on GPUs."""
    return all("cuda" in str(param.device) for param in model.parameters())


def get_tokenizer(ckpt_path, trust_remote_code=False, **kwargs):
    print(f"Initializing tokenizer from {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, trust_remote_code=trust_remote_code, **kwargs
    )

    # can't set attribute 'pad_token' for "<unk>"
    # We skip this step for Nemo models
    if tokenizer.pad_token != "<unk>" or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, f"Pad token for {ckpt_path} cannot be set!"

    return tokenizer

def build_quant_cfg(
    qformat,
    kv_cache_qformat,
    awq_block_size,
    model_type,
    quant_cfg_choices,
    kv_quant_cfg_choices,
):
    quant_cfg = {}
    assert qformat in quant_cfg_choices, (
        f"Unsupported quantization format: {qformat} with {kv_cache_qformat} KV cache"
    )

    quant_cfg = quant_cfg_choices[qformat]
    
    if "awq" in qformat:
        quant_cfg = copy.deepcopy(quant_cfg_choices[qformat])
        weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
        if isinstance(weight_quantizer, list):
            weight_quantizer = weight_quantizer[0]

        # If awq_block_size argument is provided, update weight_quantizer
        if awq_block_size:
            weight_quantizer["block_sizes"][-1] = awq_block_size

        # Coarser optimal scale search seems to resolve the overflow in TRT-LLM for some models
        if qformat == "w4a8_awq" and model_type in ["gemma", "mpt"]:
            quant_cfg["algorithm"] = {"method": "awq_lite", "alpha_step": 1}

    enable_quant_kv_cache = kv_cache_qformat != "none"
    print(f"{'Enable' if enable_quant_kv_cache else 'Disable'} KV cache quantization")
    # Check if any bmm_quantizer is in the quant_cfg. If so, we need to enable the bmm_quantizer.
    if enable_quant_kv_cache:
        quant_cfg = mtq.update_quant_cfg_with_kv_cache_quant(
            quant_cfg,
            getattr(mtq, kv_quant_cfg_choices[kv_cache_qformat])["quant_cfg"],
        )

    # Gemma 7B has accuracy regression using alpha 1. We set 0.5 instead.
    if model_type == "gemma" and "int8_sq" in qformat:
        quant_cfg["algorithm"] = {"method": "smoothquant", "alpha": 0.5}

    if model_type == "phi4mm":
        # Only quantize the language model
        quant_cfg["quant_cfg"]["*speech*"] = {"enable": False}
        quant_cfg["quant_cfg"]["*audio*"] = {"enable": False}
        quant_cfg["quant_cfg"]["*image*"] = {"enable": False}
        quant_cfg["quant_cfg"]["*vision*"] = {"enable": False}

    return quant_cfg

def is_enc_dec(model_type) -> bool:
    """Return if the model is a encoder-decoder model."""
    return model_type in ["t5", "bart", "whisper"]

def copy_custom_model_files(source_path: str, export_path: str, trust_remote_code: bool = False):
    """Copy custom model files (configuration_*.py, modeling_*.py, *.json, etc.) from source to export directory.

    This function copies custom Python files and JSON configuration files that are needed for
    models with custom code. It excludes config.json and model.safetensors.index.json as these
    are typically handled separately by the model export process.

    Args:
        source_path: Path to the original model directory or HuggingFace model ID
        export_path: Path to the exported model directory
        trust_remote_code: Whether trust_remote_code was used (only copy files if True)
    """
    if not trust_remote_code:
        return

    # Resolve the source path (handles both local paths and HF model IDs)
    resolved_source_path = _resolve_model_path(source_path, trust_remote_code)

    source_dir = Path(resolved_source_path)
    export_dir = Path(export_path)

    if not source_dir.exists():
        if resolved_source_path != source_path:
            print(
                f"Warning: Could not find local cache for HuggingFace model '{source_path}' "
                f"(resolved to '{resolved_source_path}')"
            )
        else:
            print(f"Warning: Source directory '{source_path}' does not exist")
        return

    if not export_dir.exists():
        print(f"Warning: Export directory {export_path} does not exist")
        return

    # Common patterns for custom model files that need to be copied
    custom_file_patterns = [
        "configuration_*.py",
        "modeling*.py",
        "tokenization_*.py",
        "processing_*.py",
        "image_processing*.py",
        "feature_extraction_*.py",
        "*.json",
    ]

    copied_files = []
    for pattern in custom_file_patterns:
        for file_path in source_dir.glob(pattern):
            if file_path.is_file():
                # Skip config.json and model.safetensors.index.json as they're handled separately
                if file_path.name in ["config.json", "model.safetensors.index.json"]:
                    continue
                dest_path = export_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest_path)
                    copied_files.append(file_path.name)
                    print(f"Copied custom model file: {file_path.name}")
                except Exception as e:
                    print(f"Warning: Failed to copy {file_path.name}: {e}")

    if copied_files:
        print(f"Successfully copied {len(copied_files)} custom model files to {export_path}")
    else:
        print("No custom model files found to copy")

def _resolve_model_path(model_name_or_path: str, trust_remote_code: bool = False) -> str:
    """Resolve a model name or path to a local directory path.

    If the input is already a local directory, returns it as-is.
    If the input is a HuggingFace model ID, attempts to resolve it to the local cache path.

    Args:
        model_name_or_path: Either a local directory path or HuggingFace model ID
        trust_remote_code: Whether to trust remote code when loading the model

    Returns:
        Local directory path to the model files
    """
    # If it's already a local directory, return as-is
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    # Try to resolve HuggingFace model ID to local cache path
    try:
        # First try to load the config to trigger caching
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        # The config object should have the local path information
        # Try different ways to get the cached path
        if hasattr(config, "_name_or_path") and os.path.isdir(config._name_or_path):
            return config._name_or_path

        # Alternative: use snapshot_download if available
        if snapshot_download is not None:
            try:
                local_path = snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=["*.py", "*.json"],  # Only download Python files and config
                )
                return local_path
            except Exception as e:
                print(f"Warning: Could not download model files using snapshot_download: {e}")

        # Fallback: try to find in HuggingFace cache
        from transformers.utils import TRANSFORMERS_CACHE

        # Look for the model in the cache directory
        cache_pattern = os.path.join(TRANSFORMERS_CACHE, "models--*")
        cache_dirs = glob.glob(cache_pattern)

        # Convert model name to cache directory format
        model_cache_name = model_name_or_path.replace("/", "--")
        for cache_dir in cache_dirs:
            if model_cache_name in cache_dir:
                # Look for the snapshots directory
                snapshots_dir = os.path.join(cache_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Get the latest snapshot
                    snapshot_dirs = [
                        d
                        for d in os.listdir(snapshots_dir)
                        if os.path.isdir(os.path.join(snapshots_dir, d))
                    ]
                    if snapshot_dirs:
                        latest_snapshot = max(snapshot_dirs)  # Use lexicographically latest
                        snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
                        return snapshot_path

    except Exception as e:
        print(f"Warning: Could not resolve model path for {model_name_or_path}: {e}")

    # If all else fails, return the original path
    # This will cause the copy function to skip with a warning
    return model_name_or_path