import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from transformers import PreTrainedTokenizerBase

from surogate.core.config.sft_config import SFTConfig
from surogate.core.datasets.datasets import disable_datasets_caching
from surogate.core.datasets.loader import load_dataset_with_config, pre_process, post_process, concat_datasets, \
    shuffle_dataset
from surogate.core.datasets.preprocessor.encode import EncodePreprocessor, AddLengthPreprocessor
from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.utils.command import SurogateCommand
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.np_utils import get_seed
from rich.console import Console
from rich.text import Text


logger = get_logger()

TOKENIZE_HASH_FILE = ".tokenize_hash"

# Default maximum tokens per output file (100M tokens)
DEFAULT_MAX_TOKENS_PER_FILE = 100_000_000

# Global variable for multiprocessing worker
_worker_template_processor = None
_worker_tokenizer = None


def _dataset_config_to_dict(ds_config) -> dict:
    """Extract hashable fields from a dataset config."""
    base = {
        "path": ds_config.path,
        "subset": ds_config.subset,
        "split": ds_config.split,
        "type": str(ds_config.type),
        "samples": ds_config.samples,
    }
    # Add type-specific fields
    if hasattr(ds_config, 'text_field'):
        base["text_field"] = ds_config.text_field
    if hasattr(ds_config, 'instruction_field'):
        base["instruction_field"] = ds_config.instruction_field
        base["input_field"] = ds_config.input_field
        base["output_field"] = ds_config.output_field
        base["system_prompt_type"] = str(ds_config.system_prompt_type)
        base["system_prompt_field"] = ds_config.system_prompt_field
        base["system_prompt"] = ds_config.system_prompt
    if hasattr(ds_config, 'messages_field'):
        base["messages_field"] = ds_config.messages_field
        base["system_field"] = ds_config.system_field
        base["tools_field"] = ds_config.tools_field
        base["message_property_mappings"] = ds_config.message_property_mappings
    return base


def compute_tokenize_hash(config: SFTConfig) -> str:
    """
    Compute a hash of all parameters that affect tokenization output.

    If this hash matches a previously stored hash, tokenization can be skipped.
    """
    hash_dict = {
        # Model/tokenizer identity
        "model_name": config.model,
        # Tokenization parameters
        "sequence_len": config.sequence_len,
        "max_model_len": config.max_model_len,
        "sample_packing": config.sample_packing,
        # Dataset split configuration
        "validation_split_ratio": config.validation_split_ratio,
        "train_seed": config.train_seed,
        "eval_seed": config.eval_seed,
        # Chat template configuration
        "template": config.template,
        "use_chat_template": config.use_chat_template,
        "loss_scale": config.loss_scale,
        "padding_free": config.padding_free,
        "padding_side": config.padding_side,
        "sequence_parallel_size": config.sequence_parallel_size,
        "truncation_strategy": config.truncation_strategy,
        "max_length": config.max_length,
        "max_pixels": config.max_pixels,
        "norm_bbox": config.norm_bbox,
        # Dataset configurations
        "datasets": [_dataset_config_to_dict(ds) for ds in config.datasets],
        "validation_datasets": [_dataset_config_to_dict(ds) for ds in config.validation_datasets],
    }

    # Serialize to JSON with sorted keys for deterministic output
    hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]


def read_tokenize_hash(output_dir: str) -> Optional[str]:
    """Read the stored tokenization hash from the output directory."""
    hash_path = os.path.join(output_dir, TOKENIZE_HASH_FILE)
    abs_hash_path = os.path.abspath(hash_path)
    if os.path.exists(hash_path):
        try:
            with open(hash_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read tokenization hash from {abs_hash_path}: {e}")
            return None
    logger.debug(f"Tokenization hash file not found at: {abs_hash_path}")
    return None


def write_tokenize_hash(output_dir: str, hash_value: str) -> None:
    """Write the tokenization hash to the output directory."""
    hash_path = os.path.join(output_dir, TOKENIZE_HASH_FILE)
    with open(hash_path, 'w') as f:
        f.write(hash_value)


def tokenized_files_exist(output_dir: str) -> bool:
    """Check if tokenized files exist in the output directory."""
    train_path = os.path.join(output_dir, 'train.bin')
    # Also check for sharded files (train-000.bin, train-001.bin, etc.)
    train_shard_path = os.path.join(output_dir, 'train-000.bin')
    return os.path.exists(train_path) or os.path.exists(train_shard_path)


def _init_worker(model_name: str, template_processor_state: Optional[dict] = None):
    """Initialize worker process with tokenizer and optional template processor."""
    from transformers import AutoTokenizer
    global _worker_tokenizer, _worker_template_processor
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Template processor state can be passed if needed for complex processing
    _worker_template_processor = template_processor_state


def _tokenize_example_worker(args: tuple) -> dict:
    """Worker function for multiprocessing tokenization.

    Args:
        args: Tuple of (example_dict, seq_len) where example_dict contains
              'input_ids' and 'labels' keys.

    Returns:
        Dictionary with 'tokens', 'mask', and 'position_ids' arrays.
    """
    example, seq_len = args
    input_ids = np.asarray(example['input_ids'], dtype=np.int32)
    labels = np.asarray(example['labels'], dtype=np.int32)

    # Create mask: 1 where we want to compute loss (labels != -100), 0 otherwise
    assistant_mask = (labels != -100).astype(np.int32)
    mask = _to_input_mask(assistant_mask)

    return {'tokens': input_ids, 'mask': mask}


class TokenizedDataFileWriter:
    def __init__(self, file_name: str,  vocab_size: int, masking: bool = False, non_overlapping: bool = False):
        self.file_name = file_name
        self.file_handle = None
        self.n_tokens = 0
        self.vocab_size = vocab_size
        self.has_masks = masking
        self.non_overlapping = non_overlapping
        self.mask_list = []
        self.mask_rest = None
        self.pos_ids_list = []

    def __enter__(self):
        self.file_handle = open(self.file_name, "wb+")
        # reserve space for the file header
        self.file_handle.write(('*' * 1023 + '\n').encode("ascii"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Format:
        # [Header 1024 bytes]
        # [Tokens (INT32) ... ]
        # [PositionIDs (INT32) ... ]
        # [Masks (Packed Bits) ... ] (Optional)
        
        self._write_position_ids()
        
        if self.has_masks:
            self._write_masks()
        self._write_header()
        self.file_handle.close()
        self.file_handle = None

    def add_document(self, tokens: np.ndarray, position_ids: np.ndarray, mask: Optional[np.ndarray] = None):
        assert self.file_handle is not None
        if mask is not None and self.has_masks is False:
            raise ValueError("Cannot add masking to a file that was not created with masking enabled")
        elif mask is None and self.has_masks is True:
            raise ValueError("Cannot add maskless tokens to a file that was created with masking enabled")

        tokens = np.array(tokens , dtype=np.int32)
        assert tokens.ndim == 1
        
        position_ids = np.array(position_ids, dtype=np.int32)
        assert position_ids.ndim == 1
        assert len(position_ids) == len(tokens)
        
        # Buffer position IDs for later writing
        self.pos_ids_list.append(position_ids)

        if mask is not None:
            mask = np.array(mask)
            assert len(mask) == len(tokens)
            self._record_mask(mask)

        # Write tokens immediately
        self.file_handle.write(tokens.tobytes())
        self.n_tokens += len(tokens)
        if self.n_tokens >= 2**31:
            raise RuntimeError("cannot have more than 2**31 tokens in a single file")

    def _record_mask(self, mask: np.ndarray):
        mask = mask.astype(np.bool_)
        if self.mask_rest is not None:
            full_mask = np.concatenate([self.mask_rest, mask]).astype(np.bool_, copy=False)
        else:
            full_mask = mask

        full_bytes = len(full_mask) // 8 * 8
        mask_bytes = full_mask[:full_bytes]
        self.mask_rest = full_mask[full_bytes:]
        self.mask_list.append(np.packbits(mask_bytes, bitorder='little'))
        
    def _write_position_ids(self):
        # Write all buffered position IDs immediately after the token block
        # Since usage pattern is append-only, we can just write chunks.
        for chunk in self.pos_ids_list:
             self.file_handle.write(chunk.tobytes())
        self.pos_ids_list = []

    def _write_masks(self):
        if self.mask_rest is not None and len(self.mask_rest) > 0:
            self.mask_list.append(np.packbits(self.mask_rest, bitorder='little'))
        for part in self.mask_list:
            self.file_handle.write(part.tobytes())

    def _write_header(self):
        assert self.file_handle is not None
        self.file_handle.seek(0)
        header_str = "BIN.TOK\n"  # 8 bytes
        version = 3 # Bump version for PositionID support
        bytes_per_token = 4
        self.file_handle.write(header_str.encode("ascii"))
        # Header layout (int32 each, starting at offset 8):
        # [2] version, [3] bytes_per_token, [4] n_tokens, [5] vocab_size, [6] has_masks, [7] non_overlapping
        self.file_handle.write(np.array([version, bytes_per_token, self.n_tokens, self.vocab_size, self.has_masks, self.non_overlapping], dtype=np.int32).tobytes())
        self.file_handle.seek(256*4)


def _to_input_mask(assistant_token_mask: np.ndarray) -> np.ndarray:
    """
    Convert a token-aligned mask (train on token positions) to the input-aligned mask
    expected by `DataLoader`, which masks targets based on the corresponding input position.

    DataLoader loads:
      inputs  = tokens[s : s+T]
      targets = tokens[s+1 : s+T+1]
    and applies mask bits to input positions `s..s+T-1`.

    To train on assistant tokens as targets, we set:
      input_mask[i] = assistant_token_mask[i+1]
    and force the last mask bit to 0 so we never train across chunk boundaries.
    """
    if assistant_token_mask.ndim != 1:
        raise ValueError("assistant_token_mask must be 1D")
    if assistant_token_mask.size == 0:
        return assistant_token_mask.astype(np.int32)

    out = np.zeros_like(assistant_token_mask, dtype=np.int32)
    out[:-1] = assistant_token_mask[1:].astype(np.int32)
    out[-1] = 0
    return out


def tokenize_messages(tokenizer: PreTrainedTokenizerBase, messages: list[dict], seq_len: int) -> dict:
    """
    Tokenize a chat conversation and pad/truncate to fixed seq_len.

    Returns tokens, mask, and position_ids all of length seq_len.
    """
    # Apply chat template to get the full conversation tokens.
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="np")[0]

    # Determine where the final assistant response begins.
    # Convention: only train on the final assistant message.
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected last message role == 'assistant'")
    prompt_messages = messages[:-1]
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="np",
    )[0]
    prompt_len = len(prompt_tokens)

    # Create token-level assistant mask: 0 for prompt, 1 for assistant response tokens.
    # NOTE: Our DataLoader applies mask bits to the *input* positions, but targets are
    # shifted by +1 token (next-token prediction). To train on assistant tokens as
    # targets, we later shift this mask by one position (see `_to_input_mask`).
    assistant_mask = np.zeros(len(tokens), dtype=np.int32)
    if prompt_len < len(tokens):
        assistant_mask[prompt_len:] = 1

    # Generate Position IDs (0, 1, ..., N-1)
    position_ids = np.arange(len(tokens), dtype=np.int32)

    # Pad/truncate to seq_len.
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
        assistant_mask = assistant_mask[:seq_len]
        position_ids = position_ids[:seq_len]
        # Ensure last token is EOS and included in loss.
        if tokenizer.eos_token_id is not None:
            tokens[-1] = tokenizer.eos_token_id
            assistant_mask[-1] = 1
    else:
        pad_len = seq_len - len(tokens)
        pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        tokens = np.pad(tokens, (0, pad_len), mode="constant", constant_values=pad_val)
        assistant_mask = np.pad(assistant_mask, (0, pad_len), mode="constant", constant_values=0)
        # For padding, we usually set position_ids to 0
        position_ids = np.pad(position_ids, (0, pad_len), mode="constant", constant_values=0)

    # Shift token-level mask to input-aligned mask bits used by DataLoader.
    mask = _to_input_mask(assistant_mask)
    return {"tokens": tokens, "mask": mask, "position_ids": position_ids}


def tokenize_messages_unpadded(tokenizer: PreTrainedTokenizerBase, messages: list[dict], max_len: int) -> dict:
    """
    Tokenize a single chat record without padding.

    Returns variable-length `tokens` and an input-aligned `mask` (same length as tokens),
    suitable for later packing into fixed-size chunks.
    """
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="np")[0]

    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Expected last message role == 'assistant'")
    prompt_messages = messages[:-1]
    prompt_tokens = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="np",
    )[0]
    prompt_len = len(prompt_tokens)

    assistant_mask = np.zeros(len(tokens), dtype=np.int32)
    if prompt_len < len(tokens):
        assistant_mask[prompt_len:] = 1

    # Generate Position IDs unpadded (0, ..., N-1)
    position_ids = np.arange(len(tokens), dtype=np.int32)

    # Truncate very long records to max_len, force EOS as a terminator.
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        assistant_mask = assistant_mask[:max_len]
        position_ids = position_ids[:max_len]
        if tokenizer.eos_token_id is not None:
            tokens[-1] = tokenizer.eos_token_id
            assistant_mask[-1] = 1

    # Ensure the record ends with EOS so boundaries are well-defined for packing.
    if tokenizer.eos_token_id is not None and (len(tokens) == 0 or int(tokens[-1]) != int(tokenizer.eos_token_id)):
        tokens = np.concatenate([tokens, np.array([tokenizer.eos_token_id], dtype=tokens.dtype)])
        assistant_mask = np.concatenate([assistant_mask, np.array([1], dtype=assistant_mask.dtype)])
        # Extend position_ids
        next_pos = position_ids[-1] + 1 if len(position_ids) > 0 else 0
        position_ids = np.concatenate([position_ids, np.array([next_pos], dtype=position_ids.dtype)])

    mask = _to_input_mask(assistant_mask)
    return {"tokens": tokens.astype(np.int32), "mask": mask.astype(np.int32), "position_ids": position_ids.astype(np.int32)}


def pack_and_write(
    writer: TokenizedDataFileWriter,
    docs: Iterable[dict],
    seq_len: int,
    pad_token_id: int,
) -> None:
    """
    Pack variable-length token/mask docs into fixed-size `seq_len` sequences and write them.
    """
    cur_tokens: list[np.ndarray] = []
    cur_masks: list[np.ndarray] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur_tokens, cur_masks, cur_len
        if cur_len == 0:
            return
        tokens = np.concatenate(cur_tokens).astype(np.int32)
        mask = np.concatenate(cur_masks).astype(np.int32)

        # IMPORTANT: For packed sequences, position_ids must be monotonic within the packed chunk
        # (0..len-1, then 0 for padding). Resetting position IDs per original example (as in the
        # unpadded docs) changes RoPE phases and hurts training.
        if tokens.size > seq_len:
            tokens = tokens[:seq_len]
            mask = mask[:seq_len]
            pos_ids = np.arange(seq_len, dtype=np.int32)
        else:
            pos_ids = np.arange(tokens.size, dtype=np.int32)
            if tokens.size < seq_len:
                pad_len = seq_len - tokens.size
                tokens = np.pad(tokens, (0, pad_len), mode="constant", constant_values=pad_token_id)
                mask = np.pad(mask, (0, pad_len), mode="constant", constant_values=0)
                pos_ids = np.pad(pos_ids, (0, pad_len), mode="constant", constant_values=0)

        writer.add_document(tokens=tokens, position_ids=pos_ids, mask=mask)
        cur_tokens = []
        cur_masks = []
        cur_len = 0

    for doc in docs:
        tokens = np.asarray(doc["tokens"], dtype=np.int32)
        mask = np.asarray(doc["mask"], dtype=np.int32)

        if tokens.ndim != 1 or mask.ndim != 1 or tokens.size != mask.size:
            raise ValueError("doc tokens/mask must be 1D and same length")
        if tokens.size == 0:
            continue

        if tokens.size > seq_len:
            # Too long: write as its own chunk (truncate, pad not needed).
            flush()
            writer.add_document(tokens=tokens[:seq_len], position_ids=np.arange(seq_len, dtype=np.int32), mask=mask[:seq_len])
            continue

        if cur_len + tokens.size > seq_len:
            flush()

        cur_tokens.append(tokens)
        cur_masks.append(mask)
        cur_len += tokens.size

    flush()


def write_padded(
    writer: TokenizedDataFileWriter,
    docs: Iterable[dict],
    seq_len: int,
    pad_token_id: int,
) -> None:
    """
    Write each doc as a separate padded sequence (no packing).
    Used for validation datasets where per-example metrics matter.
    """
    for doc in docs:
        tokens = np.asarray(doc["tokens"], dtype=np.int32)
        mask = np.asarray(doc["mask"], dtype=np.int32)

        if tokens.ndim != 1 or mask.ndim != 1 or tokens.size != mask.size:
            raise ValueError("doc tokens/mask must be 1D and same length")
        if tokens.size == 0:
            continue

        # Truncate if too long
        if tokens.size > seq_len:
            tokens = tokens[:seq_len]
            mask = mask[:seq_len]

        # Pad if too short
        if tokens.size < seq_len:
            pad_len = seq_len - tokens.size
            tokens = np.pad(tokens, (0, pad_len), mode="constant", constant_values=pad_token_id)
            mask = np.pad(mask, (0, pad_len), mode="constant", constant_values=0)

        # Position IDs: 0..actual_len-1, then 0 for padding
        actual_len = min(doc["tokens"].size if hasattr(doc["tokens"], 'size') else len(doc["tokens"]), seq_len)
        pos_ids = np.zeros(seq_len, dtype=np.int32)
        pos_ids[:actual_len] = np.arange(actual_len, dtype=np.int32)

        writer.add_document(tokens=tokens, position_ids=pos_ids, mask=mask)

def debug_labels(example, tokenizer, text_only=False):
    """Debug labels using Rich library Token Pill design for better readability.

    Args:
        example: Dataset example with 'input_ids' and 'labels' keys
        tokenizer: Tokenizer for decoding token IDs
        text_only: If True, only show text without token IDs
    """
    # Force color output and use full terminal width
    console = Console(force_terminal=True, force_interactive=True, width=None, legacy_windows=False)

    # Get the input_ids, labels, and attention_mask from the dataset
    input_ids = example["input_ids"]
    labels = example["labels"]
    target_mask = example.pop("target_mask", None)

    output = Text()
    target_labels_count = 0

    for input_id, label_id in zip(input_ids, labels, strict=False):
        decoded_token = tokenizer.decode(input_id)

        # --- 1. Sanitize Special Characters ---
        # Explicitly show newlines as symbols so the debug flow isn't broken
        display_text = decoded_token.replace('\n', '⏎').replace('\r', '')
        if display_text.strip() == "":
            display_text = "␣"  # Symbol for pure space if needed
        elif decoded_token == " ":
            display_text = " "

        # --- 2. Determine Logic/Style ---
        if label_id == -100:
            # MASKED (Prompt) -> White
            main_style = "white"
            id_style = "dim white"
            border_color = "white"
            tag = "M"  # Masked
        elif label_id == input_id:
            # TRAIN (Loss calculated) -> Green
            main_style = "bold green"
            id_style = "green"
            border_color = "green"
            tag = "T"  # Train
            target_labels_count += 1
        else:
            # Different label (e.g., padding with label=0 but different token_id) -> Yellow
            # Or ERROR (Mismatch) -> could be red, but let's use yellow for padding
            if label_id == 0:
                main_style = "yellow"
                id_style = "dim yellow"
                border_color = "yellow"
                tag = "P"  # Padding
            else:
                # True error/mismatch
                main_style = "bold white on red"
                id_style = "white on red"
                border_color = "red"
                tag = "?"

        # --- 3. Construct the "Pill" ---
        # Format: [ Text | ID ] or just [ Text ] if text_only
        output.append("[", style=f"dim {border_color}")
        output.append(display_text, style=main_style)

        if not text_only:
            output.append("|", style=f"dim {border_color}")
            output.append(str(input_id), style=id_style)

        output.append("]", style=f"dim {border_color}")

        # Add a tiny space between pills for readability
        output.append(" ")

    # Print the formatted output with wrapping at full terminal width
    console.print(output, soft_wrap=True)
    console.print()  # Extra newline

    # Print summary
    total_len = len(input_ids)
    console.print("=" * console.width, style="cyan")
    console.print("DEBUG SUMMARY:", style="bold cyan")
    console.print(f"  Total input len: {total_len}")
    console.print(f"  Count of trained labels: {target_labels_count}")
    console.print(f"  Trained ratio: {target_labels_count/total_len*100:.1f}%")
    if target_mask:
        target_mask_positions = sum(m[0] for m in target_mask)
        console.print(f"  Number of positions in target_mask: {target_mask_positions}")

    console.print("Legend:", style="bold cyan", end=" ")
    console.print("[M]", style="white", end="=Masked (prompt), ")
    console.print("[T]", style="bold green", end="=Trained (response), ")
    console.print("[P]", style="yellow", end="=Padding")
    console.print()
    console.print("=" * console.width, style="cyan")
    console.print()

    return output

class TokenizeDatasets(SurogateCommand):
    template_processor: ChatTemplateProcessor
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, config: SFTConfig, args: DictDefault):
        super().__init__(config=config, args=args)
        config.__post_init__()

        self._prepare_chat_template()

    def _prepare_chat_template(self) -> None:
        template_processor = self.config.get_template_processor(self.config.tokenizer)
        template_processor.set_mode('train')
        if template_processor.use_model:
            template_processor.model = self.config.model
        
        if self.config.model_template.is_multimodal and (
                self.config.padding_free or self.config.sample_packing) and not template_processor.support_padding_free:
            raise ValueError(f'Template `{self.config.template}` does not support padding free or packing.')
        
        self.template_processor = template_processor

    def _load_and_encode_datasets(self):
        """Load, preprocess, and encode datasets. Returns (train_dataset, val_dataset)."""
        train_datasets, val_datasets = [], []
        train_seed = np.random.RandomState(self.config.train_seed)
        eval_seed = np.random.RandomState(self.config.eval_seed)

        with disable_datasets_caching():
            for ds_config in self.config.datasets:
                dataset = load_dataset_with_config(ds_config)

                dataset = pre_process(dataset, ds_config, num_proc=self.config.dataloader_num_workers)
                train_dataset, val_dataset = post_process(
                    dataset,
                    dataset_sample=ds_config.samples,
                    split_dataset_ratio=self.config.validation_split_ratio,
                    random_state=train_seed,
                )
                train_datasets.append(train_dataset)
                val_datasets.append(val_dataset)

            for ds_config in self.config.validation_datasets:
                dataset = load_dataset_with_config(ds_config)
                dataset = pre_process(dataset, ds_config, num_proc=self.config.dataloader_num_workers)
                _, val_dataset = post_process(
                    dataset,
                    dataset_sample=ds_config.samples,
                    split_dataset_ratio=1.0,
                    random_state=eval_seed,
                )
                val_datasets.append(val_dataset)

            train_dataset = concat_datasets(train_datasets)
            train_dataset = shuffle_dataset(
                train_dataset, seed=get_seed(train_seed), buffer_size=1000)

            val_dataset = concat_datasets(val_datasets)
            val_dataset = shuffle_dataset(
                val_dataset, seed=get_seed(eval_seed), buffer_size=1000)

            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset)

        return train_dataset, val_dataset

    def run(self):
        # Check if we can skip tokenization based on hash
        current_hash = compute_tokenize_hash(self.config)
        stored_hash = read_tokenize_hash(self.config.output_dir)
        files_exist = tokenized_files_exist(self.config.output_dir)

        logger.debug(f"Tokenization cache check: current_hash={current_hash}, stored_hash={stored_hash}, files_exist={files_exist}")

        # If debug flag is set, always load and process datasets
        if self.args.get('debug', False):
            train_dataset, val_dataset = self._load_and_encode_datasets()
            logger.info("Debug: printing labels for first 5 train dataset rows")
            for i, row in enumerate(train_dataset):
                if i >= 5:
                    break
                debug_labels(row, self.config.tokenizer)

            # If tokenization can be skipped (and debug was the only reason to run), return now
            if stored_hash == current_hash and files_exist:
                logger.info(f"Tokenization hash unchanged ({current_hash}), skipping tokenization write.")
                return

            # Otherwise, continue to write the tokenized files
            logger.info(f"Writing tokenized files (hash={current_hash})...")
            self._write_bin_tok(train_dataset, os.path.join(self.config.output_dir, 'train.bin'), packing=True)
            if val_dataset is not None:
                self._write_bin_tok(val_dataset, os.path.join(self.config.output_dir, 'eval.bin'), packing=False)

            # Write the hash after successful tokenization
            write_tokenize_hash(self.config.output_dir, current_hash)
            logger.info(f"Tokenization complete. Hash saved: {current_hash}")
            return

        # No debug flag: check if we can skip entirely
        if stored_hash == current_hash and files_exist:
            logger.info(f"Tokenization hash unchanged ({current_hash}), skipping tokenization.")
            return

        # Log why we're not skipping
        if stored_hash is None:
            logger.info(f"No stored tokenization hash found, tokenizing dataset (hash={current_hash})...")
        elif stored_hash != current_hash:
            logger.info(f"Tokenization config changed (old={stored_hash}, new={current_hash}), re-tokenizing...")
        elif not files_exist:
            logger.info(f"Tokenized files not found, tokenizing dataset (hash={current_hash})...")

        # Load and encode datasets
        train_dataset, val_dataset = self._load_and_encode_datasets()

        # Write BIN.TOK files
        self._write_bin_tok(train_dataset, os.path.join(self.config.output_dir, 'train.bin'), packing=True)
        if val_dataset is not None:
            self._write_bin_tok(val_dataset, os.path.join(self.config.output_dir, 'eval.bin'), packing=False)

        # Write the hash after successful tokenization
        write_tokenize_hash(self.config.output_dir, current_hash)
        logger.info(f"Tokenization complete. Hash saved: {current_hash}")

    def _encode_dataset(self, train_dataset, val_dataset):
        template_processor = self.template_processor

        datasets = [train_dataset, val_dataset]
        origin_template_model = template_processor.model
        template_processor.model = None  # Avoid serializing the model.
        
        if self.config.truncation_strategy == 'split':
            if self.config.use_chat_template:
                raise ValueError(
                    'truncation_strategy=split is currently only supported for plain text model pretraining')
            
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            preprocessor_cls = EncodePreprocessor if self.config.truncation_strategy == 'split' else AddLengthPreprocessor
            preprocessor = preprocessor_cls(template=template_processor)
            batch_size = 100 if self.config.model_template.is_multimodal else 1000
            
            dataset = preprocessor(
                dataset,
                num_proc=self.config.dataloader_num_workers,
                load_from_cache_file=False,
                strict=False,
                batch_size=batch_size)

            datasets[i] = dataset

        template_processor.model = origin_template_model
        return datasets

    def _write_bin_tok(self, dataset, out_path: str, packing: bool = True,
                       max_tokens_per_file: Optional[int] = None,
                       num_workers: Optional[int] = None) -> None:
        """Write dataset to BIN.TOK format file(s).

        Args:
            dataset: The dataset to write.
            out_path: Output file path. For multi-file output, this is used as a base
                     (e.g., 'train.bin' becomes 'train-000.bin', 'train-001.bin', etc.)
            packing: If True, pack multiple examples into sequences (for training).
                     If False, pad each example individually (for validation).
            max_tokens_per_file: Maximum tokens per output file. If None or the total
                                tokens fit in one file, writes a single file.
                                Default is 100M tokens per file for training data.
            num_workers: Number of parallel workers for tokenization. If None, uses
                        half of available CPUs (capped at 8).
        """
        vocab_size = self.config.tokenizer.vocab_size
        seq_len = self.config.sequence_len or self.config.max_model_len
        pad_token_id = self.config.tokenizer.pad_token_id if self.config.tokenizer.pad_token_id is not None else 0

        # Set defaults for multi-file processing
        if max_tokens_per_file is None and packing:
            max_tokens_per_file = DEFAULT_MAX_TOKENS_PER_FILE

        if num_workers is None:
            num_workers = max(1, min(mp.cpu_count() // 2, 8))

        def iter_docs():
            for row in dataset:
                input_ids = np.asarray(row['input_ids'], dtype=np.int32)
                labels = np.asarray(row['labels'], dtype=np.int32)
                # Create mask: 1 where we want to compute loss (labels != -100), 0 otherwise
                # Shift by 1 position for input-aligned mask (as in _to_input_mask)
                assistant_mask = (labels != -100).astype(np.int32)
                mask = _to_input_mask(assistant_mask)
                yield {'tokens': input_ids, 'mask': mask}

        # Determine output path structure
        out_dir = os.path.dirname(out_path)
        base_name = os.path.basename(out_path)
        name_without_ext = base_name.rsplit('.', 1)[0] if '.' in base_name else base_name
        ext = '.bin'

        # non_overlapping=True for validation (padded, non-packed) so dataloader uses correct chunk count
        non_overlapping = not packing

        # Use multi-file writer for training data with packing
        if packing and max_tokens_per_file:
            self._write_multi_file(
                iter_docs(),
                out_dir=out_dir,
                name_prefix=name_without_ext,
                vocab_size=vocab_size,
                seq_len=seq_len,
                pad_token_id=pad_token_id,
                max_tokens_per_file=max_tokens_per_file,
                non_overlapping=non_overlapping
            )
        else:
            # Single file output for validation or small datasets
            with TokenizedDataFileWriter(out_path, vocab_size, masking=True, non_overlapping=non_overlapping) as writer:
                if packing:
                    pack_and_write(writer, iter_docs(), seq_len=seq_len, pad_token_id=pad_token_id)
                else:
                    write_padded(writer, iter_docs(), seq_len=seq_len, pad_token_id=pad_token_id)

    def _write_multi_file(
        self,
        docs: Iterable[dict],
        out_dir: str,
        name_prefix: str,
        vocab_size: int,
        seq_len: int,
        pad_token_id: int,
        max_tokens_per_file: int,
        non_overlapping: bool = False
    ) -> int:
        """Write tokenized documents to multiple files, splitting when token limit is reached.

        Args:
            docs: Iterable of document dictionaries with 'tokens' and 'mask' keys.
            out_dir: Output directory.
            name_prefix: Prefix for output files (e.g., 'train' -> 'train-000.bin').
            vocab_size: Vocabulary size for header.
            seq_len: Sequence length for packing.
            pad_token_id: Padding token ID.
            max_tokens_per_file: Maximum tokens per file before creating a new shard.
            non_overlapping: Whether chunks should be non-overlapping.

        Returns:
            Total number of tokens written across all files.
        """
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        file_index = 0
        total_tokens = 0
        current_writer = None
        current_tokens = []
        current_masks = []
        current_len = 0

        def get_output_path(idx: int) -> str:
            return os.path.join(out_dir, f"{name_prefix}-{idx:03d}.bin")

        def flush_to_writer():
            nonlocal current_tokens, current_masks, current_len
            if current_len == 0:
                return

            tokens = np.concatenate(current_tokens).astype(np.int32)
            mask = np.concatenate(current_masks).astype(np.int32)

            if tokens.size > seq_len:
                tokens = tokens[:seq_len]
                mask = mask[:seq_len]
                pos_ids = np.arange(seq_len, dtype=np.int32)
            else:
                pos_ids = np.arange(tokens.size, dtype=np.int32)
                if tokens.size < seq_len:
                    pad_len = seq_len - tokens.size
                    tokens = np.pad(tokens, (0, pad_len), mode="constant", constant_values=pad_token_id)
                    mask = np.pad(mask, (0, pad_len), mode="constant", constant_values=0)
                    pos_ids = np.pad(pos_ids, (0, pad_len), mode="constant", constant_values=0)

            current_writer.add_document(tokens=tokens, position_ids=pos_ids, mask=mask)
            current_tokens = []
            current_masks = []
            current_len = 0

        def start_new_file():
            nonlocal current_writer, file_index
            if current_writer is not None:
                flush_to_writer()
                current_writer.__exit__(None, None, None)
                logger.info(f"Completed file {get_output_path(file_index - 1)} with {current_writer.n_tokens:,} tokens")

            output_path = get_output_path(file_index)
            current_writer = TokenizedDataFileWriter(output_path, vocab_size, masking=True, non_overlapping=non_overlapping)
            current_writer.__enter__()
            file_index += 1

        # Start first file
        start_new_file()

        for doc in docs:
            tokens = np.asarray(doc["tokens"], dtype=np.int32)
            mask = np.asarray(doc["mask"], dtype=np.int32)

            if tokens.ndim != 1 or mask.ndim != 1 or tokens.size != mask.size:
                raise ValueError("doc tokens/mask must be 1D and same length")
            if tokens.size == 0:
                continue

            # Check if we need to start a new file
            if current_writer.n_tokens + current_len >= max_tokens_per_file:
                flush_to_writer()
                total_tokens += current_writer.n_tokens
                start_new_file()

            # Handle documents longer than seq_len
            if tokens.size > seq_len:
                flush_to_writer()
                current_writer.add_document(
                    tokens=tokens[:seq_len],
                    position_ids=np.arange(seq_len, dtype=np.int32),
                    mask=mask[:seq_len]
                )
                continue

            # Check if adding this doc would exceed seq_len
            if current_len + tokens.size > seq_len:
                flush_to_writer()

            current_tokens.append(tokens)
            current_masks.append(mask)
            current_len += tokens.size

        # Flush remaining data
        flush_to_writer()
        if current_writer is not None:
            total_tokens += current_writer.n_tokens
            current_writer.__exit__(None, None, None)
            logger.info(f"Completed file {get_output_path(file_index - 1)} with {current_writer.n_tokens:,} tokens")

        logger.info(f"Multi-file write complete: {file_index} files, {total_tokens:,} total tokens")
        return total_tokens


def tokenize_main(config: SFTConfig, args: DictDefault):
    TokenizeDatasets(config, args).run()