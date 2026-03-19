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
from surogate.core.datasets.preprocessor.encode import EncodePreprocessor
from surogate.core.model.chat_templates.processor import ChatTemplateProcessor
from surogate.utils.command import SurogateCommand
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.np_utils import get_seed
from rich.console import Console
from rich.text import Text

try:
    from surogate._surogate import Tokenizer as NativeTokenizer
    HAS_NATIVE_TOKENIZER = True
except ImportError:
    HAS_NATIVE_TOKENIZER = False

logger = get_logger()

TOKENIZE_HASH_FILE = ".tokenize_hash"

# Default maximum tokens per output file (100M tokens)
DEFAULT_MAX_TOKENS_PER_FILE = 100_000_000

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


def _pack_buffer_to_sequence(
    token_list: list[np.ndarray],
    mask_list: list[np.ndarray],
    buffer_len: int,
    seq_len: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack buffered tokens and masks into a fixed-size sequence with position IDs.

    Args:
        token_list: List of token arrays to concatenate
        mask_list: List of mask arrays to concatenate
        buffer_len: Total length of buffered tokens
        seq_len: Target sequence length
        pad_token_id: Token ID to use for padding

    Returns:
        Tuple of (tokens, position_ids, mask) arrays, each of length seq_len
    """
    if buffer_len == 0:
        raise ValueError("Cannot pack empty buffer")

    tokens = np.concatenate(token_list).astype(np.int32)
    mask = np.concatenate(mask_list).astype(np.int32)

    # Per-document position IDs: each packed document gets 0..doc_len-1.
    # This enables compute_doc_masking() to detect document boundaries via
    # non-consecutive transitions, triggering Flash Attention varlen with
    # cu_seqlens for proper document-level attention isolation.
    pos_segments = [np.arange(len(arr), dtype=np.int32) for arr in token_list]
    pos_ids = np.concatenate(pos_segments)

    if tokens.size > seq_len:
        tokens = tokens[:seq_len]
        mask = mask[:seq_len]
        pos_ids = pos_ids[:seq_len]
    elif tokens.size < seq_len:
        pad_len = seq_len - tokens.size
        tokens = np.pad(tokens, (0, pad_len), mode="constant", constant_values=pad_token_id)
        mask = np.pad(mask, (0, pad_len), mode="constant", constant_values=0)
        # Continue position IDs monotonically through padding so that
        # compute_doc_masking() sees consecutive transitions and absorbs
        # padding into the last real document (no spurious length-1 docs).
        last_pos = int(pos_ids[-1]) if pos_ids.size > 0 else -1
        pad_pos = np.arange(last_pos + 1, last_pos + 1 + pad_len, dtype=np.int32)
        pos_ids = np.concatenate([pos_ids, pad_pos])

    return tokens, pos_ids, mask


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

        tokens, pos_ids, mask = _pack_buffer_to_sequence(
            cur_tokens, cur_masks, cur_len, seq_len, pad_token_id
        )
        writer.add_document(tokens=tokens, position_ids=pos_ids, mask=mask)
        cur_tokens = []
        cur_masks = []
        cur_len = 0

    for doc in docs:
        # tokens and mask are already numpy arrays from iter_docs()
        tokens = doc["tokens"]
        mask = doc["mask"]

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
        # tokens and mask are already numpy arrays from iter_docs()
        tokens = doc["tokens"]
        mask = doc["mask"]

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

class NativeEncodePreprocessor:
    """Fast dataset encoder using the C++ tokenizer's encode_for_training.

    Drop-in replacement for EncodePreprocessor that bypasses the Python
    ChatTemplateProcessor entirely. Uses the Jinja chat template + BPE
    tokenizer in C++ with multi-threaded batch encoding.

    Iterates the dataset directly instead of using dataset.map() to avoid
    pickling the C++ tokenizer object across process boundaries.
    """

    def __init__(self, native_tokenizer: 'NativeTokenizer', loss_strategy: str = 'default'):
        self.native_tokenizer = native_tokenizer
        self.loss_strategy = loss_strategy

    def __call__(self, dataset, batch_size: int = 10000, desc: str = "Encoding", **kwargs):
        from datasets import Dataset as HfDataset
        from tqdm import tqdm

        all_input_ids = []
        all_labels = []
        n_skipped = 0
        n_total = len(dataset)

        with tqdm(total=n_total, desc=desc, unit=" examples") as pbar:
            for batch in dataset.iter(batch_size=batch_size):
                messages_batch = batch['messages']

                try:
                    results = self.native_tokenizer.encode_for_training_batch(
                        messages_batch, strategy=self.loss_strategy)
                except Exception as e:
                    # Fall back to one-by-one to isolate bad examples
                    logger.warning(f"Batch encoding failed ({e}), falling back to row-by-row")
                    results = []
                    for msgs in messages_batch:
                        try:
                            results.append(self.native_tokenizer.encode_for_training(
                                msgs, strategy=self.loss_strategy))
                        except Exception:
                            n_skipped += 1
                            results.append({'input_ids': [], 'labels': []})

                for result in results:
                    ids = result['input_ids']
                    if len(ids) == 0:
                        continue
                    all_input_ids.append(ids)
                    all_labels.append(result['labels'])

                pbar.update(len(messages_batch))

        if n_skipped > 0:
            logger.warning(f"Skipped {n_skipped} examples due to encoding errors")

        return HfDataset.from_dict({
            'input_ids': all_input_ids,
            'labels': all_labels,
        })


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
            template_processor.model = self.config._model
        
        if self.config.model_template.is_multimodal and (
                self.config.padding_free or self.config.sample_packing) and not template_processor.support_padding_free:
            raise ValueError(f'Template `{self.config.template}` does not support padding free or packing.')
        
        self.template_processor = template_processor

    def _load_and_encode_datasets(self):
        """Load, preprocess, and encode datasets. Returns (train_dataset, val_dataset)."""
        import time
        train_datasets, val_datasets = [], []
        train_seed = np.random.RandomState(self.config.train_seed)
        eval_seed = np.random.RandomState(self.config.eval_seed)
        has_validation_datasets = len(self.config.validation_datasets) > 0

        # Get node sharding info for distributed training (set by distributed.py)
        node_rank = getattr(self.config, '_node_rank', None)
        num_nodes = getattr(self.config, '_num_nodes', None)

        with disable_datasets_caching():
            for ds_config in self.config.datasets:
                t0 = time.perf_counter()
                # Shard training data across nodes for distributed training
                dataset = load_dataset_with_config(
                    ds_config,
                    num_workers=self.config.dataloader_num_workers,
                    node_rank=node_rank,
                    num_nodes=num_nodes,
                )
                t1 = time.perf_counter()

                dataset = pre_process(dataset, ds_config, num_proc=self.config.dataloader_num_workers)
                t2 = time.perf_counter()

                train_dataset, val_dataset = post_process(
                    dataset,
                    dataset_sample=ds_config.samples,
                    split_dataset_ratio=self.config.validation_split_ratio if not has_validation_datasets else 0.0,
                    random_state=train_seed,
                )
                t3 = time.perf_counter()
                logger.info(f"Dataset '{ds_config.path}': load={t1-t0:.2f}s, pre_process={t2-t1:.2f}s, post_process={t3-t2:.2f}s")

                train_datasets.append(train_dataset)
                if val_dataset is not None:
                    val_datasets.append(val_dataset)

            for ds_config in self.config.validation_datasets:
                # Validation datasets are NOT sharded - all nodes get full eval data
                # for consistent evaluation metrics across nodes
                dataset = load_dataset_with_config(ds_config, num_workers=self.config.dataloader_num_workers)
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

            if len(val_datasets) > 0:
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

        if self.args.get('debug', False):
            self.config.validation_datasets = []
            for ds_config in self.config.datasets:
                ds_config.samples = 10

            # If debug flag is set, always load and process datasets            
            train_dataset, _ = self._load_and_encode_datasets()
            logger.info("Debug: printing labels for first 5 train dataset rows")
            for i, row in enumerate(train_dataset):
                if i >= 5:
                    break
                debug_labels(row, self.config.tokenizer)
            
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
        logger.info("Writing tokenized train files...")
        self._write_bin_tok(train_dataset, os.path.join(self.config.output_dir, 'train.bin'), packing=self.config.sample_packing)
        if val_dataset is not None:
            logger.info("Writing tokenized validation files...")
            self._write_bin_tok(val_dataset, os.path.join(self.config.output_dir, 'eval.bin'), packing=False)

        # Write the hash after successful tokenization
        write_tokenize_hash(self.config.output_dir, current_hash)
        logger.info(f"Tokenization complete. Hash saved: {current_hash}")

    def _can_use_native_encoder(self) -> bool:
        """Check if we can use the fast C++ encoder instead of the Python one."""
        if not HAS_NATIVE_TOKENIZER:
            return False
        if self.config.model_template.is_multimodal:
            return False
        # Only base strategies are supported by the native encoder
        loss_scale = getattr(self.config, 'loss_scale', 'default')
        if loss_scale not in ('default', 'last_round', 'all'):
            return False
        return True

    def _encode_dataset(self, train_dataset, val_dataset):
        if self._can_use_native_encoder():
            return self._encode_dataset_native(train_dataset, val_dataset)
        return self._encode_dataset_python(train_dataset, val_dataset)

    def _encode_dataset_native(self, train_dataset, val_dataset):
        """Encode datasets using the fast C++ tokenizer."""
        import time
        native_tok = NativeTokenizer.from_pretrained(self.config.model_dir)

        loss_strategy = getattr(self.config, 'loss_scale', 'default')
        preprocessor = NativeEncodePreprocessor(native_tok, loss_strategy=loss_strategy)

        datasets = [train_dataset, val_dataset]
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            split_name = 'train' if i == 0 else 'validation'
            t0 = time.perf_counter()
            datasets[i] = preprocessor(dataset, batch_size=10000, desc=f"Encoding {split_name}")
            t1 = time.perf_counter()
            n = len(datasets[i])
            logger.info(f"Encoded {split_name}: {n} examples in {t1-t0:.2f}s ({n/(t1-t0):.0f} examples/s)")
        return datasets

    def _encode_dataset_python(self, train_dataset, val_dataset):
        """Encode datasets using the Python ChatTemplateProcessor (fallback)."""
        template_processor = self.template_processor

        datasets = [train_dataset, val_dataset]
        origin_template_model = template_processor.model
        template_processor.model = None  # Avoid serializing the model with IPC when mapping the dataset.

        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            logger.info(f"Encoding {'train' if i == 0 else 'validation'} dataset...")
            preprocessor = EncodePreprocessor(template=template_processor)
            batch_size = 100 if self.config.model_template.is_multimodal else 10000

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

        Vectorized implementation: bulk-converts all examples to numpy, concatenates,
        builds position IDs and masks, slices into seq_len chunks, and writes files.
        """
        from tqdm import tqdm

        vocab_size = self.config.tokenizer.vocab_size
        seq_len = self.config.sequence_len or self.config.max_model_len
        pad_token_id = self.config.tokenizer.pad_token_id if self.config.tokenizer.pad_token_id is not None else 0

        if max_tokens_per_file is None:
            max_tokens_per_file = DEFAULT_MAX_TOKENS_PER_FILE

        non_overlapping = not packing
        out_dir = os.path.dirname(out_path)
        base_name = os.path.basename(out_path)
        name_without_ext = base_name.rsplit('.', 1)[0] if '.' in base_name else base_name
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Bulk collect all doc tokens/masks and their lengths
        all_tokens = []
        all_masks = []
        doc_lengths = []
        n_docs = len(dataset)

        for batch in tqdm(dataset.iter(batch_size=5000), desc="Preparing",
                          total=(n_docs + 4999) // 5000, unit=" batches"):
            for input_ids, labels in zip(batch['input_ids'], batch['labels']):
                tokens = np.asarray(input_ids, dtype=np.int32)
                lab = np.asarray(labels, dtype=np.int32)
                if tokens.size == 0:
                    continue
                mask = _to_input_mask((lab != -100).astype(np.int32))
                all_tokens.append(tokens)
                all_masks.append(mask)
                doc_lengths.append(tokens.size)

        if not all_tokens:
            return

        if packing:
            self._write_packed_vectorized(
                all_tokens, all_masks, doc_lengths,
                out_dir, name_without_ext, vocab_size, seq_len,
                pad_token_id, max_tokens_per_file, non_overlapping,
            )
        else:
            self._write_padded_vectorized(
                all_tokens, all_masks, doc_lengths,
                out_dir, name_without_ext, vocab_size, seq_len,
                pad_token_id, max_tokens_per_file, non_overlapping,
            )

    def _write_packed_vectorized(
        self, all_tokens, all_masks, doc_lengths,
        out_dir, name_prefix, vocab_size, seq_len,
        pad_token_id, max_tokens_per_file, non_overlapping,
    ):
        """Vectorized packing: concat all docs, build position IDs, slice into sequences."""
        from tqdm import tqdm

        # Build per-doc position IDs (0..len-1 for each doc)
        pos_segments = [np.arange(l, dtype=np.int32) for l in doc_lengths]

        # Concatenate everything into big flat arrays
        big_tokens = np.concatenate(all_tokens)
        big_mask = np.concatenate(all_masks)
        big_pos = np.concatenate(pos_segments)
        total = big_tokens.size

        # Pad to multiple of seq_len
        remainder = total % seq_len
        if remainder != 0:
            pad_len = seq_len - remainder
            big_tokens = np.pad(big_tokens, (0, pad_len), constant_values=pad_token_id)
            big_mask = np.pad(big_mask, (0, pad_len), constant_values=0)
            # Continue position IDs monotonically through padding
            last_pos = int(big_pos[-1]) if big_pos.size > 0 else -1
            big_pos = np.concatenate([big_pos, np.arange(last_pos + 1, last_pos + 1 + pad_len, dtype=np.int32)])

        n_seqs = big_tokens.size // seq_len

        # Reshape into (n_seqs, seq_len) for bulk slicing
        tokens_2d = big_tokens.reshape(n_seqs, seq_len)
        mask_2d = big_mask.reshape(n_seqs, seq_len)
        pos_2d = big_pos.reshape(n_seqs, seq_len)

        # Write to file(s)
        file_index = 0
        total_tokens = 0

        def get_path(idx):
            return os.path.join(out_dir, f"{name_prefix}-{idx:03d}.bin")

        writer = None
        seqs_per_file = max(1, max_tokens_per_file // seq_len)

        for start in tqdm(range(0, n_seqs, seqs_per_file), desc="Writing", unit=" shards"):
            end = min(start + seqs_per_file, n_seqs)
            output_path = get_path(file_index)
            with TokenizedDataFileWriter(output_path, vocab_size, masking=True, non_overlapping=non_overlapping) as writer:
                for i in range(start, end):
                    writer.add_document(tokens=tokens_2d[i], position_ids=pos_2d[i], mask=mask_2d[i])
                total_tokens += writer.n_tokens
                logger.info(f"Completed file {output_path} with {writer.n_tokens:,} tokens")
            file_index += 1

        logger.info(f"Multi-file write complete: {file_index} files, {total_tokens:,} total tokens")

    def _write_padded_vectorized(
        self, all_tokens, all_masks, doc_lengths,
        out_dir, name_prefix, vocab_size, seq_len,
        pad_token_id, max_tokens_per_file, non_overlapping,
    ):
        """Vectorized padded write: each doc is individually padded/truncated."""
        from tqdm import tqdm

        file_index = 0
        total_tokens = 0
        seqs_per_file = max(1, max_tokens_per_file // seq_len)

        def get_path(idx):
            return os.path.join(out_dir, f"{name_prefix}-{idx:03d}.bin")

        n_docs = len(all_tokens)
        writer = None

        for start in tqdm(range(0, n_docs, seqs_per_file), desc="Writing", unit=" shards"):
            end = min(start + seqs_per_file, n_docs)
            output_path = get_path(file_index)
            with TokenizedDataFileWriter(output_path, vocab_size, masking=True, non_overlapping=non_overlapping) as writer:
                for i in range(start, end):
                    tokens = all_tokens[i]
                    mask = all_masks[i]

                    # Truncate
                    if tokens.size > seq_len:
                        tokens = tokens[:seq_len]
                        mask = mask[:seq_len]

                    actual_len = tokens.size

                    # Pad
                    if tokens.size < seq_len:
                        p = seq_len - tokens.size
                        tokens = np.pad(tokens, (0, p), constant_values=pad_token_id)
                        mask = np.pad(mask, (0, p), constant_values=0)

                    pos_ids = np.zeros(seq_len, dtype=np.int32)
                    pos_ids[:actual_len] = np.arange(actual_len, dtype=np.int32)
                    writer.add_document(tokens=tokens, position_ids=pos_ids, mask=mask)

                total_tokens += writer.n_tokens
                logger.info(f"Completed file {output_path} with {writer.n_tokens:,} tokens")
            file_index += 1

        logger.info(f"Multi-file write complete: {file_index} files, {total_tokens:,} total tokens")



def tokenize_main(config: SFTConfig, args: DictDefault):
    TokenizeDatasets(config, args).run()