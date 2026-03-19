import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from surogate._surogate import Tokenizer as NativeTokenizer
from surogate.core.config.sft_config import SFTConfig
from surogate.core.datasets.datasets import disable_datasets_caching
from surogate.core.datasets.loader import load_dataset_with_config, pre_process, post_process, concat_datasets, \
    shuffle_dataset
from surogate.utils.command import SurogateCommand
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger
from surogate.utils.np_utils import get_seed

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
        "model_name": config.model,
        "sequence_len": config.sequence_len,
        "max_model_len": config.max_model_len,
        "sample_packing": config.sample_packing,
        "validation_split_ratio": config.validation_split_ratio,
        "train_seed": config.train_seed,
        "eval_seed": config.eval_seed,
        "loss_scale": config.loss_scale,
        "padding_free": config.padding_free,
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

def debug_labels(input_ids, labels, tokenizer, text_only=False):
    """Debug labels using Rich library Token Pill design for better readability.

    Args:
        input_ids: Token ID list
        labels: Label ID list (-100 = masked)
        tokenizer: Native tokenizer for decoding token IDs
        text_only: If True, only show text without token IDs
    """
    from rich.console import Console
    from rich.text import Text

    console = Console(force_terminal=True, force_interactive=True, width=None, legacy_windows=False)
    output = Text()
    target_labels_count = 0

    for input_id, label_id in zip(input_ids, labels):
        decoded_token = tokenizer.decode([input_id])

        display_text = decoded_token.replace('\n', '\u23ce').replace('\r', '')
        if display_text.strip() == "":
            display_text = "\u2423"
        elif decoded_token == " ":
            display_text = " "

        if label_id == -100:
            main_style = "white"
            border_color = "white"
        elif label_id == input_id:
            main_style = "bold green"
            border_color = "green"
            target_labels_count += 1
        elif label_id == 0:
            main_style = "yellow"
            border_color = "yellow"
        else:
            main_style = "bold white on red"
            border_color = "red"

        output.append("[", style=f"dim {border_color}")
        output.append(display_text, style=main_style)
        if not text_only:
            output.append("|", style=f"dim {border_color}")
            output.append(str(input_id), style=f"dim {border_color}")
        output.append("]", style=f"dim {border_color}")
        output.append(" ")

    console.print(output, soft_wrap=True)
    console.print()

    total_len = len(input_ids)
    console.print("=" * console.width, style="cyan")
    console.print("DEBUG SUMMARY:", style="bold cyan")
    console.print(f"  Total input len: {total_len}")
    console.print(f"  Count of trained labels: {target_labels_count}")
    console.print(f"  Trained ratio: {target_labels_count/total_len*100:.1f}%")
    console.print("Legend:", style="bold cyan", end=" ")
    console.print("[M]", style="white", end="=Masked (prompt), ")
    console.print("[T]", style="bold green", end="=Trained (response), ")
    console.print("[P]", style="yellow", end="=Padding")
    console.print()
    console.print("=" * console.width, style="cyan")
    console.print()


def _encode_and_prepare_native(
    native_tokenizer,
    dataset,
    loss_strategy: str = 'default',
    batch_size: int = 10000,
    desc: str = "Encoding",
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Single-pass encode + prepare: messages → numpy arrays ready for writing.

    Pipelines C++ encoding (GIL-released) with Python numpy conversion using
    a background thread. Returns (all_tokens, all_masks, doc_lengths) directly,
    skipping the intermediate HfDataset.
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor

    all_tokens = []
    all_masks = []
    doc_lengths = []
    n_skipped = 0
    n_total = len(dataset)

    def encode_batch(messages_batch):
        """Submit to C++ encoder (releases GIL internally)."""
        try:
            return native_tokenizer.encode_for_training_batch(
                messages_batch, strategy=loss_strategy)
        except Exception as e:
            logger.warning(f"Batch encoding failed ({e}), falling back to row-by-row")
            results = []
            for msgs in messages_batch:
                try:
                    results.append(native_tokenizer.encode_for_training(
                        msgs, strategy=loss_strategy))
                except Exception:
                    results.append(None)
            return results

    def collect_results(results):
        """Convert encoding results to numpy arrays."""
        nonlocal n_skipped
        for result in results:
            if result is None:
                n_skipped += 1
                continue
            ids = result['input_ids']
            if len(ids) == 0:
                continue
            tokens = np.asarray(ids, dtype=np.int32)
            labels = np.asarray(result['labels'], dtype=np.int32)
            mask = _to_input_mask((labels != -100).astype(np.int32))
            all_tokens.append(tokens)
            all_masks.append(mask)
            doc_lengths.append(tokens.size)

    with tqdm(total=n_total, desc=desc, unit=" examples") as pbar:
        # Pipeline: encode batch N+1 in background while collecting batch N
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending_future = None
            pending_count = 0

            for batch in dataset.iter(batch_size=batch_size):
                messages_batch = batch['messages']

                # Submit encoding to thread (C++ releases GIL)
                future = executor.submit(encode_batch, messages_batch)

                # While C++ encodes current batch, collect previous results
                if pending_future is not None:
                    collect_results(pending_future.result())
                    pbar.update(pending_count)

                pending_future = future
                pending_count = len(messages_batch)

            # Collect last batch
            if pending_future is not None:
                collect_results(pending_future.result())
                pbar.update(pending_count)

    if n_skipped > 0:
        logger.warning(f"Skipped {n_skipped} examples due to encoding errors")

    return all_tokens, all_masks, doc_lengths


class TokenizeDatasets(SurogateCommand):

    def __init__(self, config: SFTConfig, args: DictDefault):
        super().__init__(config=config, args=args)
        config.__post_init__()

    def _load_raw_datasets(self):
        """Load and preprocess datasets. Returns raw (train_dataset, val_dataset) with 'messages' column."""
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

            val_dataset = None
            if len(val_datasets) > 0:
                val_dataset = concat_datasets(val_datasets)
                val_dataset = shuffle_dataset(
                    val_dataset, seed=get_seed(eval_seed), buffer_size=1000)

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
            train_dataset, _ = self._load_raw_datasets()
            native_tok = NativeTokenizer.from_pretrained(self.config.model_dir)
            loss_strategy = getattr(self.config, 'loss_scale', 'default')
            logger.info("Debug: printing labels for first 5 train dataset rows")
            count = 0
            for batch in train_dataset.iter(batch_size=1):
                if count >= 5:
                    break
                msgs = batch['messages'][0]
                result = native_tok.encode_for_training(msgs, strategy=loss_strategy)
                if result is not None:
                    debug_labels(result['input_ids'], result['labels'], native_tok)
                    count += 1
            return

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

        # Load, encode and write datasets using the native C++ tokenizer
        train_dataset, val_dataset = self._load_raw_datasets()
        self._encode_and_write_native(train_dataset, val_dataset)

        # Write the hash after successful tokenization
        write_tokenize_hash(self.config.output_dir, current_hash)
        logger.info(f"Tokenization complete. Hash saved: {current_hash}")

    def _encode_and_write_native(self, train_dataset, val_dataset):
        """Encode and write datasets using the fast C++ tokenizer.

        Single-pass pipeline: encode → numpy → pack → write.
        No intermediate HfDataset, no double iteration.
        """
        import time
        native_tok = NativeTokenizer.from_pretrained(self.config.model_dir)
        loss_strategy = getattr(self.config, 'loss_scale', 'default')

        seq_len = self.config.sequence_len or self.config.max_model_len
        vocab_size = self.config.tokenizer.vocab_size
        pad_token_id = self.config.tokenizer.pad_token_id if self.config.tokenizer.pad_token_id is not None else 0
        max_tokens_per_file = DEFAULT_MAX_TOKENS_PER_FILE

        for dataset, split_name, packing in [
            (train_dataset, 'train', self.config.sample_packing),
            (val_dataset, 'validation', False),
        ]:
            if dataset is None:
                continue

            t0 = time.perf_counter()
            all_tokens, all_masks, doc_lengths = _encode_and_prepare_native(
                native_tok, dataset, loss_strategy=loss_strategy,
                batch_size=10000, desc=f"Encoding {split_name}",
            )
            t1 = time.perf_counter()
            n = len(all_tokens)
            logger.info(f"Encoded {split_name}: {n} examples in {t1-t0:.2f}s ({n/(t1-t0):.0f} examples/s)")

            if n == 0:
                continue

            out_dir = self.config.output_dir
            name_prefix = 'train' if split_name == 'train' else 'eval'
            non_overlapping = not packing

            if packing:
                self._write_packed_vectorized(
                    all_tokens, all_masks, doc_lengths,
                    out_dir, name_prefix, vocab_size, seq_len,
                    pad_token_id, max_tokens_per_file, non_overlapping,
                )
            else:
                self._write_padded_vectorized(
                    all_tokens, all_masks, doc_lengths,
                    out_dir, name_prefix, vocab_size, seq_len,
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