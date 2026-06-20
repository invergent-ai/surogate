"""Preference-pair tokenization for offline DPO.

Each preference row is {prompt, chosen, rejected}. We render prompt+chosen and
prompt+rejected, mask the prompt (loss only on the response continuation), and
lay each sequence out as its own right-padded row of width `max_len`. A
micro-batch of B rows is therefore B/2 atomic pairs (pair k = rows 2k, 2k+1);
the trainer (surogate/dpo/data.py packing → step_dpo_native) keeps both rows of a
pair in the same optimizer step.

Engine conventions (must match step_dpo_native / the dpo_dloss kernel):
- targets[j] = input_ids[j+1]; targets[last] = 0 (unused, masked).
- loss_mask[j] = 1 iff position j holds a scored response token. The logprob of
  token j is read from losses[j-1] (shifted layout), so the FIRST response token
  (position = prompt_len) and the LAST response token are both scored.
- position_ids restart at 0 per row; padding uses position 0 and loss_mask 0.
  Right padding + causal attention means padding never affects real-token logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from surogate.utils.logger import get_logger

logger = get_logger()


@dataclass
class PrefBatch:
    """Tokenized preference pairs, one sequence per row (chosen=2k, rejected=2k+1)."""

    max_len: int
    n_pairs: int
    input_ids: np.ndarray  # int32 [2*n_pairs, max_len]
    targets: np.ndarray  # int32 [2*n_pairs, max_len]
    loss_mask: np.ndarray  # uint8 [2*n_pairs, max_len]
    position_ids: np.ndarray  # int32 [2*n_pairs, max_len]
    seq_len: np.ndarray  # int32 [2*n_pairs] real (unpadded) length per row

    @property
    def n_seq(self) -> int:
        return int(self.input_ids.shape[0])


def _render(tok, prompt: Any, response: str) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, response_ids). `prompt` is a string or a messages list."""
    if isinstance(prompt, list):
        prompt_ids = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    else:
        prompt_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    response_ids = tok(response, add_special_tokens=False)["input_ids"]
    return list(prompt_ids), list(response_ids)


def _layout_sequence(prompt_ids: list[int], response_ids: list[int], max_len: int, pad_id: int):
    """Left-truncate the prompt so the response is preserved; return (ids, prompt_len)
    or None if no response token survives."""
    ids = prompt_ids + response_ids
    prompt_len = len(prompt_ids)
    if len(ids) > max_len:
        drop = len(ids) - max_len
        ids = ids[drop:]
        prompt_len = max(0, prompt_len - drop)
    if prompt_len >= len(ids):  # response fully truncated away
        return None
    return ids, prompt_len


def tokenize_preference_pairs(rows: list[dict], tok, max_len: int, pad_id: int | None = None) -> PrefBatch:
    """Tokenize {prompt, chosen, rejected} rows into a one-sequence-per-row PrefBatch.

    Rows whose prompt+response cannot fit a single response token in `max_len`
    are dropped (logged). Raises ValueError if no pair survives.
    """
    if pad_id is None:
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id; pass pad_id explicitly")

    seqs: list[tuple[list[int], int]] = []  # (ids, prompt_len), chosen then rejected per kept pair
    dropped = 0
    for row in rows:
        prompt = row["prompt"]
        laid = []
        for key in ("chosen", "rejected"):
            p_ids, r_ids = _render(tok, prompt, row[key])
            res = _layout_sequence(p_ids, r_ids, max_len, pad_id)
            if res is None:
                laid = []
                break
            laid.append(res)
        if len(laid) != 2:
            dropped += 1
            continue
        seqs.extend(laid)

    n_pairs = len(seqs) // 2
    if n_pairs == 0:
        raise ValueError("tokenize_preference_pairs: no preference pair fit in max_len")
    if dropped:
        logger.warning("tokenize_preference_pairs: dropped %d row(s) that did not fit max_len=%d", dropped, max_len)

    n_seq = 2 * n_pairs
    input_ids = np.full((n_seq, max_len), pad_id, dtype=np.int32)
    targets = np.zeros((n_seq, max_len), dtype=np.int32)
    loss_mask = np.zeros((n_seq, max_len), dtype=np.uint8)
    position_ids = np.zeros((n_seq, max_len), dtype=np.int32)
    seq_len = np.zeros((n_seq,), dtype=np.int32)

    for k, (ids, prompt_len) in enumerate(seqs):
        L = len(ids)
        input_ids[k, :L] = ids
        # targets[j] = ids[j+1]; targets[L-1..] stay 0 (unused / masked).
        if L >= 2:
            targets[k, : L - 1] = ids[1:L]
        loss_mask[k, prompt_len:L] = 1  # response tokens (incl. the last) are scored
        position_ids[k, :L] = np.arange(L, dtype=np.int32)
        seq_len[k] = L

    return PrefBatch(
        max_len=max_len,
        n_pairs=n_pairs,
        input_ids=input_ids,
        targets=targets,
        loss_mask=loss_mask,
        position_ids=position_ids,
        seq_len=seq_len,
    )
