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

import hashlib
import json
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
    ref: np.ndarray | None = None  # float32 [2*n_pairs, max_len] reference logprobs (filled by precompute)

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


def _diff_span(chosen_ids: list[int], rejected_ids: list[int]) -> tuple[int, int, int, int] | None:
    """Longest-common-prefix/suffix diff of two token lists.

    Returns (c_start, c_end, r_start, r_end) — the half-open differing spans in
    each list — or None if the lists are identical. The common suffix is bounded
    so the spans never underrun the prefix (all-diff worst case keeps full spans).
    """
    if chosen_ids == rejected_ids:
        return None
    lcp = 0
    for a, b in zip(chosen_ids, rejected_ids):
        if a != b:
            break
        lcp += 1
    lcs = 0
    max_lcs = min(len(chosen_ids), len(rejected_ids)) - lcp
    while lcs < max_lcs and chosen_ids[-1 - lcs] == rejected_ids[-1 - lcs]:
        lcs += 1
    return lcp, len(chosen_ids) - lcs, lcp, len(rejected_ids) - lcs


def tokenize_preference_pairs(
    rows: list[dict], tok, max_len: int, pad_id: int | None = None, span_mask: bool = False
) -> PrefBatch:
    """Tokenize {prompt, chosen, rejected} rows into a one-sequence-per-row PrefBatch.

    Rows whose prompt+response cannot fit a single response token in `max_len`
    are dropped (logged). Raises ValueError if no pair survives.

    span_mask=True confines loss_mask to the tokens where chosen and rejected
    DIFFER (common prefix and suffix excluded). For minimal word-substitution
    pairs this makes the DPO gradient structurally unable to shift style,
    length, or language of the surrounding text — only the substituted form is
    scored. Identical-tokenization rows are dropped (no contrastive signal).
    """
    if pad_id is None:
        pad_id = tok.pad_token_id
        if pad_id is None:
            pad_id = tok.eos_token_id
        if pad_id is None:
            raise ValueError("tokenizer has neither pad_token_id nor eos_token_id; pass pad_id explicitly")

    seqs: list[tuple[list[int], int]] = []  # (ids, prompt_len), chosen then rejected per kept pair
    spans: list[tuple[int, int] | None] = []  # per-sequence (start, end) into the RESPONSE, or None = all
    dropped = 0
    dropped_identical = 0
    for row in rows:
        prompt = row["prompt"]
        laid = []
        resp_ids = []
        for key in ("chosen", "rejected"):
            p_ids, r_ids = _render(tok, prompt, row[key])
            res = _layout_sequence(p_ids, r_ids, max_len, pad_id)
            if res is None:
                laid = []
                break
            laid.append(res)
            resp_ids.append(r_ids)
        if len(laid) != 2:
            dropped += 1
            continue
        if span_mask:
            d = _diff_span(resp_ids[0], resp_ids[1])
            if d is None:
                dropped_identical += 1
                continue
            # Adjust spans for any response-head tokens lost to left-truncation.
            adj = []
            for (ids, p_len), r_full, (s, e) in zip(laid, resp_ids, [d[:2], d[2:]]):
                head_dropped = len(r_full) - (len(ids) - p_len)
                s2, e2 = max(0, s - head_dropped), e - head_dropped
                if e2 <= s2:  # differing span truncated away — no contrastive signal
                    adj = []
                    break
                adj.append((s2, e2))
            if len(adj) != 2:
                dropped += 1
                continue
            spans.extend(adj)
        else:
            spans.extend([None, None])
        seqs.extend(laid)

    n_pairs = len(seqs) // 2
    if n_pairs == 0:
        raise ValueError("tokenize_preference_pairs: no preference pair fit in max_len")
    if dropped:
        logger.warning("tokenize_preference_pairs: dropped %d row(s) that did not fit max_len=%d", dropped, max_len)
    if dropped_identical:
        logger.warning(
            "tokenize_preference_pairs: dropped %d row(s) with identical chosen/rejected tokenization "
            "(span_mask needs a differing span)",
            dropped_identical,
        )

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
        span = spans[k]
        if span is None:
            loss_mask[k, prompt_len:L] = 1  # response tokens (incl. the last) are scored
        else:
            a = min(prompt_len + span[0], L - 1)
            b = min(prompt_len + span[1], L)
            loss_mask[k, a : max(b, a + 1)] = 1  # only the differing span is scored
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


# --- Reference log-prob precompute + sidecar cache --------------------------
# DPO needs the frozen start model's per-token log-probs for both chosen and
# rejected. They depend only on (start model, tokenization layout), so we compute
# them ONCE and cache to a sidecar. The cache key must change whenever anything
# upstream changes (model, max_len, prompt rendering, or the rows themselves) so a
# stale sidecar can never silently feed wrong references into training.

RENDER_VERSION = 1  # bump if _render / _layout_sequence semantics change


def rows_digest(rows: list[dict]) -> str:
    """Stable digest of the preference rows (order-sensitive)."""
    h = hashlib.sha256()
    for r in rows:
        h.update(
            json.dumps(
                [r.get("prompt"), r.get("chosen"), r.get("rejected")], ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        )
    return h.hexdigest()[:16]


def sidecar_hash(
    *, model: str, max_len: int, n_rows: int, rows_digest: str = "", render_version: int = RENDER_VERSION
) -> str:
    """Cache key for a reference-logprob sidecar."""
    payload = {
        "model": model,
        "max_len": max_len,
        "n_rows": n_rows,
        "rows_digest": rows_digest,
        "render_version": render_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def precompute_ref_logprobs(trainer, batch: PrefBatch, engine_b: int, host_rows: int = 1) -> np.ndarray:
    """Reference per-token log-probs of the frozen start checkpoint, in the shifted
    layout (ref[k, j] = logprob of token j+1).

    WARNING — batch-invariant recipes only (e.g. bf16). NOT used by the DPO trainer,
    which computes the reference INLINE per micro-step (see surogate/dpo/trainer.py).
    Offline precompute is INCORRECT under fp8-hybrid: fp8 derives the activation scale
    from the current batch's amax, so these fixed sequential chunks scale each pair
    differently from the training loop's reshuffled batches → a spurious nonzero margin
    at init. Kept only for batch-invariant (bf16) experiments / debugging.

    Uses `trainer.compute_ref_logprobs_dpo` — the SAME fused-loss forward that
    `step_dpo_native` runs, with no backward. Must be called BEFORE any optimizer step:
    LoRA B is zero-initialised, so at init the adapter contributes nothing and the
    forward equals the start checkpoint = π_ref. Rows are fed in fixed
    `[host_rows * engine_b, max_len]` chunks (one B-block per data-parallel rank).
    """
    n_seq, max_len = batch.input_ids.shape
    chunk = engine_b * max(1, host_rows)
    ref = np.zeros((n_seq, max_len), dtype=np.float32)
    for start in range(0, n_seq, chunk):
        end = min(start + chunk, n_seq)
        rows = end - start
        ids = np.zeros((chunk, max_len), dtype=np.int32)
        tgt = np.zeros((chunk, max_len), dtype=np.int32)
        pos = np.zeros((chunk, max_len), dtype=np.int32)
        ids[:rows] = batch.input_ids[start:end]
        tgt[:rows] = batch.targets[start:end]
        pos[:rows] = batch.position_ids[start:end]
        out = trainer.compute_ref_logprobs_dpo(ids, tgt, position_ids=pos)
        ref[start:end] = np.asarray(out, dtype=np.float32)[:rows]
    return ref


def save_ref_sidecar(path: str, key: str, ref: np.ndarray) -> None:
    np.savez(path, key=np.array(key), ref=ref.astype(np.float32))


def load_ref_sidecar(path: str, expected_key: str) -> np.ndarray | None:
    """Return cached reference logprobs if the sidecar matches `expected_key`,
    else None (caller recomputes). Returns None on any read error."""
    import os

    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        if str(data["key"]) != expected_key:
            logger.warning("DPO ref sidecar key mismatch (%s); recomputing", path)
            return None
        return data["ref"]
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("DPO ref sidecar unreadable (%s: %s); recomputing", path, exc)
        return None
