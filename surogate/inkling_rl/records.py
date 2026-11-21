"""Rollout records: the kit's TrainingBatch (rollouts/step_<N>/rollouts.bin) -> per-row arrays the step needs.

A TrainingSample (kit loop eff_samples.to_training_sample) is ONE row: prompt_ids (the first prompt, no loss) +
completion_ids (every later token: sampled spans mask 1 with the engine's log-probs, observation / re-rendered prompt
tokens mask 0 with log-prob 0.0), advantage, per-token policy versions (-1 where the model did not sample), and the
engine's routing per position: int16 little-endian, C order [len(prompt) + len(completion), layers, top_k], -1 rows where
nothing was captured (the last sampled token never has a row). For Inkling layers = 48 VISITS (engine capture per visit,
serve-inkling targets/inkling2/impl/config.h, api/ops/inkling_moe.h).

Token ids are the ENGINE's rendered ids (prompt_token_ids of each call): rule 20 (the trainer sees the exact effort line
the engine sampled with) holds because nothing here renders or re-tokenises text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import TOP_K, VISITS
from .kitsrc import types_api


@dataclass
class Row:
    ids: np.ndarray                 # int64 [T]: prompt + completion
    loss_mask: np.ndarray           # bool [T]
    engine_lp: np.ndarray           # float64 [T]: the engine's log-prob of ids[t] on sampled tokens, 0 elsewhere
    advantage: float
    versions: np.ndarray | None     # int32 [T]: policy version per token, -1 unsampled; None = not collected
    teacher_lp: np.ndarray | None   # float64 [T] or None
    routed: np.ndarray | None       # int16 [T, R, K] or None
    trial_idx: int | None = None
    group_idx: int | None = None
    notes: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.ids.shape[0])


def decode_batch(data: bytes):
    import msgspec

    _, TrainingBatch = types_api()
    return msgspec.msgpack.decode(data, type=TrainingBatch)


def encode_batch(batch) -> bytes:
    import msgspec

    return msgspec.msgpack.encode(batch)


def row_from_sample(s) -> Row:
    P, C = len(s.prompt_ids), len(s.completion_ids)
    ids = np.asarray(list(s.prompt_ids) + list(s.completion_ids), dtype=np.int64)
    mask = np.zeros(P + C, dtype=bool)
    mask[P:] = np.asarray(s.completion_mask, dtype=bool)
    lp = np.zeros(P + C, dtype=np.float64)
    lp[P:] = np.asarray(s.completion_logprobs, dtype=np.float64)
    versions = None
    if s.completion_versions is not None:
        versions = np.full(P + C, -1, dtype=np.int32)
        versions[P:] = np.asarray(s.completion_versions, dtype=np.int32)
    teacher = None
    if s.teacher_logprobs is not None:
        tl = np.asarray(s.teacher_logprobs, dtype=np.float64)
        if tl.size != P + C:                      # the kit trainer's batch.py:93-94 asserts len == len(input_ids)
            raise ValueError(f"teacher_logprobs has {tl.size} entries for {P + C} tokens")
        teacher = tl
    routed, notes = None, {}
    if s.routed_experts is not None and s.routed_experts_dims is not None:
        R, K = (int(x) for x in s.routed_experts_dims)
        raw = np.frombuffer(s.routed_experts, dtype="<i2")
        if raw.size != (P + C) * R * K:
            notes["r3_bad_row"] = f"routing bytes {raw.size} != tokens {P + C} x {R} x {K}"
        else:
            routed = raw.reshape(P + C, R, K)
    return Row(ids=ids, loss_mask=mask, engine_lp=lp, advantage=float(s.advantage or 0.0), versions=versions,
               teacher_lp=teacher, routed=routed, trial_idx=s.trial_idx, group_idx=s.group_idx, notes=notes)


def load_rows(path: str) -> tuple[int, list[Row]]:
    batch = decode_batch(open(path, "rb").read())
    return int(batch.step), [row_from_sample(s) for s in batch.examples]


@dataclass
class Replay:
    ids: np.ndarray        # int64 [T, VISITS, TOP_K]; uncovered positions filled with a valid placeholder (see covered)
    covered: np.ndarray    # bool [T]: the engine captured this position's route


def replay_rows(row: Row, visits: int = VISITS, top_k: int = TOP_K) -> tuple[Replay | None, dict]:
    """The row's routing as replay input, and the kit's R3 counters for it (trainer r3-py.diff semantics):
    r3_shape_mismatches: dims != [visits, top_k]; r3_bad_rows: unusable bytes; r3_replayed_tokens: covered positions;
    r3_rows: rows replayed; r3_uncovered_interior: uncovered positions before the last loss target (they run the
    trainer's own routing in the kit's native replay; a backend that cannot mix must refuse the row)."""
    c = {"r3_rows": 0.0, "r3_replayed_tokens": 0.0, "r3_bad_rows": 0.0, "r3_shape_mismatches": 0.0,
         "r3_uncovered_interior": 0.0}
    if "r3_bad_row" in row.notes:
        c["r3_bad_rows"] = 1.0
        return None, c
    if row.routed is None:
        return None, c
    T, R, K = row.routed.shape
    if (R, K) != (visits, top_k):
        c["r3_shape_mismatches"] = 1.0
        return None, c
    covered = row.routed[:, 0, 0] >= 0
    ids = row.routed.astype(np.int64)
    # a position's route only feeds the logits of LATER tokens: positions after the last loss target never matter
    last = int(np.flatnonzero(row.loss_mask)[-1]) if row.loss_mask.any() else 0
    c["r3_uncovered_interior"] = float((~covered[:last]).sum())
    fill = np.arange(top_k, dtype=np.int64)          # placeholder ids for uncovered positions (never a loss input
    ids[~covered] = fill                             # when r3_uncovered_interior == 0)
    c["r3_rows"] = 1.0
    c["r3_replayed_tokens"] = float(covered.sum())
    return Replay(ids=ids, covered=covered), c
