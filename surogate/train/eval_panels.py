"""Held-out evaluation panels: per-panel and per-row teacher-forced NLL at fixed training steps.

``validation_datasets`` merge every validation set into one shuffled loader and report one loss; a paired
comparison of two training arms needs each panel's loss and each row's NLL at the same steps. A panel is a
jsonl of conversations (the training format: ``messages`` + optional ``tools``; scored on the assistant tokens
exactly as training scores them) or of pre-tokenized rows (``{"id", "ids", "spans"}``: token i is scored for i in
a span, the merge-check format of rl/gate/nll). Every row is one window (positions from 0), truncated to
``sequence_len``, scored with the trainer's per-token log-probabilities (chunk by chunk under sequence_chunks;
chunks after the last scored token are skipped).

Config (SFT)::

    eval_panels:
      - {name: P_in_workplace, path: .../pin_workplace.jsonl, per_token: true}
      - {name: P_ret_mergecheck, path: .../pret_mergecheck.tok.jsonl}
    eval_panels_steps: 4        # step 0 (before the first update), every 4 steps, and after the last step

Writes to <output_dir>:
  eval_panels.jsonl          one line per row and eval: step, panel, id, tokens, nll (mean over its scored tokens)
  eval_panels_summary.jsonl  one line per panel and eval: token-weighted NLL, row-mean NLL, rows, tokens, seconds
                             (with an n-gram memory also its statistics over the eval forwards)
  eval_tokens_step<S>.jsonl.gz  for ``per_token`` panels at step 0 and the final eval: per row the scored input
                             positions j (predicting token j + 1), log p, and the training-split counts of the
                             bigram and trigram ending at j under the memory's context rule (a predecessor before
                             the row start or equal to EOS cuts it and every older one, which read as EOS), counted
                             over every token of the training windows (<output_dir>/train-*.bin) -- the n-grams the
                             memory read while training.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from surogate.utils.logger import get_logger

logger = get_logger()

EOS_DEFAULT = 248044
_SHIFT = np.uint64(18)  # token ids < 2^18


def _row_id(raw: dict[str, Any], index: int) -> str:
    for key in ("id", "row_id", "conversation_id", "task_id"):
        if raw.get(key) is not None:
            return str(raw[key])
    return str(index)


def context_keys(ids: np.ndarray, eos: int) -> tuple[np.ndarray, np.ndarray]:
    """(bigram, trigram) keys of the n-grams ending at every position of one window, the memory's context rule."""
    t = np.asarray(ids, dtype=np.int64)
    n = len(t)
    p1 = np.full(n, -1, dtype=np.int64)
    p2 = np.full(n, -1, dtype=np.int64)
    p1[1:] = t[:-1]
    p2[2:] = t[:-2]
    cut1 = (p1 < 0) | (p1 == eos)
    cut2 = cut1 | (p2 < 0) | (p2 == eos)
    c1 = np.where(cut1, eos, p1).astype(np.uint64)
    c2 = np.where(cut2, eos, p2).astype(np.uint64)
    bi = t.astype(np.uint64) | (c1 << _SHIFT)
    tri = bi | (c2 << (_SHIFT * np.uint64(2)))
    return bi, tri


def read_bin_windows(path: str, eos: int = EOS_DEFAULT) -> list[np.ndarray]:
    """The token windows of a BIN.TOK v3 file (one document per window starting at position 0), trailing padding
    (the vocab-id marker, or EOS -- the pad token -- with a zero mask) stripped."""
    with open(path, "rb") as f:
        head = f.read(1024)
        if head[:8] != b"BIN.TOK\n":
            raise ValueError(f"{path}: not a BIN.TOK file")
        _, _, version, bpt, n, vocab, has_masks, _ = struct.unpack("<8i", head[:32])
        if version != 3 or bpt != 4:
            raise ValueError(f"{path}: BIN.TOK v{version} with {bpt}-byte tokens")
        tok = np.frombuffer(f.read(4 * n), dtype=np.int32)
        pos = np.frombuffer(f.read(4 * n), dtype=np.int32)
        mask = np.unpackbits(np.frombuffer(f.read((n + 7) // 8), dtype=np.uint8), bitorder="little")[:n] \
            if has_masks else np.ones(n, dtype=np.uint8)
    starts = list(np.flatnonzero(pos == 0)) + [n]
    pad = (tok == vocab) | ((tok == eos) & (mask == 0))
    out = []
    for s, e in zip(starts[:-1], starts[1:]):
        real = np.flatnonzero(~pad[s:e])
        if len(real):
            out.append(np.array(tok[s:s + int(real[-1]) + 1]))
    return out


class NgramCounts:
    def __init__(self, windows: list[np.ndarray], eos: int):
        bis, tris = [], []
        for w in windows:
            b, t = context_keys(w, eos)
            bis.append(b)
            tris.append(t)
        bi = np.concatenate(bis) if bis else np.zeros(0, np.uint64)
        tri = np.concatenate(tris) if tris else np.zeros(0, np.uint64)
        self.bi_keys, self.bi_counts = np.unique(bi, return_counts=True)
        self.tri_keys, self.tri_counts = np.unique(tri, return_counts=True)
        self.positions = int(len(bi))

    @staticmethod
    def _lookup(keys, counts, q):
        if len(keys) == 0:
            return np.zeros(len(q), dtype=np.int64)
        i = np.searchsorted(keys, q)
        i = np.minimum(i, len(keys) - 1)
        return np.where(keys[i] == q, counts[i], 0).astype(np.int64)

    def lookup(self, ids: np.ndarray, eos: int):
        b, t = context_keys(ids, eos)
        return self._lookup(self.bi_keys, self.bi_counts, b), self._lookup(self.tri_keys, self.tri_counts, t)


class EvalPanels:
    def __init__(self, config, trainer) -> None:
        self.config = config
        self.trainer = trainer
        self.seq_len = int(config.sequence_len)
        self.every = int(getattr(config, "eval_panels_steps", None) or config.eval_steps or 0)
        out = Path(config.output_dir)
        self.out_rows = out / "eval_panels.jsonl"
        self.out_summary = out / "eval_panels_summary.jsonl"
        self.out_dir = out
        self.done_steps: set[int] = set()
        if int(getattr(config, "gpus", 1) or 1) != 1 or int(config.per_device_train_batch_size) != 1:
            raise ValueError("eval_panels: needs gpus: 1 and per_device_train_batch_size: 1 (one row per window)")
        ngram = getattr(config, "ngram_memory", None) or {}
        self.eos = int((ngram.get("meta") or {}).get("eos_token_id", EOS_DEFAULT))
        self.panels: list[dict[str, Any]] = []
        for spec in config.eval_panels:
            rows = self._load(spec)
            self.panels.append({"name": str(spec["name"]), "rows": rows, "per_token": bool(spec.get("per_token"))})
            logger.info(f"[eval_panels] {spec['name']}: {len(rows)} rows, {sum(int(m.sum()) for _, _, m in rows):,} "
                        f"scored tokens from {spec['path']}")
        self.counts: NgramCounts | None = None
        if any(p["per_token"] for p in self.panels):
            files = sorted(glob.glob(os.path.join(config.output_dir, "train-*.bin")))
            if not files:
                raise ValueError(f"eval_panels: per_token needs the training windows ({config.output_dir}/train-*.bin)")
            t0 = time.time()
            windows = [w for f in files for w in read_bin_windows(f, self.eos)]
            self.counts = NgramCounts(windows, self.eos)
            logger.info(f"[eval_panels] training n-gram counts: {len(windows)} windows, {self.counts.positions:,} "
                        f"positions, {len(self.counts.bi_keys):,} bigrams, {len(self.counts.tri_keys):,} trigrams "
                        f"({time.time() - t0:.1f} s)")

    # -------------------------------------------------------------------------------------------- loading
    def _load(self, spec: dict[str, Any]):
        path = str(spec["path"])
        raws = [json.loads(line) for line in open(path) if line.strip()]
        if spec.get("max_rows"):
            raws = raws[: int(spec["max_rows"])]
        if not raws:
            raise ValueError(f"eval_panels: {path} holds no rows")
        if "ids" in raws[0]:
            out = []
            for k, raw in enumerate(raws):
                ids = np.asarray(raw["ids"], dtype=np.int32)
                mask = np.zeros(len(ids), dtype=bool)
                for a, b in raw.get("spans") or [[1, len(ids)]]:
                    mask[int(a):int(b)] = True
                mask[0] = False
                out.append((_row_id(raw, k), ids, mask))
            return out
        return self._encode_conversations(spec, raws)

    def _encode_conversations(self, spec, raws):
        """The training tokenization (dataset config -> pre_process -> native encoder), row order kept."""
        from surogate._surogate import Tokenizer as NativeTokenizer
        from surogate.core.config.dataset_config import create_dataset_config
        from surogate.core.datasets.loader import load_dataset_with_config, pre_process
        from surogate.train.tokenize import _conversation_json, _tool_jsons
        from surogate.utils.dict import DictDefault

        ds_cfg = create_dataset_config(DictDefault({"path": str(spec["path"]), "type": "conversation",
                                                    "messages_field": spec.get("messages_field", "messages"),
                                                    "tools_field": spec.get("tools_field", "tools")}))
        dataset = pre_process(load_dataset_with_config(ds_cfg, num_workers=1), ds_cfg, num_proc=1)
        if spec.get("max_rows"):
            dataset = dataset.select(range(min(len(dataset), int(spec["max_rows"]))))
        if len(dataset) != len(raws):
            raise ValueError(f"eval_panels: {spec['path']}: {len(raws)} rows read, {len(dataset)} after pre-processing")
        tok = NativeTokenizer.from_pretrained(self.config.model_dir)
        strategy = getattr(self.config, "loss_scale", "default")
        out = []
        for k, row in enumerate(dataset):
            [res] = tok.encode_for_training_json_batch([_conversation_json(row["messages"])],
                                                       [_tool_jsons(row.get("tools"))], strategy=strategy)
            if res is None or len(res["input_ids"]) == 0:
                raise ValueError(f"eval_panels: {spec['path']} row {k} does not encode")
            ids = np.asarray(res["input_ids"], dtype=np.int32)
            mask = np.asarray(res["labels"], dtype=np.int32) != -100
            mask[0] = False
            out.append((_row_id(raws[k], k), ids, mask))
        return out

    # -------------------------------------------------------------------------------------------- running
    def due(self, step: int, start_step: int) -> bool:
        if step in self.done_steps:
            return False
        if step == 0:
            return True
        return self.every > 0 and step % self.every == 0 and step > start_step

    def _xy(self, ids: np.ndarray, mask: np.ndarray):
        n = min(len(ids), self.seq_len)
        x = np.zeros((1, self.seq_len), dtype=np.int32)
        y = np.full((1, self.seq_len), -100, dtype=np.int32)
        x[0, :n] = ids[:n]
        y[0, : n - 1] = np.where(mask[1:n], ids[1:n], -100)  # y[j] = ids[j + 1] where token j + 1 is scored
        return x, y

    def _memory_stats(self) -> dict[str, Any] | None:
        if not getattr(self.config, "ngram_memory", None):
            return None
        try:
            return dict(self.trainer.get_ngram_memory_stats(0, True))
        except Exception:
            return None

    def run(self, step: int, final: bool = False) -> dict[str, float]:
        self.done_steps.add(step)
        pending = self._memory_stats()  # training forwards since the last log (kept out of the eval's)
        if pending and pending.get("tokens"):
            logger.info(f"[eval_panels] memory statistics before the eval: {json.dumps(pending)}")
        dump_tokens = self.counts is not None and (step == 0 or final)
        tok_out = gzip.open(self.out_dir / f"eval_tokens_step{step}.jsonl.gz", "wt") if dump_tokens else None
        result = {}
        self.out_rows.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.out_rows, "a") as rows_out, open(self.out_summary, "a") as sum_out:
                for panel in self.panels:
                    name = panel["name"]
                    t0 = time.time()
                    nll_sum = 0.0
                    tokens = 0
                    row_nlls = []
                    for rid, ids, mask in panel["rows"]:
                        x, y = self._xy(ids, mask)
                        scored = np.flatnonzero(y[0] != -100)
                        if len(scored) == 0:
                            continue
                        lp = np.asarray(self.trainer.compute_logprobs(x, y, use_lora=True)[0], dtype=np.float64)
                        token_lp = lp[scored]
                        nll = float(-token_lp.mean())
                        n = int(len(scored))
                        nll_sum += nll * n
                        tokens += n
                        row_nlls.append(nll)
                        rows_out.write(json.dumps({"step": step, "panel": name, "id": rid, "tokens": n,
                                                   "nll": nll}) + "\n")
                        if tok_out is not None and panel["per_token"]:
                            bi, tri = self.counts.lookup(x[0, : min(len(ids), self.seq_len)], self.eos)
                            tok_out.write(json.dumps({"step": step, "panel": name, "id": rid,
                                                      "pos": scored.tolist(),
                                                      "lp": [round(float(v), 6) for v in token_lp],
                                                      "bi": bi[scored].tolist(), "tri": tri[scored].tolist()}) + "\n")
                    rows_out.flush()
                    summary = {"step": step, "panel": name, "nll": nll_sum / max(tokens, 1),
                               "row_mean_nll": float(np.mean(row_nlls)) if row_nlls else 0.0, "rows": len(row_nlls),
                               "tokens": tokens, "seconds": round(time.time() - t0, 1)}
                    stats = self._memory_stats()
                    if stats:
                        summary["memory"] = stats
                    sum_out.write(json.dumps(summary) + "\n")
                    sum_out.flush()
                    result[name] = summary["nll"]
                    logger.info(f"[eval_panels] step {step} {name}: NLL {summary['nll']:.5f} over {tokens:,} tokens "
                                f"({len(row_nlls)} rows, {summary['seconds']} s)")
        finally:
            if tok_out is not None:
                tok_out.close()
        return result
