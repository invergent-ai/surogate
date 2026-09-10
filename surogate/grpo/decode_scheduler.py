"""Schedule token steps and reusable prompt state on one resident trainer."""

import itertools
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np


class DecodeCapacityError(RuntimeError):
    """A request exceeds the configured cache token or VRAM budget."""


@dataclass(eq=False)
class Submission:
    session: int
    tokens: np.ndarray
    reset: bool
    future: Future
    sampling: dict | None
    prefill: bool
    key: bytes = b""
    offset: int = 0
    started: bool = False
    chunk_limit: int = 0


class DecodeScheduler:
    def __init__(self, trainer, *, max_batch, prefill_chunk, token_budget, prefix_entries=32):
        self.trainer = trainer
        self.max_batch = max_batch
        self.max_tokens = trainer.batch_size * trainer.seq_length
        self.prefill_chunk = min(prefill_chunk, self.max_tokens)
        self.token_budget = token_budget
        if max_batch <= 0 or self.prefill_chunk <= 0 or token_budget <= 0 or prefix_entries < 0:
            raise ValueError("decode scheduler capacities must be positive (prefix_entries may be zero)")
        self.prefix_entries = prefix_entries if hasattr(trainer, "cache_decode_prefix") else 0
        self.prefixes = OrderedDict()
        self.completed_prefixes = set()
        self.histories = {}
        self.prefill_owners = {}
        self.condition = threading.Condition()
        self.pending = deque()
        self.releases = deque()
        self.ids = itertools.count(1)
        self.lengths = {}
        self.decoding = set()
        self.closed = False
        self.decode_streak = 0
        self.rounds = self.batched_rounds = self.max_observed_batch = 0
        self.prefill_tokens = self.prefix_hits = self.prefix_tokens = self.prefill_splits = 0
        self.completed_saves = self.completed_hits = 0
        self.thread = threading.Thread(target=self._run, name="shared-model-decode", daemon=True)
        self.thread.start()

    def new_session(self):
        with self.condition:
            if self.closed:
                raise RuntimeError("decode scheduler is closed")
            return next(self.ids)

    def step(self, session, tokens, reset=False, sampling=None):
        tokens = np.asarray(tokens, dtype=np.int32)
        if tokens.ndim != 1 or not tokens.size:
            raise ValueError("decode input must be a nonempty token vector")
        future = Future()
        request = Submission(session, tokens, reset, future, sampling, reset or len(tokens) > 1,
                             tokens.tobytes() if reset and self.prefix_entries and tokens.size > 1 else b"")
        with self.condition:
            if self.closed:
                raise RuntimeError("decode scheduler is closed")
            self.pending.append(request)
            self.condition.notify()
        return future.result()

    def release(self, session, *, cache=False):
        future = Future()
        with self.condition:
            if self.closed:
                return
            self.releases.append((session, future, cache))
            self.condition.notify()
        future.result()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()

    def invalidate_prefixes(self):
        """Called after draining requests and resetting native state for training."""
        with self.condition:
            if self.lengths or self.pending:
                raise RuntimeError("prefix invalidation requires drained decode requests")
            self.prefixes.clear()
            self.completed_prefixes.clear()
            self.histories.clear()
            self.prefill_owners.clear()
            self.decoding.clear()

    def summary(self):
        with self.condition:
            return dict(decode_rounds=self.rounds, batched_decode_rounds=self.batched_rounds,
                        max_decode_batch=self.max_observed_batch, prefill_tokens=self.prefill_tokens,
                        prefix_cache_hits=self.prefix_hits, prefix_cached_tokens=self.prefix_tokens,
                        completed_turn_cache_saves=self.completed_saves, completed_turn_cache_hits=self.completed_hits,
                        prefill_chunk_splits=self.prefill_splits)

    def _take_batch(self):
        def ready(item):
            return not item.prefill or (item.key and item.key[:-4] in self.prefixes)

        decodes = [item for item in self.pending if ready(item)]
        prefills = [item for item in self.pending if not ready(item) and
                    (not item.key or self.prefill_owners.get(item.key, item.session) == item.session)]
        # Bound decode bursts so sustained generation cannot starve new prompts,
        # including when max_batch is one.
        if decodes and (not prefills or self.decode_streak < 8):
            candidates, quota = decodes, self.max_tokens
            decode_round = True
            self.decode_streak += 1
        else:
            candidates = prefills
            decode_round = False
            quota = max(1, self.prefill_chunk // (4 if self.decoding else 1))
            self.decode_streak = 0
        if not candidates:
            return []
        sampled = candidates[0].sampling is not None
        selected, keys = [], set()
        for item in candidates:
            if (item.sampling is not None) != sampled or (not decode_round and item.key and item.key in keys):
                continue
            selected.append(item)
            if item.key and not decode_round:
                keys.add(item.key)
                self.prefill_owners[item.key] = item.session
            if len(selected) == min(self.max_batch, quota):
                break
        # Divide prefill work among prompts, reducing chunk size during decode.
        # Power-of-two chunks limit the number of compiled execution shapes.
        chunk = min(self.prefill_chunk, max(1, quota // len(selected))) if not decode_round else 1
        if not decode_round:
            chunk = 1 << (chunk.bit_length() - 1)
        for item in selected:
            self.pending.remove(item)
        return [(item, chunk) for item in selected]

    def _restore(self, item):
        if item.started:
            return
        item.started = True
        if item.reset and item.session in self.lengths:
            self.trainer.release_decode_sessions([item.session])
            self.lengths.pop(item.session)
            self.decoding.discard(item.session)
        if item.reset and self.prefix_entries:
            self.histories[item.session] = bytearray()
        if not item.key:
            return
        # Execute the final prompt token with this request's parameters and RNG;
        # cached state never stores another request's sampling draw.
        for key in sorted(self.prefixes, key=len, reverse=True):
            if len(key) >= len(item.key) or not item.key.startswith(key):
                continue
            length = len(key) // 4
            if sum(self.lengths.values()) + length >= self.token_budget:
                continue
            prefix = self.prefixes[key]
            if self.trainer.restore_decode_prefix(prefix, item.session):
                item.offset, item.reset = length, False
                self.lengths[item.session] = length
                self.histories[item.session] = bytearray(key)
                self.prefixes.move_to_end(key)
                self.prefix_hits += 1
                self.completed_hits += key in self.completed_prefixes
                self.prefix_tokens += length
                return
            self.prefixes.pop(key)
            self.completed_prefixes.discard(key)
            self.trainer.release_decode_prefixes([prefix])

    def _cache(self, item):
        if not item.key or item.offset == len(item.tokens):
            return
        key = item.key[:item.offset * 4]
        self._cache_prefix(item.session, key)

    def _cache_prefix(self, session, key, *, completed=False):
        if not self.prefix_entries or not key:
            return
        if key in self.prefixes:
            self.prefixes.move_to_end(key)
            if completed:
                self.completed_prefixes.add(key)
            return
        while len(self.prefixes) >= self.prefix_entries:
            old_key, prefix = self.prefixes.popitem(last=False)
            self.completed_prefixes.discard(old_key)
            self.trainer.release_decode_prefixes([prefix])
        prefix = next(self.ids)
        if self.trainer.cache_decode_prefix(session, prefix):
            self.prefixes[key] = prefix
            if completed:
                self.completed_prefixes.add(key)
                self.completed_saves += 1

    def _forget(self, item):
        if item.key and self.prefill_owners.get(item.key) == item.session:
            self.prefill_owners.pop(item.key)

    def _fail(self, item, error):
        self.trainer.release_decode_sessions([item.session])
        self.lengths.pop(item.session, None)
        self.histories.pop(item.session, None)
        self.decoding.discard(item.session)
        self._forget(item)
        if not item.future.done():
            item.future.set_exception(error)

    @staticmethod
    def _sampling(item, count):
        return {"enabled": False} if item.sampling is not None and item.offset + count < len(item.tokens) else item.sampling

    def _execute(self, selected):
        batch = []
        try:
            total = sum(self.lengths.values())
            for item, chunk in selected:
                before = self.lengths.get(item.session, 0)
                self._restore(item)
                total += self.lengths.get(item.session, 0) - before
                remaining = len(item.tokens) - item.offset
                count = min(chunk, item.chunk_limit or self.prefill_chunk, remaining,
                            max(0, self.token_budget - total))
                if item.key and remaining > 1:
                    count = min(count, remaining - 1)
                if not count:
                    total -= self.lengths.get(item.session, 0)
                    self._fail(item, DecodeCapacityError("shared decode token or VRAM budget exhausted"))
                    continue
                total += count
                batch.append([item, count])
            if hasattr(self.trainer, "admit_decode_sessions"):
                checking = list(batch)
                while checking:
                    sampled = checking[0][0].sampling is not None
                    kwargs = {"sampling": [self._sampling(item, n) for item, n in checking]} if sampled else {}
                    admitted = self.trainer.admit_decode_sessions(
                        np.asarray([item.session for item, _ in checking], dtype=np.int64),
                        np.asarray([n for _, n in checking], dtype=np.int32),
                        np.asarray([item.reset and item.offset == 0 for item, _ in checking], dtype=np.int32), **kwargs)
                    retry = []
                    for entry, fits in zip(checking, admitted, strict=True):
                        item, n = entry
                        if fits:
                            continue
                        if item.prefill and n > 1:
                            entry[1] = max(1, n // 2)
                            item.chunk_limit = entry[1]
                            self.prefill_splits += 1
                            retry.append(entry)
                        else:
                            batch.remove(entry)
                            self._fail(item, DecodeCapacityError("shared decode token or VRAM budget exhausted"))
                    checking = retry
            if not batch:
                return
            offsets = np.cumsum([0] + [n for _, n in batch], dtype=np.int32)
            args = (np.asarray([item.session for item, _ in batch], dtype=np.int64),
                    np.concatenate([item.tokens[item.offset:item.offset + n] for item, n in batch]), offsets,
                    np.asarray([item.reset and item.offset == 0 for item, _ in batch], dtype=np.int32))
            results = (self.trainer.decode_batch_sample(*args, [self._sampling(item, n) for item, n in batch])
                       if batch[0][0].sampling is not None else self.trainer.decode_batch_logits(*args))
            self.rounds += 1
            self.batched_rounds += len(batch) > 1
            self.max_observed_batch = max(self.max_observed_batch, len(batch))
            for item, n in batch:
                self.lengths[item.session] = self.lengths.get(item.session, 0) + n
                if self.prefix_entries:
                    self.histories[item.session].extend(item.tokens[item.offset:item.offset + n].tobytes())
                item.offset += n
                self.prefill_tokens += n if item.prefill else 0
                self._cache(item)
            # Finish cache operations before waking callers, so an execution
            # error cannot race a successful caller submitting its next token.
            for row, (item, _) in enumerate(batch):
                if item.offset == len(item.tokens):
                    self.decoding.add(item.session)
                    self._forget(item)
                    item.future.set_result(results[row])
                else:
                    with self.condition:
                        self.pending.append(item)
        except Exception as exc:
            with self.condition:
                failed = {item for item, _ in selected}
                self.pending = deque(item for item in self.pending if item not in failed)
            for item, _ in selected:
                try:
                    self._fail(item, exc)
                except Exception:
                    if not item.future.done():
                        item.future.set_exception(exc)

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending or self.releases)
                if not self.closed and self.pending and not self.releases:
                    deadline = time.monotonic() + 0.001
                    while len(self.pending) < self.max_batch and not self.releases and not self.closed:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.condition.wait(remaining)
                stopping = self.closed
                releases = list(self.releases)
                self.releases.clear()
                if stopping:
                    for item in self.pending:
                        item.future.set_exception(RuntimeError("decode scheduler is closed"))
                    self.pending.clear()
                    batch = []
                else:
                    batch = self._take_batch()
            if releases or stopping:
                try:
                    ids = list(self.lengths) if stopping else [session for session, _, _ in releases]
                    if not stopping:
                        for session, _, cache in releases:
                            if cache and session in self.histories:
                                self._cache_prefix(session, bytes(self.histories[session]), completed=True)
                    self.trainer.release_decode_sessions(ids)
                    for session in ids:
                        self.lengths.pop(session, None)
                        self.histories.pop(session, None)
                        self.decoding.discard(session)
                    if stopping and self.prefix_entries:
                        self.trainer.release_decode_prefixes(list(self.prefixes.values()))
                        self.prefixes.clear()
                        self.completed_prefixes.clear()
                    for _, future, _ in releases:
                        future.set_result(None)
                except Exception as exc:
                    # Release remains mandatory if optional caching failed.
                    try:
                        self.trainer.release_decode_sessions(ids)
                    except Exception:
                        pass
                    finally:
                        for session in ids:
                            self.lengths.pop(session, None)
                            self.histories.pop(session, None)
                            self.decoding.discard(session)
                    for _, future, _ in releases:
                        future.set_exception(exc)
            if stopping:
                return
            if batch:
                self._execute(batch)
