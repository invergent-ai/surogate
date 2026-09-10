"""Batch token steps from independent HTTP workers on one resident trainer.

HTTP workers own request parameters, RNG and streaming. The compute thread owns native
sessions and serializes cache mutations with graph execution. Each worker waits
only for its current chunk, so requests join and leave between decode rounds.
"""

import itertools
import threading
import time
from collections import deque
from concurrent.futures import Future

import numpy as np


class DecodeCapacityError(RuntimeError):
    """A request exceeds the configured cache token or VRAM budget."""


class DecodeScheduler:
    def __init__(self, trainer, *, max_batch, prefill_chunk, token_budget):
        self.trainer = trainer
        self.max_batch = max_batch
        self.max_tokens = trainer.batch_size * trainer.seq_length
        self.prefill_chunk = min(prefill_chunk, self.max_tokens)
        self.token_budget = token_budget
        if max_batch <= 0 or self.prefill_chunk <= 0 or token_budget <= 0:
            raise ValueError("decode scheduler capacities must be positive")
        self.condition = threading.Condition()
        self.pending = deque()
        self.releases = deque()
        self.ids = itertools.count(1)
        self.lengths = {}
        self.closed = False
        self.rounds = self.batched_rounds = self.max_observed_batch = 0
        self.thread = threading.Thread(target=self._run, name="shared-model-decode", daemon=True)
        self.thread.start()

    def new_session(self):
        with self.condition:
            if self.closed:
                raise RuntimeError("decode scheduler is closed")
            return next(self.ids)

    def step(self, session, tokens, reset=False, sampling=None):
        tokens = np.asarray(tokens, dtype=np.int32)
        result = None
        for start in range(0, len(tokens), self.prefill_chunk):
            future = Future()
            chunk = tokens[start : start + self.prefill_chunk]
            params = sampling
            if sampling is not None and start + len(chunk) < len(tokens):
                params = {"enabled": False}
            with self.condition:
                if self.closed:
                    raise RuntimeError("decode scheduler is closed")
                self.pending.append((session, chunk, reset and start == 0, future, params))
                self.condition.notify()
            result = future.result()
        return result

    def release(self, session):
        future = Future()
        with self.condition:
            if self.closed:
                return
            self.releases.append((session, future))
            self.condition.notify()
        future.result()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()

    def summary(self):
        with self.condition:
            return dict(
                decode_rounds=self.rounds,
                batched_decode_rounds=self.batched_rounds,
                max_decode_batch=self.max_observed_batch,
            )

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending or self.releases)
                if self.closed:
                    for _, _, _, future, _ in self.pending:
                        future.set_exception(RuntimeError("decode scheduler is closed"))
                    releases = list(self.releases)
                    self.pending.clear()
                    self.releases.clear()
                    stopping = True
                    batch = []
                else:
                    stopping = False
                    # Briefly coalesce token steps. HTTP streaming and request
                    # preparation cannot hold the compute lock.
                    deadline = time.monotonic() + 0.001
                    while self.pending and len(self.pending) < self.max_batch and not self.releases:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.condition.wait(remaining)
                    releases = list(self.releases)
                    self.releases.clear()
                    batch, tokens = [], 0
                    while self.pending and len(batch) < self.max_batch:
                        item = self.pending[0]
                        if tokens + len(item[1]) > self.max_tokens or (
                            batch and (item[4] is None) != (batch[0][4] is None)
                        ):
                            break
                        batch.append(self.pending.popleft())
                        tokens += len(item[1])
            if releases or stopping:
                try:
                    ids = list(self.lengths) if stopping else [session for session, _ in releases]
                    self.trainer.release_decode_sessions(ids)
                    for session in ids:
                        self.lengths.pop(session, None)
                    for _, future in releases:
                        future.set_result(None)
                except Exception as exc:
                    for _, future in releases:
                        future.set_exception(exc)
            if stopping:
                return
            if not batch:
                continue
            try:
                lengths = self.lengths.copy()
                accepted = []
                rejected = []
                total = sum(lengths.values())
                for item in batch:
                    session, tokens, reset, future, _ = item
                    before = lengths.get(session, 0)
                    after = (0 if reset else before) + len(tokens)
                    if total + after - before > self.token_budget:
                        rejected.append(item)
                        total -= before
                        lengths.pop(session, None)
                        continue
                    total += after - before
                    lengths[session] = after
                    accepted.append(item)
                if accepted and hasattr(self.trainer, "admit_decode_sessions"):
                    admitted = self.trainer.admit_decode_sessions(
                        np.asarray([item[0] for item in accepted], dtype=np.int64),
                        np.asarray([len(item[1]) for item in accepted], dtype=np.int32),
                        np.asarray([item[2] for item in accepted], dtype=np.int32),
                    )
                    rejected.extend(item for item, fits in zip(accepted, admitted, strict=True) if not fits)
                    accepted = [item for item, fits in zip(accepted, admitted, strict=True) if fits]
                for session, _, _, future, _ in rejected:
                    self.trainer.release_decode_sessions([session])
                    self.lengths.pop(session, None)
                    lengths.pop(session, None)
                    future.set_exception(DecodeCapacityError("shared decode token or VRAM budget exhausted"))
                batch = accepted
                if not batch:
                    continue
                offsets = np.cumsum([0] + [len(item[1]) for item in batch], dtype=np.int32)
                args = (
                    np.asarray([item[0] for item in batch], dtype=np.int64),
                    np.concatenate([item[1] for item in batch]),
                    offsets,
                    np.asarray([item[2] for item in batch], dtype=np.int32),
                )
                results = (
                    self.trainer.decode_batch_sample(*args, [item[4] for item in batch])
                    if batch[0][4] is not None
                    else self.trainer.decode_batch_logits(*args)
                )
                self.lengths = lengths
                with self.condition:
                    self.rounds += 1
                    self.batched_rounds += len(batch) > 1
                    self.max_observed_batch = max(self.max_observed_batch, len(batch))
                for row, (_, _, _, future, _) in enumerate(batch):
                    future.set_result(results[row])
            except Exception as exc:
                # The native batch invalidates participating sessions after an
                # execution failure. Release also covers host-side admission errors.
                try:
                    self.trainer.release_decode_sessions([item[0] for item in batch])
                except Exception:
                    pass  # Report the original execution error to every waiting caller.
                for session, _, _, future, _ in batch:
                    self.lengths.pop(session, None)
                    if not future.done():
                        future.set_exception(exc)
