"""Concurrent admission, chunking, release, failure and shutdown semantics."""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from surogate.grpo.decode_scheduler import DecodeScheduler


class Trainer:
    batch_size = 2
    seq_length = 128

    def __init__(self):
        self.states = {}
        self.batches = []
        self.fail = False

    def decode_batch_logits(self, ids, tokens, offsets, resets):
        if self.fail:
            raise RuntimeError("injected decode failure")
        self.batches.append(ids.tolist())
        result = []
        for i, session in enumerate(ids):
            if resets[i]:
                self.states[session] = []
            self.states[session].extend(tokens[offsets[i] : offsets[i + 1]].tolist())
            result.append([sum(self.states[session]), len(self.states[session])])
        return np.array(result, dtype=np.float32)

    def release_decode_sessions(self, ids):
        for session in ids:
            self.states.pop(session, None)


@pytest.fixture
def scheduler():
    trainer = Trainer()
    scheduler = DecodeScheduler(trainer, max_batch=4, prefill_chunk=8, token_budget=512)
    yield scheduler, trainer
    scheduler.close()


def test_changing_membership_and_chunked_prefills(scheduler):
    scheduler, trainer = scheduler
    barrier = threading.Barrier(3)

    def run(value, rounds):
        session = scheduler.new_session()
        barrier.wait(timeout=5)
        try:
            # Prompt lengths are deliberately different and exceed one chunk.
            tokens = [value] * (9 + value)
            logits = scheduler.step(session, tokens, True)
            assert logits.tolist() == [sum(tokens), len(tokens)]
            for _ in range(rounds):
                logits = scheduler.step(session, [value])
                tokens.append(value)
                assert logits.tolist() == [sum(tokens), len(tokens)]
        finally:
            scheduler.release(session)

    with ThreadPoolExecutor(3) as pool:
        futures = [pool.submit(run, i, 6 - i) for i in (1, 2, 3)]
        for future in futures:
            future.result(timeout=10)
    assert not trainer.states
    assert any(len(batch) > 1 for batch in trainer.batches)
    assert scheduler.summary()["batched_decode_rounds"] > 0


def test_failure_releases_sessions_and_worker_recovers(scheduler):
    scheduler, trainer = scheduler
    session = scheduler.new_session()
    scheduler.step(session, [2, 3], True)
    trainer.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        scheduler.step(session, [4])
    assert not trainer.states
    trainer.fail = False
    assert scheduler.step(scheduler.new_session(), [5], True).tolist() == [5, 1]


def test_budget_and_shutdown_release_owned_state(scheduler):
    scheduler, trainer = scheduler
    scheduler.token_budget = 3
    session = scheduler.new_session()
    scheduler.step(session, [1, 2], True)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        scheduler.step(session, [3, 4])
    assert not trainer.states
    scheduler.step(scheduler.new_session(), [2], True)
    scheduler.close()
    assert not trainer.states
    with pytest.raises(RuntimeError, match="closed"):
        scheduler.new_session()
