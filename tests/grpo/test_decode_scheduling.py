"""Decode latency, prompt fairness, adaptive admission and shared prefix semantics."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from surogate.grpo.decode_scheduler import DecodeScheduler
from tests.grpo.test_decode_scheduler import Trainer


class PrefixTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.prefixes = {}

    def cache_decode_prefix(self, session, prefix):
        self.prefixes[prefix] = self.states[session].copy()
        return True

    def restore_decode_prefix(self, prefix, session):
        if prefix not in self.prefixes:
            return False
        self.states[session] = self.prefixes[prefix].copy()
        return True

    def release_decode_prefixes(self, ids):
        for prefix in ids:
            self.prefixes.pop(prefix, None)


def test_prefix_reuse_coalesces_prompts_and_preserves_divergent_tails():
    trainer = PrefixTrainer()
    scheduler = DecodeScheduler(trainer, max_batch=4, prefill_chunk=8, token_budget=512, prefix_entries=8)
    prompt = list(range(1, 30))
    try:
        sessions = [scheduler.new_session() for _ in range(3)]
        with ThreadPoolExecutor(3) as pool:
            futures = [pool.submit(scheduler.step, s, prompt, True) for s in sessions]
            for future in futures:
                assert future.result(timeout=5).tolist() == [sum(prompt), len(prompt)]
        assert scheduler.summary()["prefill_tokens"] == len(prompt) + 2
        assert scheduler.summary()["prefix_cached_tokens"] == 2 * (len(prompt) - 1)
        for session, tail in zip(sessions, (3, 5, 7), strict=True):
            assert scheduler.step(session, [tail]).tolist() == [sum(prompt) + tail, len(prompt) + 1]
            scheduler.release(session)
        extended = prompt + [91, 92]
        session = scheduler.new_session()
        assert scheduler.step(session, extended, True).tolist() == [sum(extended), len(extended)]
        scheduler.release(session)
        trainer.prefixes.clear()  # Training/import invalidated native snapshots.
        assert scheduler.step(scheduler.new_session(), prompt, True).tolist() == [sum(prompt), len(prompt)]
        assert len(trainer.prefixes) <= 8
    finally:
        scheduler.close()
    assert not trainer.states and not trainer.prefixes


def test_prefix_disabled_and_lru_eviction():
    for entries in (0, 1):
        trainer = PrefixTrainer()
        scheduler = DecodeScheduler(trainer, max_batch=2, prefill_chunk=8, token_budget=512, prefix_entries=entries)
        try:
            for prompt in ([1] * 17, [2] * 19, [1] * 17):
                session = scheduler.new_session()
                assert scheduler.step(session, prompt, True).tolist() == [sum(prompt), len(prompt)]
                scheduler.release(session)
                assert len(trainer.prefixes) <= entries
            assert scheduler.summary()["prefix_cache_hits"] == 0
        finally:
            scheduler.close()


def test_prefill_retries_smaller_chunks_under_memory_pressure():
    trainer = Trainer()
    scheduler = DecodeScheduler(trainer, max_batch=4, prefill_chunk=8, token_budget=512)
    admissions = []

    def admit(ids, counts, resets):
        admissions.extend(counts.tolist())
        return [n <= 2 for n in counts]

    trainer.admit_decode_sessions = admit
    try:
        prompt = list(range(19))
        assert scheduler.step(scheduler.new_session(), prompt, True).tolist() == [sum(prompt), len(prompt)]
        assert max(admissions) == 8 and 2 in admissions
        assert admissions[:3] == [8, 4, 2] and max(admissions[3:]) <= 2
        assert scheduler.summary()["prefill_chunk_splits"] > 0
    finally:
        scheduler.close()


def test_decode_overtakes_queued_prefill_and_prefill_shrinks():
    trainer = Trainer()
    scheduler = DecodeScheduler(trainer, max_batch=4, prefill_chunk=8, token_budget=512)
    a, b, prompt_session = [scheduler.new_session() for _ in range(3)]
    scheduler.step(a, [1], True)
    scheduler.step(b, [2], True)
    entered, resume = threading.Event(), threading.Event()
    original, calls = trainer.decode_batch_logits, []

    def blocked(ids, tokens, offsets, resets):
        calls.append((ids.tolist(), np.diff(offsets).tolist()))
        if len(calls) == 1:
            entered.set()
            assert resume.wait(5)
        return original(ids, tokens, offsets, resets)

    trainer.decode_batch_logits = blocked
    try:
        with ThreadPoolExecutor(3) as pool:
            first = pool.submit(scheduler.step, a, [3])
            assert entered.wait(5)
            prefill = pool.submit(scheduler.step, prompt_session, [4] * 19, True)
            decode = pool.submit(scheduler.step, b, [5])
            with scheduler.condition:
                assert scheduler.condition.wait_for(lambda: len(scheduler.pending) == 2, timeout=5)
            resume.set()
            assert first.result(timeout=5).tolist() == [4, 2]
            assert decode.result(timeout=5).tolist() == [7, 2]
            assert prefill.result(timeout=5).tolist() == [76, 19]
        assert calls[1][0] == [b]
        assert max(n for ids, counts in calls if prompt_session in ids for n in counts) <= 2
    finally:
        resume.set()
        scheduler.close()


def test_sustained_decode_does_not_starve_prefill():
    trainer = Trainer()
    original = trainer.decode_batch_logits

    def paced(*args):
        time.sleep(0.002)
        return original(*args)

    trainer.decode_batch_logits = paced
    scheduler = DecodeScheduler(trainer, max_batch=1, prefill_chunk=16, token_budget=512)
    session = scheduler.new_session()
    scheduler.step(session, [1], True)
    start = threading.Event()

    def generate():
        start.set()
        for _ in range(80):
            scheduler.step(session, [1])

    try:
        with ThreadPoolExecutor(2) as pool:
            decode = pool.submit(generate)
            assert start.wait(5)
            prefill = pool.submit(scheduler.step, scheduler.new_session(), [2] * 8, True)
            assert prefill.result(timeout=5).tolist() == [16, 8]
            assert not decode.done()
            decode.result(timeout=5)
    finally:
        scheduler.close()


def test_failed_prefill_unblocks_identical_waiting_prompts():
    trainer = PrefixTrainer()
    original = trainer.decode_batch_logits
    attempts = 0

    def fail_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected first prefill failure")
        return original(*args)

    trainer.decode_batch_logits = fail_once
    scheduler = DecodeScheduler(trainer, max_batch=4, prefill_chunk=8, token_budget=512)
    try:
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(scheduler.step, scheduler.new_session(), [7] * 19, True) for _ in range(2)]
            failures = 0
            for future in futures:
                try:
                    assert future.result(timeout=5).tolist() == [133, 19]
                except RuntimeError as error:
                    assert "injected first prefill" in str(error)
                    failures += 1
            assert failures == 1
    finally:
        scheduler.close()


def test_prefix_hits_cannot_bypass_logical_token_budget():
    trainer = PrefixTrainer()
    scheduler = DecodeScheduler(trainer, max_batch=4, prefill_chunk=16, token_budget=40)
    try:
        healthy = scheduler.new_session()
        assert scheduler.step(healthy, [3] * 30, True).tolist() == [90, 30]
        with pytest.raises(RuntimeError, match="budget exhausted"):
            scheduler.step(scheduler.new_session(), [3] * 30, True)
        assert scheduler.step(healthy, [7]).tolist() == [97, 31]
    finally:
        scheduler.close()
