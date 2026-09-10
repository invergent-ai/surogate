"""Shared pages, private hybrid state and window reclamation on real CUDA execution."""

import gc
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from surogate.grpo.decode_scheduler import DecodeScheduler
from tests.grpo.test_batched_decode_gpu import CASES, decode, make_trainer

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
SELECTED = os.environ.get("SUROGATE_SHARED_CASES", "all").split(",")


def assert_policy(actual, expected, case):
    actual = actual.astype(np.float64)
    expected = expected.astype(np.float64)
    actual -= np.logaddexp.reduce(actual)
    expected -= np.logaddexp.reduce(expected)
    assert np.isfinite(actual).all()
    if case != "glm":
        assert np.sqrt(np.mean((actual - expected) ** 2)) < 0.03
    np.testing.assert_allclose(actual, expected, atol=1e-5 if case == "glm" else 0.1, rtol=0)


@pytest.mark.parametrize("case", list(CASES) + ["glm"] if SELECTED == ["all"] else SELECTED)
def test_forked_pages_and_recurrent_state_preserve_divergent_requests(tmp_path, case):
    trainer = make_trainer(tmp_path, case, graphs=True)
    rng = np.random.default_rng(87)
    for length in (127, 128, 163):
        trainer.reset_decode_state()
        prompt = rng.integers(3, 97, size=length, dtype=np.int32)
        decode(trainer, [1], [prompt], [1])
        before = trainer.get_decode_batch_stats()
        assert trainer.cache_decode_prefix(1, 100)
        assert trainer.restore_decode_prefix(100, 2)
        assert trainer.restore_decode_prefix(100, 3)
        shared = trainer.get_decode_batch_stats()
        assert shared["pool_used_bytes"] == before["pool_used_bytes"]
        assert shared["sessions"] == 3 and shared["prefix_entries"] == 1
        # Build independent histories using the same prefill shape. This isolates
        # sharing from existing BF16/MoE routing differences between full prefill
        # and incremental execution (covered separately by policy regressions).
        for session in (11, 12, 13, 14):
            decode(trainer, [session], [prompt], [1])
        for sessions, sizes in (([3, 2], [9, 1]), ([2, 1, 3], [1, 7, 1]), ([3, 1, 2], [1, 1, 1])):
            chunks = [rng.integers(3, 97, size=n, dtype=np.int32) for n in sizes]
            actual = decode(trainer, sessions, chunks, [0] * len(sessions))
            independent = decode(trainer, [s + 10 for s in sessions], chunks, [0] * len(sessions))
            np.testing.assert_array_equal(actual, independent)
        # Later restores still see the original prefix after all branches wrote.
        assert trainer.restore_decode_prefix(100, 4)
        actual = decode(trainer, [4], [np.array([31])], [0])[0]
        np.testing.assert_array_equal(actual, decode(trainer, [14], [np.array([31])], [0])[0])
        assert trainer.get_decode_batch_stats()["prefix_hits"] >= 3
        trainer.release_decode_prefixes([100])
        trainer.release_decode_sessions([1, 2, 3, 4, 11, 12, 13, 14])
        assert trainer.get_decode_batch_stats()["prefix_entries"] == 0
        assert trainer.get_decode_batch_stats()["pool_used_bytes"] == 0
    trainer.reset_decode_state()
    assert trainer.get_decode_batch_stats()["pool_used_bytes"] == 0
    assert not trainer.restore_decode_prefix(100, 9)
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", ["llama", "qwen3_5", "glm"])
def test_prefix_restore_copies_sampling_counts_and_keeps_rng_per_request(tmp_path, case):
    from surogate.grpo.shared_model import sample_logits
    from tests.grpo.test_decode_sampling_gpu import DEFAULTS

    trainer = make_trainer(tmp_path, case, graphs=True)
    prompt = [7, 7, 9] * 43
    decode(trainer, [1], [prompt], [1])
    assert trainer.cache_decode_prefix(1, 100)
    for session in (2, 3, 12, 13):
        assert trainer.restore_decode_prefix(100, session)
    history = {2: prompt.copy(), 3: prompt.copy()}
    for step in range(4):
        sessions, chunks = [3, 2], [[7], [13]]
        logits = decode(trainer, [13, 12], chunks, [0, 0])
        params = DEFAULTS | dict(temperature=0.7, top_p=0.8, repetition_penalty=1.3,
                                presence_penalty=0.4, frequency_penalty=0.15, top_logprobs=5, logit_bias={7: 2.0})
        expected, requests = [], []
        for row, session in enumerate(sessions):
            history[session] += chunks[row]
            seed = 10 * step + session
            expected.append(sample_logits(logits[row], history[session], params, [], np.random.default_rng(seed)))
            requests.append(params | dict(uniform=float(np.random.default_rng(seed).random())))
        sampled = trainer.decode_batch_sample(np.array(sessions, dtype=np.int64),
                                               np.array(chunks, dtype=np.int32).ravel(),
                                               np.array([0, 1, 2], dtype=np.int32), np.zeros(2, dtype=np.int32), requests)
        for actual, reference in zip(sampled, expected, strict=True):
            assert actual["status"] == 0 and actual["token"] == reference["token"]
            assert actual["top_ids"] == reference["top_ids"]
            np.testing.assert_allclose(actual["logprob"], reference["logprob"], atol=1e-8, rtol=0)
    trainer.reset_decode_state()
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", ["gemma3", "gpt_oss"])
def test_sliding_windows_recycle_pages_under_a_fixed_cache_budget(tmp_path, case):
    config = CASES[case] | dict(layer_types=["sliding_attention"] * 2, max_position_embeddings=1024)
    trainer = make_trainer(tmp_path, case, graphs=True, sequence=1024, config=config)
    rng = np.random.default_rng(53)
    history = rng.integers(3, 97, size=900, dtype=np.int32)
    decode(trainer, [1], [history[:1]], [1])
    first = trainer.get_decode_batch_stats()
    # One retained page and one new page per layer suffice across each boundary.
    trainer.set_decode_cache_budget(first["pool_used_bytes"] * 2 + first["auxiliary_bytes"])
    start = 1
    for end in range(65, 898, 64):
        actual = decode(trainer, [1], [history[start:end]], [0])[0]
        inputs = np.zeros((2, 1024), dtype=np.int32)
        inputs[0, :end] = history[:end]
        expected = trainer.next_token_logits(inputs, np.array([end - 1, 0], dtype=np.int32))[0]
        assert_policy(actual, expected, case)
        stats = trainer.get_decode_batch_stats()
        assert stats["pool_used_bytes"] <= first["pool_used_bytes"] * 2
        start = end
    assert stats["recycled_window_pages"] >= 10
    assert stats["reused_pages"] > 0
    trainer.reset_decode_state()
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", ["llama", "qwen3_5", "glm"])
def test_scheduler_reuses_prefixes_across_completed_requests(tmp_path, case):
    trainer = make_trainer(tmp_path, case, graphs=True)
    scheduler = DecodeScheduler(trainer, max_batch=4, prefill_chunk=64, token_budget=1024, prefix_entries=8)
    prompt = [7, 9, 13] * 45

    def run(tail):
        session = scheduler.new_session()
        try:
            scheduler.step(session, prompt, True)
            return scheduler.step(session, [tail])
        finally:
            scheduler.release(session)

    try:
        with ThreadPoolExecutor(3) as pool:
            results = list(pool.map(run, (17, 19, 23)))
        for actual, token in zip(results, (17, 19, 23), strict=True):
            assert_policy(actual, trainer.decode_logits(np.array(prompt + [token], dtype=np.int32), reset=True), case)
        assert scheduler.summary()["prefix_cache_hits"] == 2
        assert scheduler.summary()["prefill_tokens"] == len(prompt) + 2
        assert trainer.get_decode_batch_stats()["sessions"] == 0
        trainer.reset_decode_state()
        scheduler.invalidate_prefixes()
        assert trainer.get_decode_batch_stats()["prefix_entries"] == 0
        run(29)
    finally:
        scheduler.close()
    trainer.reset_decode_state()
    del trainer
    gc.collect()


def test_prefix_pressure_evicts_snapshots_without_canceling_active_requests(tmp_path):
    trainer = make_trainer(tmp_path, "llama")
    prompt = np.array([7] * 127, dtype=np.int32)
    decode(trainer, [1], [prompt], [1])
    initial = trainer.get_decode_batch_stats()
    trainer.set_decode_cache_budget(initial["pool_used_bytes"] * 2 + initial["auxiliary_bytes"] * 3)
    assert trainer.cache_decode_prefix(1, 100)
    assert trainer.restore_decode_prefix(100, 2)
    # Both tails need a private partial page; inactive prefixes are evicted first.
    decode(trainer, [1, 2], [[11], [13]], [0, 0])
    stats = trainer.get_decode_batch_stats()
    assert stats["prefix_evictions"] > 0 and stats["sessions"] == 2
    assert stats["cache_allocated_bytes"] <= stats["cache_budget_bytes"]
    assert not trainer.restore_decode_prefix(100, 3)
    trainer.reset_decode_state()
    del trainer
    gc.collect()


@pytest.mark.parametrize("case", ["llama", "qwen3_5", "glm"])
def test_failed_optional_snapshot_does_not_leak_page_references(tmp_path, case):
    trainer = make_trainer(tmp_path, case)
    decode(trainer, [1], [[7, 9, 13]], [1])
    initial = trainer.get_decode_batch_stats()
    # Allow some table copies, then fail when copying private recurrent/count
    # buffers. The incomplete snapshot must release its references and charges.
    trainer.set_decode_cache_budget(initial["pool_used_bytes"] + initial["auxiliary_bytes"] + 128)
    assert not trainer.cache_decode_prefix(1, 100)
    failed = trainer.get_decode_batch_stats()
    assert failed["pool_used_bytes"] == initial["pool_used_bytes"]
    assert failed["auxiliary_bytes"] == initial["auxiliary_bytes"]
    assert failed["prefix_entries"] == 0 and failed["sessions"] == 1
    actual = decode(trainer, [1], [[19]], [0])
    assert np.isfinite(actual).all()
    assert trainer.get_decode_batch_stats()["copy_on_write_pages"] == 0
    trainer.reset_decode_state()
    del trainer
    gc.collect()
