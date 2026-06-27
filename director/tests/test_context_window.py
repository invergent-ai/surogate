"""Tests for long-context token windowing (the Qwen3-0.6B routing-context fix)."""

from __future__ import annotations

from director.fugu.model import window_token_ids

IDS = list(range(1000))  # a long transcript: tokens 0..999


def test_short_sequence_unchanged():
    assert window_token_ids([1, 2, 3], window=100, head_tokens=10, strategy="head_tail") == [1, 2, 3]


def test_head_tail_keeps_goal_and_recent():
    out = window_token_ids(IDS, window=100, head_tokens=20, strategy="head_tail")
    assert len(out) == 100
    assert out[:20] == list(range(20))          # goal/system prefix preserved
    assert out[20:] == list(range(920, 1000))   # most-recent tail preserved
    assert 500 not in out                        # stale middle dropped


def test_tail_keeps_only_recent():
    out = window_token_ids(IDS, window=50, head_tokens=10, strategy="tail")
    assert out == list(range(950, 1000))         # recent only (not the stale beginning)


def test_tail_is_decision_relevant_not_first_n():
    # the OLD bug was right-truncation (first-N) which drops recent turns
    out = window_token_ids(IDS, window=100, head_tokens=0, strategy="head_tail")
    assert out[-1] == 999  # the latest token is always retained
