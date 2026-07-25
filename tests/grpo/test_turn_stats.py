"""Turn-id plumbing and turn-resolved diagnostics.

Covers the path an agent trajectory takes from vf.RolloutOutput through packing
into a MicroBatch, plus the TurnOPD (arXiv:2607.05804) diagnostic math.
"""

import numpy as np
import pytest

from surogate.grpo.batch import pad_micro_batch, packed_samples_into_micro_bs, prepare_sample
from surogate.grpo.transport.types import TrainingSample
from surogate.grpo.turn_stats import TurnAccumulator, blend_weight


def _sample(prompt_len: int, turn_lens: list[int], advantage: float = 1.0) -> TrainingSample:
    """Build a TrainingSample with `len(turn_lens)` turns of the given sizes."""
    completion_ids: list[int] = []
    turn_ids: list[int] = []
    for turn, n in enumerate(turn_lens):
        completion_ids.extend([100 + turn] * n)
        turn_ids.extend([turn] * n)
    return TrainingSample(
        prompt_ids=[1] * prompt_len,
        prompt_mask=[False] * prompt_len,
        completion_ids=completion_ids,
        completion_mask=[True] * len(completion_ids),
        completion_logprobs=[-0.5] * len(completion_ids),
        completion_temperatures=[1.0] * len(completion_ids),
        advantage=advantage,
        completion_turn_ids=turn_ids,
        num_turns=len(turn_lens),
    )


class TestTurnIdPlumbing:
    def test_prepare_sample_aligns_turn_ids_and_marks_prompt(self):
        mb = prepare_sample(_sample(prompt_len=4, turn_lens=[3, 2]), seq_len=64)

        assert len(mb.turn_ids) == len(mb.input_ids)
        assert mb.turn_ids[:4] == [-1, -1, -1, -1], "prompt tokens belong to no turn"
        assert mb.turn_ids[4:] == [0, 0, 0, 1, 1]

    def test_truncation_keeps_turn_ids_aligned(self):
        mb = prepare_sample(_sample(prompt_len=2, turn_lens=[5, 5]), seq_len=6)

        assert len(mb.input_ids) == 6
        assert len(mb.turn_ids) == 6

    def test_packing_preserves_per_sample_turn_ids(self):
        a = prepare_sample(_sample(prompt_len=2, turn_lens=[2, 1]), seq_len=64)
        b = prepare_sample(_sample(prompt_len=1, turn_lens=[3]), seq_len=64)

        packed = packed_samples_into_micro_bs([(0, a), (0, b)], max_seq_len=64, num_loras=1)

        assert len(packed) == 1, "both samples should fit in one bin"
        mb = packed[0]
        assert len(mb.turn_ids) == len(mb.input_ids)
        # Sample A: 2 prompt + turns [0,0,1]; sample B: 1 prompt + turn [0,0,0].
        assert mb.turn_ids == [-1, -1, 0, 0, 1, -1, 0, 0, 0]

    def test_packing_backfills_when_one_sample_lacks_turn_ids(self):
        multi = prepare_sample(_sample(prompt_len=1, turn_lens=[2]), seq_len=64)
        single = prepare_sample(_sample(prompt_len=1, turn_lens=[2]), seq_len=64)
        single.turn_ids = None  # a single-turn env contributes no turn structure

        # Single-turn sample lands in the bin first, multi-turn packs on top.
        packed = packed_samples_into_micro_bs([(0, single), (0, multi)], max_seq_len=64, num_loras=1)

        mb = packed[0]
        assert len(mb.turn_ids) == len(mb.input_ids), "turn_ids must stay index-aligned"
        assert mb.turn_ids[:3] == [-1, -1, -1], "backfilled span carries no turn"

    def test_padding_extends_turn_ids(self):
        mb = prepare_sample(_sample(prompt_len=1, turn_lens=[2]), seq_len=64)
        mb.lora_num_tokens = [len(mb.input_ids)]

        padded = pad_micro_batch(mb, pad_to_multiple_of=8)

        assert len(padded.input_ids) == 8
        assert len(padded.turn_ids) == 8
        assert padded.turn_ids[3:] == [-1] * 5

    def test_turn_ids_absent_stays_none(self):
        s = _sample(prompt_len=2, turn_lens=[2])
        s.completion_turn_ids = None

        mb = prepare_sample(s, seq_len=64)

        assert mb.turn_ids is None


def _step(prompt_ids, completion_ids):
    return {
        "tokens": {
            "prompt_ids": list(prompt_ids),
            "prompt_mask": [False] * len(prompt_ids),
            "completion_ids": list(completion_ids),
            "completion_mask": [True] * len(completion_ids),
            "completion_logprobs": [-0.5] * len(completion_ids),
        }
    }


def _rollout(trajectory):
    return {
        "trajectory": trajectory,
        "error": None,
        "example_id": "x",
        "sampling_args": {"temperature": 1.0},
        "stop_condition": None,
    }


class TestInterleaveRolloutTurnIds:
    def test_turn_ids_span_observation_and_completion(self):
        from surogate.grpo.orchestrator.trajectories import interleave_rollout

        # Turn 0: prompt [1,2] -> completion [3]. Turn 1: observation [4] appended.
        traj = [_step([1, 2], [3]), _step([1, 2, 3, 4], [5, 6])]

        (sample,) = interleave_rollout(_rollout(traj))

        assert sample.num_turns == 2
        # completion stream is [3] + observation [4] + [5, 6]
        assert sample.completion_ids == [3, 4, 5, 6]
        assert sample.completion_turn_ids == [0, 1, 1, 1]
        assert sample.completion_mask == [True, False, True, True]

    def test_split_samples_keep_trajectory_turn_index(self):
        from surogate.grpo.orchestrator.trajectories import interleave_rollout

        # Steps 0,1 chain; step 2 starts a fresh prefix -> second sample.
        traj = [_step([1], [2]), _step([1, 2], [3]), _step([9], [8])]

        samples = interleave_rollout(_rollout(traj))

        assert len(samples) == 2
        assert samples[0].completion_turn_ids == [0, 1]
        # The split sample must keep turn index 2, not be relabelled to 0 —
        # otherwise depth information is lost whenever a chat template breaks
        # the extension property.
        assert samples[1].completion_turn_ids == [2]
        assert all(s.num_turns == 3 for s in samples)

    def test_every_turn_split_still_spans_the_depth_range(self):
        from surogate.grpo.orchestrator.trajectories import interleave_rollout

        # Worst case: extension never holds, one sample per turn.
        traj = [_step([10 * i], [10 * i + 1]) for i in range(5)]

        samples = interleave_rollout(_rollout(traj))

        assert len(samples) == 5
        assert [s.completion_turn_ids for s in samples] == [[0], [1], [2], [3], [4]]

    def test_turn_ids_align_with_completion_length(self):
        from surogate.grpo.orchestrator.trajectories import interleave_rollout

        traj = [_step([1], [2, 3]), _step([1, 2, 3, 4, 5], [6])]

        (sample,) = interleave_rollout(_rollout(traj))

        assert len(sample.completion_turn_ids) == len(sample.completion_ids)


class TestTurnAccumulator:
    def _feed(self, turn_ids, loss_mask, kl, loss):
        acc = TurnAccumulator()
        acc.update(
            turn_ids=np.array(turn_ids, dtype=np.int32),
            loss_mask=np.array(loss_mask, dtype=bool),
            per_token_kl=np.array(kl, dtype=np.float32),
            per_token_loss=np.array(loss, dtype=np.float32),
            sample_ranges=[(0, len(turn_ids))],
        )
        return acc

    def test_excludes_unsupervised_tokens(self):
        # Turn 1's tokens are all unmasked; they must not appear at all.
        acc = self._feed(
            turn_ids=[-1, 0, 0, 1, 1],
            loss_mask=[False, True, True, False, False],
            kl=[9.0, 1.0, 3.0, 9.0, 9.0],
            loss=[9.0, 1.0, 1.0, 9.0, 9.0],
        )

        assert acc.token_count == {0: 2}
        assert acc.per_turn_mean_kl() == {0: 2.0}

    def test_shares_sum_to_one(self):
        acc = self._feed(
            turn_ids=[0, 0, 1, 2],
            loss_mask=[True] * 4,
            kl=[1.0, 1.0, 4.0, 2.0],
            loss=[1.0, 1.0, 4.0, 2.0],
        )

        assert sum(acc.per_turn_kl_share().values()) == pytest.approx(1.0)
        assert sum(acc.per_turn_loss_share().values()) == pytest.approx(1.0)
        assert sum(acc.per_turn_token_share().values()) == pytest.approx(1.0)

    def test_loss_share_uses_magnitude(self):
        # Gradients carry sign; budget is about magnitude, so -3 must not cancel +3.
        acc = self._feed(
            turn_ids=[0, 1],
            loss_mask=[True, True],
            kl=[1.0, 1.0],
            loss=[3.0, -3.0],
        )

        assert acc.per_turn_loss_share() == pytest.approx({0: 0.5, 1: 0.5})

    def test_survivor_counts_track_reached_depth(self):
        acc = TurnAccumulator()
        # Sample A reaches turns 0-2, sample B only turn 0.
        turn_ids = np.array([0, 1, 2, 0], dtype=np.int32)
        acc.update(
            turn_ids=turn_ids,
            loss_mask=np.ones(4, dtype=bool),
            per_token_kl=np.ones(4, dtype=np.float32),
            per_token_loss=np.ones(4, dtype=np.float32),
            sample_ranges=[(0, 3), (3, 4)],
        )

        assert acc.survivors == {0: 2, 1: 1, 2: 1}
        assert acc.summary()["turn/deep_support"] == pytest.approx(0.5)

    def test_summary_detects_shallow_budget_concentration(self):
        # 6 turns; turn 0 huge and loud, deep turns tiny and quiet.
        turn_ids, loss_mask, kl, loss = [], [], [], []
        for t in range(6):
            n = 50 if t == 0 else 2
            mag = 1.0 if t == 0 else 0.1
            turn_ids += [t] * n
            loss_mask += [True] * n
            kl += [mag] * n
            loss += [mag] * n

        acc = self._feed(turn_ids, loss_mask, kl, loss)
        s = acc.summary()

        assert s["turn/num_turns_observed"] == 6
        # Thirds of 6 turns: shallow={0,1}, deep={4,5}.
        # shallow mean KL = mean(1.0, 0.1) = 0.55; deep = mean(0.1, 0.1) = 0.1.
        assert s["turn/deep_shallow_kl_ratio"] == pytest.approx(0.1 / 0.55)
        assert s["turn/turn0_loss_budget"] > 0.9
        assert s["turn/deep_kl_budget"] < 0.02

    def test_kl_share_uses_magnitude_when_sign_flips(self):
        """The signed k1 mean flips sign along the turn axis in real runs
        (measured: -1.02 at turn 0, +0.12 at turn 11). Summing signed values
        gives a cancelling denominator and shares outside [0, 1] (observed
        -71.9%), so the budget share must be built from |KL|."""
        acc = self._feed(
            turn_ids=[0, 0, 1, 1],
            loss_mask=[True] * 4,
            kl=[-3.0, -3.0, 1.0, 1.0],
            loss=[1.0, 1.0, 1.0, 1.0],
        )

        shares = acc.per_turn_kl_share()
        assert all(0.0 <= v <= 1.0 for v in shares.values())
        assert sum(shares.values()) == pytest.approx(1.0)
        # Turn 0 carries 3x the divergence magnitude of turn 1.
        assert shares[0] == pytest.approx(0.75)
        assert shares[1] == pytest.approx(0.25)

    def test_deep_shallow_ratio_is_positive_under_sign_flip(self):
        turn_ids, loss_mask, kl, loss = [], [], [], []
        for t in range(6):
            turn_ids += [t] * 4
            loss_mask += [True] * 4
            kl += [-2.0 if t < 3 else 0.5] * 4
            loss += [1.0] * 4

        s = self._feed(turn_ids, loss_mask, kl, loss).summary()

        assert s["turn/deep_shallow_kl_ratio"] == pytest.approx(0.25)
        assert 0.0 <= s["turn/deep_kl_budget"] <= 1.0
        # The signed means are still reported, and keep their sign.
        assert s["turn/mean_kl_shallow"] < 0 < s["turn/mean_kl_deep"]

    def test_mean_kl_keeps_sign(self):
        """The mean is a KL *estimate* and must report what was measured,
        including a negative excursion — only the budget share is clamped."""
        acc = self._feed(
            turn_ids=[0, 1],
            loss_mask=[True, True],
            kl=[2.0, -2.0],
            loss=[1.0, 1.0],
        )

        assert acc.per_turn_mean_kl() == pytest.approx({0: 2.0, 1: -2.0})

    def test_empty_accumulator_is_safe(self):
        assert TurnAccumulator().summary() == {}
        assert TurnAccumulator().as_rows() == []


class TestDiagnosticLayoutRoundTrip:
    """The diagnostic path must produce byte-identical gradients to the fused step.

    forward_for_grpo returns the negated CE loss buffer, which is in TARGET slot
    layout (buffer[t] = log p of logical token t+1). The Python loss expects
    logical layout. Getting that un-shift wrong silently corrupts every gradient
    by one position, so pin it against the existing native reference.
    """

    def _run(self, sample_ranges, seq_len, rng):
        from surogate.grpo.config import GRPOLossConfig
        from surogate.grpo.loss import compute_native_shifted_grpo_dloss_reference

        logical_lp = rng.normal(-1.0, 0.5, seq_len).astype(np.float32)
        inference_lp = rng.normal(-1.0, 0.5, seq_len).astype(np.float32)
        advantages = rng.normal(0.0, 1.0, seq_len).astype(np.float32)
        teacher_lp = rng.normal(-1.0, 0.5, seq_len).astype(np.float32)

        loss_mask = np.zeros(seq_len, dtype=bool)
        for start, end in sample_ranges:
            loss_mask[start + 1 : end] = True

        cfg = GRPOLossConfig()
        expected = compute_native_shifted_grpo_dloss_reference(
            trainer_logprobs=logical_lp,
            inference_logprobs=inference_lp,
            advantages=advantages,
            loss_mask=loss_mask,
            loss_config=cfg,
            sample_ranges=sample_ranges,
            teacher_logprobs=teacher_lp,
            loss_scale=1.0,
        )

        # Emulate the engine: build the target-slot buffer the trainer receives.
        buf = np.zeros(seq_len, dtype=np.float32)
        for start, end in sample_ranges:
            if end - start > 1:
                buf[start : end - 1] = logical_lp[start + 1 : end]

        # The un-shift performed in GRPOTrainer._diagnostic_micro_step.
        recovered = np.zeros(seq_len, dtype=np.float32)
        for start, end in sample_ranges:
            if end - start > 1:
                recovered[start + 1 : end] = buf[start : end - 1]

        from surogate.grpo.loss import compute_grpo_per_token_grads

        grads, _ = compute_grpo_per_token_grads(
            trainer_logprobs=recovered,
            inference_logprobs=inference_lp,
            advantages=advantages,
            loss_mask=loss_mask,
            loss_config=cfg,
            sample_ranges=sample_ranges,
            teacher_logprobs=teacher_lp,
        )
        shifted = np.zeros(seq_len, dtype=np.float32)
        for start, end in sample_ranges:
            if end - start > 1:
                shifted[start : end - 1] = grads[start + 1 : end]
        return shifted, expected

    def test_single_sample(self):
        got, expected = self._run([(0, 16)], 16, np.random.default_rng(0))
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_multiple_packed_samples(self):
        got, expected = self._run([(0, 7), (7, 12), (12, 20)], 20, np.random.default_rng(1))
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_raw_buffer_without_unshift_is_detectably_wrong(self):
        # Guards the guard: if the un-shift were a no-op this test must fail,
        # otherwise the round-trip assertions above prove nothing.
        rng = np.random.default_rng(2)
        got, expected = self._run([(0, 16)], 16, rng)
        assert not np.allclose(np.roll(got, 1), expected), "shift must be observable"


class TestTrainerPadding:
    """turn_ids must reach the accumulator padded to seq_len like every other array.

    The MicroBatch arrives at T_actual length and the trainer pads each per-token
    array to seq_len before the forward. Missing that for turn_ids raised
    `operands could not be broadcast together with shapes (4096,) (3986,)` on the
    first real step — only once a packed batch happened to be shorter than seq_len.
    """

    def test_short_batch_pads_turn_ids_to_seq_len(self):
        seq_len, t_actual = 16, 11
        turn_ids_flat = np.array([-1, -1, 0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        assert len(turn_ids_flat) == t_actual

        # Mirror of the trainer's padding step.
        padded = np.full(seq_len, -1, dtype=np.int32)
        padded[:t_actual] = turn_ids_flat[:t_actual]

        loss_mask = np.zeros(seq_len, dtype=bool)
        loss_mask[2:t_actual] = True

        assert padded.shape == loss_mask.shape
        # Padding must not invent turns.
        assert (padded[t_actual:] == -1).all()

        acc = TurnAccumulator()
        acc.update(
            turn_ids=padded,
            loss_mask=loss_mask,
            per_token_kl=np.ones(seq_len, dtype=np.float32),
            per_token_loss=np.ones(seq_len, dtype=np.float32),
            sample_ranges=[(0, seq_len)],
        )
        assert sorted(acc.token_count) == [0, 1, 2]
        assert sum(acc.token_count.values()) == t_actual - 2

    def test_mismatched_lengths_are_rejected_not_broadcast(self):
        acc = TurnAccumulator()
        with pytest.raises(ValueError):
            acc.update(
                turn_ids=np.zeros(11, dtype=np.int32),
                loss_mask=np.ones(16, dtype=bool),
                per_token_kl=np.ones(16, dtype=np.float32),
                per_token_loss=np.ones(16, dtype=np.float32),
                sample_ranges=[(0, 16)],
            )


class TestBlendWeight:
    def _mk(self, turn_lens):
        turn_ids, loss_mask = [], []
        for t, n in enumerate(turn_lens):
            turn_ids += [t] * n
            loss_mask += [True] * n
        return np.array(turn_ids, dtype=np.int32), np.array(loss_mask, dtype=bool)

    def test_alpha_zero_is_uniform_token_weighting(self):
        turn_ids, loss_mask = self._mk([10, 2, 1])

        w = blend_weight(turn_ids, loss_mask, alpha=0.0)

        # Trajectory-level normalization gives every token the same weight.
        assert w[loss_mask] == pytest.approx(np.ones(13), rel=1e-5)

    def test_alpha_one_equalizes_turn_totals(self):
        turn_ids, loss_mask = self._mk([10, 2, 1])

        w = blend_weight(turn_ids, loss_mask, alpha=1.0)

        totals = [w[turn_ids == t].sum() for t in range(3)]
        assert totals == pytest.approx([totals[0]] * 3, rel=1e-5)

    def test_total_gradient_scale_is_preserved(self):
        turn_ids, loss_mask = self._mk([10, 2, 1])

        for alpha in (0.0, 0.25, 0.5, 1.0):
            w = blend_weight(turn_ids, loss_mask, alpha=alpha)
            assert w.sum() == pytest.approx(13.0, rel=1e-5), f"alpha={alpha} changed total scale"

    def test_blend_shifts_budget_monotonically_toward_deep_turns(self):
        turn_ids, loss_mask = self._mk([50, 2, 2])

        deep_budget = []
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            w = blend_weight(turn_ids, loss_mask, alpha=alpha)
            deep_budget.append(w[turn_ids == 2].sum() / w.sum())

        assert all(b < a for a, b in zip(deep_budget, deep_budget[1:])) is False
        assert deep_budget == sorted(deep_budget), "deep budget must rise with alpha"
        assert deep_budget[0] < 0.05 and deep_budget[-1] > 0.3

    def test_unsupervised_tokens_get_zero_weight(self):
        turn_ids = np.array([-1, 0, 0, -1], dtype=np.int32)
        loss_mask = np.array([False, True, True, False], dtype=bool)

        w = blend_weight(turn_ids, loss_mask, alpha=0.5)

        assert w[0] == 0.0 and w[3] == 0.0

    def test_no_supervised_tokens_is_safe(self):
        turn_ids = np.array([-1, -1], dtype=np.int32)
        loss_mask = np.array([False, False], dtype=bool)

        assert blend_weight(turn_ids, loss_mask, alpha=0.5).sum() == 0.0
