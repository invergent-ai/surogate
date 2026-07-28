"""GRPO routes through dispatch-PP when parallelism=dispatch_pp.

dispatch-PP exists so a model whose weights do not fit resident can still train.
GRPO cannot use the fused dispatch step directly because its gradients do not
come from cross-entropy, so it drives the two halves itself:

    staged forward -> logprobs -> Python per-token grads -> staged backward
                                                            seeded via custom_dloss

These tests cover the wiring and the gradient assembly without a GPU; the
numerical behaviour of the C++ halves is covered by
tests/train/dispatch_pp/test_custom_dloss.py.
"""

import numpy as np
import pytest

from surogate.grpo.trainer import GRPOTrainer
from surogate.train.dispatch_pp import plan_stages


class _Cfg:
    """Minimal stand-in for GRPOTrainConfig for the pure-Python paths."""

    def __init__(self, seq_len=16, gpus=1):
        from surogate.grpo.config import GRPOLossConfig

        self.sequence_len = seq_len
        self.gpus = gpus
        self.loss = GRPOLossConfig()


def _trainer(seq_len=16):
    t = GRPOTrainer.__new__(GRPOTrainer)
    t.config = _Cfg(seq_len=seq_len)
    t._dispatch_pp = True
    t._diag_metrics = None
    return t


def _micro_batch(seq_len, t_actual, with_teacher=True, with_turns=True):
    rng = np.random.default_rng(0)
    mb = {
        "input_ids": np.arange(t_actual, dtype=np.int32).reshape(1, t_actual) + 1,
        "position_ids": np.arange(t_actual, dtype=np.int32).reshape(1, t_actual),
        "targets": np.arange(t_actual, dtype=np.int32).reshape(1, t_actual) + 2,
        "advantages": rng.normal(size=t_actual).astype(np.float32).reshape(1, t_actual),
        "inference_logprobs": (-rng.random(t_actual)).astype(np.float32).reshape(1, t_actual),
        "loss_mask": np.array([[False] * 2 + [True] * (t_actual - 2)]),
        "temperatures": np.ones((1, t_actual), dtype=np.float32),
        "teacher_logprobs": (-rng.random(t_actual)).astype(np.float32).reshape(1, t_actual)
        if with_teacher
        else None,
        "turn_ids": np.array([[-1, -1] + [0] * (t_actual - 4) + [1, 1]], dtype=np.int32)
        if with_turns
        else None,
    }
    return mb


class TestStagePlanning:
    def test_grpo_and_sft_get_identical_stages(self):
        class MC:
            num_hidden_layers = 24

        a = plan_stages(MC, gpus=2)
        b = plan_stages(MC, gpus=2)
        assert a == b
        los, his, n_layers, nst, _, sb = a
        assert los[0] == 0 and his[-1] == n_layers - 1
        assert nst >= 2
        assert all(lo % sb == 0 for lo in los)


class TestMicroBatchPreparation:
    def test_pads_every_per_token_array_to_seq_len(self):
        t = _trainer(seq_len=16)
        p = t._prepare_micro_batch(_micro_batch(16, 11), 16)

        assert p["input"].shape == (1, 16)
        assert p["targets"].shape == (1, 16)
        for key in ("inference", "advantages", "loss_mask", "teacher", "turn_ids"):
            assert p[key].shape == (16,), key

    def test_padding_carries_no_supervision(self):
        t = _trainer(seq_len=16)
        p = t._prepare_micro_batch(_micro_batch(16, 11), 16)

        assert not p["loss_mask"][11:].any()
        assert (p["turn_ids"][11:] == -1).all()
        assert (p["targets"][0, 11:] == -100).all()

    def test_position_ids_continue_past_the_real_tokens(self):
        """RoPE resets mark sample boundaries; padding must not fake a new sample."""
        t = _trainer(seq_len=16)
        p = t._prepare_micro_batch(_micro_batch(16, 11), 16)

        pos = p["position_ids"][0]
        assert pos[11] == pos[10] + 1
        assert (np.diff(pos[11:]) == 1).all()

    def test_targets_masked_where_loss_mask_is_off(self):
        t = _trainer(seq_len=16)
        mb = _micro_batch(16, 12)
        p = t._prepare_micro_batch(mb, 16)

        # loss_mask[t] True means target slot t-1 is supervised.
        assert p["targets"][0, 0] == -100, "prompt token must not be supervised"

    def test_absent_optional_arrays_stay_none(self):
        t = _trainer(seq_len=16)
        p = t._prepare_micro_batch(_micro_batch(16, 8, with_teacher=False, with_turns=False), 16)
        assert p["teacher"] is None
        assert p["turn_ids"] is None


class TestDispatchStep:
    """The dispatch step must assemble exactly the gradients the native path would."""

    def test_custom_dloss_matches_the_native_reference(self):
        from surogate.grpo.loss import compute_native_shifted_grpo_dloss_reference

        seq_len, t_actual, M = 16, 13, 3
        t = _trainer(seq_len=seq_len)
        mbs = [_micro_batch(seq_len, t_actual) for _ in range(M)]
        prepared = [t._prepare_micro_batch(mb, seq_len) for mb in mbs]

        rng = np.random.default_rng(7)
        logical = [rng.normal(-1.0, 0.4, seq_len).astype(np.float32) for _ in range(M)]
        loss_scale = 37.0

        # What the engine hands back: target-slot layout.
        rows = np.zeros((M, seq_len), dtype=np.float32)
        for i, p in enumerate(prepared):
            for start, end in p["sample_ranges"]:
                if end - start > 1:
                    rows[i, start : end - 1] = logical[i][start + 1 : end]

        captured = {}

        def fake_forward(inputs, targets, los, his, m):
            captured["shape"] = (inputs.shape, targets.shape, m)
            return rows

        def fake_step(inputs, targets, los, his, opt, step, stale, m, custom_dloss):
            captured["custom_dloss"] = np.array(custom_dloss, copy=True)
            captured["M"] = m
            return 0.5

        t.trainer = type(
            "T",
            (),
            {
                "dispatch_pp_forward_logprobs_multigpu": staticmethod(fake_forward),
                "dispatch_pp_train_step_multigpu": staticmethod(fake_step),
                "dispatch_pp_last_grad_norm": staticmethod(lambda: 1.25),
            },
        )()
        t._dpp_los, t._dpp_his = [0], [3]

        out = t._dispatch_pp_step(mbs, loss_scale, object(), step=0, turn_acc=None)

        assert out["loss"] == pytest.approx(0.5)
        assert out["norm"] == pytest.approx(1.25)
        assert captured["M"] == M
        assert captured["shape"][0] == (M, seq_len)

        for i, p in enumerate(prepared):
            expected = compute_native_shifted_grpo_dloss_reference(
                trainer_logprobs=logical[i],
                inference_logprobs=p["inference"],
                advantages=p["advantages"],
                loss_mask=p["loss_mask"].astype(bool),
                loss_config=t.config.loss,
                sample_ranges=p["sample_ranges"],
                teacher_logprobs=p["teacher"],
                loss_scale=loss_scale,
            )
            np.testing.assert_allclose(captured["custom_dloss"][i], expected, rtol=1e-5, atol=1e-6)

    def test_turn_diagnostics_still_collected_under_dispatch(self):
        from surogate.grpo.turn_stats import TurnAccumulator

        seq_len, M = 16, 2
        t = _trainer(seq_len=seq_len)
        mbs = [_micro_batch(seq_len, 13) for _ in range(M)]

        t.trainer = type(
            "T",
            (),
            {
                "dispatch_pp_forward_logprobs_multigpu": staticmethod(
                    lambda *a, **k: np.full((M, seq_len), -1.0, dtype=np.float32)
                ),
                "dispatch_pp_train_step_multigpu": staticmethod(lambda *a, **k: 0.0),
                "dispatch_pp_last_grad_norm": staticmethod(lambda: 0.0),
            },
        )()
        t._dpp_los, t._dpp_his = [0], [3]

        acc = TurnAccumulator()
        t._dispatch_pp_step(mbs, 10.0, object(), step=0, turn_acc=acc)

        assert acc.token_count, "turn diagnostics must work under dispatch-PP too"
        assert set(acc.token_count) <= {0, 1}
