"""GSPO (loss.variant: gspo), the trial count a GSPO step is divided by, and the KL-anchor guards.

GSPO (Zheng et al. 2025, arXiv 2507.18071) in its GSPO-token form (eqs. 13-14): one importance ratio per sequence,
s_i = exp(mean over its loss tokens of log(pi_theta / pi_old)), PPO's min of the unclipped and the clipped term, a
token's gradient A_t * s_i / |y_i|, and the step divided by its trial count. The references here are written from the
paper's equations, independently of surogate/grpo/loss.py: finite differences of the objective, torch autograd of the
GSPO-token objective with its stop-gradients, and a tiny model trained through the CE-backward seed the engine uses.
"""

from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch

from surogate.grpo.config import GRPOLossConfig
from surogate.grpo.loss import compute_grpo_per_token_grads, shift_to_target

LOW, HIGH = 0.05, 0.06  # wide clips, so the finite differences below stay on one side of them


def _cfg(low=LOW, high=HIGH, **kw):
    return GRPOLossConfig(variant="gspo", seq_clip_low=low, seq_clip_high=high, **kw)


def _sequence(rng, n_prompt, n_target, ratio):
    """Old and current log-probs of one sequence whose loss tokens have importance ratio `ratio` (geometric mean)."""
    n = n_prompt + n_target
    old = -rng.random(n) * 2.0
    delta = rng.normal(0.0, 0.2, n)
    delta[n_prompt:] += np.log(ratio) - delta[n_prompt:].mean()
    mask = np.zeros(n, bool)
    mask[n_prompt:] = True
    return old + delta, old, mask


def _objective(lp, old, adv, mask, ranges, low=LOW, high=HIGH):
    """GSPO's objective (eq. 5) for one advantage per sequence, summed over the sequences (no 1/G)."""
    total = 0.0
    for start, end in ranges:
        m = mask[start:end]
        s = np.exp(np.mean((lp[start:end] - old[start:end])[m]))
        a = adv[start:end][m][0]
        total += min(s * a, float(np.clip(s, 1.0 - low, 1.0 + high)) * a)
    return total


def _finite_differences(f, x, h=1e-6):
    g = np.zeros_like(x)
    for t in range(len(x)):
        up, down = x.copy(), x.copy()
        up[t] += h
        down[t] -= h
        g[t] = (f(up) - f(down)) / (2.0 * h)
    return g


class TestConfig:
    def test_gspo_has_no_probability_mask_and_no_kl_term(self):
        cfg = GRPOLossConfig(variant="gspo")
        assert (cfg.ipo_mask_low, cfg.ipo_mask_high, cfg.kl_tau) == (2.0, 2.0, 0.0)
        assert (cfg.seq_clip_low, cfg.seq_clip_high) == (3e-4, 4e-4), "the paper's clipping ranges by default"

    @pytest.mark.parametrize("low, high", [(0.0, 4e-4), (1.0, 4e-4), (3e-4, 0.0)])
    def test_rejects_clipping_ranges_outside_the_ratio(self, low, high):
        with pytest.raises(ValueError, match="seq_clip"):
            GRPOLossConfig(variant="gspo", seq_clip_low=low, seq_clip_high=high)

    def test_rejects_per_token_terms(self):
        with pytest.raises(ValueError, match="gspo"):
            GRPOLossConfig(variant="gspo", replay_tau=0.1)


class TestGradient:
    @pytest.mark.parametrize(
        "ratio, adv, clipped",
        [
            (1.01, 0.7, False),   # inside the clips
            (1.01, -0.4, False),
            (1.20, 0.7, True),    # above 1 + high with A > 0: the clipped term is the min, no gradient
            (0.85, -0.4, True),   # below 1 - low with A < 0: likewise
            (0.85, 0.7, False),   # below with A > 0: the unclipped term is the min, the gradient stays
            (1.20, -0.4, False),  # above with A < 0: likewise
        ],
    )
    def test_matches_finite_differences_of_the_objective(self, ratio, adv, clipped):
        rng = np.random.default_rng(1)
        lp, old, mask = _sequence(rng, 4, 9, ratio)
        advantages = np.full(len(lp), adv)
        ranges = [(0, len(lp))]

        g, metrics = compute_grpo_per_token_grads(lp, old, advantages, mask, _cfg(), ranges)
        fd = _finite_differences(lambda x: _objective(x, old, advantages, mask, ranges), lp)

        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-9)
        assert (not g.any()) == clipped
        if not clipped:  # each token gets A * s / |y|: the sequence counts once, whatever its length
            s = np.exp(np.mean((lp - old)[mask]))
            np.testing.assert_allclose(g[mask], adv * s / mask.sum(), rtol=1e-6)
        assert metrics["seq_log_ratio"] == pytest.approx(np.log(ratio), abs=1e-12)
        side = "is_masked_high" if adv > 0 else "is_masked_low"
        assert metrics[side] == (1.0 if clipped else 0.0)

    def test_pi_old_from_the_trainer_sets_the_ratio_to_one(self):
        """seq_ratio_old: trainer -- pi_old = sg[pi_theta]: no clip, each token A / |y|; the metric keeps the engine's."""
        rng = np.random.default_rng(6)
        lp, old, mask = _sequence(rng, 3, 8, 0.9)  # far below the clips
        advantages = np.full(len(lp), -0.4)
        g, metrics = compute_grpo_per_token_grads(lp, old, advantages, mask, _cfg(seq_ratio_old="trainer"),
                                                  [(0, len(lp))])
        np.testing.assert_allclose(g[mask], -0.4 / mask.sum(), rtol=1e-6)
        assert not g[~mask].any()
        assert metrics["is_masked_low"] == 0.0 and metrics["seq_log_ratio"] == pytest.approx(np.log(0.9), abs=1e-12)

    def test_packed_sequences_each_get_their_own_ratio(self):
        rng = np.random.default_rng(2)
        parts = [_sequence(rng, 3, 6, 1.02), _sequence(rng, 2, 11, 0.97), _sequence(rng, 5, 4, 1.3)]
        lp = np.concatenate([p[0] for p in parts])
        old = np.concatenate([p[1] for p in parts])
        mask = np.concatenate([p[2] for p in parts])
        bounds = np.cumsum([0] + [len(p[0]) for p in parts])
        ranges = list(zip(bounds[:-1], bounds[1:]))
        advantages = np.concatenate([np.full(len(p[0]), a) for p, a in zip(parts, (0.5, -0.3, 0.8))])

        g, metrics = compute_grpo_per_token_grads(lp, old, advantages, mask, _cfg(), ranges)
        fd = _finite_differences(lambda x: _objective(x, old, advantages, mask, ranges), lp)

        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-9)
        assert not g[ranges[2][0]:].any(), "the third sequence (s = 1.3, A > 0) is clipped"
        assert metrics["is_masked_high"] == pytest.approx(1 / 3)

    def test_per_token_advantages_follow_the_gspo_token_objective(self):
        """With the teacher term the advantage varies per token: GSPO-token (eq. 13), with its stop-gradients."""
        rng = np.random.default_rng(3)
        parts = [_sequence(rng, 3, 15, 1.03), _sequence(rng, 4, 12, 0.96)]
        lp = np.concatenate([p[0] for p in parts])
        old = np.concatenate([p[1] for p in parts])
        mask = np.concatenate([p[2] for p in parts])
        teacher = lp + rng.normal(0.0, 0.8, len(lp))
        bounds = np.cumsum([0] + [len(p[0]) for p in parts])
        ranges = list(zip(bounds[:-1], bounds[1:]))
        advantages = np.concatenate([np.full(len(p[0]), a) for p, a in zip(parts, (0.05, -0.04))])
        cfg = _cfg(low=0.02, high=0.02, teacher_tau=0.3)

        g, _ = compute_grpo_per_token_grads(lp, old, advantages, mask, cfg, ranges, teacher_logprobs=teacher)

        x = torch.tensor(lp, dtype=torch.float64, requires_grad=True)
        objective = 0.0
        for start, end in ranges:
            m = torch.tensor(mask[start:end])
            xs = x[start:end]
            s = torch.exp((xs - torch.tensor(old[start:end]))[m].mean())
            s_t = s.detach() * torch.exp(xs - xs.detach())  # sg[s_i] * pi_t / sg[pi_t]
            # The teacher term is part of the advantage (no gradient through it), as in the token-level variants.
            a_t = cfg.adv_tau * torch.tensor(advantages[start:end]) + cfg.teacher_tau * (
                torch.tensor(teacher[start:end]) - xs.detach())
            term = torch.minimum(s_t * a_t, torch.clamp(s_t, 1.0 - 0.02, 1.0 + 0.02) * a_t)
            objective = objective + term[m].mean()
        objective.backward()

        np.testing.assert_allclose(g, x.grad.numpy(), rtol=1e-6, atol=1e-9)
        a_tok = advantages + cfg.teacher_tau * (teacher - lp)
        assert (a_tok[mask] > 0).any() and (a_tok[mask] < 0).any(), "the per-token advantages take both signs"

    def test_a_tiny_model_gets_the_gspo_gradient_through_the_ce_backward_seed(self):
        """The engine turns the per-token seed into dlogits = seed * (softmax - onehot) at the target slot."""
        torch.manual_seed(0)
        V, D = 13, 6
        emb = torch.randn(V, D, dtype=torch.float64, requires_grad=True)
        out = torch.randn(D, V, dtype=torch.float64, requires_grad=True)
        lens, prompts, advs = [9, 7, 11], [3, 2, 4], [0.6, -0.5, 0.3]
        bounds = np.cumsum([0] + lens)
        ranges = list(zip(bounds[:-1], bounds[1:]))
        T = int(bounds[-1])
        ids = torch.randint(V, (T,))
        mask = np.zeros(T, bool)
        for (start, end), p in zip(ranges, prompts):
            mask[start + p:end] = True
        advantages = np.concatenate([np.full(n, a) for n, a in zip(lens, advs)])

        def forward():
            logits = emb[ids] @ out  # slot t predicts token t + 1 of the same sample
            lsm = torch.log_softmax(logits, dim=-1)
            lp = torch.zeros(T, dtype=torch.float64)
            for start, end in ranges:
                lp[start + 1:end] = lsm[start:end - 1].gather(1, ids[start + 1:end, None])[:, 0]
            return logits, lp

        _, lp0 = forward()
        old = lp0.detach().numpy() + np.random.default_rng(4).normal(0.0, 0.02, T)
        G = len(ranges)

        # Reference: L = -(1/G) * sum_i min(s_i A_i, clip(s_i) A_i), the paper's eq. 5, by autograd.
        _, lp = forward()
        loss = 0.0
        for (start, end), a in zip(ranges, advs):
            m = torch.tensor(mask[start:end])
            s = torch.exp((lp[start:end] - torch.tensor(old[start:end]))[m].mean())
            loss = loss - torch.minimum(s * a, torch.clamp(s, 1.0 - LOW, 1.0 + HIGH) * a) / G
        loss.backward()
        ref = (emb.grad.clone(), out.grad.clone())
        emb.grad = None
        out.grad = None

        # Ours: the per-token seed, shifted to the target slots and divided by the trial count, through the CE backward.
        logits, lp = forward()
        g, _ = compute_grpo_per_token_grads(lp.detach().numpy(), old, advantages, mask, _cfg(), ranges)
        seed = torch.tensor(shift_to_target(g, ranges) / G)
        targets = torch.zeros(T, dtype=torch.long)
        for start, end in ranges:
            targets[start:end - 1] = ids[start + 1:end]
        onehot = torch.nn.functional.one_hot(targets, V).double()
        dlogits = seed[:, None] * (torch.softmax(logits.detach(), dim=-1) - onehot)
        logits.backward(dlogits)

        # The seed is float32 (as the engine receives it): agreement to float32 rounding.
        torch.testing.assert_close(emb.grad, ref[0], rtol=1e-5, atol=1e-8)
        torch.testing.assert_close(out.grad, ref[1], rtol=1e-5, atol=1e-8)
        assert ref[1].abs().max() > 1e-3, "the reference gradient is not trivially zero"


def _sample(n_prompt, n_target, trial, advantage=0.25):
    from surogate.grpo.transport.types import TrainingSample

    return TrainingSample(
        prompt_ids=[1] * n_prompt,
        prompt_mask=[False] * n_prompt,
        completion_ids=[2] * n_target,
        completion_mask=[True] * n_target,
        completion_logprobs=[-0.5] * n_target,
        completion_temperatures=[1.0] * n_target,
        advantage=advantage,
        trial_idx=trial,
    )


def _wire_rows(samples, seq_len, single_sample_bins):
    """Samples through msgspec, the packer's preparation, packing and padding, msgspec again and the data loader."""
    from surogate.grpo.batch import pad_micro_batch, packed_samples_into_micro_bs, prepare_sample
    from surogate.grpo.data import microbatch_to_numpy
    from surogate.grpo.transport.types import MicroBatch, TrainingSample

    samples = msgspec.msgpack.decode(msgspec.msgpack.encode(samples), type=list[TrainingSample])
    prepared = [(0, prepare_sample(s, seq_len)) for s in samples]
    packed = packed_samples_into_micro_bs(prepared, seq_len, 1, single_sample_bins=single_sample_bins)
    packed = [pad_micro_batch(m, 8) for m in packed]
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(packed), type=list[MicroBatch])
    return [microbatch_to_numpy(m) for m in wire]


class TestTrialCount:
    @pytest.mark.parametrize("single_sample_bins", [True, False])
    def test_each_trial_counts_once(self, single_sample_bins):
        from surogate.grpo.trainer import _step_trial_count

        # Trial 0 in two rows, trials 1 and 2 in one each, one sample without an index (a trial of its own).
        samples = [_sample(3, 5, 0), _sample(4, 6, 0), _sample(2, 3, 1), _sample(3, 4, None), _sample(2, 2, 2)]
        rows = _wire_rows(samples, 20, single_sample_bins)

        assert _step_trial_count(rows) == 4
        if not single_sample_bins:
            assert any(r["trial_idx"] is not None and (r["trial_idx"] == -1).any() for r in rows), "padding is -1"

    def test_samples_without_indices_count_one_each(self):
        from surogate.grpo.trainer import _step_trial_count

        rows = _wire_rows([_sample(3, 5, None), _sample(2, 4, None), _sample(2, 3, None)], 64, False)
        assert all(r["trial_idx"] is None for r in rows)
        assert _step_trial_count(rows) == 3

    def test_samples_without_loss_tokens_do_not_count(self):
        from surogate.grpo.trainer import _step_trial_count

        s = _sample(3, 4, 5)
        s.completion_mask = [False] * 4
        assert _step_trial_count(_wire_rows([s, _sample(2, 3, 6)], 32, True)) == 1

    def test_an_old_producer_without_the_field_still_decodes(self):
        from surogate.grpo.transport.types import TrainingSample

        old = msgspec.msgpack.encode([[[1], [False], [2], [True], [-0.5], [1.0], None, 0.1]])
        (s,) = msgspec.msgpack.decode(old, type=list[TrainingSample])
        assert s.trial_idx is None and s.advantage == pytest.approx(0.1)


class _Cfg:
    def __init__(self, seq_len, loss):
        self.sequence_len = seq_len
        self.trainer_seq_len = seq_len
        self.gpus = 1
        self.teacher_from_base = False
        self.loss = loss


class TestDispatchStep:
    def test_custom_dloss_is_the_gspo_seed_over_the_trial_count(self, monkeypatch):
        from surogate.grpo.trainer import GRPOTrainer, _step_trial_count

        seq_len = 32
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.config = _Cfg(seq_len, _cfg())
        t._dispatch_pp = True
        t._diag_metrics = None
        rows = _wire_rows([_sample(3, 9, 0, 0.2), _sample(4, 12, 0, 0.2), _sample(2, 7, 1, -0.5)], seq_len, True)
        prepared = [t._prepare_micro_batch(r, seq_len) for r in rows]
        rng = np.random.default_rng(5)
        logical = [(p["inference"] + rng.normal(0.0, 0.01, seq_len)).astype(np.float32) for p in prepared]
        slots = np.stack([shift_to_target(lp, p["sample_ranges"]) for lp, p in zip(logical, prepared)])
        captured = []

        def fake_step(inputs, targets, los, his, opt, step, stale, m, custom_dloss, **kwargs):
            captured.append(np.array(custom_dloss, copy=True))
            return 0.0

        done = []

        def fake_forward(inputs, targets, los, his, m, **kwargs):
            first = sum(done)
            done.append(m)
            return slots[first:first + m]

        t.trainer = SimpleNamespace(
            dispatch_pp_forward_logprobs_multigpu=fake_forward,
            dispatch_pp_train_step_multigpu=fake_step,
            dispatch_pp_last_grad_norm=lambda: 0.0,
        )
        t._dpp_los, t._dpp_his = [0], [3]
        monkeypatch.setenv("SUROGATE_DPP_CALL_TOKENS", str(seq_len))  # one row per call

        trials = _step_trial_count(rows)
        t._dispatch_pp_step(rows, float(trials), object(), step=0, turn_acc=None)

        assert trials == 2 and len(captured) == 3
        for p, lp, dloss in zip(prepared, logical, captured):
            g, _ = compute_grpo_per_token_grads(lp, p["inference"], p["advantages"], p["loss_mask"].astype(bool),
                                                t.config.loss, p["sample_ranges"])
            np.testing.assert_allclose(dloss[0], shift_to_target(g, p["sample_ranges"]) / trials, rtol=1e-6)
        m = t._diag_metrics
        assert "seq_log_ratio" in m and m["total_tokens"] == 28 and m["policy_sample_count"] == 3


class TestAnchorGuards:
    @staticmethod
    def _config(teacher_from_base=False, teacher_tau=0.0, variant="cispo"):
        return SimpleNamespace(teacher_from_base=teacher_from_base,
                               loss=GRPOLossConfig(variant=variant, teacher_tau=teacher_tau))

    def test_a_base_pass_without_the_anchor_is_refused(self):
        from surogate.grpo.trainer import _check_loss_options

        with pytest.raises(ValueError, match="teacher_from_base: false"):
            _check_loss_options(self._config(teacher_from_base=True, teacher_tau=0.0), dispatch_pp=True)
        _check_loss_options(self._config(teacher_from_base=True, teacher_tau=0.3), dispatch_pp=True)
        _check_loss_options(self._config(), dispatch_pp=True)  # no anchor, no base pass

    def test_gspo_needs_dispatch_pp(self):
        from surogate.grpo.trainer import _check_loss_options

        with pytest.raises(ValueError, match="gspo needs parallelism: dispatch_pp"):
            _check_loss_options(self._config(variant="gspo"), dispatch_pp=False)
        _check_loss_options(self._config(variant="gspo"), dispatch_pp=True)

    @pytest.mark.parametrize("tau, with_teacher, warns", [(0.3, False, True), (0.0, True, True), (0.0, False, False),
                                                           (0.3, True, False)])
    def test_rows_and_anchor_disagreeing_warn_once(self, monkeypatch, tau, with_teacher, warns):
        import surogate.grpo.trainer as trainer_mod
        from surogate.grpo.trainer import GRPOTrainer

        said = []
        monkeypatch.setattr(trainer_mod.logger, "warning", lambda msg, *a, **k: said.append(msg))
        t = GRPOTrainer.__new__(GRPOTrainer)
        t.config = self._config(teacher_tau=tau)
        rows = [{"teacher_logprobs": np.zeros((1, 4), np.float32) if with_teacher else None} for _ in range(3)]
        t._check_teacher_rows(rows)
        t._check_teacher_rows(rows)
        assert len(said) == (1 if warns else 0)
