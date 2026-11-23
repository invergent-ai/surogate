"""Lane T8, S2 pass line (PLAN-SKYRL.md): SkyRL's update on the fake model, ours against a reference written with SkyRL's OWN loss functions (AST-loaded,
pinned): old log-probs at the start weights, n mini-batches (1 and 4) each with its own token mean and one AdamW step (lr 2e-5). Per mini-batch the
gradient within rel 1e-5 and the masters after the batch within rel 1e-5 of the reference; GRPO advantages from t8_skyrl; the mini-batch cut never splits
a group; the stale gate (first mini-batch == old log-probs, bitwise) and the replay gate on the old log-probs both fire on a bad input."""
import copy
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from surogate.inkling_rl.backend_fake import FakeInkling  # noqa: E402
from surogate.inkling_rl.shim import MasterAdamW  # noqa: E402
from surogate.inkling_rl.step import StepConfig  # noqa: E402
import t8_skyrl as KS  # noqa: E402
import t8_skyrl_cpu as SR  # noqa: E402
import t8_step as TS  # noqa: E402
import t8_step_cpu as C  # noqa: E402

LR = 2e-5


class FakeW(C.FakeWorker):
    """Its own model (worker threads run autograd at the same time; a shared module would race on .grad); the trainer copies the work values in."""

    def logprobs_row(self, row):
        with torch.no_grad():
            lp, _ = self.be.forward(torch.as_tensor(row.ids), replay=None, lora=self.be.lora)
        return lp[1:].numpy().astype(np.float32)


class FakeTrainer:
    def __init__(self, cfg, n_workers):
        self.be = FakeInkling(seed=0, rank=4, alpha=8)
        self.shapes = [{k: tuple(d[k].shape) for k in TS.LORA_KEYS} for d in self.be.lora]
        self.layout = TS.ParamLayout(self.shapes, list(range(len(self.shapes))), cfg.train_keys)
        self.flats = [torch.zeros(self.layout.size) for _ in range(n_workers)]
        self.workers = [FakeW(cfg, f, self.layout) for f in self.flats]
        self.be = self.workers[0].be
        self.handles = [TS.InlineHandle(w) for w in self.workers]
        self.cfg = cfg
        self.work = self.be.parameters()
        self.opt = MasterAdamW(self.work, lr=LR, weight_decay=0.0, max_grad_norm=1.0)
        self._sync()

    def _sync(self):
        with torch.no_grad():
            for w in self.workers[1:]:
                for p, q in zip(w.be.parameters(), self.work):
                    p.copy_(q)

    def logprobs(self, rows):
        return [self.handles[0].call("logprobs_row", r) for r in rows]

    def compute(self, rows, version, step_cfg, seeds="gspo", minibatch_index=0):
        return TS.compute_step(self.handles, self.layout, self.flats, self.shapes, self.cfg, rows, version, step_cfg, seeds, minibatch_index)

    def apply(self, out, lr):
        for g in self.opt.opt.param_groups:
            g["lr"] = lr
        for p, g in zip(self.work, out.grads):
            p.grad = g
        n = self.opt.step()
        self._sync()
        return n


def reference(rows, version, n_mb, S):
    """SkyRL's own loss, on the fake model, with the same mini-batch cut and the same optimizer."""
    be = FakeInkling(seed=0, rank=4, alpha=8)
    opt = MasterAdamW(be.parameters(), lr=LR, weight_decay=0.0, max_grad_norm=1.0)
    with torch.no_grad():
        olds = [be.forward(torch.as_tensor(r.ids), lora=be.lora)[0] for r in rows]
    grads, norms = [], []
    for part in TS.split_minibatches(rows, n_mb):
        N = sum(int(rows[i].loss_mask.sum()) for i in part)
        for p in be.parameters():
            p.grad = None
        total = 0.0
        for i in part:
            r = rows[i]
            lp, _ = be.forward(torch.as_tensor(r.ids), lora=be.lora)
            mask = torch.as_tensor(r.loss_mask.astype(np.float32))[None]
            adv = torch.full((1, r.T), float(r.advantage)) / max(N, 1)        # apply_loss_reduction_to_advantages_minibatch, token_mean, whole mini-batch
            loss, _ = S.ppo_policy_loss(lp[None], olds[i][None], adv, SR.Cfg(), loss_mask=mask)
            total = total + loss
        total.backward()
        grads.append([p.grad.detach().clone() for p in be.parameters()])
        norms.append(opt.step())
    return grads, norms, [m.detach().clone() for m in opt.masters]


def rel(a, b):
    a, b = a.double(), b.double()
    nb = float(b.norm())
    return float((a - b).norm()) / nb if nb > 0 else float(a.norm())


def make(version):
    rows = C.make_rows(12, 5, version)
    for k, r in enumerate(rows):
        r.group_idx = k // 3                                 # 4 groups of 3 samples
    scores = np.random.default_rng(7).choice([0.0, 1.0], size=len(rows))
    adv = KS.grpo_advantages(scores, [r.group_idx for r in rows])
    for r, a in zip(rows, adv):
        r.advantage = float(a)
    return rows


def main():
    _, S = SR.reference_modules() if hasattr(SR, "reference_modules") else (None, None)
    if S is None:
        S = load_S()
    version = 3
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    cfg = TS.T8Config(head="plain", max_row_tokens=10 ** 6, segment_keys=(), run_dir=tmp, visits=5, top_k=2, K=5)
    for n_mb in (1, 4):
        rows = make(version)
        parts = TS.split_minibatches(rows, n_mb)
        assert sorted(sum(parts, [])) == list(range(len(rows)))
        for p in parts:
            assert rows[p[0]].group_idx != rows[p[0] - 1].group_idx or p[0] == 0, "a cut splits a group"
        g_ref, n_ref, m_ref = reference(copy.deepcopy(rows), version, n_mb, S)
        tr = FakeTrainer(cfg, 2)
        res = TS.ppo_batch(tr, copy.deepcopy(rows), version, LR, n_mb, StepConfig(max_policy_lag=None))
        assert len(res) == n_mb and all(o.sense == "descent" for o, _ in res)
        wg = max(rel(a, b) for (o, _), gr in zip(res, g_ref) for a, b in zip(o.grads, gr))
        wm = max(rel(a, b) for a, b in zip(tr.opt.masters, m_ref))
        wn = max(abs(n - r) / max(abs(r), 1e-30) for (_, n), r in zip(res, n_ref))
        print(f"n_minibatches {n_mb}: worst gradient rel {wg:.2e}, masters rel {wm:.2e}, grad-norm rel {wn:.2e}; clip ratio {res[-1][0].metrics.get('ppo_clip_ratio')}")
        assert wg <= 1e-5 and wm <= 1e-5 and wn <= 1e-5
        if n_mb == 4:
            assert any(m > 0 for m in (float((tr.opt.masters[i] - FakeInkling(seed=0, rank=4, alpha=8).parameters()[i].detach()).abs().max()) for i in range(len(m_ref))))
    # gates
    rows = make(version)
    tr = FakeTrainer(cfg, 1)
    olds = tr.logprobs(rows)
    for r, o in zip(rows, olds):
        r.notes["old_lp"] = o
    bad = copy.deepcopy(rows)
    bad[2].notes["old_lp"] = bad[2].notes["old_lp"].copy()
    bad[2].notes["old_lp"][int(np.flatnonzero(bad[2].loss_mask[1:])[0])] += 1e-3
    for what, rr in (("STALE GATE", bad),):
        try:
            tr.compute(rr, version, StepConfig(max_policy_lag=None), seeds="ppo", minibatch_index=0)
            raise AssertionError(f"{what} did not fire")
        except TS.T8GateError as e:
            assert what in str(e), str(e)
    bad2 = copy.deepcopy(rows)
    t = int(np.flatnonzero(bad2[4].loss_mask)[0])
    bad2[4].engine_lp[t] += 1e-3
    try:
        tr.compute(bad2, version, StepConfig(max_policy_lag=None), seeds="ppo", minibatch_index=1)
        raise AssertionError("REPLAY GATE on the old log-probs did not fire")
    except TS.T8GateError as e:
        assert "REPLAY GATE" in str(e), str(e)
    print("gates: PASS")
    # review fixes: attempt-level GRPO, contiguity, head, all-or-nothing
    rows = make(version)
    for r in rows:
        r.trial_idx = r.trial_idx // 2 if False else r.trial_idx
    multi = copy.deepcopy(rows[:6])
    for k, r in enumerate(multi):
        r.trial_idx = k // 2                                # 3 attempts of two rows each
        r.group_idx = 0 if k < 4 else 1                     # group 0: attempts 0, 1; group 1: attempt 2
    KS.grpo_advantages_rows(multi, {0: 1.0, 1: 0.0, 2: 1.0})
    want = KS.grpo_advantages([1.0, 0.0, 1.0], [0, 0, 1])
    assert [r.advantage for r in multi] == [want[0]] * 2 + [want[1]] * 2 + [want[2]] * 2, [r.advantage for r in multi]
    try:
        KS.grpo_advantages_rows(multi, {0: 1.0, 1: 0.0})
        raise AssertionError("a missing score passed")
    except ValueError:
        pass
    inter = copy.deepcopy(rows)
    for k, r in enumerate(inter):
        r.group_idx = k % 3
    tr = FakeTrainer(cfg, 2)
    for bad_rows, what in ((inter, "not contiguous"),):
        try:
            TS.ppo_batch(tr, bad_rows, version, LR, 2, StepConfig(max_policy_lag=None))
            raise AssertionError(what)
        except ValueError as e:
            assert what in str(e), str(e)
    tr_c = FakeTrainer(TS.T8Config(head="centred", max_row_tokens=10 ** 6, segment_keys=(), run_dir=tmp, visits=5, top_k=2, K=5), 1)
    try:
        TS.ppo_batch(tr_c, copy.deepcopy(rows), version, LR, 1, StepConfig(max_policy_lag=None))
        raise AssertionError("ppo with the centred head")
    except ValueError:
        pass
    m0 = [m.detach().clone() for m in tr.opt.masters]
    bad3 = copy.deepcopy(rows)
    t = int(np.flatnonzero(bad3[-1].loss_mask)[0])
    bad3[-1].engine_lp[t] += 1e-3                           # the replay gate fires in the LAST mini-batch, after earlier optimizer steps
    try:
        TS.ppo_batch(tr, bad3, version, LR, 4, StepConfig(max_policy_lag=None))
        raise AssertionError("no gate")
    except TS.T8GateError:
        pass
    assert all(torch.equal(a, b) for a, b in zip(tr.opt.masters, m0)) and not tr.opt.opt.state, "not rolled back"
    res = TS.ppo_batch(tr, copy.deepcopy(rows), version, LR, 1, StepConfig(max_policy_lag=None))
    assert res[0][0].metrics["stale_checked_rows"] == len(rows)
    print("review fixes: PASS")
    print("T8 PPO S2 CPU TEST: PASS")
    return 0


def load_S():
    tu = SR.load("torch_utils_faae9039.py.txt", ["masked_mean", "safe_exp_delta"], {"torch": torch})
    from collections import defaultdict
    env = {"torch": torch, "np": np, "defaultdict": defaultdict, "Optional": __import__("typing").Optional, "Tuple": tuple, "List": list, "Float": object,
           "masked_mean": tu.masked_mean, "safe_exp_delta": tu.safe_exp_delta, "AlgorithmConfig": object,
           "apply_off_policy_correction": lambda loss, old, roll, mask, cfg: (loss, mask, {}), "PolicyLossType": None}
    return SR.load("ppo_utils_faae9039.py.txt", ["ppo_policy_loss", "reduce_loss"], env)


if __name__ == "__main__":
    sys.exit(main())
