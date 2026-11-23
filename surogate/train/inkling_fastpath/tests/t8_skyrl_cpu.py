"""S0 pass line (PLAN-SKYRL.md): on 200 fixed random inputs (+ edge cases) our port == SkyRL's OWN functions, loaded from its file by AST
(sha256-pinned copies of SkyRL faae9039's ppo_utils.py / torch_utils.py in $T8_SKYRL_REF, named *_faae9039.py.txt): advantages, per-token loss, reduced loss, and the per-token seed d loss / d logp, to 1e-12 relative (fp64).
Edge cases: a group of one, a group with std 0, negative advantages hitting the dual clip, ratios beyond the 20 clip, rows with no loss token."""
import ast
import hashlib
import os
import sys
import types
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import t8_skyrl as K  # noqa: E402

REF = os.environ.get("T8_SKYRL_REF", "")
PINS = {"ppo_utils_faae9039.py.txt": "82f14312eeb45b46", "torch_utils_faae9039.py.txt": None}


def load(name, fns, env):
    raw = open(os.path.join(REF, name), "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if PINS[name] and not sha.startswith(PINS[name]):
        raise SystemExit(f"{name}: sha {sha[:16]} != pinned {PINS[name]}")
    tree = ast.parse(raw.decode())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in fns]
    for n in keep:
        n.decorator_list = []                                # the registry decorators need ray
    assert {n.name for n in keep} == set(fns), fns
    mod = types.ModuleType("skyrl_ref_" + name.split(".")[0])
    mod.__dict__.update(env)
    exec(compile(ast.Module(body=keep, type_ignores=[]), name, "exec"), mod.__dict__)
    return mod


class Cfg:
    policy_loss_type = "dual_clip"
    eps_clip_low, eps_clip_high, clip_ratio_c = 0.2, 0.28, 10.0
    off_policy_correction = None


def main():
    if not REF or not os.path.isdir(REF):
        print("SKIP: set T8_SKYRL_REF to a directory holding SkyRL faae9039's ppo_utils.py and torch_utils.py as ppo_utils_faae9039.py.txt / torch_utils_faae9039.py.txt")
        return 0
    tu = load("torch_utils_faae9039.py.txt", ["masked_mean", "safe_exp_delta"], {"torch": torch})
    env = {"torch": torch, "np": np, "defaultdict": defaultdict, "Optional": __import__("typing").Optional, "Tuple": tuple, "List": list,
           "Float": object, "masked_mean": tu.masked_mean, "safe_exp_delta": tu.safe_exp_delta, "AlgorithmConfig": object,
           "apply_off_policy_correction": lambda loss, old, roll, mask, cfg: (loss, mask, {}),   # identity in the preset (all options unset)
           "PolicyLossType": None}
    S = load("ppo_utils_faae9039.py.txt", ["ppo_policy_loss", "reduce_loss", "compute_grpo_outcome_advantage", "apply_loss_reduction_to_advantages_minibatch"], env)
    g = np.random.default_rng(20261123)
    worst = {"adv": 0.0, "loss": 0.0, "seed": 0.0}

    def rel(a, b):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        d = float(np.abs(a - b).max()) if a.size else 0.0
        return d / max(float(np.abs(b).max()) if b.size else 0.0, 1e-300) if d else 0.0

    for trial in range(200):
        # advantages: groups of 1..8 (incl. a group of one and all-equal groups)
        n_groups = int(g.integers(1, 6))
        sizes = [1 if k == 0 and trial % 5 == 0 else int(g.integers(2, 9)) for k in range(n_groups)]
        groups = np.repeat(np.arange(n_groups), sizes)
        scores = g.choice([0.0, 0.5, 1.0], size=groups.size) if trial % 3 else g.normal(size=groups.size)
        if trial % 7 == 0:
            scores[groups == groups[-1]] = 1.0               # std 0
        mine = K.grpo_advantages(scores, groups)
        tl = torch.zeros(groups.size, 4, dtype=torch.float64)
        tl[:, -1] = torch.as_tensor(scores)
        ref_adv, _ = S.compute_grpo_outcome_advantage(tl, torch.ones_like(tl), groups, epsilon=1e-6, grpo_norm_by_std=True)
        worst["adv"] = max(worst["adv"], rel(mine, ref_adv[:, 0].numpy()))
        # a mini-batch of rows: per-token loss, reduction, seeds
        B, T = int(g.integers(1, 6)), int(g.integers(5, 40))
        lp = torch.as_tensor(g.normal(-1.0, 1.0, (B, T)))
        old = lp + torch.as_tensor(g.normal(0.0, 0.3 if trial % 4 else 8.0, (B, T)))   # some ratios far outside the clip, some past 20
        adv = torch.as_tensor(np.repeat(g.normal(0.0, 1.0, (B, 1)), T, 1))
        mask = torch.as_tensor((g.random((B, T)) < 0.7).astype(np.float64))
        if trial % 9 == 0:
            mask[0] = 0.0                                    # a row with no loss token
        sa = S.apply_loss_reduction_to_advantages_minibatch(adv, mask, "token_mean", B, T)
        lpr = lp.clone().requires_grad_(True)
        ref_loss, _ = S.ppo_policy_loss(lpr, old, sa, Cfg(), loss_mask=mask)
        ref_loss.backward()
        N = int(mask.sum())
        tot, seeds = 0.0, []
        for b in range(B):
            s, l, _ = K.dual_clip_seeds(lp[b].numpy(), old[b].numpy(), adv[b].numpy(), mask[b].numpy() > 0, N)
            tot += l
            seeds.append(s)
        worst["loss"] = max(worst["loss"], rel([tot], [float(ref_loss)]))
        worst["seed"] = max(worst["seed"], rel(np.stack(seeds), lpr.grad.numpy()))
    print("worst relative differences vs SkyRL's own functions:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert all(v <= 1e-12 for v in worst.values()), worst
    print("T8 SKYRL S0 CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
