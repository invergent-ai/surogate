"""S0: SkyRL's policy loss, advantage and reduction for the Inkling-2-Mini trainer (SkyRL faae9039, preset `lora`).

Written from SkyRL's code, line for line, in fp64 on the host (the per-row seeds of our step):
- GRPO advantage (`ppo_utils.py:1411-1460`, `compute_grpo_outcome_advantage`): per group, mean of the scores and torch.std (unbiased), both in fp32 as SkyRL's calls give them; a group of
  ONE gets mean 0, std 1; advantage = (score - mean) / (std + eps), eps 1e-6, with `grpo_norm_by_std=True` (`mini:105`); without it, score - mean.
- dual-clip PPO (`ppo_utils.py:566-597`): ratio = exp(clamp(logp - old, -20, 20)) (`torch_utils.py:188-193`, computed in fp32 there),
  surr1 = ratio A, surr2 = clamp(ratio, 1 - eps_low, 1 + eps_high) A, loss = -min(surr1, surr2); where A < 0: min(-A c, loss).
  Off-policy correction: identity in the preset (all of its options unset; `off_policy_correction_utils.py:422-442`).
- token mean (`ppo_utils.py:1212-1250`): advantages / the MINI-BATCH's loss tokens, then a plain masked sum.
Our step needs d loss / d logp per token: `dual_clip_seeds` returns it, with torch's own subgradient choices at the kinks (taken from torch autograd on
the same expression, so it matches SkyRL's backward exactly).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SkyRLLoss:
    eps_clip_low: float = 0.2      # mini:107
    eps_clip_high: float = 0.28    # mini:108
    clip_ratio_c: float = 10.0     # mini:109
    ratio_delta_clip: float = 20.0  # torch_utils.py:188
    grpo_eps: float = 1e-6         # ppo_utils.py:1415
    grpo_norm_by_std: bool = True  # mini:105


def grpo_advantages(scores, groups, cfg: SkyRLLoss = SkyRLLoss()) -> np.ndarray:
    """scores [n] (one scalar reward per sample), groups [n] (the prompt / task id) -> advantages [n], fp64."""
    s = torch.as_tensor(np.asarray(scores, dtype=np.float64))
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    out = torch.empty_like(s)
    for g, idx in by.items():
        v = s[idx]
        lst = [s[i] for i in idx]
        if len(idx) == 1:
            mean, std = torch.tensor(0.0), torch.tensor(1.0)
        else:  # SkyRL's exact calls (the list keeps the scores' dtype: fp64 here; SkyRL's production rewards are fp32)
            mean, std = torch.mean(torch.tensor(lst)), torch.std(torch.tensor([lst]))
        out[idx] = (v - mean) / (std + cfg.grpo_eps) if cfg.grpo_norm_by_std else v - mean
    return out.numpy()


def grpo_advantages_rows(rows: list, score_by_trial: dict, cfg: SkyRLLoss = SkyRLLoss()) -> None:
    """GRPO over ATTEMPTS, broadcast to their rows (review M3; SkyRL's step-wise mode does the same, trainer.py:1254-1290): one score per trial_idx,
    groups = the rows' group_idx (one per trial), then row.advantage = its attempt's advantage. A group of one attempt gets mean 0, std 1 (SkyRL)."""
    trial_group = {}
    for r in rows:
        if r.trial_idx is None or r.group_idx is None:
            raise ValueError("every row needs trial_idx and group_idx")
        if trial_group.setdefault(r.trial_idx, r.group_idx) != r.group_idx:
            raise ValueError(f"trial {r.trial_idx} has rows in two groups")
    trials = sorted(trial_group, key=str)
    missing = [t for t in trials if t not in score_by_trial]
    if missing:
        raise ValueError(f"no score for trials {missing[:5]}")
    adv = grpo_advantages([float(score_by_trial[t]) for t in trials], [trial_group[t] for t in trials], cfg)
    by = dict(zip(trials, adv))
    for r in rows:
        r.advantage = float(by[r.trial_idx])


def _ratio(lp, old, cfg):
    # SkyRL: safe_exp_delta(delta, clip=20.0, out_dtype=log_probs.dtype): clamp in fp32, exp in fp32, cast back
    return torch.clamp((lp - old).to(torch.float32), -cfg.ratio_delta_clip, cfg.ratio_delta_clip).exp().to(lp.dtype)


def dual_clip_token_loss(lp: torch.Tensor, old: torch.Tensor, adv: torch.Tensor, cfg: SkyRLLoss = SkyRLLoss()) -> torch.Tensor:
    """Per-token loss before the reduction (SkyRL's expression, same op order)."""
    ratio = _ratio(lp, old, cfg)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - cfg.eps_clip_low, 1 + cfg.eps_clip_high) * adv
    loss = -torch.min(surr1, surr2)
    pg_losses3 = -adv * cfg.clip_ratio_c
    clip2 = torch.min(pg_losses3, loss)
    return torch.where(adv < 0, clip2, loss)


def dual_clip_seeds(lp: np.ndarray, old: np.ndarray, adv_tok: np.ndarray, mask: np.ndarray, n_minibatch_tokens: int,
                    cfg: SkyRLLoss = SkyRLLoss()):
    """One row of a mini-batch. lp, old, adv_tok, mask: [T] (adv_tok = the row's advantage at every token; mask = its loss tokens).
    Returns (seeds [T] fp64 = d loss / d lp, the row's contribution to the mini-batch loss, metrics). loss = sum_t mask_t L_t / N_minibatch."""
    lp_t = torch.as_tensor(np.asarray(lp, dtype=np.float64)).requires_grad_(True)
    old_t = torch.as_tensor(np.asarray(old, dtype=np.float64))
    m = torch.as_tensor(np.asarray(mask, dtype=np.float64))
    a = torch.as_tensor(np.asarray(adv_tok, dtype=np.float64)) / max(int(n_minibatch_tokens), 1)  # ppo_utils.py:1247-1248 (clamp(min=1))
    tok = dual_clip_token_loss(lp_t, old_t, a, cfg)
    loss = (tok * m).sum()                                    # reduce_loss, ppo_utils.py:1212-1216
    loss.backward()
    with torch.no_grad():
        ratio = _ratio(lp_t, old_t, cfg)
        surr1, surr2 = ratio * a, ratio.clamp(1 - cfg.eps_clip_low, 1 + cfg.eps_clip_high) * a
        clipped = float(((-surr2 > -surr1).double() * m).sum())
    met = {"loss": float(loss), "clipped_tokens": clipped, "loss_tokens": float(m.sum()),
           "ratio_max": float((ratio * m).max()) if bool(m.any()) else 1.0}
    return lp_t.grad.numpy().copy(), float(loss), met


BLOCKS = {"before": range(0, 6), "inside": range(6, 18), "after": range(18, 24)}  # physical layers: prelude, looped core, coda


def first_order_by_block(grad_ascent: list, delta: list, keys: tuple, blocks: dict = BLOCKS) -> dict:
    """The loop's follow read: the predicted first-order change <g, dθ> per layer block and in all.
    grad_ascent: per (layer, key) in the trainer's gradient order (layer-major, `keys` per layer), the gradient of the quantity whose change is
    predicted (e.g. sum over the trained rows of A x sum_t log pi: StepOut.grads of seeds="weights" with weight = the advantage, sense "ascent";
    or -N x StepOut.grads of the FIRST ppo mini-batch, where the ratio is exactly 1 and nothing is clipped). delta: the update, same order
    (e.g. masters after - masters before, scaled as the read needs). Returns {"before", "inside", "after", "all"}: float64 sums."""
    nk = len(keys)
    if len(grad_ascent) != len(delta) or len(delta) % nk:
        raise ValueError("gradient and update lists differ")
    out = {b: 0.0 for b in blocks}
    for i, (g, d) in enumerate(zip(grad_ascent, delta)):
        L = i // nk
        v = float((torch.as_tensor(g).double() * torch.as_tensor(d).double()).sum())
        for b, rng in blocks.items():
            if L in rng:
                out[b] += v
    out["all"] = sum(out[b] for b in blocks)
    return out
