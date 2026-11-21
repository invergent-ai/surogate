"""One kit step on a list of rows, backend-agnostic (the trainer-side pieces of pkg-async-r3, re-implemented from the kit's
diffs; the loss itself is the deployed loss.py, loaded by kitsrc).

Order, as surogate/grpo/trainer.py 87ffd39f + recipe/trainer-patches {async-decoupled-ppo, gspo-token-aggregation, r3-dpp}:
 1. off-policy drop (trainer.py:853-881): a row whose trained tokens carry several versions, or one outside
    [version - max_policy_lag, version], loses its loss mask; a row whose trained tokens carry no version (all -1) is kept
    and counted as unversioned.
 2. step-start score (seq_ratio_old step_start): the backend's log-probs at the weights the step starts from;
    engine_mismatch_kl = mean over rows (with loss tokens) of mean_t (e^r - r - 1), r = lp_start - lp_engine
    (trainer.py:711-724); engine_log_ratio = mean over rows of mean_t r.
 3. token TIS (r3-py.diff behavior_weights, level "token"): w_t = min(cap, exp(clip(lp_start - lp_engine, -50, 50))) on
    loss tokens, 1 elsewhere; engine_behavior_w_{mean,min,max,capped,tokens} over the step's loss tokens.
 4. pi_old := lp_start; update pass: the backend's log-probs WITH grad at the same weights; per-row seeds from the
    deployed compute_grpo_per_token_grads(lp_new, lp_start, A, mask, cfg, [(0, T)], teacher); seeds *= w (TIS);
    loss = -sum_t seed_t * lp_new_t / scale, scale = the step's loss tokens after the drop (aggregation token).
 5. optimizer step by the caller (fp32 AdamW, grad clip 1.0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .kitsrc import loss_api
from .records import Row, replay_rows


@dataclass
class StepConfig:
    variant: str = "gspo"
    seq_ratio_old: str = "step_start"
    seq_clip_low: float = 3.0e-4
    seq_clip_high: float = 4.0e-4
    teacher_tau: float = 0.3
    behavior_is_cap: float | None = 2.0
    behavior_is_level: str = "token"
    aggregation: str = "token"
    max_policy_lag: int | None = 1

    @classmethod
    def from_train_yaml(cls, cfg: dict) -> "StepConfig":
        loss = dict(cfg.get("loss") or {})
        out = cls(max_policy_lag=cfg.get("max_policy_lag"))
        for k in ("variant", "seq_ratio_old", "seq_clip_low", "seq_clip_high", "teacher_tau", "behavior_is_cap",
                  "behavior_is_level", "aggregation"):
            if k in loss:
                setattr(out, k, loss[k])
        if out.variant != "gspo" or out.seq_ratio_old != "step_start":
            raise ValueError("the Inkling shim implements the kit's gspo + seq_ratio_old step_start path only")
        if out.behavior_is_level not in ("token", "sequence"):
            raise ValueError(out.behavior_is_level)
        return out

    def loss_config(self):
        GRPOLossConfig, _ = loss_api()
        return GRPOLossConfig(variant=self.variant, seq_ratio_old=self.seq_ratio_old, seq_clip_low=self.seq_clip_low,
                              seq_clip_high=self.seq_clip_high, teacher_tau=self.teacher_tau,
                              behavior_is_cap=self.behavior_is_cap, aggregation=self.aggregation)


def drop_off_policy(rows: list[Row], version: int | None, lag: int | None) -> dict[str, int]:
    counts = {"samples": 0, "tokens": 0, "unversioned": 0}
    if lag is None or version is None:
        return counts
    for r in rows:
        trained = r.loss_mask
        if not trained.any():
            continue
        vs = r.versions[trained] if r.versions is not None else np.array([], np.int32)
        if not len(vs) or (vs < 0).all():
            counts["unversioned"] += 1
            continue
        distinct = set(int(v) for v in vs)
        if len(distinct) > 1 or not version - lag <= next(iter(distinct)) <= version:
            counts["samples"] += 1
            counts["tokens"] += int(trained.sum())
            r.loss_mask = np.zeros_like(r.loss_mask)
    return counts


def behavior_weights_token(prox: np.ndarray, beh: np.ndarray, mask: np.ndarray, cap: float):
    w = np.ones(prox.shape, dtype=np.float32)
    if not mask.any():
        return w, np.zeros(0, np.float32)
    r = np.clip(prox[mask].astype(np.float64) - beh[mask].astype(np.float64), -50.0, 50.0)
    x = np.minimum(cap, np.exp(r)).astype(np.float32)
    w[mask] = x
    return w, x


def behavior_weights_sequence(prox: np.ndarray, beh: np.ndarray, mask: np.ndarray, cap: float):
    w = np.ones(prox.shape, dtype=np.float32)
    if not mask.any():
        return w, np.zeros(0, np.float32)
    x = float(min(cap, math.exp(float(np.mean(prox[mask].astype(np.float64) - beh[mask].astype(np.float64))))))
    w[mask] = x
    return w, np.asarray([x], np.float32)


def mismatch(lp_start: np.ndarray, lp_engine: np.ndarray, mask: np.ndarray) -> tuple[float, float] | None:
    if not mask.any():
        return None
    r = lp_start[mask].astype(np.float64) - lp_engine[mask].astype(np.float64)
    return float(np.mean(np.exp(r) - r - 1.0)), float(np.mean(r))


@dataclass
class StepResult:
    metrics: dict = field(default_factory=dict)
    loss_tokens: int = 0
    rows: int = 0


def run_step(backend, rows: list[Row], version: int | None, cfg: StepConfig, log=print) -> StepResult:
    """Steps 1-4 of the module docstring; leaves the gradients in the backend's parameters (the caller steps the
    optimizer). backend: score(row, replay) -> np.ndarray [T] (no grad); logprobs_grad(row, replay) -> torch [T]."""
    import torch

    _, grads_fn = loss_api()
    lcfg = cfg.loss_config()
    out = StepResult()
    m = out.metrics
    dropped = drop_off_policy(rows, version, cfg.max_policy_lag)
    m["offpolicy_dropped_samples"] = float(dropped["samples"])
    m["offpolicy_dropped_tokens"] = float(dropped["tokens"])
    m["unversioned_samples"] = float(dropped["unversioned"])
    if dropped["samples"]:
        log(f"max_policy_lag {cfg.max_policy_lag}: dropped {dropped['samples']} samples ({dropped['tokens']} loss tokens) "
            f"not sampled by policy version {version - cfg.max_policy_lag}..{version} alone")
    r3 = {"r3_rows": 0.0, "r3_replayed_tokens": 0.0, "r3_bad_rows": 0.0, "r3_shape_mismatches": 0.0,
          "r3_uncovered_interior": 0.0}
    live = [i for i, r in enumerate(rows) if r.loss_mask.any()]
    replays = {}
    for i in live:
        rp, c = replay_rows(rows[i])
        replays[i] = rp
        for k, v in c.items():
            r3[k] += v
    # step-start score (pass 1) -> mismatch metric, TIS weights, pi_old
    kl, lr_, bw = [], [], []
    starts, wts = {}, {}
    for i in live:
        r = rows[i]
        lp = np.asarray(backend.score(r, replays[i]), dtype=np.float64)
        mm = mismatch(lp, r.engine_lp, r.loss_mask)
        if mm is not None:
            kl.append(mm[0])
            lr_.append(mm[1])
        if cfg.behavior_is_cap is not None:
            fn = behavior_weights_token if cfg.behavior_is_level == "token" else behavior_weights_sequence
            wts[i], x = fn(lp, r.engine_lp, r.loss_mask, float(cfg.behavior_is_cap))
            if len(x):
                bw.append(x)
        starts[i] = lp
    if kl:
        m["engine_mismatch_kl"] = float(np.mean(kl))
        m["engine_log_ratio"] = float(np.mean(lr_))
    if bw:
        b = np.concatenate(bw)
        cap = float(cfg.behavior_is_cap)
        m.update({"engine_behavior_w_mean": float(np.mean(b)), "engine_behavior_w_min": float(np.min(b)),
                  "engine_behavior_w_max": float(np.max(b)), "engine_behavior_w_capped": float(np.mean(b >= cap)),
                  "engine_behavior_w_tokens": float(b.size)})
    scale = float(max(sum(int(rows[i].loss_mask.sum()) for i in live), 1))
    if cfg.aggregation != "token":
        trials = {rows[i].trial_idx if rows[i].trial_idx is not None else ("row", i) for i in live}
        scale = float(max(len(trials), 1))
    # update pass (pass 2)
    agg = {"policy_loss": 0.0, "mismatch_kl": [], "is_masked": [], "seq_log_ratio": [], "total_tokens": 0}
    for i in live:
        r = rows[i]
        lp_t = backend.logprobs_grad(r, replays[i])
        lp_new = lp_t.detach().double().cpu().numpy()
        adv = np.full(r.T, r.advantage, dtype=np.float64)
        seeds, met = grads_fn(lp_new, starts[i], adv, r.loss_mask, lcfg, [(0, r.T)], teacher_logprobs=r.teacher_lp)
        if i in wts:
            seeds = seeds * wts[i]
        seeds_t = torch.as_tensor(seeds, dtype=torch.float32, device=lp_t.device)
        loss = -(seeds_t * lp_t).sum() / scale
        loss.backward()
        n = int(r.loss_mask.sum())
        agg["policy_loss"] += met["policy_loss"] * n
        agg["total_tokens"] += n
        agg["mismatch_kl"].append((met["mismatch_kl"], n))
        agg["is_masked"].append((met["is_masked"], n))
        mk = r.loss_mask
        agg["seq_log_ratio"].append(float(np.mean(lp_new[mk] - starts[i][mk])) if mk.any() else 0.0)
    tot = max(agg["total_tokens"], 1)
    m["policy_loss"] = agg["policy_loss"] / tot
    m["mismatch_kl"] = sum(v * n for v, n in agg["mismatch_kl"]) / tot
    m["is_masked"] = sum(v * n for v, n in agg["is_masked"]) / tot
    m["seq_log_ratio"] = float(np.mean(agg["seq_log_ratio"])) if agg["seq_log_ratio"] else 0.0
    m["total_tokens"] = agg["total_tokens"]
    m["expected_loss_scale"] = scale
    # the kit counts R3 per pass over a row (gated_start criterion 2: replayed = passes x covered); two passes here
    r3["r3_rows"] *= 2
    r3["r3_replayed_tokens"] *= 2
    m.update({k: float(v) for k, v in r3.items()})
    out.loss_tokens = agg["total_tokens"]
    out.rows = len(live)
    return out
