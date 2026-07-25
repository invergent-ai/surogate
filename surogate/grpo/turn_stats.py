"""Turn-resolved supervision diagnostics for multi-turn GRPO / on-policy distillation.

Replicates the measurements in TurnOPD (arXiv:2607.05804) §4 against our own
runs, so the decision to adopt turn-level loss budgeting rests on our numbers
rather than theirs:

  * per-turn reverse KL           (their Fig. 2)
  * per-turn realized loss share  (their Eq. 11 / Fig. 4)
  * deep/shallow ratios + survivor support for the deepest third (their Table 1)

Nothing here changes the gradient. It only measures where the supervision
signal lives along the turn axis, which is currently unobservable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TurnAccumulator:
    """Accumulates per-turn statistics across the micro-batches of one step.

    Turn indices are 0-based and dense within a sample (see
    ``interleave_rollout``). Only supervised tokens (loss_mask=True) count —
    prompt, observation and padding tokens carry no gradient and would dilute
    every ratio if included.
    """

    kl_sum: dict[int, float] = field(default_factory=dict)
    # Magnitude of the same per-token KL. The k1 integrand is signed and its sign
    # varies along the turn axis (measured: -1.02 at turn 0, +0.12 at turn 11),
    # so signed sums cannot form a budget share — the denominator is a sum of
    # cancelling terms. "How much supervision signal lives at turn t" is a
    # magnitude, so budget shares use this and the signed mean stays for reading.
    abs_kl_sum: dict[int, float] = field(default_factory=dict)
    loss_sum: dict[int, float] = field(default_factory=dict)
    token_count: dict[int, int] = field(default_factory=dict)
    # Number of distinct samples that reached each turn — the survivor count.
    survivors: dict[int, int] = field(default_factory=dict)
    n_samples: int = 0

    def update(
        self,
        turn_ids: np.ndarray,
        loss_mask: np.ndarray,
        per_token_kl: np.ndarray,
        per_token_loss: np.ndarray,
        sample_ranges: list[tuple[int, int]],
    ) -> None:
        """Fold one micro-batch into the accumulator.

        Args:
            turn_ids: [T] int, -1 outside model turns
            loss_mask: [T] bool, True = supervised token
            per_token_kl: [T] float, reverse KL at each token
            per_token_loss: [T] float, realized loss magnitude at each token
            sample_ranges: (start, end) per packed sample, for survivor counting
        """
        supervised = loss_mask & (turn_ids >= 0)
        if not supervised.any():
            return

        turns = turn_ids[supervised]
        kls = per_token_kl[supervised]
        losses = per_token_loss[supervised]

        for t in np.unique(turns):
            sel = turns == t
            key = int(t)
            self.kl_sum[key] = self.kl_sum.get(key, 0.0) + float(kls[sel].sum())
            self.abs_kl_sum[key] = self.abs_kl_sum.get(key, 0.0) + float(np.abs(kls[sel]).sum())
            self.loss_sum[key] = self.loss_sum.get(key, 0.0) + float(np.abs(losses[sel]).sum())
            self.token_count[key] = self.token_count.get(key, 0) + int(sel.sum())

        # Survivor support: count each sample once per turn it actually reached.
        for start, end in sample_ranges:
            span = supervised[start:end]
            if not span.any():
                continue
            self.n_samples += 1
            for t in np.unique(turn_ids[start:end][span]):
                key = int(t)
                self.survivors[key] = self.survivors.get(key, 0) + 1

    def _turns(self) -> list[int]:
        return sorted(self.token_count)

    def per_turn_mean_kl(self) -> dict[int, float]:
        """Mean reverse KL per supervised token, by turn (their Fig. 2)."""
        return {t: self.kl_sum[t] / max(self.token_count[t], 1) for t in self._turns()}

    def per_turn_loss_share(self) -> dict[int, float]:
        """Fraction of realized |dloss| mass per turn — where the optimizer spends.

        Our objective is not their pure reverse-KL loss (we fold the teacher term
        into the advantage under an IPO mask and a squared-KL trust region), so
        the gradient magnitude actually reaching the LM head is the honest
        analogue of their Eq. 11 for this stack.
        """
        total = sum(self.loss_sum.values())
        if total <= 0:
            return {t: 0.0 for t in self._turns()}
        return {t: self.loss_sum[t] / total for t in self._turns()}

    def per_turn_mean_abs_kl(self) -> dict[int, float]:
        """Mean |reverse-KL| per supervised token — divergence magnitude by turn."""
        return {t: self.abs_kl_sum[t] / max(self.token_count[t], 1) for t in self._turns()}

    def per_turn_kl_share(self) -> dict[int, float]:
        """Fraction of total divergence magnitude per turn — their Eq. 11.

        Uses |k1| rather than signed k1: the share is a budget, so it must be a
        partition of a non-negative total. Signed sums cancel across turns and
        produce shares outside [0, 1] (observed: -71.9%).
        """
        total = sum(self.abs_kl_sum.values())
        if total <= 0:
            return {t: 0.0 for t in self._turns()}
        return {t: self.abs_kl_sum[t] / total for t in self._turns()}

    def per_turn_token_share(self) -> dict[int, float]:
        """Fraction of supervised tokens contributed by each turn (Eq. 10)."""
        total = sum(self.token_count.values())
        if total <= 0:
            return {t: 0.0 for t in self._turns()}
        return {t: self.token_count[t] / total for t in self._turns()}

    def summary(self) -> dict[str, float]:
        """Table 1 row: deep/shallow ratios over the deepest vs shallowest third.

        The turn axis is split into thirds by turn index. "Deep support" is the
        survivor coverage of the deepest third relative to turn 0, and "deep loss
        budget" is the share of realized loss mass it receives.
        """
        turns = self._turns()
        if not turns:
            return {}

        n = len(turns)
        third = max(n // 3, 1)
        shallow, deep = turns[:third], turns[-third:]

        mean_kl = self.per_turn_mean_kl()
        mean_abs_kl = self.per_turn_mean_abs_kl()
        loss_share = self.per_turn_loss_share()
        kl_share = self.per_turn_kl_share()

        # Ratio uses magnitude so it stays a meaningful "deep turns carry X% of
        # the divergence a shallow turn does" even when the signed mean flips.
        shallow_kl = float(np.mean([mean_abs_kl[t] for t in shallow]))
        deep_kl = float(np.mean([mean_abs_kl[t] for t in deep]))

        base_survivors = max(self.survivors.get(turns[0], 0), 1)
        deep_survivors = float(np.mean([self.survivors.get(t, 0) for t in deep]))

        return {
            "turn/num_turns_observed": float(n),
            "turn/max_turn": float(turns[-1]),
            # Table 1 columns
            "turn/deep_shallow_kl_ratio": deep_kl / shallow_kl if shallow_kl > 0 else 0.0,
            "turn/deep_support": deep_survivors / base_survivors,
            "turn/deep_kl_budget": float(sum(kl_share[t] for t in deep)),
            # Our own optimizer-budget analogue
            "turn/deep_loss_budget": float(sum(loss_share[t] for t in deep)),
            "turn/shallow_loss_budget": float(sum(loss_share[t] for t in shallow)),
            "turn/turn0_loss_budget": loss_share.get(turns[0], 0.0),
            "turn/mean_abs_kl_shallow": shallow_kl,
            "turn/mean_abs_kl_deep": deep_kl,
            # Signed means: negative = teacher assigns higher probability to the
            # student's own sampled tokens at that depth.
            "turn/mean_kl_shallow": float(np.mean([mean_kl[t] for t in shallow])),
            "turn/mean_kl_deep": float(np.mean([mean_kl[t] for t in deep])),
        }

    def as_rows(self) -> list[dict[str, float]]:
        """Per-turn detail, one row per turn — for offline plotting."""
        mean_kl = self.per_turn_mean_kl()
        mean_abs_kl = self.per_turn_mean_abs_kl()
        loss_share = self.per_turn_loss_share()
        kl_share = self.per_turn_kl_share()
        token_share = self.per_turn_token_share()
        return [
            {
                "turn": float(t),
                "mean_kl": mean_kl[t],
                "mean_abs_kl": mean_abs_kl[t],
                "loss_share": loss_share[t],
                "kl_share": kl_share[t],
                "token_share": token_share[t],
                "tokens": float(self.token_count[t]),
                "survivors": float(self.survivors.get(t, 0)),
            }
            for t in self._turns()
        ]


def blend_weight(turn_ids: np.ndarray, loss_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Per-token weight for TurnOPD's progressive turn normalization (§5.2).

    ``alpha=0`` reproduces trajectory-level (token-count) weighting exactly;
    ``alpha=1`` gives every turn equal total weight. Weights are normalized so
    they sum to the supervised token count, which keeps the overall gradient
    scale identical to the unweighted case — only the *distribution* across
    turns changes.

    This is provided so the diagnostic can report the counterfactual budget
    shift. It does not affect training until wired into the loss.
    """
    weights = np.zeros_like(turn_ids, dtype=np.float32)
    supervised = loss_mask & (turn_ids >= 0)
    if not supervised.any():
        return weights

    turns = turn_ids[supervised]
    uniq, counts = np.unique(turns, return_counts=True)
    n_total = counts.sum()
    n_turns = len(uniq)

    # q_traj = n_t / N and q_turn = 1/T are per-turn budgets; dividing by n_t
    # turns each into the per-token weight for tokens of that turn.
    per_turn_weight = {
        int(t): ((1.0 - alpha) * (c / n_total) + alpha * (1.0 / n_turns)) / c for t, c in zip(uniq, counts)
    }

    w = np.array([per_turn_weight[int(t)] for t in turns], dtype=np.float32)
    w *= n_total / w.sum()  # preserve total gradient scale
    weights[supervised] = w
    return weights
