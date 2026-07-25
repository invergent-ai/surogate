"""Summarize turn_stats.jsonl into the TurnOPD Table 1 comparison.

Usage: python examples/turnopd/analyze.py outputs/turnopd_diag/turn_stats.jsonl
"""

import json
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def phase_of(step, max_step):
    if max_step < 3:
        return "all"
    if step < max_step / 3:
        return "early"
    if step < 2 * max_step / 3:
        return "mid"
    return "late"


def thirds(turns):
    n = len(turns)
    k = max(n // 3, 1)
    return turns[:k], turns[-k:]


def summarize(rows, label):
    by_turn_tokens = defaultdict(float)
    by_turn_kl_mass = defaultdict(float)
    by_turn_loss_mass = defaultdict(float)
    by_turn_survivors = defaultdict(float)
    steps = set()

    for r in rows:
        t = int(r["turn"])
        steps.add(r["step"])
        by_turn_tokens[t] += r["tokens"]
        # Budget shares must partition a non-negative total, so use |KL|. The
        # signed k1 mean flips sign along the turn axis; summing it produces
        # cancelling denominators and shares outside [0, 1].
        by_turn_kl_mass[t] += r.get("mean_abs_kl", abs(r["mean_kl"])) * r["tokens"]
        by_turn_loss_mass[t] += r["loss_share"]
        by_turn_survivors[t] += r["survivors"]

    turns = sorted(by_turn_tokens)
    if not turns:
        return None
    shallow, deep = thirds(turns)

    def mean_kl(ts):
        tok = sum(by_turn_tokens[t] for t in ts)
        return sum(by_turn_kl_mass[t] for t in ts) / tok if tok else 0.0

    kl_total = sum(by_turn_kl_mass.values())
    loss_total = sum(by_turn_loss_mass.values())
    base_surv = by_turn_survivors[turns[0]] or 1.0

    shallow_kl, deep_kl = mean_kl(shallow), mean_kl(deep)
    return {
        "phase": label,
        "n_steps": len(steps),
        "max_turn": turns[-1],
        "deep_shallow_kl": deep_kl / shallow_kl if shallow_kl else 0.0,
        "deep_support": sum(by_turn_survivors[t] for t in deep) / len(deep) / base_surv,
        "deep_kl_budget": sum(by_turn_kl_mass[t] for t in deep) / kl_total if kl_total else 0.0,
        "deep_loss_budget": sum(by_turn_loss_mass[t] for t in deep) / loss_total if loss_total else 0.0,
        "turn0_loss_budget": by_turn_loss_mass[turns[0]] / loss_total if loss_total else 0.0,
        "deep_token_share": sum(by_turn_tokens[t] for t in deep) / sum(by_turn_tokens.values()),
    }


def main(path):
    rows = load(path)
    if not rows:
        print(f"no rows in {path}")
        return
    max_step = max(r["step"] for r in rows)

    print(f"\n{'=' * 92}")
    print("TurnOPD Table 1 replication — our stack, our env")
    print(f"{'=' * 92}")
    hdr = f"{'Phase':<8}{'steps':>6}{'maxT':>6}{'deep/shallow KL':>18}{'deep support':>14}{'deep KL budget':>16}{'deep loss budget':>18}"
    print(hdr)
    print("-" * 92)

    buckets = defaultdict(list)
    for r in rows:
        buckets[phase_of(r["step"], max_step)].append(r)

    for label in ("early", "mid", "late", "all"):
        subset = buckets.get(label) if label != "all" else rows
        if not subset:
            continue
        s = summarize(subset, label)
        if s is None:
            continue
        print(
            f"{s['phase']:<8}{s['n_steps']:>6}{s['max_turn']:>6}"
            f"{s['deep_shallow_kl']:>17.0%}{s['deep_support']:>14.1%}"
            f"{s['deep_kl_budget']:>16.1%}{s['deep_loss_budget']:>18.1%}"
        )

    print("-" * 92)
    print("Paper's reference values (ALFWorld / Multi-Hop Search):")
    print(f"{'ALFWorld early':<26}{'31%':>17}{'23.0%':>14}{'3.6%':>16}")
    print(f"{'ALFWorld late':<26}{'42%':>17}{'18.2%':>14}{'4.5%':>16}")
    print(f"{'MultiHop early':<26}{'90%':>17}{'17.3%':>14}{'12.9%':>16}")
    print(f"{'MultiHop late':<26}{'92%':>17}{'15.5%':>14}{'11.1%':>16}")

    # Per-turn detail, averaged over the run.
    print(f"\n{'=' * 92}")
    print("Per-turn profile (run average)")
    print(f"{'=' * 92}")
    per_turn = defaultdict(lambda: {"tokens": 0.0, "kl": 0.0, "abskl": 0.0, "loss": 0.0, "surv": 0.0, "n": 0})
    for r in rows:
        d = per_turn[int(r["turn"])]
        d["tokens"] += r["tokens"]
        d["kl"] += r["mean_kl"] * r["tokens"]
        d["abskl"] += r.get("mean_abs_kl", abs(r["mean_kl"])) * r["tokens"]
        d["loss"] += r["loss_share"]
        d["surv"] += r["survivors"]
        d["n"] += 1

    loss_total = sum(d["loss"] for d in per_turn.values())
    tok_total = sum(d["tokens"] for d in per_turn.values())
    base = per_turn[min(per_turn)]["surv"] or 1.0
    print(
        f"{'turn':>5}{'tokens':>10}{'tok share':>11}{'mean KL':>10}{'mean |KL|':>11}{'loss share':>12}{'survival':>10}"
    )
    print("-" * 92)
    for t in sorted(per_turn):
        d = per_turn[t]
        mean_kl = d["kl"] / max(d["tokens"], 1)
        mean_abs = d["abskl"] / max(d["tokens"], 1)
        loss_share = d["loss"] / loss_total if loss_total else 0.0
        print(
            f"{t:>5}{d['tokens']:>10.0f}{d['tokens'] / tok_total:>11.1%}"
            f"{mean_kl:>10.3f}{mean_abs:>11.3f}{loss_share:>12.1%}{d['surv'] / base:>10.1%}"
        )
    print("\n  mean KL is signed (log pi_S - log pi_T); negative = teacher assigns")
    print("  higher probability to the student's own sampled tokens at that depth.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/turnopd_diag/turn_stats.jsonl")
