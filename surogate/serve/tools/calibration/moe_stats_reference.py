"""PyTorch reference for the engine's MoE calibration statistics (api/ops/calibration.h).

  python -m surogate.serve.tools.calibration.moe_stats_reference DUMP_DIR [--device cuda]

DUMP_DIR is what `sinfer_moe_stats_test --dump DUMP_DIR` writes for its W8 case: every round's
inputs (x.bf16), the router (router.bf16), the active experts' int8 codes (experts.i8; each weight
is code * 2^-9 exactly, as the engine decodes it) and, under stats/<case>/, the statistics the
engine dumped. This recomputes them from first principles in FP64 -- the router's softmax top-k
renormalised, each selected expert's SwiGLU output y = W_down (silu(W_gate x) * (W_up x)) -- and
compares: counts and tokens exactly, gate sums to FP32 rounding, the norm sums to the engine's
BF16/FP32 arithmetic. Prints a JSON report and exits non-zero on a mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def bf16_file(path: Path, shape: tuple[int, ...]) -> torch.Tensor:
    bits = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(bits.view(np.float32).reshape(shape).copy()).to(torch.float64)


def npy(stats: Path, name: str) -> np.ndarray:
    return np.load(stats / name)


def reference(dump: Path, device: str) -> dict:
    case = json.loads((dump / "case.json").read_text())
    hidden, experts, top_k = case["hidden"], case["experts"], case["top_k"]
    inter, row, step = case["intermediate"], case["layer_row"], float(case["weight_step"])
    active, widths = case["active"], case["widths"]
    tokens = sum(widths)
    x = bf16_file(dump / "x.bf16", (tokens, hidden)).to(device)
    router = bf16_file(dump / "router.bf16", (experts + 1, hidden)).to(device)
    codes = np.fromfile(dump / "experts.i8", dtype=np.int8)
    per_expert = 2 * inter * hidden + hidden * inter
    if codes.size != per_expert * len(active):
        raise ValueError(f"experts.i8 holds {codes.size} codes, expected {per_expert * len(active)}")

    logits = x @ router[:experts].T  # [tokens, experts], FP64
    # The engine ranks the logits, ties to the lower id; softmax over the winners alone.
    order = torch.argsort(-logits, dim=1, stable=True)
    ids = order[:, :top_k]
    selected = torch.gather(logits, 1, ids)
    alpha = torch.softmax(selected, dim=1)

    counts = np.zeros(experts, dtype=np.uint64)
    gate = np.zeros(experts)
    norm = np.zeros(experts)
    gate_norm = np.zeros(experts)
    flat_ids = ids.reshape(-1)
    flat_alpha = alpha.reshape(-1)
    token_of = torch.arange(tokens, device=device).repeat_interleave(top_k)
    for index, expert in enumerate(active):
        block = torch.from_numpy(codes[index * per_expert:(index + 1) * per_expert].astype(np.float64)).to(device) * step
        gate_up = block[:2 * inter * hidden].reshape(2 * inter, hidden)
        down = block[2 * inter * hidden:].reshape(hidden, inter)
        mask = flat_ids == expert
        if not bool(mask.any()):
            continue
        xs = x[token_of[mask]]
        g, u = (xs @ gate_up[:inter].T), (xs @ gate_up[inter:].T)
        y = (torch.nn.functional.silu(g) * u) @ down.T
        lengths = torch.linalg.vector_norm(y, dim=1)
        a = flat_alpha[mask]
        counts[expert] = int(mask.sum())
        gate[expert] = float(a.sum())
        norm[expert] = float(lengths.sum())
        gate_norm[expert] = float((a * lengths).sum())
    unexpected = set(flat_ids.unique().tolist()) - set(active)
    if unexpected:
        raise ValueError(f"the reference routed to experts without weights: {sorted(unexpected)[:8]}")

    stats = Path(case["stats_dir"])
    if not stats.is_dir():
        stats = dump / "stats" / stats.name
    engine = {name: npy(stats, name + ".npy") for name in ("counts", "gate_sum", "norm_sum", "gate_norm_sum", "tokens")}
    report = {"case": case["stats_dir"], "tokens": tokens, "rounds": widths}

    def rel(a, b):
        mask = b != 0
        return float(np.max(np.abs(a[mask] - b[mask]) / np.abs(b[mask]))) if mask.any() else 0.0

    report["tokens_match"] = int(engine["tokens"][row]) == tokens
    report["counts_match"] = bool(np.array_equal(engine["counts"][row].astype(np.uint64), counts))
    report["gate_sum_max_rel"] = rel(engine["gate_sum"][row], gate)
    report["norm_sum_max_rel"] = rel(engine["norm_sum"][row], norm)
    report["gate_norm_sum_max_rel"] = rel(engine["gate_norm_sum"][row], gate_norm)
    report["reap_rank_agreement"] = float(np.mean(
        np.argsort(-np.where(counts > 0, gate_norm / np.maximum(counts, 1), -1))[:20] ==
        np.argsort(-np.where(engine["counts"][row] > 0, engine["gate_norm_sum"][row] / np.maximum(engine["counts"][row], 1), -1))[:20]))
    report["ok"] = (report["tokens_match"] and report["counts_match"] and report["gate_sum_max_rel"] < 1e-5
                    and report["norm_sum_max_rel"] < 2e-2 and report["gate_norm_sum_max_rel"] < 2e-2)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    report = reference(args.dump, args.device)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
