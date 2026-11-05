"""QSA parity: the engine's sparse attention (block selection and attention output) against
transformers' Qwen4ExpTextQSAIndexer, on the engine's own dumped inputs.

  python -m surogate.serve.tools.parity.qwen4exp.qsa_parity --dump DIR --release DIR --layer L \
      [--prefill-rows 256] [--json OUT]

DIR holds an engine run with SUROGATE_SERVE_QSA_DUMP=DIR SUROGATE_SERVE_QSA_DUMP_LAYER=L (and
--enforce-eager, so decode steps are dumped too; csrc/src/serve/family/impl/runtime/qsa_dump.h).
The sequence is rebuilt from the dumps by cache position: the indexer input hidden states, and
the attention's query, key and value after their norms and rope. The reference indexer runs the
release's BF16 weights (index_qk_proj, q_layernorm, k_layernorm) with the model's own rotary
embedding, and selects per query row exactly as the transformers code does: complete 4-token
blocks of the visible tokens, mean-pooled raw keys, k_layernorm, rope at the block's first
position, scores sum_h relu(q_h . k_block) / sqrt(head_dim), the top budget/4 blocks, plus the
incomplete tail. For each checked row it reports whether the engine's block set equals the
reference's, and the engine's attention output (before the output gate) against the reference
attention over the reference selection and over the engine's own selection -- which separates a
selection error from an attention-kernel error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


def load(dump: Path, call: int, layer: int, name: str):
    path = dump / f"c{call}_L{layer}_{name}.npy"
    return np.load(path) if path.exists() else None


def bf16(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.uint16).view(np.int16)).view(torch.bfloat16)


def rotated(result):
    """apply_rotary_pos_emb returns the queries alone when no keys are given, else a pair."""
    return result[0] if isinstance(result, tuple) else result


def release_tensor(release: Path, name: str) -> torch.Tensor:
    from safetensors import safe_open
    index = json.loads((release / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(release / index[name]), "pt") as handle:
        return handle.get_tensor(name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--prefill-rows", type=int, default=256, help="prefill rows checked per call (spread evenly)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--first-call", type=int, default=0,
                    help="ignore calls numbered below this (the engine's startup forwards)")
    args = ap.parse_args(argv)

    from transformers import AutoConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp as ref

    calls = [json.loads(line) for line in (args.dump / "index.jsonl").read_text().splitlines() if line.strip()]
    calls = [c for c in calls if c["layer"] == args.layer and c["call"] >= args.first_call]
    if not calls:
        raise SystemExit(f"no dumped calls for layer {args.layer}")
    config = AutoConfig.from_pretrained(str(args.release))
    text = getattr(config, "text_config", config)

    # Rebuild the sequence by cache position.
    per_call = []
    max_position = -1
    for c in calls:
        positions = load(args.dump, c["call"], args.layer, "cache_positions").reshape(-1).astype(np.int64)
        per_call.append(positions)
        max_position = max(max_position, int(positions.max()))
    n = max_position + 1
    hidden = text.hidden_size
    head_dim, q_heads, kv_heads = calls[0]["head_dim"], calls[0]["q_heads"], calls[0]["kv_heads"]
    H = torch.zeros(n, hidden, dtype=torch.bfloat16)
    K = torch.zeros(n, kv_heads, head_dim, dtype=torch.bfloat16)
    V = torch.zeros(n, kv_heads, head_dim, dtype=torch.bfloat16)
    seen = np.zeros(n, dtype=bool)
    for c, positions in zip(calls, per_call):
        h = bf16(load(args.dump, c["call"], args.layer, "h")).reshape(-1, hidden)
        k = bf16(load(args.dump, c["call"], args.layer, "k")).reshape(-1, kv_heads, head_dim)
        v = bf16(load(args.dump, c["call"], args.layer, "v")).reshape(-1, kv_heads, head_dim)
        idx = torch.from_numpy(positions)
        H[idx], K[idx], V[idx] = h, k, v
        seen[positions] = True
    if not seen.all():
        raise SystemExit(f"positions {np.flatnonzero(~seen)[:8]}... were not dumped (dump every chunk of the prompt)")

    # The reference indexer with the release's BF16 weights and the model's rotary embedding.
    indexer = ref.Qwen4ExpTextQSAIndexer(text, args.layer)
    prefix = f"model.language_model.layers.{args.layer}.self_attn.indexer."
    with torch.no_grad():
        for name, module in (("index_qk_proj", indexer.index_qk_proj), ("q_layernorm", indexer.q_layernorm),
                             ("k_layernorm", indexer.k_layernorm)):
            module.weight.copy_(release_tensor(args.release, prefix + name + ".weight").to(module.weight.dtype))
    indexer = indexer.to(torch.bfloat16)
    rotary = ref.Qwen4ExpTextRotaryEmbedding(text)
    position_ids = torch.arange(n)[None, None, :].expand(3, 1, n)
    cos, sin = rotary(H[None], position_ids)  # [1, n, rotary]
    with torch.no_grad():
        qk = indexer.index_qk_proj(H)
        q_idx, raw_k = torch.split(qk, [indexer.index_n_heads * indexer.index_head_dim,
                                        indexer.index_kv_heads * indexer.index_head_dim], dim=-1)
        q_idx = indexer.q_layernorm(q_idx.reshape(1, n, indexer.index_n_heads, indexer.index_head_dim))
        q_idx = rotated(ref.apply_rotary_pos_emb(q_idx, cos=cos, sin=sin, unsqueeze_dim=2))[0].float()  # [n, heads, d]
        raw_k = raw_k.reshape(n, indexer.index_head_dim)
        ratio, budget_blocks = indexer.compress_ratio, indexer.block_topk
        blocks = n // ratio
        pooled = raw_k[: blocks * ratio].view(blocks, ratio, -1).float().mean(dim=1).to(raw_k.dtype)
        pooled = indexer.k_layernorm(pooled)
        starts = torch.arange(blocks) * ratio
        block_keys = rotated(ref.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0, starts], sin=sin[0, starts]))
        block_keys = block_keys.squeeze(1).float()  # [blocks, d]

    scale = calls[0]["attention_scale"]
    group = q_heads // kv_heads

    def attend(row_q: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        keys, values = K[tokens].float(), V[tokens].float()  # [t, kv, d]
        out = torch.empty(q_heads, head_dim)
        for hq in range(q_heads):
            kv = hq // group
            weights = torch.softmax((keys[:, kv] @ row_q[hq].float()) * scale, dim=0)
            out[hq] = weights @ values[:, kv]
        return out

    def tokens_of(block_set, position: int) -> torch.Tensor:
        complete = (position + 1) // ratio
        parts = [torch.arange(b * ratio, b * ratio + ratio) for b in sorted(block_set) if b < complete]
        parts.append(torch.arange(complete * ratio, position + 1))
        return torch.cat(parts) if parts else torch.arange(0)

    report = {"layer": args.layer, "positions": n, "calls": len(calls), "budget_blocks": budget_blocks,
              "phases": {}}
    rows_out = []
    for c, positions in zip(calls, per_call):
        phase = "decode" if c["columns"] <= 4 else "prefill"
        T = positions.shape[0]
        mask = load(args.dump, c["call"], args.layer, "mask")
        q_att = bf16(load(args.dump, c["call"], args.layer, "q")).reshape(T, q_heads, head_dim)
        a_eng = bf16(load(args.dump, c["call"], args.layer, "attention")).reshape(T, q_heads, head_dim).float()
        chosen = range(T) if phase == "decode" or T <= args.prefill_rows else \
            sorted(set(np.linspace(0, T - 1, args.prefill_rows).astype(int).tolist()))
        for i in chosen:
            p = int(positions[i])
            complete = (p + 1) // ratio
            scores = None
            if complete > budget_blocks:
                scores = torch.relu(q_idx[p] @ block_keys[:complete].T).sum(dim=0) / math.sqrt(indexer.index_head_dim)
                ref_set = set(scores.topk(min(budget_blocks, complete)).indices.tolist())
            else:
                ref_set = set(range(complete))
            if mask is not None:
                words = mask[i].astype(np.uint64)
                bits = [w * 32 + b for w in range(words.shape[0]) for b in range(32) if (int(words[w]) >> b) & 1]
                eng_set = {b for b in bits if b < complete}
                tail_marked = any(b >= complete for b in bits)
            else:
                eng_set, tail_marked = set(range(complete)), False
            same = eng_set == ref_set
            boundary = {}
            if scores is not None and not same:
                order = torch.argsort(scores, descending=True)
                rank = torch.empty_like(order)
                rank[order] = torch.arange(order.numel())
                cut = float(scores[order[budget_blocks - 1]])
                extra = sorted(eng_set - ref_set)
                missing = sorted(ref_set - eng_set)
                boundary = {"extra_max_rank": int(max(rank[b] for b in extra)) if extra else None,
                            "missing_min_rank": int(min(rank[b] for b in missing)) if missing else None,
                            "extra_max_rel_gap": float(max((cut - float(scores[b])) / max(abs(cut), 1e-30) for b in extra)) if extra else 0.0,
                            "differing": len(extra)}
            out_ref = attend(q_att[i], tokens_of(ref_set, p))
            out_eng_sel = attend(q_att[i], tokens_of(eng_set, p))
            def rel(a, b):
                return float((a - b).norm() / max(b.norm(), 1e-12))
            rows_out.append({"call": c["call"], "phase": phase, "position": p, "sparse": complete > budget_blocks,
                             "selection_equal": same, "selected_engine": len(eng_set), "selected_reference": len(ref_set),
                             "jaccard": len(eng_set & ref_set) / max(1, len(eng_set | ref_set)),
                             "tail_bits": tail_marked, **boundary,
                             "attention_vs_reference_selection": rel(a_eng[i], out_ref),
                             "attention_vs_engine_selection": rel(a_eng[i], out_eng_sel)})
    for phase in ("prefill", "decode"):
        for sparse in (False, True):
            rs = [r for r in rows_out if r["phase"] == phase and r["sparse"] == sparse]
            if not rs:
                continue
            key = f"{phase}_{'sparse' if sparse else 'dense'}"
            report["phases"][key] = {
                "rows": len(rs),
                "selection_equal": sum(r["selection_equal"] for r in rs),
                "jaccard_mean": float(np.mean([r["jaccard"] for r in rs])),
                "jaccard_min": float(np.min([r["jaccard"] for r in rs])),
                "attention_rel_err_vs_reference_selection_median": float(np.median([r["attention_vs_reference_selection"] for r in rs])),
                "attention_rel_err_vs_reference_selection_max": float(np.max([r["attention_vs_reference_selection"] for r in rs])),
                "attention_rel_err_vs_engine_selection_max": float(np.max([r["attention_vs_engine_selection"] for r in rs])),
            }
    diff = [r for r in rows_out if r.get("differing")]
    if diff:
        def pct(key, q):
            values = [r[key] for r in diff if r.get(key) is not None]
            return float(np.percentile(values, q)) if values else None
        report["boundary"] = {"rows_differing": len(diff),
                              "differing_blocks_median": pct("differing", 50), "differing_blocks_max": pct("differing", 100),
                              "extra_max_rank_median": pct("extra_max_rank", 50), "extra_max_rank_p99": pct("extra_max_rank", 99),
                              "extra_max_rank_max": pct("extra_max_rank", 100),
                              "missing_min_rank_min": min(r["missing_min_rank"] for r in diff if r.get("missing_min_rank") is not None),
                              "extra_rel_score_gap_median": pct("extra_max_rel_gap", 50),
                              "extra_rel_score_gap_max": pct("extra_max_rel_gap", 100)}
    report["worst_rows"] = sorted(rows_out, key=lambda r: (r["jaccard"], -r["attention_vs_reference_selection"]))[:12]
    text_out = json.dumps(report, indent=1)
    if args.json:
        args.json.write_text(text_out)
        (args.json.with_suffix(".rows.json")).write_text(json.dumps(rows_out))
    print(text_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
