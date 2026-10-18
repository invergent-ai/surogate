"""Trainer side of the n-gram memory parity gate G2 (rl/ngram-transplant-2026-10-18/INTERFACE.md, E2).

Scores the engine's own token ids (ENGINE.jsonl from ``surogate/serve/tools/parity/qwen3_5_moe/ple_parity.py
score``: ``{"id", "ids", "logprobs"}``) with the trainer's forward of the base model plus a trained memory
adapter, and writes ``{"id", "ids", "logprobs", "logprobs_base"}`` (``logprobs[0] = null``, ``logprobs[i] =
log p(ids[i] | ids[:i])``; ``logprobs_base`` the same forward with the adapter detached, i.e. the base model).
Each row is one conversation (positions from 0, a fresh n-gram context), as the engine serves it.

    python -m surogate.dsl.ngram_memory_score --model MODEL_DIR --adapter ADAPTER_DIR \\
        --engine ENGINE.jsonl --out TRAINER.jsonl [--recipe bf16] [--max-len 32768]

then ``ple_parity.py compare ENGINE.jsonl TRAINER.jsonl --base ENGINE_BASE.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def memory_option_from_adapter(adapter_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(ngram_memory option, adapter_config) of a trainer adapter directory."""
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    spec = cfg.get("surogate_ngram_memory")
    if not spec:
        raise ValueError(f"{adapter_dir} carries no n-gram memory (no surogate_ngram_memory in adapter_config.json)")
    meta = {k: spec[k] for k in ("multipliers", "head_offsets", "head_vocab_sizes", "ngram_size", "heads_per_ngram",
                                 "head_dim", "row_count", "eos_token_id", "layer")}
    meta["table_offset"] = int(spec.get("table_offset", 0))
    meta["conv_kernel"] = int(spec.get("conv_kernel", 4))
    option = {"table": spec["table"], "meta": meta, "layer": int(spec["layer"]),
              "zero_token_ids": list(spec.get("zero_token_ids", [])),
              "table_residency": spec.get("table_residency", "host"), "init_std": float(spec.get("init_std", 0.02)),
              "only": not cfg.get("target_modules")}
    if spec.get("row_shuffle_a") is not None:  # the control arm scores with its own relabelling
        option["row_shuffle"] = {"a": int(spec["row_shuffle_a"]), "b": int(spec.get("row_shuffle_b", 0))}
    return option, cfg


def build_scorer(model_dir: str, adapter_dir: str, seq_len: int, recipe: str = "bf16", chunks: int = 1):
    from surogate import _surogate as sg
    from surogate.dsl.ir_builder import build_dsl_ir_for_model
    from surogate.kernels.jit_compile import compile_jit_kernels
    from surogate.utils.hf import get_model_weights_path

    option, cfg = memory_option_from_adapter(adapter_dir)
    opts = sg.RuntimeOptions(recipe=recipe, use_cuda_graphs=False, offload_master=False, offload_grads=False,
                             offload_optimizer=False, offload_residual=False, shard_gradients=True)
    opts.recompute = "true"
    if chunks > 1:  # long rows: per-token log-probs chunk by chunk (KV prefix / GDN state carried)
        if seq_len % chunks:
            raise ValueError(f"seq_len {seq_len} is not a multiple of {chunks} chunks")
        opts.sequence_chunks = chunks
        import os

        os.environ.setdefault("SUROGATE_ROPE_MAX_SEQ", str(seq_len))
    opts.dsl_ir_json = build_dsl_ir_for_model(model_dir, extra_config={"ngram_memory": option})
    manifests = compile_jit_kernels(opts.dsl_ir_json)
    if manifests:
        opts.jit_kernel_manifests = manifests
    from safetensors import safe_open

    targets = list(cfg.get("target_modules") or [])
    with safe_open(str(Path(adapter_dir) / "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        dtype = str(f.get_slice(keys[0]).get_dtype()).lower()
    lora_config = sg.LoRAAdapterConfig(rank=int(cfg.get("r", 8)), alpha=float(cfg.get("lora_alpha", 16)), dropout=0.0,
                                       dtype="bf16" if "bf16" in dtype else "fp32", target_modules=targets,
                                       train_router=any(".mlp.gate.lora_" in k for k in keys))
    tr = sg.SurogateTrainer(ngpu=1, config=sg.PretrainedConfig.from_pretrained(model_dir, "bf16"), options=opts,
                            batch_size=1, seq_len=seq_len // max(chunks, 1), grad_accum=1, memcpy_all_gather=True,
                            memcpy_send_recv=True, lora_config=lora_config, qlora_config=None)
    tr.import_weights(get_model_weights_path(model_dir))
    tr.import_adapter(str(Path(adapter_dir) / "adapter_model.safetensors"))
    return tr


def score_rows(tr, rows, seq_len: int):
    import numpy as np

    for row in rows:
        ids = [int(t) for t in row["ids"]]
        n = len(ids)
        if n > seq_len:
            raise ValueError(f"row {row['id']}: {n} tokens exceed --max-len {seq_len}")
        x = np.zeros((1, seq_len), dtype=np.int32)
        y = np.full((1, seq_len), -100, dtype=np.int32)
        x[0, :n] = ids
        y[0, : n - 1] = ids[1:]
        with_memory = tr.compute_logprobs(x, y, use_lora=True)[0]
        base = tr.compute_logprobs(x, y, use_lora=False)[0]
        yield {"id": row["id"], "ids": ids, "logprobs": [None] + [float(v) for v in with_memory[: n - 1]],
               "logprobs_base": [None] + [float(v) for v in base[: n - 1]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--engine", required=True, help="ENGINE.jsonl from ple_parity.py score (its ids are scored)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--recipe", default="bf16")
    ap.add_argument("--max-len", type=int, default=0, help="trainer sequence length (default: longest row, rounded)")
    ap.add_argument("--chunk", type=int, default=4096, help="tokens per sequence chunk for rows longer than one chunk")
    a = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(a.engine) if line.strip()]
    longest = max(len(r["ids"]) for r in rows)
    if a.max_len:
        seq_len = a.max_len
    elif longest <= a.chunk:
        seq_len = (longest + 127) // 128 * 128
    else:
        seq_len = (longest + a.chunk - 1) // a.chunk * a.chunk
    chunks = max(1, seq_len // a.chunk) if seq_len > a.chunk else 1
    tr = build_scorer(a.model, a.adapter, seq_len, a.recipe, chunks=chunks)
    with open(a.out, "w") as out:
        for rec in score_rows(tr, rows, seq_len):
            lp = [v for v in rec["logprobs"] if v is not None]
            lb = [v for v in rec["logprobs_base"] if v is not None]
            print(f"{rec['id']}: {len(rec['ids'])} tokens, NLL {-sum(lp) / max(len(lp), 1):.5f} "
                  f"(base {-sum(lb) / max(len(lb), 1):.5f})", flush=True)
            out.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
