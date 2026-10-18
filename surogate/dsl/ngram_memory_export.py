"""Export a trained hashed n-gram memory for the engine (rl/ngram-transplant-2026-10-18/INTERFACE.md, section 2).

Reads a trainer adapter directory (``adapter_model.safetensors`` + ``adapter_config.json`` with
``surogate_ngram_memory``) and writes ``OUT_DIR/ngram_memory.safetensors`` (Flash-Next layout for the
memory's layer: projections BF16, gains FP32 folded, a zero convolution, ones for its norm, the hash constants
as I64) and ``OUT_DIR/ngram_memory.json`` (geometry and the frozen table it reads). The table itself is not
copied: the engine maps the referenced file.

    python -m surogate.dsl.ngram_memory_export ADAPTER_DIR OUT_DIR [--table-sha256 HEX | --hash-table]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

FORMAT = "surogate-ngram-memory-v1"
PEFT_PREFIX = "base_model.model."
TRAINED = ("key_proj", "value_proj", "norm_key", "norm_query")


def _sha256(path: str, chunk: int = 1 << 26) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _commit() -> str:
    env = os.environ.get("SUROGATE_BUILD_COMMIT")
    if env:
        return env
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def export_ngram_memory(adapter_dir: str | os.PathLike, out_dir: str | os.PathLike, *,
                        table_sha256: str | None = None, hash_table: bool = False,
                        hidden_size: int | None = None) -> dict[str, Any]:
    """Write the engine inputs; returns the json written."""
    import torch
    from safetensors.torch import load_file, save_file

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    spec = cfg.get("surogate_ngram_memory")
    if not spec:
        raise ValueError(f"{adapter_dir}/adapter_config.json has no surogate_ngram_memory")
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    names = spec["tensors"]
    missing = [names[k] for k in TRAINED if names[k] not in tensors]
    if missing:
        raise ValueError(f"adapter lacks the memory tensors {missing}")
    wk, wv = tensors[names["key_proj"]].float(), tensors[names["value_proj"]].float()
    gk, gq = tensors[names["norm_key"]].float(), tensors[names["norm_query"]].float()
    C, E = wk.shape
    if wv.shape != (C, E) or gk.shape != (C,) or gq.shape != (C,):
        raise ValueError(f"memory tensor shapes: key {tuple(wk.shape)} value {tuple(wv.shape)} gains "
                         f"{tuple(gk.shape)} {tuple(gq.shape)}")
    if E != int(spec["heads"]) * int(spec["head_dim"]):
        raise ValueError(f"projection width {E} != heads * head_dim {spec['heads']} * {spec['head_dim']}")
    if hidden_size is not None and C != hidden_size:
        raise ValueError(f"projection rows {C} != hidden_size {hidden_size}")
    for t in (wk, wv, gk, gq):
        if not torch.isfinite(t).all():
            raise ValueError("the memory tensors hold non-finite values")
    if not names["key_proj"].startswith(PEFT_PREFIX) or not names["key_proj"].endswith(".ple.key_proj.weight"):
        raise ValueError(f"unexpected memory tensor name {names['key_proj']}")
    # Flash-Next's layout whatever the adapter's module path (the trainer names text-only adapters model.layers.*).
    module = f"model.language_model.layers.{int(spec['layer'])}.ple."
    kernel = int(spec.get("conv_kernel", 4))
    out = {
        module + "key_proj.weight": wk.to(torch.bfloat16).contiguous(),
        module + "value_proj.weight": wv.to(torch.bfloat16).contiguous(),
        module + "norm_key.weight": gk.contiguous(),
        module + "norm_query.weight": gq.contiguous(),
        module + "norm_conv.weight": torch.ones(C, dtype=torch.float32),
        module + "conv1d.weight": torch.zeros(C, 1, kernel, dtype=torch.bfloat16),
        module + "ple_embedding.layer_multipliers": torch.tensor(
            [m - (1 << 64) if m >= (1 << 63) else m for m in spec["multipliers"]], dtype=torch.int64),
        module + "ple_embedding.ngram_heads_offsets": torch.tensor(spec["head_offsets"], dtype=torch.int64),
        module + "ple_embedding.ngram_heads_vocab_sizes": torch.tensor(spec["head_vocab_sizes"], dtype=torch.int64),
    }
    table = spec["table"]
    row_bytes = int(spec["row_bytes"])
    rows = int(spec["row_count"])
    size = os.path.getsize(table) if os.path.exists(table) else None
    if hash_table and table_sha256 is None:
        table_sha256 = _sha256(table)
    meta = {
        "format": FORMAT,
        "layer": int(spec["layer"]), "streams": 1, "hidden": int(C), "embed_dim": int(E),
        "heads": int(spec["heads"]), "heads_per_ngram": int(spec["heads_per_ngram"]),
        "ngram_size": int(spec["ngram_size"]), "head_dim": int(spec["head_dim"]),
        "eos_token_id": int(spec["eos_token_id"]),
        "image_token_id": 248056 if 248056 in spec.get("zero_token_ids", []) else None,
        "video_token_id": 248057 if 248057 in spec.get("zero_token_ids", []) else None,
        "zero_token_ids": list(spec.get("zero_token_ids", [])),
        "conv_kernel": kernel, "conv_dilation": int(spec.get("conv_dilation", spec["ngram_size"])),
        "conv_frozen_zero": True, "eps": float(spec.get("eps", 1e-6)), "gate_clamp": 1e-6,
        "norm_gain_convention": "folded",
        "table": {"path": table, "format": "IQ4_NL", "row_count": rows, "row_bytes": row_bytes,
                  "head_dim": int(spec["head_dim"]), "offset": int(spec.get("table_offset", 0)),
                  "bytes": rows * row_bytes, "file_bytes": size, "sha256": table_sha256,
                  "source": spec.get("table_source", "frozen IQ4_NL table (unmodified)")},
        "tensors": {k: {"dtype": str(v.dtype).replace("torch.", ""), "shape": list(v.shape)} for k, v in out.items()},
        "trainer_commit": _commit(),
        "adapter": str(adapter_dir.resolve()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / "ngram_memory.safetensors"), metadata={"format": FORMAT})
    (out_dir / "ngram_memory.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--table-sha256", default=None, help="the table file's sha256 (recorded as is)")
    ap.add_argument("--hash-table", action="store_true", help="compute the table file's sha256 (reads it whole)")
    ap.add_argument("--hidden-size", type=int, default=None, help="refuse a memory of another width")
    a = ap.parse_args(argv)
    meta = export_ngram_memory(a.adapter_dir, a.out_dir, table_sha256=a.table_sha256, hash_table=a.hash_table,
                               hidden_size=a.hidden_size)
    print(json.dumps({k: meta[k] for k in ("layer", "hidden", "embed_dim", "heads", "table")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
