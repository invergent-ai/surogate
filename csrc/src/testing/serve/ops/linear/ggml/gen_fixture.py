"""Fixtures for test_ggml_k: for each K-quant type, a [n, k] weight in native GGML blocks
plus its exact dequantisation by gguf-py (the oracle). Synthetic blocks are random bytes with
sane half-precision super-scales (gguf-py cannot *quantise* K types, only dequantise); real
blocks come from a GGUF on disk where one exists. Written to $SINFER_GGML_FIXTURE_DIR or
/tmp/surogate_ggml_test/<TYPE>.{meta,blocks,f32}."""
from __future__ import annotations
import os, sys, pathlib
import numpy as np
import gguf
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

OUT = pathlib.Path(os.environ.get("SINFER_GGML_FIXTURE_DIR", "/tmp/surogate_ggml_test"))
OUT.mkdir(parents=True, exist_ok=True)
REAL = pathlib.Path(os.environ.get("SINFER_GGML_FIXTURE_GGUF", "models/Qwen3.5-0.8B-Q4_K_M.gguf"))
HALF_FIELDS = {  # byte offsets of the fp16 super-scales inside each block
    T.Q2_K: (80, 82), T.Q3_K: (108,), T.Q4_K: (0, 2), T.Q5_K: (0, 2), T.Q6_K: (208,),
}

def synthetic(t: T, n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, bsize = gguf.GGML_QUANT_SIZES[t]
    blocks = rng.integers(0, 256, size=(n, k // 256, bsize), dtype=np.uint8)
    for off in HALF_FIELDS[t]:
        vals = rng.uniform(0.002, 0.03, size=(n, k // 256)).astype(np.float16)
        blocks[:, :, off:off + 2] = vals.view(np.uint8).reshape(n, k // 256, 2)
    return blocks.reshape(n, -1)

def write(t: T, blocks: np.ndarray, n: int, k: int, label: str) -> None:
    f32 = dequantize(blocks, t).astype(np.float32).reshape(n, k)
    stem = OUT / f"{t.name}_{label}"
    (stem.with_suffix(".meta")).write_text(f"{n} {k}\n")
    blocks.tofile(stem.with_suffix(".blocks")); f32.tofile(stem.with_suffix(".f32"))
    print(f"  {stem.name:<22} [{n},{k}]  {blocks.nbytes/1e6:6.2f} MB blocks")

for i, t in enumerate((T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K)):
    write(t, synthetic(t, 256, 2048, 100 + i), 256, 2048, "synthetic")
    write(t, synthetic(t, 63, 512, 200 + i), 63, 512, "odd")   # odd row count: the 2-row CTA's guard
if REAL.exists():
    r = gguf.GGUFReader(str(REAL))
    seen = set()
    for tensor in r.tensors:
        t = tensor.tensor_type
        if t.name in ("Q4_K", "Q5_K", "Q6_K") and t not in seen and len(tensor.shape) == 2:
            k, n = int(tensor.shape[0]), int(tensor.shape[1])
            rows = min(n, 512)
            data = np.asarray(tensor.data).reshape(n, -1)[:rows]
            write(t, np.ascontiguousarray(data), rows, k, "real")
            seen.add(t)
else:
    print(f"  (no real GGUF at {REAL}; synthetic only)")
