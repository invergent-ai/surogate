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
class _Q2_0:
    """Stand-in for the installed gguf-py, which does not know GGML type 42 yet."""
    name = "Q2_0"
    def __hash__(self): return hash("Q2_0")
    def __eq__(self, other): return getattr(other, "name", None) == "Q2_0"


Q2_0 = getattr(T, "Q2_0", None) or _Q2_0()
QUANT_SIZES = dict(gguf.GGML_QUANT_SIZES)
QUANT_SIZES.setdefault(Q2_0, (64, 18))
# gguf-py calls F16 a one-value "block" of two bytes. The engine reads it in 32-value windows,
# because that is the unit every other route here already works in, and the two descriptions
# cover exactly the same bytes.
QUANT_SIZES[T.F16] = (32, 64)
HALF_FIELDS = {  # byte offsets of the fp16 super-scales inside each block
    T.Q2_K: (80, 82), T.Q3_K: (108,), T.Q4_K: (0, 2), T.Q5_K: (0, 2), T.Q6_K: (208,),
    # Q4_1/Q5_1 are 32-value blocks, not superblocks: (d, m) sit at the front of each.
    T.Q4_1: (0, 2), T.Q5_1: (0, 2), T.Q8_0: (0,), T.IQ4_NL: (0,), T.Q4_0: (0,), T.Q5_0: (0,),
    # The IQ superblocks keep `d` at the front, except IQ1_M, which spreads its block scale over
    # the top nibbles of its four scale words (handled below); the ternary pair keeps it last.
    T.IQ2_XXS: (0,), T.IQ2_XS: (0,), T.IQ2_S: (0,), T.IQ3_XXS: (0,), T.IQ3_S: (0,), T.IQ1_S: (0,),
    T.IQ1_M: (), T.IQ4_XS: (0,), T.TQ1_0: (52,), T.TQ2_0: (64,), T.MXFP4: (), T.NVFP4: (),
    T.Q1_0: (0,), Q2_0: (0,),
    # F16 has no scale field at all; its values are drawn directly below.
    T.F16: (),
}

#: Values a block of each type holds. Only the K-quants are superblocks.
BLOCK_VALUES = {T.Q4_1: 32, T.Q5_1: 32, T.Q8_0: 32, T.IQ4_NL: 32, T.Q4_0: 32, T.Q5_0: 32,
                T.MXFP4: 32, T.NVFP4: 64, T.Q1_0: 128, Q2_0: 64, T.F16: 32}

#: Every format the engine reads where it lies. Q2_0 is absent from the installed gguf-py, so
#: its oracle is the hand decoder below rather than gguf.quants.dequantize.
TYPES = (T.Q2_K, T.Q3_K, T.Q4_K, T.Q5_K, T.Q6_K, T.Q8_0, T.Q4_1, T.Q5_1, T.IQ4_NL, T.Q4_0, T.Q5_0,
         T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ1_S, T.IQ1_M, T.IQ4_XS,
         T.TQ1_0, T.TQ2_0, T.MXFP4, T.NVFP4, T.Q1_0, Q2_0, T.F16)


def synthetic(t: T, n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, bsize = QUANT_SIZES[t]
    per = BLOCK_VALUES.get(t, 256)
    blocks = rng.integers(0, 256, size=(n, k // per, bsize), dtype=np.uint8)
    for off in HALF_FIELDS[t]:
        vals = rng.uniform(0.002, 0.03, size=(n, k // per)).astype(np.float16)
        blocks[:, :, off:off + 2] = vals.view(np.uint8).reshape(n, k // per, 2)
    if t == T.IQ1_M:
        # the block scale is the top nibble of each of the four 16-bit scale words (bytes 48..55)
        vals = rng.uniform(0.002, 0.03, size=(n, k // per)).astype(np.float16).view(np.uint16)
        words = blocks[:, :, 48:56].copy().view(np.uint16).reshape(n, k // per, 4)
        for i in range(4):
            words[:, :, i] = (words[:, :, i] & 0x0FFF) | (((vals >> (4 * i)) & 0xF) << 12)
        blocks[:, :, 48:56] = words.view(np.uint8).reshape(n, k // per, 8)
    if t == T.F16:
        # Every byte pair is a value, so draw halves rather than fixing up random bytes: the
        # fixture then spans both signs and a realistic magnitude range.
        vals = rng.uniform(-1.0, 1.0, size=(n, k // per, per)).astype(np.float16)
        return vals.view(np.uint8).reshape(n, -1)
    if t == T.MXFP4:
        # E8M0 exponents near one, so the tile stays representable in BF16
        blocks[:, :, 0] = rng.integers(118, 128, size=(n, k // per), dtype=np.uint8)
    if t == T.NVFP4:
        # UE4M3 sub-scales below one and never the NaN code
        blocks[:, :, 0:4] = rng.integers(0x08, 0x38, size=(n, k // per, 4), dtype=np.uint8)
    return blocks.reshape(n, -1)


def dequantize_q1_0(blocks: np.ndarray, n: int, k: int) -> np.ndarray:
    """gguf-py names Q1_0 but cannot dequantise it: 128 sign bits under one fp16 scale."""
    b = blocks.reshape(n, k // 128, 18)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(n, k // 128, 1)
    bits = np.unpackbits(b[:, :, 2:18], axis=-1, bitorder="little").reshape(n, k // 128, 128)
    return (d * np.where(bits == 1, 1.0, -1.0).astype(np.float32)).reshape(n, k)


def dequantize_q2_0(blocks: np.ndarray, n: int, k: int) -> np.ndarray:
    """gguf-py has no Q2_0 yet: 64 values under one fp16 scale, code - 1 in {-1, 0, 1, 2}."""
    b = blocks.reshape(n, k // 64, 18)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(n, k // 64, 1)
    qs = b[:, :, 2:18]
    codes = np.stack([(qs >> (2 * i)) & 3 for i in range(4)], axis=-1).reshape(n, k // 64, 64)
    return (d * (codes.astype(np.float32) - 1.0)).reshape(n, k)

def write(t: T, blocks: np.ndarray, n: int, k: int, label: str) -> None:
    f32 = (dequantize_q2_0(blocks, n, k) if t.name == "Q2_0"
           else dequantize_q1_0(blocks, n, k) if t.name == "Q1_0"
           else dequantize(blocks, t).astype(np.float32).reshape(n, k))
    # the artifact spells the GGUF's NVFP4 block type NVFP4_GGML, and the test loads by that name
    stem = OUT / f"{'NVFP4_GGML' if t.name == 'NVFP4' else t.name}_{label}"
    (stem.with_suffix(".meta")).write_text(f"{n} {k}\n")
    blocks.tofile(stem.with_suffix(".blocks")); f32.tofile(stem.with_suffix(".f32"))
    print(f"  {stem.name:<22} [{n},{k}]  {blocks.nbytes/1e6:6.2f} MB blocks")

for i, t in enumerate(TYPES):
    write(t, synthetic(t, 256, 2048, 100 + i), 256, 2048, "synthetic")
    write(t, synthetic(t, 63, 512, 200 + i), 63, 512, "odd")   # odd row count: the 2-row CTA's guard
#: Real tensors of every type, from whichever files on this machine carry them: the K_M
#: file for the K-quants, unsloth's UD mixtures for the IQ family. Comma-separated.
def _hub(name: str) -> list[pathlib.Path]:
    """The named GGUF wherever it sits: the repo's models/ or the Hugging Face cache."""
    hub = pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return sorted(p for p in [pathlib.Path("models") / name, *hub.glob(f"models--*/snapshots/*/{name}")]
                  if p.is_file())


REAL_FILES = ([pathlib.Path(x) for x in os.environ["SINFER_GGML_FIXTURE_GGUFS"].split(",") if x]
              if "SINFER_GGML_FIXTURE_GGUFS" in os.environ else
              [REAL] + [p for name in ("Qwen3-0.6B-UD-IQ1_M.gguf", "Qwen3-0.6B-UD-IQ2_M.gguf",
                                       "Qwen3-0.6B-IQ4_XS.gguf", "Qwen3-0.6B-UD-IQ3_XXS.gguf",
                                       # the XL mixes keep their most sensitive tensors at F16
                                       "Qwen3.5-0.8B-UD-Q8_K_XL.gguf")
                        for p in _hub(name)[:1]])
REAL_TYPES = {t.name for t in TYPES}
seen = set()
for real in REAL_FILES:
    if not real.exists():
        print(f"  (no real GGUF at {real})")
        continue
    r = gguf.GGUFReader(str(real))
    for tensor in r.tensors:
        t = tensor.tensor_type
        if t.name in REAL_TYPES and t.name not in ("Q2_K", "Q3_K") and t not in seen and len(tensor.shape) == 2:
            k, n = int(tensor.shape[0]), int(tensor.shape[1])
            rows = min(n, 512)
            # gguf-py hands back a typed array for the unquantised types and raw bytes for
            # everything else; the fixture is always the stored bytes.
            raw = np.asarray(tensor.data)
            raw = raw if raw.dtype == np.uint8 else raw.view(np.uint8)
            data = raw.reshape(n, -1)[:rows]
            write(t, np.ascontiguousarray(data), rows, k, "real")
            seen.add(t)
            if tensor.name == "token_embd.weight" and real == REAL:
                # the whole table: the lm_head's row count, which the small cases never reach
                write(t, np.ascontiguousarray(raw.reshape(n, -1)), n, k, "big")
