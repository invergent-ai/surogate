"""Inkling expert-LoRA adapters in the serving engine's layout (numpy only, so it runs on every host).

Layout the engine binds (serve-inkling e035ba7b; lora_registry.cpp:406):
  base_model.model.model.layers.{L}.mlp.experts.{e}.gate_up_proj.lora_A.weight  [r, 2304]
  base_model.model.model.layers.{L}.mlp.experts.{e}.gate_up_proj.lora_B.weight  [2304, r]  rows INTERLEAVED: gate j = 2j,
                                                                                            up j = 2j + 1 (w13's order)
  base_model.model.model.layers.{L}.mlp.experts.{e}.down_proj.lora_A.weight     [r, 1152]
  base_model.model.model.layers.{L}.mlp.experts.{e}.down_proj.lora_B.weight     [2304, r]
L = physical layer 0..23 (a core layer's adapter applies on all three passes), e = 0..127; adapter_config.json PEFT LORA.

In-memory layout ("rig layout", the learnability rig's rig.make_lora): one dict per physical layer (None = no adapter there)
  {"A13": [E, r, H], "B13": [E, 2I, r], "A2": [E, r, I], "B2": [E, H, r]} float32.

The engine's fold at load (ops/wrapper/lora_store.cpp:197-212): A_bank = bf16(bf16(A) * alpha / r), B_bank = bf16(B), with
float_to_bf16 = round to nearest even on the float32 word (lora_store.cpp:60-65). With alpha / r = 2 (r16 / a32) the
scale is a power of two, so the fold is exact: A_bank = 2 * bf16(A). A trainer forward that uses bf16(A_master),
bf16(B_master) and the scale 2 therefore sees the engine's bank values.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

import numpy as np

PREFIX = "base_model.model.model.layers"
KEYS = ("A13", "B13", "A2", "B2")
# (rig key, module, A or B) in the order tensors are written per expert
_PARTS = (("A13", "gate_up_proj", "A"), ("B13", "gate_up_proj", "B"), ("A2", "down_proj", "A"), ("B2", "down_proj", "B"))


def tensor_name(layer: int, expert: int, module: str, ab: str) -> str:
    return f"{PREFIX}.{layer}.mlp.experts.{expert}.{module}.lora_{ab}.weight"


# ---------------------------------------------------------------------------------------------------------- bf16
def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round to nearest even exactly as the engine's float_to_bf16."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("adapter values must be finite")
    w = x.view(np.uint32).astype(np.uint64)
    w = (w + 0x7FFF + ((w >> 16) & 1)) & 0xFFFFFFFF
    return (w >> 16).astype(np.uint16)


def bf16_bits_to_f32(b: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def round_bf16(x: np.ndarray) -> np.ndarray:
    """float32 values rounded to the nearest bf16 (RNE), kept as float32."""
    return bf16_bits_to_f32(f32_to_bf16_bits(x)).reshape(np.shape(x))


def engine_bank(A: np.ndarray, B: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The engine store's fold of one module's (A, B) as float32 values: bf16(bf16(A) * scale), bf16(B)."""
    a = round_bf16(A)
    return round_bf16((a * np.float32(scale)).astype(np.float32)), round_bf16(B)


# --------------------------------------------------------------------------------------------- safetensors (numpy)
_DT = {"F32": (np.float32, 4), "BF16": (np.uint16, 2), "F16": (np.float16, 2)}


def write_safetensors(path: str, tensors: dict[str, tuple[str, np.ndarray]], metadata: dict | None = None) -> None:
    """tensors: name -> (dtype "F32" | "BF16", array; BF16 given as uint16 bit patterns). Names sorted, data contiguous,
    header padded with spaces to a multiple of 8 (the safetensors format). Written to path + '.part', then renamed."""
    header, offset, order = {}, 0, sorted(tensors)
    for name in order:
        dt, arr = tensors[name]
        nbytes = int(arr.size) * _DT[dt][1]
        header[name] = {"dtype": dt, "shape": [int(s) for s in arr.shape], "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for name in order:
            dt, arr = tensors[name]
            f.write(np.ascontiguousarray(arr, dtype=_DT[dt][0]).tobytes())
    os.replace(tmp, path)


def read_safetensors(path: str) -> dict[str, tuple[str, np.ndarray]]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, h in header.items():
            if name == "__metadata__":
                continue
            s, e = h["data_offsets"]
            f.seek(base + s)
            arr = np.frombuffer(f.read(e - s), dtype=_DT[h["dtype"]][0]).reshape(h["shape"])
            out[name] = (h["dtype"], arr)
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- export / import
def adapter_config(rank: int, alpha: int) -> dict:
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": int(rank), "lora_alpha": int(alpha),
            "lora_dropout": 0.0, "fan_in_fan_out": False, "bias": "none", "use_rslora": False,
            "target_modules": ["gate_up_proj", "down_proj"]}


def export_adapter(layers: list, out_dir: str, rank: int, alpha: int, dtype: str = "BF16",
                   metadata: dict | None = None) -> str:
    """Write layers (rig layout, float32 masters or any float array) as the engine's PEFT adapter in out_dir.
    dtype "BF16" (the kit's broadcast_dtype bf16: RNE of the masters) or "F32". Returns the safetensors sha256."""
    if dtype not in ("BF16", "F32"):
        raise ValueError(dtype)
    tensors = {}
    for L, d in enumerate(layers):
        if d is None:
            continue
        E = int(np.shape(d["A13"])[0])
        for key, module, ab in _PARTS:
            t = np.asarray(d[key], dtype=np.float32)
            if t.shape[0] != E:
                raise ValueError(f"layer {L} {key}: {t.shape[0]} experts, A13 has {E}")
            bits = f32_to_bf16_bits(t) if dtype == "BF16" else np.ascontiguousarray(t)
            for e in range(E):
                tensors[tensor_name(L, e, module, ab)] = (dtype, bits[e])
        if np.shape(d["A13"])[1] != rank or np.shape(d["B13"])[2] != rank:
            raise ValueError(f"layer {L}: rank differs from {rank}")
    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "adapter_model.safetensors")
    write_safetensors(st, tensors, metadata)
    cfg = os.path.join(out_dir, "adapter_config.json")
    with open(cfg + ".part", "w") as f:
        json.dump(adapter_config(rank, alpha), f, indent=2)
    os.replace(cfg + ".part", cfg)
    return sha256_file(st)


def import_adapter(in_dir: str, physical_layers: int = 24) -> tuple[list, dict]:
    """Read an adapter written in the engine's names back into the rig layout (float32 values). Accepts the fused
    gate_up_proj form only (what export_adapter writes)."""
    cfg = json.load(open(os.path.join(in_dir, "adapter_config.json")))
    ts = read_safetensors(os.path.join(in_dir, "adapter_model.safetensors"))
    per: dict[int, dict[str, dict[int, np.ndarray]]] = {}
    for name, (dt, arr) in ts.items():
        if not name.startswith(PREFIX + "."):
            raise ValueError(f"unexpected tensor {name}")
        rest = name[len(PREFIX) + 1:].split(".")
        # L, mlp, experts, e, module, lora_X, weight
        L, e, module, ab = int(rest[0]), int(rest[3]), rest[4], rest[5][-1]
        key = {("gate_up_proj", "A"): "A13", ("gate_up_proj", "B"): "B13", ("down_proj", "A"): "A2",
               ("down_proj", "B"): "B2"}[(module, ab)]
        vals = bf16_bits_to_f32(arr).reshape(arr.shape) if dt == "BF16" else np.asarray(arr, dtype=np.float32)
        per.setdefault(L, {}).setdefault(key, {})[e] = vals
    layers: list = [None] * physical_layers
    for L, d in per.items():
        layers[L] = {k: np.stack([d[k][e] for e in sorted(d[k])]) for k in KEYS}
    return layers, cfg


def init_masters(seed: int, layers=tuple(range(24)), experts: int = 128, hidden: int = 2304, inter: int = 1152,
                 rank: int = 16, b_std: float = 0.0, a_std: float | None = None) -> list:
    """numpy masters for tests / synthetic adapters: A = PEFT kaiming-uniform U(+-1/sqrt(fan_in)) (or N(0, a_std)),
    B = 0 (or N(0, b_std)). NOT the trainer's init: a trainer's zero adapter is exported from ITS own masters."""
    g = np.random.default_rng(seed)
    out: list = [None] * 24
    for L in layers:
        def a(n_in):
            if a_std is not None:
                return (g.standard_normal((experts, rank, n_in)) * a_std).astype(np.float32)
            return ((g.random((experts, rank, n_in)) * 2 - 1) / np.sqrt(n_in)).astype(np.float32)

        def b(n_out):
            return (g.standard_normal((experts, n_out, rank)) * b_std).astype(np.float32)

        out[L] = {"A13": a(hidden), "B13": b(2 * inter), "A2": a(inter), "B2": b(hidden)}
    return out
