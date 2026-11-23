"""Lane T8, S5: the adapter file for SkyRL's LoRA shape at the owner's ranks (STEP-API.md ADAPTER FORMAT + agent-ELORA's answer 07:54), PEFT-native form:
  adapter_config: r 64, lora_alpha 32, rank_pattern {experts: 16} (no alpha_pattern: the experts' scale is 32 / 16 = 2, attention 32 / 64 = 0.5);
  tensors per PHYSICAL layer L (the kit's prefix, as the expert names the engine already binds):
    <P>.L.attn.{wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv, wo_ud}.lora_A.weight [64, in] / .lora_B.weight [out, 64]  (in 2304; wo_ud 3072)
    <P>.L.mlp.experts.<e>.{gate_up_proj, down_proj}.lora_{A,B}.weight                                                    (rank 16; the kit's writer)
B unscaled: the engine folds the scale into A at upload, bank A = bf16(bf16(A) x scale) = what the trainer's banks hold (power-of-two scales: exact).
The engine serves it with --enable-lora --max-lora-rank 64 --max-expert-lora-rank 16 (agent-ELORA, inkling/ELORA 12e16b22)."""
from __future__ import annotations

import json
import os

import numpy as np

from surogate.inkling_rl import adapter as AD

SLICE_NAME = {"q": "wq_du", "r": "wr_du", "g": "gate_attn", "s": "attn_sink_proj", "k": "wk_dv", "v": "wv_dv", "o": "wo_ud"}
NAME_SLICE = {v: k for k, v in SLICE_NAME.items()}
EXPERT_PATTERN = r"mlp\.experts\.\d+\.(gate_up_proj|down_proj)"


def attn_name(layer: int, s: str, ab: str) -> str:
    return f"{AD.PREFIX}.{layer}.attn.{SLICE_NAME[s]}.lora_{ab}.weight"


def export_combined(layers: list, out_dir: str, attn_rank: int, alpha: float, expert_rank: int = 16, dtype: str = "BF16",
                    metadata: dict | None = None) -> str:
    """layers: per physical layer a dict with A13 / B13 / A2 / B2 (experts) and qA / qB / ... / oB (attention), float32 masters."""
    if dtype not in ("BF16", "F32"):
        raise ValueError(dtype)
    tensors = {}
    for L, d in enumerate(layers):
        E = int(np.shape(d["A13"])[0])
        if np.shape(d["A13"])[1] != expert_rank:
            raise ValueError(f"layer {L}: expert rank {np.shape(d['A13'])[1]} != {expert_rank}")
        for key, module, ab in AD._PARTS:
            t = np.asarray(d[key], dtype=np.float32)
            bits = AD.f32_to_bf16_bits(t) if dtype == "BF16" else np.ascontiguousarray(t)
            for e in range(E):
                tensors[AD.tensor_name(L, e, module, ab)] = (dtype, bits[e])
        for s in SLICE_NAME:
            for ab in ("A", "B"):
                t = np.asarray(d[f"{s}{ab}"], dtype=np.float32)
                r = t.shape[0] if ab == "A" else t.shape[1]
                if r != attn_rank:
                    raise ValueError(f"layer {L} {s}{ab}: rank {r} != {attn_rank}")
                tensors[attn_name(L, s, ab)] = (dtype, AD.f32_to_bf16_bits(t) if dtype == "BF16" else np.ascontiguousarray(t))
    os.makedirs(out_dir, exist_ok=True)
    st = os.path.join(out_dir, "adapter_model.safetensors")
    AD.write_safetensors(st, tensors, metadata)
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": int(attn_rank), "lora_alpha": float(alpha), "lora_dropout": 0.0,
           "fan_in_fan_out": False, "bias": "none", "use_rslora": False,
           "target_modules": ["gate_up_proj", "down_proj"] + [SLICE_NAME[s] for s in SLICE_NAME],
           "rank_pattern": {EXPERT_PATTERN: int(expert_rank)}}
    path = os.path.join(out_dir, "adapter_config.json")
    with open(path + ".part", "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(path + ".part", path)
    return AD.sha256_file(st)


def import_combined(in_dir: str, physical_layers: int = 24) -> tuple[list, dict]:
    """Back into the trainer's per-layer dicts (float32 values). Refuses names it does not know."""
    cfg = json.load(open(os.path.join(in_dir, "adapter_config.json")))
    ts = AD.read_safetensors(os.path.join(in_dir, "adapter_model.safetensors"))
    per: dict = {}
    for name, (dt, arr) in ts.items():
        if not name.startswith(AD.PREFIX + "."):
            raise ValueError(f"unexpected tensor {name}")
        rest = name[len(AD.PREFIX) + 1:].split(".")
        L = int(rest[0])
        vals = AD.bf16_bits_to_f32(arr).reshape(arr.shape) if dt == "BF16" else np.asarray(arr, dtype=np.float32)
        if rest[1] == "attn":
            per.setdefault(L, {})[f"{NAME_SLICE[rest[2]]}{rest[3][-1]}"] = vals
        elif rest[1] == "mlp":
            e, module, ab = int(rest[3]), rest[4], rest[5][-1]
            key = {("gate_up_proj", "A"): "A13", ("gate_up_proj", "B"): "B13", ("down_proj", "A"): "A2", ("down_proj", "B"): "B2"}[(module, ab)]
            per.setdefault(L, {}).setdefault(key, {})[e] = vals
        else:
            raise ValueError(f"unexpected tensor {name}")
    out = []
    for L in range(physical_layers):
        d = per[L]
        out.append({k: (np.stack([v[e] for e in sorted(v)]) if isinstance(v, dict) else v) for k, v in d.items()})
    return out, cfg
