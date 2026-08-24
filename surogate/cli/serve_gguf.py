# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# GGUF ingest for `surogate serve` (design/serve-engine-plan.md §5.1, §5.2b).
#
# Strategy (v0): bridge GGUF to the exact input the vendored converter already
# accepts — a temporary HF-layout directory with BF16 safetensors shards, a
# config.json synthesized from GGUF KV metadata, and the pinned official
# frontend resources (tokenizer/chat template) fetched from the canonical repo
# (the converter SHA-256-verifies them). The unpatched converter then runs with
# all of its own preflight checks. Costs one temporary BF16 materialization on
# disk (~2 bytes/param, deleted after conversion); a reader-injection path that
# avoids the temp copy is a tracked follow-up. K-quant sources are dequantized
# to BF16 and re-encoded by the recipe (double quantization vs the original
# BF16 checkpoint — documented; native K-quant repack is the plan's §5.2 path).

from __future__ import annotations

import json
import shutil
from pathlib import Path

# Canonical repos for the pinned frontend resources per registered target key.
_OFFICIAL_REPO = {
    "qwen3_6_27b": "Qwen/Qwen3.6-27B",
    "qwen3_8_27b": "Qwen/Qwen3.8-27B",
    "qwen3_6_35b_a3b": "Qwen/Qwen3.6-35B-A3B",
}

_RESOURCE_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "config.json",  # official config: authoritative geometry, overrides KV synth
]

_SHARD_BYTES = 8 << 30


def _arch_kv(reader, arch: str, key: str, default=None):
    field = reader.get_field(f"{arch}.{key}")
    if field is None:
        return default
    return field.contents()


def read_gguf_summary(gguf_path: Path) -> dict:
    """Cheap metadata pass: architecture + geometry from GGUF KV, no tensor data."""
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path), "r")
    arch_field = reader.get_field("general.architecture")
    arch = arch_field.contents() if arch_field is not None else ""
    summary = {
        "architecture": arch,
        "hidden_size": _arch_kv(reader, arch, "embedding_length", 0),
        "num_hidden_layers": _arch_kv(reader, arch, "block_count", 0),
        "num_attention_heads": _arch_kv(reader, arch, "attention.head_count", 0),
        "num_key_value_heads": _arch_kv(reader, arch, "attention.head_count_kv", 0),
        "tensor_count": len(reader.tensors),
        "quant_types": sorted({t.tensor_type.name for t in reader.tensors}),
    }
    del reader
    return summary


def _hf_name_map(arch: str, n_layers: int) -> dict[str, str]:
    """gguf tensor name -> HF tensor name, via gguf-py's canonical mapping."""
    import gguf

    model_arch = None
    for candidate in gguf.MODEL_ARCH:
        if gguf.MODEL_ARCH_NAMES.get(candidate) == arch:
            model_arch = candidate
            break
    if model_arch is None:
        raise SystemExit(f"surogate serve: GGUF architecture '{arch}' has no gguf-py mapping.")

    tmap = gguf.get_tensor_name_map(model_arch, n_layers)
    # tmap.mapping is the FORWARD map: {hf_or_gguf alias -> (MODEL_TENSOR,
    # gguf base name)}, with the gguf name itself included as an alias. Invert
    # it, preferring the canonical HF spelling among the aliases.
    def hf_preference(name: str) -> int:
        if name.startswith(("model.", "lm_head", "language_model.")):
            return 0
        return 1

    reverse: dict[str, str] = {}
    for alias, (_tid, gguf_base) in tmap.mapping.items():
        if alias == gguf_base:
            continue
        prev = reverse.get(gguf_base)
        if prev is None or hf_preference(alias) < hf_preference(prev):
            reverse[gguf_base] = alias

    out: dict[str, str] = {}
    # Base names are stored without the trailing ".weight"/".bias"; GGUF tensor
    # names carry the suffix. Emit both suffixed forms.
    for gguf_base, hf_base in reverse.items():
        for suffix in (".weight", ".bias"):
            out[gguf_base + suffix] = hf_base + suffix
    return out


def build_hf_dir_from_gguf(gguf_path: Path, target_key: str, work_dir: Path, *, echo=print) -> Path:
    """Materialize a temporary HF-layout model dir from a GGUF file."""
    import numpy as np
    import torch
    from gguf import GGUFReader
    from gguf.quants import dequantize
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pinned official frontend resources (converter hash-verifies these).
    repo = _OFFICIAL_REPO[target_key]
    echo(f"surogate serve: fetching pinned frontend resources from {repo}")
    for fname in _RESOURCE_FILES:
        try:
            src = hf_hub_download(repo, fname)
        except Exception:
            continue  # optional files (e.g. video preprocessor) may not exist
        shutil.copy(src, work_dir / fname)
    if not (work_dir / "config.json").is_file():
        raise SystemExit(f"surogate serve: could not fetch config.json from {repo}.")

    # 2. Dequantize tensors to BF16 and write sharded safetensors with HF names.
    reader = GGUFReader(str(gguf_path), "r")
    arch = reader.get_field("general.architecture").contents()
    n_layers = int(_arch_kv(reader, arch, "block_count", 0))
    name_map = _hf_name_map(arch, n_layers)

    weight_map: dict[str, str] = {}
    total_bytes = 0
    shard_idx = 0
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    n_tensors = len(reader.tensors)

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        shard_idx += 1
        fname = f"model-{shard_idx:05d}.safetensors"
        save_file(shard, str(work_dir / fname))
        for key in shard:
            weight_map[key] = fname
        shard = {}
        shard_bytes = 0

    for i, tensor in enumerate(reader.tensors):
        hf_name = name_map.get(tensor.name)
        if hf_name is None:
            raise SystemExit(
                f"surogate serve: GGUF tensor '{tensor.name}' has no HF mapping for "
                f"arch '{arch}' — refusing rather than dropping weights."
            )
        data = dequantize(tensor.data, tensor.tensor_type)
        # GGUF stores dims innermost-first; HF convention is the reverse.
        array = np.ascontiguousarray(data.reshape(tuple(reversed(tensor.shape.tolist()))))
        t = torch.from_numpy(array).to(torch.bfloat16)
        shard[hf_name] = t
        shard_bytes += t.numel() * 2
        total_bytes += t.numel() * 2
        if shard_bytes >= _SHARD_BYTES:
            flush()
        if (i + 1) % 100 == 0 or i + 1 == n_tensors:
            echo(f"surogate serve: dequantized {i + 1}/{n_tensors} tensors "
                 f"({total_bytes / (1 << 30):.1f} GiB BF16)")
    flush()
    del reader

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (work_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    return work_dir


def gguf_target_key(gguf_path: Path):
    """Map a GGUF file to a registered converter target key, or None.

    Architecture strings follow llama.cpp/gguf-py naming: the Qwen3.5/3.6
    family (both Qwen3_5ForCausalLM) is 'qwen35'; MoE is 'qwen35moe'. Older
    HF-style spellings are accepted defensively. Validated against a real
    Qwen3.6-27B GGUF before this path is called supported.
    """
    s = read_gguf_summary(gguf_path)
    arch = s["architecture"]
    hidden = int(s["hidden_size"] or 0)
    layers = int(s["num_hidden_layers"] or 0)
    if arch in ("qwen35", "qwen3_6", "qwen3_5") and hidden == 5120 and layers >= 60:
        return "qwen3_6_27b"
    if arch in ("qwen38", "qwen3_8") and hidden == 5120:
        return "qwen3_8_27b"
    if arch in ("qwen35moe", "qwen3moe", "qwen3_6_moe", "qwen3_5_moe") and hidden > 0:
        return "qwen3_6_35b_a3b"
    return None
