# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Ingest layer for `surogate serve`: safetensors (HF repos) and GGUF in,
# engine-ready weights out (design/serve-engine-plan.md §5.2b).
#
# The engine's supported INPUT formats are safetensors and GGUF. The vendored
# engine core consumes a packed container; that container is an INTERNAL,
# regenerable cache artifact produced here transparently on first load — it is
# never a user-facing or interchange format. Conversion runs offline in Python
# (the vendored, tested converter tooling); the serving process stays pure C++.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Registered converter targets in the vendored engine, keyed by facts read
# from the source config.json. Model breadth beyond this set is the plan's
# generalization phase; unknown models get a clear refusal, never a crash.
@dataclass(frozen=True)
class ConverterTarget:
    key: str                 # cache identity component
    module: str              # python -m <module> under the vendored ninfer root
    display: str


def _ninfer_root() -> Path | None:
    root = Path(__file__).resolve().parent.parent.parent / "csrc" / "src" / "serve" / "ninfer"
    return root if (root / "tools" / "convert").is_dir() else None


def classify_input(spec: str) -> str:
    """'artifact' | 'gguf' | 'safetensors_dir' | 'hf_repo_id' | 'unknown'."""
    p = Path(spec)
    if p.suffix == ".ninfer" and p.is_file():
        return "artifact"
    if p.suffix == ".gguf" and p.is_file():
        return "gguf"
    if p.is_dir():
        if (p / "config.json").is_file() and any(p.glob("*.safetensors")):
            return "safetensors_dir"
        return "unknown"
    # Not a local path: treat owner/name shapes as HF repo ids.
    if not p.exists() and spec.count("/") == 1 and not spec.startswith((".", "/", "~")):
        return "hf_repo_id"
    return "unknown"


def converter_for_config(config: dict) -> ConverterTarget | None:
    """Map a source config.json to a registered converter, or None."""
    model_type = str(config.get("model_type", ""))
    hidden = int(config.get("hidden_size", 0) or 0)
    layers = int(config.get("num_hidden_layers", 0) or 0)
    quant = config.get("quantization_config") or {}
    quant_method = str(quant.get("quant_method", "")) if isinstance(quant, dict) else ""
    nvfp4 = quant_method in ("modelopt", "nvfp4") or "NVFP4" in json.dumps(quant)[:2000]

    # Registered geometries (vendored targets). Text-config nesting (VL-style
    # configs) is flattened by callers before this point.
    if model_type == "qwen3_5" and hidden == 1024 and layers == 24:
        return ConverterTarget("qwen3_5_0_8b", "tools.convert.qwen3_5_0_8b.convert",
                               "Qwen3.5-0.8B")
    if model_type in ("qwen3_5", "qwen3_6") and hidden == 5120 and layers >= 60:
        if nvfp4:
            return ConverterTarget("qwen3_6_27b_nvfp4", "tools.convert.qwen3_6_27b.convert_nvfp4",
                                   "Qwen3.6-27B (NVFP4)")
        return ConverterTarget("qwen3_6_27b", "tools.convert.qwen3_6_27b.convert", "Qwen3.6-27B")
    if model_type == "qwen3_8" and hidden == 5120:
        if nvfp4:
            return ConverterTarget("qwen3_8_27b_nvfp4", "tools.convert.qwen3_8_27b.convert_nvfp4",
                                   "Qwen3.8-27B (NVFP4)")
        return ConverterTarget("qwen3_8_27b", "tools.convert.qwen3_8_27b.convert", "Qwen3.8-27B")
    if model_type in ("qwen3_5_moe", "qwen3_6_moe") and int(config.get("num_experts", 0) or 0) > 0:
        return ConverterTarget("qwen3_6_35b_a3b", "tools.convert.qwen3_6_35b_a3b.convert",
                               "Qwen3.6-35B-A3B")
    return None


def _flatten_text_config(config: dict) -> dict:
    if not isinstance(config.get("text_config"), dict):
        return config
    # Geometry comes from text_config, but the ROOT model_type is the family
    # identity ("qwen3_5"); the nested one is the "_text" variant. Keep root's.
    merged = {**config, **config["text_config"]}
    merged["model_type"] = config.get("model_type", merged.get("model_type"))
    return merged


def source_fingerprint(model_dir: Path) -> str:
    """Cheap, stable fingerprint: config bytes + (name, size, mtime_ns) of shards.

    Plan §5.3 wants source-content hashes; hashing 50+ GB on every serve is not
    acceptable startup cost, so v0 fingerprints metadata. A corrupted-in-place
    shard with identical size+mtime evades this — accepted and documented.
    """
    h = hashlib.sha256()
    h.update((model_dir / "config.json").read_bytes())
    for shard in sorted(model_dir.glob("*.safetensors")):
        st = shard.stat()
        h.update(f"{shard.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:24]


def cache_dir() -> Path:
    return Path(os.environ.get("SUROGATE_SERVE_CACHE",
                               Path.home() / ".cache" / "surogate" / "serve"))


def resolve_hf_repo(repo_id: str) -> Path:
    """Download (or reuse) an HF snapshot with the files the converter needs."""
    from huggingface_hub import snapshot_download  # lazy: not needed for local paths

    path = snapshot_download(
        repo_id,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.txt"],
    )
    return Path(path)


def _gguf_fingerprint(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha256(f"{path.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:24]


def _ensure_from_gguf(gguf_path: Path, *, echo=print) -> Path:
    """GGUF → temp HF dir (dequant BF16) → vendored converter → cached weights.

    v0 bridge (surogate/serve/gguf/bridge.py): correctness inherits the converter's
    own preflight/hash checks; K-quant sources pay one documented
    double-quantization vs the original BF16 checkpoint. Temp BF16 shards
    (~2 bytes/param) are deleted after conversion.
    """
    from surogate.serve.gguf import bridge as serve_gguf

    root = _ninfer_root()
    if root is None:
        raise SystemExit("surogate serve: vendored engine tree not found (run from a checkout).")

    target_key = serve_gguf.gguf_target_key(gguf_path)
    if target_key is None:
        s = serve_gguf.read_gguf_summary(gguf_path)
        raise SystemExit(
            "surogate serve: this GGUF is not yet supported by the native engine.\n"
            f"  architecture={s['architecture']!r} hidden={s['hidden_size']} "
            f"layers={s['num_hidden_layers']} quants={s['quant_types']}\n"
            "  Registered today: Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B."
        )

    fp = _gguf_fingerprint(gguf_path)
    out = cache_dir() / f"{target_key}-gguf-{fp}.ninfer"
    if out.is_file() and out.stat().st_size > 0:
        echo(f"surogate serve: using cached engine weights ({out.name})")
        return out

    work = cache_dir() / f"gguf-bridge-{fp}"
    try:
        model_dir = serve_gguf.build_hf_dir_from_gguf(gguf_path, target_key, work, echo=echo)
        return _run_converter_cached(model_dir, out, echo=echo, derived_frontend=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_converter_cached(model_dir: Path, out: Path, *, echo=print,
                          derived_frontend: bool = False) -> Path:
    """Shared converter driver: model_dir (HF layout) → atomic-published `out`."""
    root = _ninfer_root()
    config = _flatten_text_config(json.loads((model_dir / "config.json").read_text()))
    target = converter_for_config(config)
    if target is None:
        raise SystemExit("surogate serve: internal error — bridged model dir maps to no converter.")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".ninfer.partial")
    tmp.unlink(missing_ok=True)
    echo(f"surogate serve: preparing engine weights for {target.display} "
         f"(one-time conversion; cached at {out})")
    cmd = [sys.executable, "-m", target.module, "--model", str(model_dir), "--out", str(tmp)]
    if os.environ.get("SUROGATE_CONVERT_DEVICE"):
        cmd += ["--device", os.environ["SUROGATE_CONVERT_DEVICE"]]
    if os.environ.get("SUROGATE_SERVE_DRY"):
        echo("DRY: cwd=" + str(root))
        echo("DRY: " + " ".join(cmd))
        raise SystemExit(0)
    env = {**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if derived_frontend:
        # GGUF-sourced: tokenizer reconstructed from KV (PATCHES.md #12).
        env["NINFER_ALLOW_DERIVED_FRONTEND"] = "1"
    result = subprocess.run(cmd, cwd=root, env=env)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"surogate serve: conversion failed (exit {result.returncode}).")
    tmp.replace(out)
    return out


def ensure_engine_weights(spec: str, *, echo=print) -> Path:
    """Resolve `spec` (safetensors dir | HF repo id | GGUF | internal artifact)
    to an engine-loadable weights file, converting through the transparent
    cache when needed. Raises SystemExit with a clear message on refusal."""
    kind = classify_input(spec)

    if kind == "artifact":
        # Internal/dev passthrough (not a supported product input).
        return Path(spec)

    if kind == "gguf":
        return _ensure_from_gguf(Path(spec).resolve(), echo=echo)

    if kind == "hf_repo_id":
        echo(f"surogate serve: resolving Hugging Face repo '{spec}'...")
        model_dir = resolve_hf_repo(spec)
    elif kind == "safetensors_dir":
        model_dir = Path(spec).resolve()
    else:
        raise SystemExit(
            f"surogate serve: cannot interpret '{spec}'. Supported inputs: a local "
            "safetensors model directory, a Hugging Face repo id, or a .gguf file."
        )

    root = _ninfer_root()
    if root is None:
        raise SystemExit("surogate serve: vendored engine tree not found (run from a checkout).")

    config = _flatten_text_config(json.loads((model_dir / "config.json").read_text()))
    target = converter_for_config(config)
    if target is None:
        raise SystemExit(
            "surogate serve: this model is not yet supported by the native engine.\n"
            f"  model_type={config.get('model_type')!r} hidden_size={config.get('hidden_size')} "
            f"layers={config.get('num_hidden_layers')}\n"
            "  Registered today: Qwen3.6-27B, Qwen3.8-27B (BF16/NVFP4), Qwen3.6-35B-A3B.\n"
            "  Model breadth is tracked in design/serve-engine-plan.md §6."
        )

    fp = source_fingerprint(model_dir)
    out = cache_dir() / f"{target.key}-{fp}.ninfer"
    if out.is_file() and out.stat().st_size > 0:
        echo(f"surogate serve: using cached engine weights ({out.name})")
        return out
    return _run_converter_cached(model_dir, out, echo=echo)
