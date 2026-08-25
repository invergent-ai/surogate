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
    gguf_repack: bool = False  # converter accepts --gguf-repack (PATCHES.md #14)


def _ninfer_root() -> Path | None:
    # Converters are part of the surogate package now; the "root" is the
    # repository root (kept for subprocess cwd/log context only).
    root = Path(__file__).resolve().parent.parent.parent
    return root if (root / "surogate" / "serve" / "tools" / "convert").is_dir() else None


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
        return ConverterTarget("qwen3_5_0_8b", "surogate.serve.tools.convert.qwen3_5_0_8b.convert",
                               "Qwen3.5-0.8B", gguf_repack=True)
    if model_type == "qwen3_5" and hidden == 2048 and layers == 24:
        return ConverterTarget("qwen3_5_2b", "surogate.serve.tools.convert.qwen3_5_2b.convert",
                               "Qwen3.5-2B", gguf_repack=True)
    if model_type == "qwen3_5" and hidden == 2560 and layers == 32:
        return ConverterTarget("qwen3_5_4b", "surogate.serve.tools.convert.qwen3_5_4b.convert",
                               "Qwen3.5-4B", gguf_repack=True)
    if model_type in ("qwen3_5", "qwen3_6") and hidden == 5120 and layers >= 60:
        if nvfp4:
            return ConverterTarget("qwen3_6_27b_nvfp4", "surogate.serve.tools.convert.qwen3_6_27b.convert_nvfp4",
                                   "Qwen3.6-27B (NVFP4)")
        return ConverterTarget("qwen3_6_27b", "surogate.serve.tools.convert.qwen3_6_27b.convert", "Qwen3.6-27B")
    if model_type == "qwen3_8" and hidden == 5120:
        if nvfp4:
            return ConverterTarget("qwen3_8_27b_nvfp4", "surogate.serve.tools.convert.qwen3_8_27b.convert_nvfp4",
                                   "Qwen3.8-27B (NVFP4)")
        return ConverterTarget("qwen3_8_27b", "surogate.serve.tools.convert.qwen3_8_27b.convert", "Qwen3.8-27B")
    if model_type in ("qwen3_5_moe", "qwen3_6_moe") and int(config.get("num_experts", 0) or 0) > 0:
        return ConverterTarget("qwen3_6_35b_a3b", "surogate.serve.tools.convert.qwen3_6_35b_a3b.convert",
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

    # Warm start: the cache name embeds the (target-independent) fingerprint,
    # so a hit returns without touching the GGUF — gguf-py's eager KV parse
    # costs ~10s on a 250k-token vocabulary and must stay off this path.
    fp = _gguf_fingerprint(gguf_path)
    for cached in cache_dir().glob(f"*-gguf-{fp}.ninfer"):
        if cached.is_file() and cached.stat().st_size > 0:
            echo(f"surogate serve: using cached engine weights ({cached.name})")
            return cached

    reader = serve_gguf.open_gguf(gguf_path)
    target_key = serve_gguf.gguf_target_key(gguf_path, reader)
    if target_key is None:
        s = serve_gguf.read_gguf_summary(gguf_path, reader)
        raise SystemExit(
            "surogate serve: this GGUF is not yet supported by the native engine.\n"
            f"  architecture={s['architecture']!r} hidden={s['hidden_size']} "
            f"layers={s['num_hidden_layers']} quants={s['quant_types']}\n"
            "  Registered today: Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B."
        )

    out = cache_dir() / f"{target_key}-gguf-{fp}.ninfer"

    # Q8_0 repack (PATCHES.md #14): for targets whose converter takes
    # --gguf-repack, plan against the converter's own recipes which candidate
    # tensors it repacks bit-exactly; the bridge dequantizes only the rest.
    repack_targets = {"qwen3_5_0_8b", "qwen3_5_2b", "qwen3_5_4b"}
    planner = _repack_planner(root, target_key) if target_key in repack_targets else None
    # No-MTP variant (PATCHES.md #15): community exports may strip nextn.
    arch = serve_gguf.read_gguf_summary(gguf_path, reader)["architecture"]
    nextn = reader.kv(f"{arch}.nextn_predict_layers", 0)
    no_mtp = target_key in repack_targets and int(nextn or 0) == 0
    work = cache_dir() / f"gguf-bridge-{fp}"
    try:
        model_dir = serve_gguf.build_hf_dir_from_gguf(
            gguf_path, target_key, work, repack_planner=planner, reader=reader, echo=echo
        )
        repack_map = model_dir / "gguf_repack.json"
        return _run_converter_cached(
            model_dir,
            out,
            echo=echo,
            derived_frontend=True,
            gguf_repack=repack_map if repack_map.is_file() else None,
            no_mtp=no_mtp,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _repack_planner(root: Path, target_key: str):
    """Repack plan via the named vendored converter's registered recipes."""
    def plan(gguf_path: Path, candidates: dict[str, str]) -> dict[str, str]:
        import importlib
        import sys as _sys
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from surogate.serve.tools.convert.common.gguf_repack import (
            REPACKABLE_TYPES,
            GgufRepackSource,
        )
        inventory = importlib.import_module(f"surogate.serve.tools.convert.{target_key}.inventory")
        recipe = importlib.import_module(f"surogate.serve.tools.convert.{target_key}.recipe")

        candidates = {
            hf: entry
            for hf, entry in candidates.items()
            if entry["type"] in REPACKABLE_TYPES
        }
        source = GgufRepackSource.from_sources(gguf_path, candidates)
        planned = source.plan(recipe.RECIPES_BY_NAME, inventory.TENSOR_SPECS)
        keep: set[str] = set()
        for name in planned:
            for src in recipe.expression_sources(recipe.RECIPES_BY_NAME[name].expression):
                keep.add(src.name)
        return {hf: candidates[hf] for hf in sorted(keep & set(candidates))}
    return plan


def _run_converter_cached(model_dir: Path, out: Path, *, echo=print,
                          derived_frontend: bool = False,
                          gguf_repack: Path | None = None,
                          no_mtp: bool = False) -> Path:
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
    if gguf_repack is not None:
        if not target.gguf_repack:
            raise SystemExit(
                "surogate serve: internal error — repack map produced for a converter "
                "without --gguf-repack support."
            )
        cmd += ["--gguf-repack", str(gguf_repack)]
    if no_mtp:
        cmd += ["--no-mtp"]
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
