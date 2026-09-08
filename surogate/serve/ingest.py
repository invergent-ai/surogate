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
from collections.abc import Mapping
from dataclasses import dataclass

from surogate.serve.cache_version import SERVING_CACHE_VERSION
from pathlib import Path

# Registered converter targets in the vendored engine, keyed by facts read
# from the source config.json. Model breadth beyond this set is the plan's
# generalization phase; unknown models get a clear refusal, never a crash.
@dataclass(frozen=True)
class ConverterTarget:
    key: str                 # cache identity component
    module: str              # python -m <module> under the vendored sinfer root
    display: str
    gguf_repack: bool = False  # converter accepts --gguf-repack (PATCHES.md #14)


def _sinfer_root() -> Path | None:
    # Converters are part of the surogate package now; the "root" is the
    # repository root (kept for subprocess cwd/log context only).
    root = Path(__file__).resolve().parent.parent.parent
    return root if (root / "surogate" / "serve" / "convert").is_dir() else None


def classify_input(spec: str) -> str:
    """'artifact' | 'gguf' | 'safetensors_dir' | 'hf_repo_id' | 'unknown'."""
    p = Path(spec)
    if p.suffix == ".sinfer" and p.is_file():
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
    # One converter for the whole interleaved gated-delta architecture, at every size and
    # generation that shares it. The checkpoint states its dimensions and its quantisation, and
    # the converter reads both, so neither size nor export picks the module.
    if model_type in ("qwen3_5", "qwen3_6", "qwen3_8") and hidden > 0 and layers > 0:
        return ConverterTarget("qwen3_5", "surogate.serve.convert.qwen3_5.convert",
                               "Qwen3.5/3.6/3.8", gguf_repack=True)
    # Any size of the plain dense Qwen3: the artifact states its dimensions and the engine
    # binds against those, so the architecture is the gate.
    if model_type == "qwen3" and hidden > 0 and layers > 0:
        return ConverterTarget("qwen3", "surogate.serve.convert.qwen3.convert", "Qwen3",
                               gguf_repack=True)
    # LFM2 interleaves attention with a short convolution; which layer is which is
    # in the checkpoint, so the architecture is the only gate here too.
    if model_type == "lfm2" and hidden > 0 and layers > 0:
        return ConverterTarget("lfm2", "surogate.serve.convert.lfm2.convert", "LFM2", gguf_repack=True)
    # Qwen3-MoE: the same attention as the dense Qwen3 over a routed mixture with no
    # always-on expert. Its GGUF keeps its experts as K-quants, so it takes the repack path.
    if model_type == "qwen3_moe" and hidden > 0 and layers > 0:
        return ConverterTarget("qwen3_moe", "surogate.serve.convert.qwen3_moe.convert",
                               "Qwen3-MoE", gguf_repack=True)
    if model_type == "llama" and hidden > 0 and layers > 0:
        return ConverterTarget("llama", "surogate.serve.convert.llama.convert", "Llama",
                               gguf_repack=True)
    if model_type in ("gemma3", "gemma3_text") and hidden > 0 and layers > 0:
        return ConverterTarget("gemma3", "surogate.serve.convert.gemma3.convert", "Gemma 3",
                               gguf_repack=True)
    if model_type in ("qwen3_5_moe", "qwen3_6_moe") and int(config.get("num_experts", 0) or 0) > 0:
        return ConverterTarget("qwen3_5_moe",
                               "surogate.serve.convert.qwen3_5_moe.convert",
                               "Qwen3.5/3.6/3.8 MoE", gguf_repack=True)
    # Gemma 4. Its five published checkpoints share `model_type` across three architectures,
    # so the shape of the model decides the target rather than its name: the mixture is told
    # by `enable_moe_block`, and the E-series from the dense sizes by the two things only it
    # carries. Nothing here reads the architecture string, because all three spell it the same.
    if (
        model_type in ("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text")
        and hidden > 0
        and layers > 0
        and config.get("enable_moe_block")
        and int(config.get("num_experts", 0) or 0) > 0
    ):
        return ConverterTarget("gemma4_moe", "surogate.serve.convert.gemma4_moe.convert",
                               "Gemma 4 mixture", gguf_repack=True)
    if (
        model_type in ("gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text")
        and hidden > 0
        and layers > 0
        and not config.get("enable_moe_block")
    ):
        # The E-series and the dense sizes share `model_type` and are different
        # architectures: the E-series carries per-layer input embeddings and a tail of
        # layers holding only a query projection. Either marker sends it to its own target.
        e_series = (int(config.get("num_kv_shared_layers", 0) or 0) > 0
                    or int(config.get("hidden_size_per_layer_input", 0) or 0) > 0)
        if e_series:
            return ConverterTarget("gemma4_e", "surogate.serve.convert.gemma4_e.convert",
                                   "Gemma 4 E-series", gguf_repack=True)
        return ConverterTarget("gemma4", "surogate.serve.convert.gemma4.convert", "Gemma 4",
                               gguf_repack=True)
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
    """Hash checkpoint/frontend metadata and the names, sizes and mtimes of weight shards.

    Plan §5.3 wants source-content hashes; hashing 50+ GB on every serve is not
    acceptable startup cost, so v0 fingerprints metadata. A corrupted-in-place
    shard with identical size+mtime evades this — accepted and documented.
    """
    h = hashlib.sha256(f"serving-cache:{SERVING_CACHE_VERSION}:".encode())
    for metadata in sorted((*model_dir.glob("*.json"), *model_dir.glob("*.jinja"))):
        h.update(metadata.name.encode())
        h.update(b"\0")
        with metadata.open("rb") as handle:
            h.update(hashlib.file_digest(handle, "sha256").digest())
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


def _gguf_fingerprint(path: Path, *extra: Path) -> str:
    def stamp(p: Path) -> str:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{st.st_mtime_ns}"

    h = hashlib.sha256(f"serving-cache:{SERVING_CACHE_VERSION}:".encode())
    h.update(":".join(stamp(p) for p in (path, *extra)).encode())
    return h.hexdigest()[:24]


def _find_mtp_gguf(gguf_path: Path) -> Path | None:
    """The NextN draft head's GGUF, if one sits with the shards.

    Unsloth publishes it under `MTP/mtp-<model>-shared-<quant>.gguf`, separately from
    the trunk. Serving it is not the same artifact as serving without it, so whether
    one is found goes into the cache fingerprint.
    """
    stem = gguf_path.name.split("-00001-of-")[0]

    def shared_prefix(candidate: Path) -> int:
        """How much of the model name the two agree on, counted only to a name boundary.

        `Qwen3-0.6B-Q4_K_M` and `Qwen3.8-Flash-Next-shared-Q8_0` share the five letters
        of "Qwen3" and are different models; requiring the agreement to end on a `-` is
        what tells that apart from `Qwen3.8-Flash-Next-UD-Q4_K_XL`, which agrees through
        `Qwen3.8-Flash-Next-`.
        """
        name   = candidate.name.removeprefix("mtp-")
        prefix = os.path.commonprefix([name, stem])
        return len(prefix) if len(prefix) > 4 and prefix.endswith("-") else 0

    def compatible(candidate: Path) -> bool:
        from surogate.serve.gguf.lean import LeanGguf
        with LeanGguf(gguf_path) as trunk, LeanGguf(candidate) as draft:
            arch = trunk.kv("general.architecture")
            if not arch or draft.kv("general.architecture") != arch:
                return False
            for key in ("embedding_length", "attention.head_count", "attention.head_count_kv",
                        "attention.key_length", "ssm.state_size", "ssm.inner_size", "ssm.time_step_rank"):
                value = trunk.kv(f"{arch}.{key}")
                if value is None or draft.kv(f"{arch}.{key}") != value:
                    return False
            text_layers = trunk.kv(f"{arch}.block_count")
            draft_layers = draft.kv(f"{arch}.block_count")
            trunk_next = trunk.kv(f"{arch}.nextn_predict_layers", 0)
            draft_next = draft.kv(f"{arch}.nextn_predict_layers", 0)
            counts = (text_layers, draft_layers, trunk_next, draft_next)
            return (all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in counts)
                    and draft_next == 1 and text_layers - trunk_next == draft_layers - draft_next)

    for directory in (gguf_path.parent, gguf_path.parent / "MTP"):
        if not directory.is_dir():
            continue
        # One head per model, published once and shared by every quant of it -- the head's
        # name carries the model, not the trunk's quant suffix, so match on what they share.
        matches = [p for p in sorted(directory.glob("mtp-*.gguf")) if shared_prefix(p) > 0 and compatible(p)]
        if matches:
            return max(matches, key=shared_prefix)
    return None


def _ensure_from_gguf(gguf_path: Path, *, reuse_cache: bool = True, echo=print) -> Path:
    """GGUF → temp HF dir (dequant BF16) → vendored converter → cached weights.

    v0 bridge (surogate/serve/gguf/bridge.py): correctness inherits the converter's
    own preflight/hash checks; K-quant sources pay one documented
    double-quantization vs the original BF16 checkpoint. Temp BF16 shards
    (~2 bytes/param) are deleted after conversion.
    """
    from surogate.serve.gguf import bridge as serve_gguf

    root = _sinfer_root()
    if root is None:
        raise SystemExit("surogate serve: vendored engine tree not found (run from a checkout).")

    # Warm start: the cache name embeds the (target-independent) fingerprint,
    # so a hit returns without touching the GGUF — gguf-py's eager KV parse
    # costs ~10s on a 250k-token vocabulary and must stay off this path.
    mtp_path = _find_mtp_gguf(gguf_path)
    fp = _gguf_fingerprint(gguf_path, *( (mtp_path,) if mtp_path is not None else () ))
    for cached in cache_dir().glob(f"*-gguf-{fp}.sinfer") if reuse_cache else ():
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
            "  Registered today: Qwen3.5/3.6/3.8 (dense and MoE), Qwen3.8-Flash-Next,\n"
            "  Qwen3 (dense and MoE), Gemma 3/4, Llama/TinyLlama, LFM2 and GLM-5-Next. A target reads its\n"
            "  dimensions from the artifact, so what has to match is the architecture rather\n"
            "  than the size -- a family with no target here has none yet."
        )

    out = cache_dir() / f"{target_key}-gguf-{fp}.sinfer"

    if target_key in ("qwen4exp", "glm5_next"):
        return _convert_gguf_native(root, gguf_path, out, target_key=target_key,
                                    mtp_path=mtp_path, reader=reader, echo=echo)

    # Q8_0 repack (PATCHES.md #14): for targets whose converter takes
    # --gguf-repack, plan against the converter's own recipes which candidate
    # tensors it repacks bit-exactly; the bridge dequantizes only the rest.
    # Every target reads its GGUF where it lies: the bridge dequantises only what a value
    # transform forces, not the whole checkpoint.
    # Who repacks is the converter's own answer, not a list kept beside it: a converter that
    # accepts `--gguf-repack` can read its GGUF where it lies, and one that cannot says so by
    # not having the flag. A hand-kept set here was what stopped the three Gemma 4 targets
    # repacking after their converters had grown the flag.
    converter_key = serve_gguf.gguf_converter_key(gguf_path, reader)
    converts_in_place = "--gguf-repack" in _converter_options(
        root, f"surogate.serve.convert.{converter_key}.convert"
    )
    planner = _repack_planner(root, converter_key) if converts_in_place else None
    # No-MTP variant (PATCHES.md #15): community exports may strip nextn. Whether the
    # converter has the flag at all is asked the same way, at the call.
    arch = serve_gguf.read_gguf_summary(gguf_path, reader)["architecture"]
    nextn = reader.kv(f"{arch}.nextn_predict_layers", 0)
    no_mtp = int(nextn or 0) == 0
    work = cache_dir() / f"gguf-bridge-{fp}"
    try:
        model_dir = serve_gguf.build_hf_dir_from_gguf(
            gguf_path, converter_key, work, repack_planner=planner, reader=reader, echo=echo
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
        # A failed conversion is examined from its bridge; keeping it costs the disk it took.
        if os.environ.get("SUROGATE_GGUF_KEEP_BRIDGE", "0") != "1":
            shutil.rmtree(work, ignore_errors=True)


def _native_gguf_frontend(reader, directory: Path, *, echo=print) -> Path:
    """Extract frontend resources from the same GGUF that supplies the weights."""
    from surogate.serve.gguf.frontend import extract_generation_config, write_frontend
    arch = reader.kv("general.architecture")
    write_frontend(reader, arch, directory, echo=echo)
    generation = extract_generation_config(reader)
    (directory / "generation_config.json").write_text(json.dumps(generation, indent=2), encoding="utf-8")
    return directory


def _convert_gguf_native(root: Path, gguf_path: Path, out: Path, *,
                         target_key: str = "qwen4exp", mtp_path: Path | None = None,
                         reader=None, echo=print) -> Path:
    """Convert a native GGUF using its own tokenizer, chat template and special IDs."""
    if reader is None:
        from surogate.serve.gguf.bridge import open_gguf
        reader = open_gguf(gguf_path)
    frontend_dir = _native_gguf_frontend(reader, cache_dir() / "frontends" / out.stem, echo=echo)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".sinfer.partial")
    tmp.unlink(missing_ok=True)
    echo(f"surogate serve: preparing engine weights for {reader.kv('general.architecture')} "
         f"(one-time conversion of the GGUF shards; cached at {out})")
    cmd = [sys.executable, "-m", f"surogate.serve.convert.{target_key}.convert",
           "--gguf", str(gguf_path), "--frontend", str(frontend_dir), "--out", str(tmp),
           "--device", os.environ.get("SUROGATE_CONVERT_DEVICE", "cuda")]
    if mtp_path is not None:
        echo(f"surogate serve: NextN draft head found, {mtp_path.name}")
        cmd += ["--mtp", str(mtp_path)]
    if os.environ.get("SUROGATE_SERVE_DRY"):
        echo("DRY: cwd=" + str(root))
        echo("DRY: " + " ".join(cmd))
        raise SystemExit(0)
    env = {**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(cmd, cwd=root, env=env)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"surogate serve: conversion failed (exit {result.returncode}).")
    tmp.replace(out)
    return out


def _gguf_geometry(recipe_module, inventory_module, gguf_path: Path):
    """The checkpoint's dimensions, for a converter that builds its recipes from them.

    The bridged `config.json` does not exist yet when the repack is planned, so this reads
    the same numbers from the GGUF that the bridge will synthesise it from.
    """
    from surogate.serve.gguf import bridge as serve_gguf
    from surogate.serve.gguf.lean import LeanGguf

    with LeanGguf(gguf_path) as reader:
        arch = serve_gguf.read_gguf_summary(gguf_path, reader)["architecture"]
        # A converter can read GGUF fields directly or consume their normalized HF spelling.
        if hasattr(inventory_module, "geometry_from_gguf"):
            return inventory_module.geometry_from_gguf(reader.kv)
        resolve = getattr(inventory_module, "geometry_from_config", None) or getattr(recipe_module, "geometry_from_config", None)
        if resolve is None:
            raise ValueError("converter must resolve checkpoint geometry before repack planning")
        config = serve_gguf.synthesised_config(reader, arch)
        tokens = reader.kv("tokenizer.ggml.tokens")
    if config is None:
        raise ValueError(f"cannot resolve {arch} geometry from GGUF metadata")
    if inventory_module.TARGET_KEY in ("qwen3_5", "qwen3_5_moe"):
        if not tokens:
            raise ValueError("GGUF must declare tokenizer tokens before repack planning")
        return resolve(config, token_domain=len(tokens))
    return resolve(config)


def _gguf_synthesised_config(gguf_path: Path) -> dict:
    """The `config.json` the bridge will write, read early so a repack can be planned
    against the objects the conversion is actually going to produce."""
    from surogate.serve.gguf import bridge as serve_gguf
    from surogate.serve.gguf.lean import LeanGguf

    with LeanGguf(gguf_path) as reader:
        arch = serve_gguf.read_gguf_summary(gguf_path, reader)["architecture"]
        config = serve_gguf.synthesised_config(reader, arch)
    if config is None:
        raise SystemExit(
            f"surogate serve: {gguf_path.name} states no config this engine can synthesise, "
            "so a repack cannot be planned against it."
        )
    return config


def _repack_planner(root: Path, target_key: str):
    """Repack plan via the named vendored converter's registered recipes."""
    def plan(gguf_path: Path, candidates: dict[str, str]) -> dict[str, str]:
        import importlib
        import sys as _sys
        if str(root) not in _sys.path:
            _sys.path.insert(0, str(root))
        from surogate.serve.convert.common.gguf_repack import (
            NATIVE_TYPES,
            REPACKABLE_TYPES,
            GgufRepackSource,
        )
        # The one definition, rather than whichever converters happen to re-export it: a
        # recipe module that reads its sources without naming this helper is not thereby
        # un-plannable.
        from surogate.serve.convert.common.recipe import expression_sources
        inventory = importlib.import_module(f"surogate.serve.convert.{target_key}.inventory")
        # Every converter keeps its recipes in `recipe.py`, so there is one place to look.
        recipe = importlib.import_module(f"surogate.serve.convert.{target_key}.recipe")
        # The plan must describe *this* checkpoint, not the size the converter registers, so
        # a converter that can build from a geometry is asked to.
        geometry = _gguf_geometry(recipe, inventory, gguf_path)
        recipes = recipe.build_recipes(geometry)
        recipes_by_name = dict(recipes) if isinstance(recipes, Mapping) else {r.object_name: r for r in recipes}
        if hasattr(inventory, "build_stored_tensor_specs"):
            tensor_specs = inventory.build_stored_tensor_specs(
                geometry, tied_output_head=bool(geometry.declared.hf_config.get("tie_word_embeddings", True)))
        elif hasattr(inventory, "build_tensor_specs"):
            tensor_specs = inventory.build_tensor_specs(geometry)
        elif hasattr(inventory, "stored_objects"):
            tensor_specs = inventory.tensor_specs(inventory.stored_objects(
                geometry, tied_output_head=recipe.tied_output_head(geometry.declared.hf_config)))
        else:
            tensor_specs = inventory.tensor_specs(inventory.declared_objects(geometry))

        candidates = {
            hf: entry
            for hf, entry in candidates.items()
            if entry["type"] in REPACKABLE_TYPES
            or (entry["type"] in NATIVE_TYPES and os.environ.get("SUROGATE_GGUF_NATIVE", "1") != "0")
        }
        from surogate.serve.convert.common.safetensors import name_spellings

        def spelled(name: str, pool: Mapping[str, object]) -> str | None:
            for spelling in name_spellings(name):
                if spelling in pool:
                    return spelling
            return None

        # The plan is a fixed point over the sources kept out of the bridge. A source one covered
        # object reads may feed an uncovered one too -- a fused parent whose halves land in
        # different types, one the plan can move and one it cannot -- and kept out of the bridge
        # the uncovered recipe finds nothing to materialise from (the converter refuses exactly
        # that). So such a source is bridged after all; and dropping it can un-cover another
        # object that read it natively, whose remaining sources must then be bridged too. The
        # loop re-plans against what is still kept until nothing more drops; it only ever
        # shrinks, so it ends.
        kept = dict(candidates)
        while True:
            source = GgufRepackSource.from_sources(gguf_path, kept)
            planned = source.plan(recipes_by_name, tensor_specs)
            native = source.plan_native(
                recipes_by_name,
                tensor_specs,
                exclude_suffixes=getattr(recipe, "NATIVE_EXCLUDE_SUFFIXES", ()),
            ) if getattr(recipe, "GGUF_NATIVE", True) else {}
            # A fused parent stored as two typed halves keeps its sources too, or the bridge
            # dequantises them and the converter can no longer see the types it split on.
            halves = (source.plan_native_halves(recipes_by_name, tensor_specs)
                      if getattr(recipe, "GGUF_NATIVE", True) else {})
            covered = set(planned) | set(native) | set(halves)
            # Recipes and the bridge may spell one tensor differently; resolve each wanted
            # source to the candidate that actually holds it before intersecting, or nothing
            # matches and the bridge dequantises weights the converter was going to read
            # straight from the file.
            keep: set[str] = set()
            for name in covered:
                for src in expression_sources(recipes_by_name[name].expression):
                    found = spelled(src.name, kept)
                    if found is not None:
                        keep.add(found)
            for name, tensor_recipe in recipes_by_name.items():
                if name in covered:
                    continue
                for src in expression_sources(tensor_recipe.expression):
                    found = spelled(src.name, kept)
                    if found is not None:
                        keep.discard(found)
            if keep == set(kept):
                break
            kept = {hf: kept[hf] for hf in sorted(keep)}
        return kept
    return plan


def _has_vision_tensors(model_dir: Path) -> bool:
    """True when the HF-layout dir holds any `model.visual.*` weight."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        return any(name.startswith("model.visual.") for name in weight_map)
    for shard in model_dir.glob("*.safetensors"):
        from safetensors import safe_open  # lazy: only for a single-shard dir
        with safe_open(str(shard), framework="pt") as reader:
            if any(name.startswith("model.visual.") for name in reader.keys()):
                return True
    return False


def _converter_options(root: Path | None, module: str) -> str:
    """The converter module's own source, to ask whether it accepts a flag."""
    if root is None:
        return ""
    source = root.joinpath(*module.split(".")).with_suffix(".py")
    try:
        return source.read_text()
    except OSError:
        return ""


def _run_converter_cached(model_dir: Path, out: Path, *, echo=print,
                          derived_frontend: bool = False,
                          gguf_repack: Path | None = None,
                          no_mtp: bool = False) -> Path:
    """Shared converter driver: model_dir (HF layout) → atomic-published `out`."""
    root = _sinfer_root()
    config = _flatten_text_config(json.loads((model_dir / "config.json").read_text()))
    target = converter_for_config(config)
    if target is None:
        raise SystemExit("surogate serve: internal error — bridged model dir maps to no converter.")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".sinfer.partial")
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
    # Only some converters have an MTP block to omit, and passing the flag to one that has
    # none is an argparse error rather than a no-op. Ask the converter, as `--no-vision` does.
    if no_mtp and "--no-mtp" in _converter_options(root, target.module):
        cmd += ["--no-mtp"]
    # A text-only export of a vision family (every community GGUF of these, so far) carries no
    # visual.* tensors; the artifact then omits vision/* entirely and the loader, which already
    # probes for a tower, refuses only `--vision` against it.
    if "--no-vision" in _converter_options(root, target.module) and not _has_vision_tensors(model_dir):
        echo("surogate serve: this checkpoint carries no vision tower — converting text-only "
             "(`--vision` will be unavailable for it).")
        cmd += ["--no-vision"]
    if os.environ.get("SUROGATE_CONVERT_DEVICE"):
        cmd += ["--device", os.environ["SUROGATE_CONVERT_DEVICE"]]
    if os.environ.get("SUROGATE_SERVE_DRY"):
        echo("DRY: cwd=" + str(root))
        echo("DRY: " + " ".join(cmd))
        raise SystemExit(0)
    env = {**os.environ, "PYTHONPATH": str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run(cmd, cwd=root, env=env)
    if result.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"surogate serve: conversion failed (exit {result.returncode}).")
    tmp.replace(out)
    return out


# --------------------------------------------------------------------------------------------
# Encoder (embedding) models
# --------------------------------------------------------------------------------------------
#
# An encoder runs one forward: no KV cache, no sampler, no decode round. It is
# served by its own binary, so it gets its own ingest entry rather than a branch
# inside the generative one. The registered converter is GGUF-native, and its
# frontend is a tokenizer and nothing else.

ENCODER_TARGETS = {
    # gguf architecture string -> (cache key, converter module, display)
    "gemma-embedding": ("gemma_embedding",
                        "surogate.serve.convert.gemma_embedding.convert",
                        "EmbeddingGemma"),
}

#: What the encoder converter reads out of a --frontend directory.
ENCODER_FRONTEND_FILES = ("tokenizer.model", "tokenizer_config.json")


def _gguf_architecture(path: Path) -> str | None:
    """The GGUF's `general.architecture`, read through the same bridge the
    generative path uses. Costs a full KV parse, so callers check the cache
    first."""
    from surogate.serve.gguf import bridge as serve_gguf

    try:
        reader = serve_gguf.open_gguf(path)
        return serve_gguf.read_gguf_summary(path, reader)["architecture"]
    except Exception:
        return None


def _encoder_frontend(gguf_path: Path, frontend: str | None) -> Path:
    """Where the tokenizer comes from, with a message instead of a stack trace."""
    if frontend is not None:
        directory = Path(frontend).expanduser().resolve()
        missing = [n for n in ENCODER_FRONTEND_FILES if not (directory / n).is_file()]
        if missing:
            raise SystemExit(
                f"surogate serve --embed: --frontend {directory} is missing {', '.join(missing)}."
            )
        return directory
    beside = gguf_path.parent
    if all((beside / name).is_file() for name in ENCODER_FRONTEND_FILES):
        return beside
    raise SystemExit(
        "surogate serve --embed: this GGUF needs a tokenizer to convert, and none was found\n"
        f"  beside it in {beside}.\n"
        "  Pass --frontend DIR pointing at the model's Hugging Face snapshot (the directory\n"
        f"  holding {' and '.join(ENCODER_FRONTEND_FILES)})."
    )


def ensure_encoder_weights(spec: str, *, frontend: str | None = None,
                           reuse_cache: bool = True, echo=print) -> Path:
    """Resolve `spec` to encoder-loadable weights, converting through the cache.

    Accepts an internal artifact (passthrough) or a `.gguf` file. Raises
    SystemExit with a clear message on anything else."""
    kind = classify_input(spec)
    if kind == "artifact":
        return Path(spec)
    if kind != "gguf":
        raise SystemExit(
            f"surogate serve --embed: cannot interpret '{spec}'. The registered encoder "
            "converter is GGUF-native, so pass a .gguf file (or an already-converted "
            "artifact).\n  Registered today: EmbeddingGemma (gemma-embedding)."
        )

    gguf_path = Path(spec).expanduser().resolve()
    architecture = _gguf_architecture(gguf_path)
    target = ENCODER_TARGETS.get(architecture or "")
    if target is None:
        raise SystemExit(
            "surogate serve --embed: this model is not yet supported by the encoder path.\n"
            f"  gguf architecture={architecture!r}\n"
            f"  Registered today: {', '.join(sorted(ENCODER_TARGETS))}."
        )
    key, module, display = target

    fp = _gguf_fingerprint(gguf_path)
    out = cache_dir() / f"{key}-gguf-{fp}.sinfer"
    if reuse_cache and out.is_file() and out.stat().st_size > 0:
        echo(f"surogate serve: using cached encoder weights ({out.name})")
        return out

    frontend_dir = _encoder_frontend(gguf_path, frontend)
    root = _sinfer_root()
    if root is None:
        raise SystemExit("surogate serve: vendored engine tree not found (run from a checkout).")

    out.parent.mkdir(parents=True, exist_ok=True)
    echo(f"surogate serve: converting {display} ({gguf_path.name}) -> {out.name}")
    cmd = [sys.executable, "-m", module, "--gguf", str(gguf_path),
           "--frontend", str(frontend_dir), "--out", str(out)]
    if os.environ.get("SUROGATE_SERVE_DRY"):
        echo("  (dry run) " + " ".join(cmd))
        return out
    result = subprocess.run(cmd, cwd=str(root))
    if result.returncode != 0 or not out.is_file():
        raise SystemExit(f"surogate serve: encoder conversion failed ({display}).")
    return out


def ensure_engine_weights(spec: str, *, reuse_cache: bool = True, echo=print) -> Path:
    """Resolve `spec` (safetensors dir | HF repo id | GGUF | internal artifact)
    to an engine-loadable weights file, converting through the transparent
    cache when needed. Raises SystemExit with a clear message on refusal."""
    kind = classify_input(spec)

    if kind == "artifact":
        # Internal/dev passthrough (not a supported product input).
        return Path(spec)

    if kind == "gguf":
        return _ensure_from_gguf(Path(spec).resolve(), reuse_cache=reuse_cache, echo=echo)

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

    root = _sinfer_root()
    if root is None:
        raise SystemExit("surogate serve: vendored engine tree not found (run from a checkout).")

    config = _flatten_text_config(json.loads((model_dir / "config.json").read_text()))
    target = converter_for_config(config)
    if target is None:
        raise SystemExit(
            "surogate serve: this model is not yet supported by the native engine.\n"
            f"  model_type={config.get('model_type')!r} hidden_size={config.get('hidden_size')} "
            f"layers={config.get('num_hidden_layers')}\n"
            "  Registered today: Qwen3.6-27B, Qwen3.8-27B (BF16/NVFP4), Qwen3.6-35B-A3B, Qwen3 (any size)."
        )

    fp = source_fingerprint(model_dir)
    out = cache_dir() / f"{target.key}-{fp}.sinfer"
    if reuse_cache and out.is_file() and out.stat().st_size > 0:
        echo(f"surogate serve: using cached engine weights ({out.name})")
        return out
    return _run_converter_cached(model_dir, out, echo=echo)
