"""Inkling-2 (``inkling2_loop``) checkpoint -> SInfer v2 artifact (ARTIFACT.md, interface I-1).

Every object is the checkpoint's words, rearranged but never re-rounded: BF16 / FP32 tensors are
bit copies (fused or stacked where the reference fuses them), NVFP4 experts keep their codes and
E4M3 block scales (the scales only move into the layout's 128x4 tiles) with the per-expert global
as the raw ``amax`` and ``outer_scale_denominator`` objects, MXFP8 experts keep codes and E8M0
scales as stored. The only computed objects are the optional FP8 attention variants
(``fp8_attention.py``), which are additional to the BF16 attention, never instead of it.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch

from surogate.serve.artifact.container import (
    ArtifactIdentity,
    ArtifactWriter,
    RAW_BYTES_V1,
    ResourceSpec,
    TensorSpec,
)
from surogate.serve.artifact.layouts import (
    encode_direct,
    encode_fp8_grid,
    encode_fp8_row_f32,
    encode_mxfp8,
    encode_nvfp4,
)

from .checkpoint import Checkpoint, Geometry
from .fp8_attention import RULE, STORAGE_FORMAT, quantize_slices
from .inventory import MTP, LOOP_STREAMS, Entry, build_inventory, covered_sources, vision_geometry

ARCHITECTURE = "inkling2"
SCHEMA = "inkling2-artifact/1"
CONFIG_RESOURCE = "inkling2/config.json"
#: (artifact resource, checkpoint file, required)
FRONTEND_RESOURCES = (
    ("frontend/tokenizer.json", "tokenizer.json", True),
    ("frontend/tokenizer_config.json", "tokenizer_config.json", True),
    ("frontend/special_tokens_map.json", "special_tokens_map.json", True),
    ("frontend/generation_config.json", "generation_config.json", True),
    ("frontend/chat_template.jinja", "chat_template.jinja", False),
    ("inkling2/source_config.json", "config.json", True),
    ("inkling2/hf_quant_config.json", "hf_quant_config.json", True),
)
_DTYPE = {"BF16": torch.bfloat16, "FP32": torch.float32}
_ST_DTYPE = {"BF16": "BF16", "FP32": "F32"}
_IDENTITY = torch.tensor(1.0, dtype=torch.float32)


def _git_commit() -> str:
    root = Path(__file__).resolve().parents[4]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "surogate/serve"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def text_geometry(g: Geometry) -> dict[str, int | float]:
    """The root ``geometry`` member: existing TextGeometry fields only (ARTIFACT.md section 2)."""
    t = g.text
    return {
        "hidden": g.hidden, "residual": g.hidden, "layers": g.layers, "intermediate": g.intermediate,
        "output_rows": g.vocab_padded, "token_domain": g.vocab, "query_heads": g.query_heads,
        "kv_heads": g.kv_heads, "head_dim": g.head_dim, "rotary_dim": 0, "rope_theta": 0.0,
        "rms_epsilon": float(t["rms_norm_eps"]), "max_context": int(t["model_max_length"]),
        "attention_scale": float(g.head_dim) ** -0.5, "sliding_window": g.sliding_window,
        "experts": g.experts, "experts_per_token": g.experts_per_token, "shared_intermediate": 0,
        "routed_scale": float(t["route_scale"]), "logit_softcap": float(t["final_logit_softcapping"]),
    }


def layer_types(g: Geometry) -> list[str]:
    return ["full_attention" if g.is_global(l) else "sliding_attention" for l in range(g.layers)]


def vision_config(ckpt: Checkpoint, vision: bool | None):
    """The checkpoint's ``vision_config`` when the vision tower is carried, else None.

    ``vision`` None carries the tower when the checkpoint has one; True requires it; False leaves it out.
    """
    if vision is False:
        return None
    config = ckpt.config.get("vision_config")
    if config is None or not ckpt.vision_names():
        if vision:
            raise ValueError("the checkpoint has no vision tower")
        return None
    return config


def config_document(ckpt: Checkpoint, *, fp8_schemes, gate_up_order, mtp, sha256: dict[str, str],
                    vision: bool = False) -> dict:
    g, t = ckpt.geometry, ckpt.geometry.text
    files = {}
    for name in ckpt.source_files():
        files[name] = {"sha256": ckpt.sha256(name, sha256), "bytes": (ckpt.dir / name).stat().st_size}
    doc = {
        "schema": SCHEMA,
        "model_type": ckpt.config["model_type"],
        "source": {"repo": "thinkingmachines/Inkling-2-Mini", "revision": "ed40e7da", "files": files},
        "converter": {"module": "surogate.serve.convert.inkling2", "git_commit": _git_commit(),
                      "options": {"attention": ["bf16", *(f"fp8-{s}" for s in fp8_schemes)],
                                  "gate_up_order": gate_up_order, "mtp": bool(mtp),
                                  "vision": bool(vision)}},
        "text": {
            "hidden": g.hidden, "layers": g.layers, "vocab_padded": g.vocab_padded, "vocab": g.vocab,
            "query_heads": g.query_heads, "kv_heads": g.kv_heads, "head_dim": g.head_dim,
            "d_rel": g.d_rel, "rel_extent": g.rel_extent, "sliding_window": g.sliding_window,
            "conv_kernel": g.conv_kernel, "rms_eps": float(t["rms_norm_eps"]),
            "qk_l2_clip": float(t["qk_l2_clip_threshold"]), "rel_l2_clip": float(t["rel_l2_clip_threshold"]),
            "log_scaling_n_floor": int(t["log_scaling_n_floor"]),
            "log_scaling_alpha": float(t["log_scaling_alpha"]),
            "logit_softcap": float(t["final_logit_softcapping"]), "experts": g.experts,
            "experts_per_token": g.experts_per_token, "expert_intermediate": g.intermediate,
            "gate_softcap": float(t["mlp_gate_softcap"]), "up_softcap": float(t["mlp_up_softcap"]),
            "route_scale": float(t["route_scale"]), "use_gate_bias": bool(t["use_gate_bias"]),
            "value_norm_no_gamma": bool(t["use_value_norm_no_gamma"]),
            "attn_res_normalize_values": bool(t["attn_res_normalize_values"]),
            "attn_res_block_layers": int(t["n_layers_per_attn_res_block"]),
            "eos_token_id": int(ckpt.config["eos_token_id"]), "max_context": int(t["model_max_length"]),
        },
        "global_layers": list(g.global_layers),
        "layers": [
            {"layer": l, "attention": "global" if g.is_global(l) else "local",
             "experts": g.expert_format[l], "core": g.is_core(l), "visits": g.layer_visits(l),
             "attn_query_rows": 0 if l == 0 else (3 if g.is_global(l) else 1)}
            for l in range(g.layers)
        ],
        "loop": {"core_start": g.core_start, "core_end": g.core_end, "passes": g.passes,
                 "gate_rank": g.gate_rank, "is_varloop": False, "streams": list(LOOP_STREAMS)},
        "visits": {"count": len(g.visits), "final_output": len(g.visits),
                   "table": [list(v) for v in g.visits]},
        "moe": {"gate_up_rows": gate_up_order,
                "nvfp4_layers": [l for l in range(g.layers) if g.expert_format[l] == "nvfp4"],
                "mxfp8_layers": [l for l in range(g.layers) if g.expert_format[l] == "mxfp8"]},
        "attention": {"bf16": True,
                      "fp8": {s: {"format": STORAGE_FORMAT[s], "rule": RULE} for s in fp8_schemes}},
    }
    if mtp and g.draft is not None:
        d = g.draft
        doc["dspark"] = {
            "layers": d.layers, "local_layers": list(d.local_layers), "hidden": d.hidden,
            "query_heads": d.query_heads, "kv_heads": d.kv_heads, "head_dim": d.head_dim,
            "intermediate": d.intermediate, "rope_theta": d.rope_theta, "sliding_window": d.sliding_window,
            "block_size": d.block_size, "markov_rank": d.markov_rank, "markov_head_type": d.markov_head_type,
            "target_visits": list(d.target_visits), "rms_eps": d.rms_eps, "max_position": d.max_position,
            "fuse_rows": d.layers + 1, "fuse_features": len(d.target_visits),
        }
    return doc


def _source(ckpt: Checkpoint, name: str) -> torch.Tensor:
    return ckpt.get_mtp(name[len(MTP):]) if name.startswith(MTP) else ckpt.get(name)


def _source_header(ckpt: Checkpoint, name: str):
    return ckpt.mtp_headers[name[len(MTP):]] if name.startswith(MTP) else ckpt.headers[name]


def gate_up_permutation(per_expert_rows: int) -> torch.Tensor:
    """Row order of one expert's w13 under ``split``: [gate 0..I-1 ; up 0..I-1] from interleaved."""
    half = per_expert_rows // 2
    return torch.cat([torch.arange(half) * 2, torch.arange(half) * 2 + 1])


def _expert_rows(tensor: torch.Tensor, entry: Entry) -> torch.Tensor:
    """[E, N, X] -> [E*N, X], with w13 rows reordered per expert under ``split``."""
    experts, rows = tensor.shape[0], tensor.shape[1]
    if entry.extra.get("gate_up_order") == "split":
        tensor = tensor.index_select(1, gate_up_permutation(rows))
    return tensor.reshape(experts * rows, -1).contiguous()


def preflight(ckpt: Checkpoint, entries: list[Entry]) -> None:
    """Every text / MTP (/ vision, when carried) checkpoint tensor is read exactly once and has the declared signature."""
    problems = []
    counts: dict[str, int] = {}
    for name in covered_sources(entries):
        counts[name] = counts.get(name, 0) + 1
    for name, count in counts.items():
        if count != 1:
            problems.append(f"{name} is read by {count} objects")
    expected = set(ckpt.text_names()) | {MTP + n for n in ckpt.mtp_headers}
    mtp_entries = any(e.name.startswith("mtp/") for e in entries)
    if not mtp_entries:
        expected = {n for n in expected if not n.startswith(MTP)}
    if any(e.name.startswith("vision/") for e in entries):
        expected |= set(ckpt.vision_names())
    missing = sorted(expected - set(counts))
    extra = sorted(set(counts) - expected)
    problems += [f"checkpoint tensor not carried: {n}" for n in missing]
    problems += [f"object reads a tensor the checkpoint lacks: {n}" for n in extra]
    for entry in entries:
        if entry.kind != "direct":
            continue
        elements = 1
        for d in entry.shape:
            elements *= d
        total = 0
        for name in entry.sources:
            if name not in counts or name in extra:
                continue
            dtype, shape = _source_header(ckpt, name)
            if dtype != _ST_DTYPE[entry.format]:
                problems.append(f"{entry.name}: source {name} is {dtype}, object is {entry.format}")
            n = 1
            for d in shape:
                n *= d
            total += n
        if total != elements:
            problems.append(f"{entry.name}: sources hold {total} values, object {entry.shape} needs {elements}")
    if problems:
        raise ValueError("inventory disagrees with the checkpoint:\n  " + "\n  ".join(problems[:20]))


def payload(ckpt: Checkpoint, entry: Entry) -> bytes:
    if entry.kind == "direct":
        parts = [_source(ckpt, s).reshape(-1) for s in entry.sources]
        tensor = parts[0] if len(parts) == 1 else torch.cat(parts)
        if tensor.dtype != _DTYPE[entry.format]:
            raise TypeError(f"{entry.name}: {tensor.dtype}, expected {entry.format}")
        return encode_direct(tensor.reshape(entry.shape), entry.format)
    if entry.kind == "nvfp4":
        codes = _source(ckpt, entry.sources[0])
        scales = _source(ckpt, entry.sources[1])
        if codes.dtype != torch.uint8 or scales.dtype != torch.float8_e4m3fn:
            raise TypeError(f"{entry.name}: expected U8 codes and E4M3 scales")
        return encode_nvfp4(_expert_rows(codes, entry), _expert_rows(scales.view(torch.uint8), entry),
                            _IDENTITY, entry.shape)
    if entry.kind == "mxfp8":
        codes = _source(ckpt, entry.sources[0])
        scales = _source(ckpt, entry.sources[1])
        if codes.dtype != torch.float8_e4m3fn or scales.dtype != torch.uint8:
            raise TypeError(f"{entry.name}: expected E4M3 codes and U8 (E8M0) scales")
        return encode_mxfp8(_expert_rows(codes.view(torch.uint8), entry), _expert_rows(scales, entry),
                            entry.shape)
    if entry.kind == "fp8":
        slices = [_source(ckpt, s) for s in entry.sources]
        q = quantize_slices(slices, entry.extra["scheme"])
        if entry.format == "FP8_E4M3FN_ROW_F32S":
            return encode_fp8_row_f32(q.codes, q.row_scales, entry.shape)
        return encode_fp8_grid(q.codes, q.grid, entry.shape, entry.format)
    raise ValueError(f"unknown entry kind {entry.kind}")


def read_sha256_file(path: str | Path | None) -> dict[str, str]:
    """``sha256sum`` output -> {basename: digest}."""
    if not path:
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[Path(parts[1].lstrip("*")).name] = parts[0]
    return out


def convert(model_dir: str | Path, out_path: str | Path, *, fp8_schemes: tuple[str, ...] = (),
            gate_up_order: str = "interleaved", mtp: bool = True, vision: bool | None = None,
            sha256_file: str | None = None, log=print) -> dict:
    started = time.time()
    ckpt = Checkpoint(model_dir)
    g = ckpt.geometry
    vc = vision_config(ckpt, vision)
    entries = build_inventory(g, fp8_schemes=fp8_schemes, gate_up_order=gate_up_order, mtp=mtp, vision=vc)
    preflight(ckpt, entries)
    doc = config_document(ckpt, fp8_schemes=fp8_schemes, gate_up_order=gate_up_order, mtp=mtp,
                          sha256=read_sha256_file(sha256_file), vision=vc is not None)
    resources: list[tuple[str, bytes]] = [
        (CONFIG_RESOURCE, (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode("utf-8"))]
    for name, file, required in FRONTEND_RESOURCES:
        path = ckpt.dir / file
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            continue
        data = path.read_bytes()
        if not data:
            raise ValueError(f"{path} is empty")
        resources.append((name, data))
    specs = [ResourceSpec(name, RAW_BYTES_V1, len(data)) for name, data in resources]
    specs += [TensorSpec(e.name, e.shape, e.format, e.layout) for e in entries]
    variant = "bf16" + "".join(f"+fp8-{s}" for s in fp8_schemes)
    identity = ArtifactIdentity(model_id="thinkingmachines/Inkling-2-Mini@ed40e7da",
                                weights_id=f"release+attn-{variant}", architecture=ARCHITECTURE)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with ArtifactWriter(partial, identity, specs, geometry=text_geometry(g), layer_types=layer_types(g),
                        vision_geometry=None if vc is None else vision_geometry(vc, g.hidden)) as writer:
        for name, data in resources:
            writer.write(name, data)
        for i, entry in enumerate(entries):
            writer.write(entry.name, payload(ckpt, entry))
            if i % 50 == 0 or entry.kind != "direct":
                log(f"[{time.time() - started:7.1f}s] {i + 1}/{len(entries)} {entry.name} {entry.format} "
                    f"{list(entry.shape)}")
    partial.replace(out)
    log(f"wrote {out} ({out.stat().st_size} bytes) in {time.time() - started:.1f}s")
    return {"objects": len(specs), "tensors": len(entries), "resources": len(resources),
            "bytes": out.stat().st_size, "seconds": time.time() - started}


__all__ = ["ARCHITECTURE", "CONFIG_RESOURCE", "FRONTEND_RESOURCES", "SCHEMA", "config_document",
           "convert", "gate_up_permutation", "layer_types", "payload", "preflight", "text_geometry",
           "vision_config"]
