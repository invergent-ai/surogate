"""Prepare Muse-Glimmer GGUF weights, retaining quantized matrices in their source files."""
from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import numpy as np
import torch

from surogate.serve.artifact import ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
from surogate.serve.artifact.layouts import encode_direct
from surogate.serve.convert.common.gguf_source import GgufSource
from surogate.serve.gguf.frontend import write_frontend, extract_generation_config


def geometry(source):
    arch = source.kv("general.architecture")
    if arch != "muse-glimmer":
        raise ValueError("expected a Muse-Glimmer GGUF")
    def get(key):
        value = source.kv(arch + "." + key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {arch}.{key}")
        return value
    def integer(key):
        value = get(key)
        if value != int(value):
            raise ValueError(f"invalid integer {arch}.{key}")
        return int(value)
    heads, kv, dim = (integer(key) for key in ("attention.head_count", "attention.head_count_kv", "attention.key_length"))
    if heads % kv or source.kv(arch + ".attention.value_length", dim) != dim:
        raise ValueError("invalid Muse-Glimmer attention geometry")
    layers = integer("block_count")
    pattern = source.kv(arch + ".attention.sliding_window_pattern", 4)
    if isinstance(pattern, int) and not isinstance(pattern, bool) and pattern > 0:
        windowed = [(i + 1) % pattern != 0 for i in range(layers)]
    elif isinstance(pattern, list) and len(pattern) == layers and all(isinstance(x, (bool, int)) and x in (0, 1) for x in pattern):
        windowed = [bool(x) for x in pattern]
    else:
        raise ValueError("invalid Muse-Glimmer sliding window schedule")
    return dict(hidden=integer("embedding_length"), residual=integer("embedding_length"),
                layers=layers, intermediate=integer("feed_forward_length"),
                query_heads=heads, kv_heads=kv, head_dim=dim,
                rotary_dim=0, sliding_rotary_dim=dim, rope_theta=0,
                sliding_rope_theta=float(get("rope.freq_base")),
                rms_epsilon=float(get("attention.layer_norm_rms_epsilon")), post_norm_epsilon=1e-8,
                attention_scale=dim ** -0.5, embedding_scale=1,
                sliding_window=integer("attention.sliding_window"),
                max_context=integer("context_length"),
                output_rows=source.tensor("token_embd.weight").shape[0],
                token_domain=len(source.kv("tokenizer.ggml.tokens")),
                logit_scale=float(get("logit_scale")),
                logit_softcap=float(get("final_logit_softcapping"))), [
                    "sliding_attention" if yes else "full_attention" for yes in windowed]


def convert(gguf, frontend, output, mmproj=None):
    source = GgufSource(Path(gguf), extra=[Path(mmproj)] if mmproj else [])
    try:
        g, layers = geometry(source)
        h, q, kv, d, m = g["hidden"], g["query_heads"], g["kv_heads"], g["head_dim"], g["intermediate"]
        objects = [("text/token_embedding", "token_embd.weight", (g["output_rows"], h), 0),
                   ("text/output_head", "output.weight", (g["output_rows"], h), 0),
                   ("text/final_norm", "output_norm.weight", (h,), 0)]
        for i in range(g["layers"]):
            entries = [(name, src, (h,), 0) for name, src in (
                ("input_norm", "attn_norm.weight"), ("post_attention_norm", "post_attention_norm.weight"),
                ("pre_feedforward_norm", "ffn_norm.weight"), ("post_feedforward_norm", "post_ffw_norm.weight"))]
            entries += [("attention/query_norm", "attn_q_norm.weight", (d,), 0),
                        ("attention/key_norm", "attn_k_norm.weight", (d,), 0),
                        ("attention/query", "attn_q.weight", (q*d, h), q),
                        ("attention/key", "attn_k.weight", (kv*d, h), kv),
                        ("attention/value", "attn_v.weight", (kv*d, h), 0),
                        ("attention/gate", "attn_gate.weight", (q*d, h), 0),
                        ("attention/output", "attn_output.weight", (h, q*d), 0),
                        ("mlp/gate", "ffn_gate.weight", (m, h), 0),
                        ("mlp/up", "ffn_up.weight", (m, h), 0),
                        ("mlp/down", "ffn_down.weight", (h, m), 0)]
            objects += [(f"text/layers/{i}/{name}", f"blk.{i}.{src}", shape, heads)
                        for name, src, shape, heads in entries]
        vg, processor = {}, None
        if mmproj:
            from .vision import vision_objects
            vg, vision, processor = vision_objects(source, g)
            objects += vision
        extras = source.tensors.keys() - {src for _, src, _, _ in objects}
        if extras:
            raise ValueError(f"unsupported Muse-Glimmer tensors: {sorted(extras)[:8]}")
        specs = []
        converted = {}
        for name, src, shape, heads in objects:
            tensor = source.tensor(src)
            if tensor.shape != shape and not (src == "v.patch_embd.weight" and tensor.shape == (shape[0], 3, math.isqrt(shape[1] // 3), math.isqrt(shape[1] // 3))):
                raise ValueError(f"{src}: expected {shape}, got {tensor.shape}")
            if len(shape) == 1 or tensor.type_name == "F32":
                value = source.float32(src).copy().reshape(shape)
                if heads:
                    order = np.arange(shape[0]).reshape(heads, shape[0] // heads // 2, 2).swapaxes(1, 2).reshape(-1)
                    value = value[order]
                converted[name] = encode_direct(torch.from_numpy(value).to(torch.bfloat16), "BF16")
                specs.append(TensorSpec(name, shape, "BF16", "contiguous-le-v1"))
                continue
            fmt = {"F16": "FP16", "F32": "FP32"}.get(tensor.type_name, tensor.type_name)
            layout = "contiguous-le-v1" if fmt in ("FP16", "FP32", "BF16") else "ggml-blocks-v1"
            if heads:
                rows = shape[0]
                order = np.arange(rows).reshape(heads, rows // heads // 2, 2).swapaxes(1, 2).reshape(-1)
                row_bytes = tensor.nbytes // rows
                runs = tuple((tensor.shard, tensor.offset + int(row) * row_bytes, row_bytes) for row in order)
            else:
                runs = ((tensor.shard, tensor.offset, tensor.nbytes),)
            specs.append(TensorSpec(name, shape, fmt, layout, runs=runs))
        front = Path(frontend) if frontend else Path(output).with_suffix(".frontend")
        if not frontend:
            write_frontend(source.readers[0], "muse-glimmer", front)
            (front / "generation_config.json").write_text(json.dumps(extract_generation_config(source.readers[0])))
        resources = {"frontend/" + name: (front / name).read_bytes() for name in
                     ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")}
        config = json.loads(resources['frontend/tokenizer_config.json'])
        config['response_template'] = {'type': 'muse_glimmer'}
        resources['frontend/tokenizer_config.json'] = json.dumps(config).encode()
        if processor:
            resources['frontend/preprocessor_config.json'] = json.dumps(processor).encode()
        specs += [ResourceSpec(name, "raw-bytes-v1", len(data)) for name, data in resources.items()]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(output, ArtifactIdentity("muse-glimmer", "groupwise-int", "muse_glimmer"),
                            specs, external=[(str(p.resolve()), p.stat().st_size) for p in source.shards],
                            geometry=g, layer_types=layers, vision_geometry=vg) as writer:
            for name, data in converted.items():
                writer.write(name, data)
            for name, data in resources.items():
                writer.write(name, data)
        Path(str(output) + '.conversion.json').write_text(json.dumps({'source': str(gguf), 'mmproj': str(mmproj) if mmproj else None}))
        return Path(output)
    finally:
        source.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True)
    p.add_argument("--frontend")
    p.add_argument("--mmproj")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    convert(args.gguf, args.frontend, args.out, args.mmproj)


if __name__ == "__main__":
    main()
