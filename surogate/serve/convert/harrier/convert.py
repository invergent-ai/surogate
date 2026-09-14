"""Convert Q8_0 Harrier Qwen3/Gemma3 GGUFs for the native CPU/GPU embedding server."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from surogate.serve.artifact import ArtifactIdentity, ArtifactWriter, ResourceSpec, TensorSpec
from surogate.serve.artifact.layouts import encode_direct, encode_row_split
from surogate.serve.artifact.numeric import get_format
from surogate.serve.convert.common.gguf_source import GgufSource
from surogate.serve.gguf.frontend import (
    _declared_merges, _spm_tokenizer_json,
    extract_tokenizer_json, synthesize_tokenizer_config,
)


def geometry_from_gguf(source):
    arch = source.kv("general.architecture")
    if arch not in ("qwen3", "gemma3", "gemma-embedding"):
        raise ValueError(f"unsupported Harrier architecture: {arch}")
    if source.kv(f"{arch}.pooling_type") != 3:
        raise ValueError("Harrier requires last-token pooling (GGUF pooling_type=3)")
    if source.kv(f"{arch}.attention.causal", True) is not True:
        raise ValueError("Harrier requires causal attention")

    def integer(name, default=None):
        value = source.kv(f"{arch}.{name}", default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{arch}.{name} must be a positive integer")
        return value

    def positive(name, default=None):
        value = source.kv(f"{arch}.{name}", default)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{arch}.{name} must be finite and positive")
        return value

    gemma = arch != "qwen3"
    layers, hidden = integer("block_count"), integer("embedding_length")
    heads, kv_heads = integer("attention.head_count"), integer("attention.head_count_kv")
    dim = integer("attention.key_length")
    if heads % kv_heads or integer("attention.value_length", dim) != dim:
        raise ValueError("unsupported Harrier query/key/value head geometry")
    vocab, embedding_hidden = source.tensor("token_embd.weight").shape
    if embedding_hidden != hidden:
        raise ValueError("token embedding shape disagrees with GGUF metadata")
    window = source.kv(f"{arch}.attention.sliding_window", 0)
    if not isinstance(window, int) or isinstance(window, bool) or window < 0:
        raise ValueError("invalid embedding sliding window")
    period = integer("attention.sliding_window_pattern", 6) if window else 1
    # Gemma 3 27B scales by hidden/query_heads, independently of its head width.
    # GGUF omits this scalar; llama.cpp applies the same architecture rule.
    scalar = hidden / heads if gemma and layers == 62 else dim
    scale = positive("attention.scale", scalar ** -0.5)
    scaling = source.kv(f"{arch}.rope.scaling.type", "none")
    if scaling not in ("none", "linear"):
        raise ValueError(f"unsupported Harrier RoPE scaling: {scaling}")
    factor = positive("rope.scaling.factor", 1.0) if scaling == "linear" else 1.0
    geometry = dict(hidden=hidden, residual=hidden, intermediate=integer("feed_forward_length"),
                    layers=layers, query_heads=heads, kv_heads=kv_heads, head_dim=dim, rotary_dim=dim,
                    output_rows=vocab, token_domain=len(source.kv("tokenizer.ggml.tokens")),
                    max_context=min(32768, integer("context_length")),
                    rms_epsilon=positive("attention.layer_norm_rms_epsilon"),
                    rope_theta=positive("rope.freq_base"), rope_frequency_scale=1.0 / factor,
                    attention_scale=scale,
                    embedding_scale=float(torch.tensor(hidden ** 0.5, dtype=torch.bfloat16)) if gemma else 1.0,
                    sliding_window=window, sliding_rope_theta=positive("rope.freq_base_swa", 10000.0))
    layer_types = ["full_attention" if not window or (i + 1) % period == 0 else "sliding_attention"
                   for i in range(layers)]
    return ("gemma3_embedding" if gemma else "qwen3_embedding"), geometry, layer_types


def frontend(source, frontend_dir=None):
    arch = source.kv("general.architecture")
    if frontend_dir is not None:
        root = Path(frontend_dir)
        names = ("tokenizer.model", "tokenizer_config.json") if (root / "tokenizer.model").is_file() else (
            "tokenizer.json", "tokenizer_config.json")
        resources = {"frontend/" + name: (root / name).read_bytes() for name in names}
        if "tokenizer.model" in names:
            return resources
    elif source.kv("tokenizer.ggml.model") == "llama":
        from surogate.serve.convert.gemma_embedding.frontend import frontend_from_gguf
        return frontend_from_gguf(source)
    else:
        def field(name):
            value = source.kv(name, False if name == "tokenizer.ggml.add_space_prefix" else None)
            return None if value is None else SimpleNamespace(contents=lambda: value)
        reader = SimpleNamespace(get_field=field)
        # Some Gemma embedding exporters label their raw SentencePiece-surface BPE
        # as gpt2. Its explicit merge list and <0xNN>/metaspace vocabulary still
        # define Gemma's normalizer, not a byte-level Llama tokenizer.
        if arch != "qwen3":
            tokenizer = _spm_tokenizer_json(reader, merges=_declared_merges(reader))
            tokenizer["model"]["vocab"] = {piece.replace(" ", "▁"): index
                                            for piece, index in tokenizer["model"]["vocab"].items()}
            for token in tokenizer["added_tokens"]:
                token["content"] = token["content"].replace(" ", "▁")
        else:
            tokenizer = extract_tokenizer_json(reader)
        config = synthesize_tokenizer_config(reader, arch)
        if arch != "qwen3":
            for token in config.get("added_tokens_decoder", {}).values():
                token["content"] = token["content"].replace(" ", "▁")
        config.pop("chat_template", None)
        config["add_eos_token"] = source.kv("tokenizer.ggml.add_eos_token", False)
        resources = {"frontend/tokenizer.json": json.dumps(tokenizer, ensure_ascii=False).encode(),
                     "frontend/tokenizer_config.json": json.dumps(config, ensure_ascii=False).encode()}
    config = json.loads(resources["frontend/tokenizer_config.json"])
    config["eos_token_id"] = source.kv("tokenizer.ggml.eos_token_id")
    resources["frontend/tokenizer_config.json"] = json.dumps(config, ensure_ascii=False).encode()
    resources["frontend/generation_config.json"] = json.dumps({"eos_token_id": config["eos_token_id"]}).encode()
    from tokenizers import Tokenizer
    checked = Tokenizer.from_str(resources["frontend/tokenizer.json"].decode())
    domain = len(source.kv("tokenizer.ggml.tokens"))
    if any(index >= domain for index in checked.get_vocab().values()):
        raise ValueError("reconstructed tokenizer introduces IDs outside the GGUF vocabulary")
    return resources


def convert(gguf, frontend_dir, out_path):
    source = GgufSource(Path(gguf))
    try:
        architecture, g, layer_types = geometry_from_gguf(source)
        gemma = architecture == "gemma3_embedding"
        resources = frontend(source, frontend_dir)
        h, d, q, kv, f = g["hidden"], g["head_dim"], g["query_heads"], g["kv_heads"], g["intermediate"]
        objects = [("text/token_embedding", "token_embd.weight", (g["output_rows"], h), False),
                   ("text/final_norm", "output_norm.weight", (h,), True)]
        for layer in range(g["layers"]):
            entries = [("input_norm", "attn_norm.weight", (h,), True),
                       ("pre_feedforward_norm", "ffn_norm.weight", (h,), True),
                       ("attention/query_norm", "attn_q_norm.weight", (d,), True),
                       ("attention/key_norm", "attn_k_norm.weight", (d,), True)]
            if gemma:
                entries += [("post_attention_norm", "post_attention_norm.weight", (h,), True),
                            ("post_feedforward_norm", "post_ffw_norm.weight", (h,), True)]
            entries += [(name, source_name, shape, False) for name, source_name, shape in (
                ("attention/query", "attn_q.weight", (q*d, h)), ("attention/key", "attn_k.weight", (kv*d, h)),
                ("attention/value", "attn_v.weight", (kv*d, h)), ("attention/output", "attn_output.weight", (h, q*d)),
                ("mlp/gate", "ffn_gate.weight", (f, h)), ("mlp/up", "ffn_up.weight", (f, h)),
                ("mlp/down", "ffn_down.weight", (h, f)))]
            objects += [(f"text/layers/{layer}/{name}", f"blk.{layer}.{src}", shape, norm)
                        for name, src, shape, norm in entries]
        specs = []
        for name, src, shape, norm in objects:
            tensor = source.tensor(src)
            if tensor.shape != shape:
                raise ValueError(f"{src}: expected {shape}, got {tensor.shape}")
            if not norm and tensor.type_name != "Q8_0":
                raise ValueError(f"{src}: Harrier serving currently requires Q8_0 weights")
            specs.append(TensorSpec(name, shape, "BF16" if norm else "W8G32_F16S",
                                    "contiguous-le-v1" if norm else "row-split-k128-v1"))
        specs += [ResourceSpec(name, "raw-bytes-v1", len(value)) for name, value in resources.items()]
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(out, ArtifactIdentity(architecture, "w8", architecture), specs,
                            geometry=g, layer_types=layer_types) as writer:
            for index, (name, src, shape, norm) in enumerate(objects):
                if norm:
                    values = torch.from_numpy(source.float32(src).copy()) - (1.0 if gemma else 0.0)
                    data = encode_direct(values.to(torch.bfloat16), "BF16")
                else:
                    codes, scales, rows = source.planes_exact(src)
                    data = encode_row_split(torch.from_numpy(codes), torch.from_numpy(scales),
                                            get_format("W8G32_F16S"), rows)
                writer.write(name, data)
                if index % 100 == 0:
                    print(f"[{index + 1}/{len(objects)}] {name}", flush=True)
            for name, data in resources.items():
                writer.write(name, data)
        return out
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--frontend")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    convert(args.gguf, args.frontend, args.out)


if __name__ == "__main__":
    main()
