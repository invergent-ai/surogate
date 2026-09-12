"""Extract representative Qwen3.5 projection blocks without dequantizing weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reader = GGUFReader(args.checkpoint)

    def field(name: str) -> int:
        return int(reader.fields["qwen35." + name].contents())

    hidden = field("embedding_length")
    query_heads = field("attention.head_count")
    kv_heads = field("attention.head_count_kv")
    key_heads = field("ssm.group_count")
    value_heads = field("ssm.time_step_rank")
    key_dim = field("ssm.state_size")
    value_rows = field("ssm.inner_size")
    blocks = field("block_count")
    mtp_field = reader.fields.get("qwen35.nextn_predict_layers")
    mtp_layers = int(mtp_field.contents()) if mtp_field is not None else 0
    main_layers = blocks - mtp_layers
    if value_heads % key_heads or value_rows % value_heads:
        raise ValueError("unsupported GDN head geometry")
    value_dim = value_rows // value_heads
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    groups = defaultdict(list)
    for layer in range(blocks):
        prefix = f"blk.{layer}."
        if prefix + "attn_q.weight" in tensors:
            kind = "mtp_attention" if layer >= main_layers else "attention"
            names = [prefix + f"attn_{part}.weight" for part in ("q", "k", "v")]
        else:
            kind = "gdn"
            names = [prefix + f"attn_{part}.weight" for part in ("qkv", "gate")]
        groups[(kind, *(tensors[name].tensor_type.name for name in names))].append(layer)

    args.output.mkdir(parents=True, exist_ok=False)
    manifest = []
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "main_layers": main_layers,
        "mtp_layers": mtp_layers,
        "hidden": hidden,
        "input": "deterministic synthetic BF16 activation; checkpoint weights",
        "cases": [],
    }

    def native_rows(name: str) -> np.ndarray:
        tensor = tensors[name]
        if len(tensor.shape) != 2 or int(tensor.shape[0]) != hidden:
            raise ValueError(f"unexpected matrix shape for {name}: {tensor.shape}")
        if tensor.tensor_type.name not in {"Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_NL"}:
            raise ValueError(f"unsupported projection format: {tensor.tensor_type.name}")
        return np.asarray(tensor.data).view(np.uint8).reshape(int(tensor.shape[1]), -1)

    # Invert GGUF's grouped-to-tiled V-head permutation on entire native rows.
    forward = np.arange(value_rows).reshape(key_heads, value_heads // key_heads, value_dim)
    inverse = np.argsort(forward.transpose(1, 0, 2).reshape(-1))
    for (kind, *formats), layers in groups.items():
        layer = layers[0]
        prefix = f"blk.{layer}."
        case = f"qwen35_{kind}_l{layer}"
        if kind.endswith("attention"):
            q_name, k_name, v_name = (prefix + f"attn_{part}.weight" for part in ("q", "k", "v"))
            qgate, key, value = (native_rows(name) for name in (q_name, k_name, v_name))
            head_dim = key.shape[0] // kv_heads
            if key.shape[0] % kv_heads or qgate.shape[0] != query_heads * 2 * head_dim:
                raise ValueError("unexpected query/gate head geometry")
            packed = qgate.reshape(query_heads, 2, head_dim, -1)
            query = packed[:, 0].reshape(query_heads * head_dim, -1)
            gate = packed[:, 1].reshape(query_heads * head_dim, -1)
            projections = [("q", q_name, query), ("k", k_name, key),
                           ("gate", q_name, gate), ("v", v_name, value)]
            min_tokens = 1
        else:
            qkv_name, z_name = prefix + "attn_qkv.weight", prefix + "attn_gate.weight"
            qkv, z = native_rows(qkv_name), native_rows(z_name)
            qk_rows = 2 * key_heads * key_dim
            if qkv.shape[0] != qk_rows + value_rows or z.shape[0] != value_rows:
                raise ValueError("unexpected GDN projection dimensions")
            qkv = np.concatenate((qkv[:qk_rows], qkv[qk_rows:][inverse]), axis=0)
            z = z[inverse]
            projections = [("qkv", qkv_name, qkv), ("z", z_name, z)]
            # Single-token GDN has a different fused projection/convolution route.
            min_tokens = 2
        manifest.append(f"{case} {hidden} {len(projections)} {min_tokens}")
        info = {"name": case, "kind": kind, "layers_with_same_formats": layers,
                "representative_layer": layer, "min_tokens": min_tokens, "projections": []}
        for label, source, data in projections:
            tensor = tensors[source]
            filename = f"{case}_{label}.blocks"
            payload = np.ascontiguousarray(data).tobytes()
            (args.output / filename).write_bytes(payload)
            manifest.append(f"{tensor.tensor_type.name} {data.shape[0]} {filename}")
            info["projections"].append({
                "label": label, "source_tensor": source, "format": tensor.tensor_type.name,
                "rows": data.shape[0], "hidden": hidden, "bytes": len(payload),
                "file": filename, "sha256": hashlib.sha256(payload).hexdigest(),
            })
        metadata["cases"].append(info)
    (args.output / "manifest.txt").write_text("\n".join(manifest) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
