"""Convert the Qwen3.8-Flash-Next GGUF (`qwen4exp`) into a `.ninfer` artifact.

GGUF-native: the four Unsloth shards are the only weight source (there is no
bridged HF checkpoint — the base model is 131 safetensors shards on the Hub).
Tensors are read by GGUF name; every object of `inventory.py` is assembled by
row algebra over the GGUF tensors and encoded in the engine's formats:

- Q8_0 tensors repack **bit-exactly** into W8G32 through the same plane
  decoders the GGUF bridge uses (row gathers are exact on grouped planes);
- Q4_K / Q5_K / Q5_1 experts are dequantised (gguf-py) and quantised to
  W8G32 (0.55 % rel-L2 — design D7);
- hyper-connection, PLE and indexer projections become BF16;
- the PLE table's IQ4_NL rows are copied verbatim as a raw resource.

The GGUF carries llama.cpp's *tiled* order of the GDN value heads; the engine
wants HF *grouped* order, so every V-indexed axis is un-tiled here.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

from surogate.serve.tools.artifact.container import ArtifactIdentity, ArtifactWriter
from surogate.serve.tools.artifact.layouts import encode_direct, encode_row_split
from surogate.serve.tools.artifact.numeric import get_format
from surogate.serve.tools.convert.common.gguf_repack import REPACKABLE_TYPES
from surogate.serve.tools.convert.common.quantize import pick_device, quantize_matrix
from surogate.serve.tools.convert.qwen3_6.common import conversion as family_conversion

from . import inventory as inv

RECIPE_ID = "qwen4exp-w8-hc-v1"
_GROUP = 32
_W8 = get_format(inv.W8)


# --------------------------------------------------------------------------------------------
# GGUF access
# --------------------------------------------------------------------------------------------


@dataclass
class _Tensor:
    reader_tensor: object
    shape: tuple[int, ...]  # logical (numpy) shape, i.e. reversed ggml ne
    type_name: str


class GgufSource:
    """All shards of one split GGUF, tensors by name."""

    def __init__(self, first_shard: Path):
        pattern = str(first_shard).replace("-00001-of-", "-*-of-")
        self.shards = sorted(glob.glob(pattern)) or [str(first_shard)]
        self.readers = [GGUFReader(path) for path in self.shards]
        self.tensors: dict[str, _Tensor] = {}
        for reader in self.readers:
            for t in reader.tensors:
                self.tensors[t.name] = _Tensor(
                    t, tuple(int(x) for x in reversed(t.shape)), t.tensor_type.name
                )
        self.fields = self.readers[0].fields

    def array_field(self, key: str) -> list[int]:
        field = self.fields[key]
        return [int(field.parts[i][0]) for i in field.data]

    def tensor(self, name: str) -> _Tensor:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise KeyError(f"GGUF has no tensor {name!r}") from error

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes as stored (uint8 for quantised types)."""
        return np.asarray(self.tensor(name).reader_tensor.data)

    def float32(self, name: str) -> np.ndarray:
        """Dequantised logical array (float32), any GGML type."""
        t = self.tensor(name)
        data = np.asarray(t.reader_tensor.data)
        if t.type_name in ("F32", "F16", "BF16"):
            if t.type_name == "BF16":
                words = data.view(np.uint16).astype(np.uint32) << 16
                return words.view(np.float32).reshape(t.shape)
            return data.astype(np.float32).reshape(t.shape)
        return np.asarray(dequantize(data, t.reader_tensor.tensor_type), dtype=np.float32).reshape(t.shape)

    def planes_exact(self, name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """(codes int8 [n, groups, 32], scales fp16 [n, groups], (n, k)) for a repackable type.

        3-D expert tensors flatten to rows = all leading dims, k = last dim."""
        t = self.tensor(name)
        if t.type_name not in REPACKABLE_TYPES:
            raise ValueError(f"{name}: {t.type_name} is not exactly repackable")
        block_bytes, decoder = REPACKABLE_TYPES[t.type_name]
        k = t.shape[-1]
        n = int(np.prod(t.shape[:-1]))
        if k % _GROUP != 0:
            raise ValueError(f"{name}: k={k} is not a multiple of {_GROUP}")
        raw = np.asarray(t.reader_tensor.data).reshape(n, k // _GROUP, block_bytes)
        codes, scales = decoder(raw)
        return np.ascontiguousarray(codes), np.ascontiguousarray(scales).reshape(n, k // _GROUP), (n, k)


# --------------------------------------------------------------------------------------------
# Row algebra
# --------------------------------------------------------------------------------------------


def untile_heads(x: np.ndarray, axis: int, k_heads: int, v_per_k: int, head_dim: int) -> np.ndarray:
    """llama.cpp's tiled V order [G0_v0, G1_v0, ..., G0_v1, ...] back to HF grouped order
    [G0_v0, G0_v1, ..., G1_v0, ...] along `axis` (length k_heads * v_per_k * head_dim)."""
    shape = list(x.shape)
    if axis < 0:
        axis += len(shape)
    if shape[axis] != k_heads * v_per_k * head_dim:
        raise ValueError(f"untile_heads: axis {axis} has {shape[axis]} != {k_heads * v_per_k * head_dim}")
    new_shape = shape[:axis] + [v_per_k, k_heads, head_dim] + shape[axis + 1 :]
    y = x.reshape(new_shape)
    y = np.swapaxes(y, axis, axis + 1)  # [k_heads, v_per_k, head_dim]
    return np.ascontiguousarray(y).reshape(shape)


def _untile_v(x: np.ndarray, axis: int) -> np.ndarray:
    """Un-tile an axis of value *channels* (48 heads x 128)."""
    return untile_heads(x, axis, inv.GDN_KEY_HEADS, inv.GDN_VALUE_HEADS // inv.GDN_KEY_HEADS, inv.GDN_HEAD_DIM)


def _untile_v_heads(x: np.ndarray, axis: int) -> np.ndarray:
    """Un-tile an axis of value *heads* (48 scalars: alpha/beta rows, a_log, dt_bias)."""
    return untile_heads(x, axis, inv.GDN_KEY_HEADS, inv.GDN_VALUE_HEADS // inv.GDN_KEY_HEADS, 1)


def w8_from_planes(codes: np.ndarray, scales: np.ndarray, shape: tuple[int, int]) -> bytes:
    return encode_row_split(torch.from_numpy(codes), torch.from_numpy(scales), _W8, shape)


def w8_from_float(weight: np.ndarray, device: torch.device, chunk_rows: int = 65536) -> bytes:
    """Quantise a logical [N, K] float32 matrix to W8G32 in row chunks (GPU), encode once."""
    n, k = weight.shape
    groups = k // _GROUP
    codes = np.empty((n, groups, _GROUP), dtype=np.int8)
    scales = np.empty((n, groups), dtype=np.float16)
    for begin in range(0, n, chunk_rows):
        end = min(n, begin + chunk_rows)
        q = quantize_matrix(torch.from_numpy(np.ascontiguousarray(weight[begin:end])), _W8, device=device)
        codes[begin:end] = q.codes.cpu().numpy()
        scales[begin:end] = q.scales.cpu().numpy()
    return w8_from_planes(codes, scales, (n, k))


def w8_object(source: GgufSource, name: str, rows: Sequence[int] | None, device: torch.device) -> bytes:
    """A W8 object from one GGUF tensor, exact when the type allows, optionally row-gathered."""
    t = source.tensor(name)
    if t.type_name in REPACKABLE_TYPES:
        codes, scales, (n, k) = source.planes_exact(name)
        if rows is not None:
            codes, scales = codes[rows], scales[rows]
        return w8_from_planes(codes, scales, (codes.shape[0], k))
    weight = source.float32(name).reshape(-1, t.shape[-1])
    if rows is not None:
        weight = weight[rows]
    return w8_from_float(weight, device)


def bf16_bytes(x: np.ndarray, shape: tuple[int, ...]) -> bytes:
    tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(shape)
    return encode_direct(tensor.to(torch.bfloat16), inv.BF16)


def f32_bytes(x: np.ndarray, shape: tuple[int, ...]) -> bytes:
    return encode_direct(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(shape), inv.FP32)


def i32_bytes(values: Sequence[int]) -> bytes:
    return encode_direct(torch.tensor(list(values), dtype=torch.int32), inv.I32)


# --------------------------------------------------------------------------------------------
# Per-object materialisation
# --------------------------------------------------------------------------------------------


def attention_rows_q_k_gate_v() -> tuple[list[int], list[int], list[int], list[int]]:
    """Row indices into the fused source [q_proj rows (12288) | k (512) | v (512)] for the
    engine's q | k | gate | v order. q_proj interleaves [q_h(256) | gate_h(256)] per head."""
    q, gate = [], []
    for head in range(inv.QUERY_HEADS):
        base = head * 2 * inv.HEAD_DIM
        q.extend(range(base, base + inv.HEAD_DIM))
        gate.extend(range(base + inv.HEAD_DIM, base + 2 * inv.HEAD_DIM))
    return q, [], gate, []


def materialize(source: GgufSource, name: str, device: torch.device) -> bytes:
    """Bytes for one artifact tensor object."""
    if name == "text/token_embedding":
        return w8_object(source, "token_embd.weight", None, device)
    if name == "text/output_head":
        return w8_object(source, "output.weight", None, device)
    if name.startswith("text/output_hc/"):
        return hyper_connection(source, "output_hc_", name.split("/")[-1], device)
    if name == "text/ple/multipliers":
        words = []
        for m in source.array_field("qwen4exp.ple.layer_multipliers"):
            words.extend((m & 0xFFFFFFFF, (m >> 32) & 0xFFFFFFFF))
        return i32_bytes([w - (1 << 32) if w >= (1 << 31) else w for w in words])
    if name == "text/ple/head_offsets":
        return i32_bytes(source.array_field("qwen4exp.ple.head_offsets"))
    if name == "text/ple/head_vocab_sizes":
        return i32_bytes(source.array_field("qwen4exp.ple.head_vocab_sizes"))

    parts = name.split("/")
    layer = int(parts[2])
    rest = "/".join(parts[3:])
    blk = f"blk.{layer}."

    if rest.startswith("hc_attn/") or rest.startswith("hc_ffn/"):
        kind = "hc_attn_" if rest.startswith("hc_attn/") else "hc_ffn_"
        return hyper_connection(source, blk + kind, rest.split("/")[-1], device)

    if rest.startswith("ple/"):
        leaf = rest.split("/")[-1]
        if leaf == "key":
            return bf16_bytes(source.float32(blk + "ple_key.weight"), (inv.HC_WIDTH, inv.PLE_EMBED))
        if leaf == "value":
            return bf16_bytes(source.float32(blk + "ple_value.weight"), (inv.HIDDEN, inv.PLE_EMBED))
        if leaf in ("norm_key", "norm_query", "norm_conv"):
            return f32_bytes(source.float32(blk + f"ple_{leaf}.weight"), (inv.HC_WIDTH,))
        if leaf == "convolution":
            conv = source.float32(blk + "ple_conv1d.weight")  # logical (10240, 4)
            return bf16_bytes(conv.T, (inv.PLE_CONV_KERNEL, inv.HC_WIDTH))

    if rest.startswith("attention/"):
        leaf = rest[len("attention/") :]
        if leaf == "query_key_gate_value":
            q_rows, _, gate_rows, _ = attention_rows_q_k_gate_v()
            q_t, k_t, v_t = (source.tensor(blk + f"attn_{x}.weight") for x in ("q", "k", "v"))
            if all(t.type_name in REPACKABLE_TYPES for t in (q_t, k_t, v_t)):
                qc, qs, _ = source.planes_exact(blk + "attn_q.weight")
                kc, ks, _ = source.planes_exact(blk + "attn_k.weight")
                vc, vs, _ = source.planes_exact(blk + "attn_v.weight")
                codes = np.concatenate([qc[q_rows], kc, qc[gate_rows], vc], axis=0)
                scales = np.concatenate([qs[q_rows], ks, qs[gate_rows], vs], axis=0)
                return w8_from_planes(codes, scales, (inv.ATTENTION_FUSED_ROWS, inv.HIDDEN))
            q = source.float32(blk + "attn_q.weight")
            fused = np.concatenate(
                [q[q_rows], source.float32(blk + "attn_k.weight"), q[gate_rows], source.float32(blk + "attn_v.weight")]
            )
            return w8_from_float(fused, device)
        if leaf == "output":
            return w8_object(source, blk + "attn_output.weight", None, device)
        if leaf in ("query_norm", "key_norm"):
            # The runtime applies these with a unit offset (HF gamma w, norm = (1+w)*x); the GGUF
            # carries the folded gamma 1+w. Every other norm keeps the folded form.
            return bf16_bytes(source.float32(blk + f"attn_{leaf[0]}_norm.weight") - 1.0, (inv.HEAD_DIM,))
        if leaf.startswith("indexer/"):
            which = leaf.split("/")[-1]
            if which == "query":
                return bf16_bytes(source.float32(blk + "indexer.q_proj.weight"), (inv.INDEXER_HEADS * inv.INDEXER_DIM, inv.HIDDEN))
            if which == "key":
                return bf16_bytes(source.float32(blk + "indexer.k_proj.weight"), (inv.INDEXER_DIM, inv.HIDDEN))
            if which == "query_norm":
                return bf16_bytes(source.float32(blk + "indexer.q_norm.weight"), (inv.INDEXER_DIM,))
            if which == "key_norm":
                return bf16_bytes(source.float32(blk + "indexer.k_norm.weight"), (inv.INDEXER_DIM,))

    if rest.startswith("gdn/"):
        leaf = rest[len("gdn/") :]
        if leaf == "query_key_value_z":
            qkv_t, z_t = source.tensor(blk + "attn_qkv.weight"), source.tensor(blk + "attn_gate.weight")
            key_rows = 2 * inv.GDN_KEY_DIM
            if qkv_t.type_name in REPACKABLE_TYPES and z_t.type_name in REPACKABLE_TYPES:
                qc, qs, _ = source.planes_exact(blk + "attn_qkv.weight")
                zc, zs, _ = source.planes_exact(blk + "attn_gate.weight")
                codes = np.concatenate([qc[:key_rows], _untile_v(qc[key_rows:], 0), _untile_v(zc, 0)])
                scales = np.concatenate([qs[:key_rows], _untile_v(qs[key_rows:], 0), _untile_v(zs, 0)])
                return w8_from_planes(codes, scales, (inv.GDN_FUSED_ROWS, inv.HIDDEN))
            qkv = source.float32(blk + "attn_qkv.weight")
            z = source.float32(blk + "attn_gate.weight")
            fused = np.concatenate([qkv[:key_rows], _untile_v(qkv[key_rows:], 0), _untile_v(z, 0)])
            return w8_from_float(fused, device)
        if leaf == "output":
            t = source.tensor(blk + "ssm_out.weight")  # logical (2560, 6144): k axis is V-tiled
            if t.type_name in REPACKABLE_TYPES:
                codes, scales, (n, k) = source.planes_exact(blk + "ssm_out.weight")
                # whole heads move (128 = 4 groups), so the permutation is exact on planes
                codes = _untile_v(codes.reshape(n, k, 1), 1).reshape(n, k // _GROUP, _GROUP)
                scales = _untile_v(np.repeat(scales, _GROUP, axis=1).reshape(n, k, 1), 1).reshape(n, k // _GROUP, _GROUP)[:, :, 0]
                return w8_from_planes(codes, scales, (n, k))
            return w8_from_float(_untile_v(source.float32(blk + "ssm_out.weight"), 1), device)
        if leaf == "a_b_projection":
            alpha = _untile_v_heads(source.float32(blk + "ssm_alpha.weight"), 0)  # (48, 2560)
            beta = _untile_v_heads(source.float32(blk + "ssm_beta.weight"), 0)
            return bf16_bytes(np.concatenate([alpha, beta]), (2 * inv.GDN_VALUE_HEADS, inv.HIDDEN))
        if leaf == "a_log":
            a_log = np.log(-_untile_v_heads(source.float32(blk + "ssm_a").reshape(inv.GDN_VALUE_HEADS), 0))
            return f32_bytes(a_log, (inv.GDN_VALUE_HEADS,))
        if leaf == "dt_bias":
            return f32_bytes(_untile_v_heads(source.float32(blk + "ssm_dt.bias").reshape(inv.GDN_VALUE_HEADS), 0), (inv.GDN_VALUE_HEADS,))
        if leaf == "convolution":
            # The conv kernel reads weight[tap * C + c] (tap-major, channel fastest), so the
            # GGUF's (10240, 4) is transposed; the value channels are re-ordered like the rows.
            conv = source.float32(blk + "ssm_conv1d.weight").T  # (4, 10240)
            key_channels = 2 * inv.GDN_KEY_DIM
            conv = np.concatenate([conv[:, :key_channels], _untile_v(conv[:, key_channels:], 1)], axis=1)
            return bf16_bytes(conv, (inv.GDN_CONV_KERNEL, inv.GDN_CONV_DIM))
        if leaf == "norm":
            return bf16_bytes(source.float32(blk + "ssm_norm.weight"), (inv.GDN_HEAD_DIM,))

    if rest.startswith("mlp/"):
        leaf = rest[len("mlp/") :]
        if leaf == "router_shared_gate":
            router = source.float32(blk + "ffn_gate_inp.weight")  # (512, 2560)
            shared = source.float32(blk + "ffn_gate_inp_shexp.weight").reshape(1, inv.HIDDEN)
            return bf16_bytes(np.concatenate([router, shared]), (inv.ROUTER_ROWS, inv.HIDDEN))
        if leaf == "routed_gate_up":
            gate_t, up_t = source.tensor(blk + "ffn_gate_exps.weight"), source.tensor(blk + "ffn_up_exps.weight")
            n_rows = inv.EXPERTS * 2 * inv.EXPERT_FFN
            if gate_t.type_name in REPACKABLE_TYPES and up_t.type_name in REPACKABLE_TYPES:
                gc, gs, (_, k) = source.planes_exact(blk + "ffn_gate_exps.weight")
                uc, us, _ = source.planes_exact(blk + "ffn_up_exps.weight")
                g = k // _GROUP
                codes = np.concatenate([gc.reshape(inv.EXPERTS, inv.EXPERT_FFN, g, _GROUP), uc.reshape(inv.EXPERTS, inv.EXPERT_FFN, g, _GROUP)], axis=1).reshape(n_rows, g, _GROUP)
                scales = np.concatenate([gs.reshape(inv.EXPERTS, inv.EXPERT_FFN, g), us.reshape(inv.EXPERTS, inv.EXPERT_FFN, g)], axis=1).reshape(n_rows, g)
                return w8_from_planes(codes, scales, (n_rows, k))
            gate = source.float32(blk + "ffn_gate_exps.weight")  # (512, 640, 2560)
            up = source.float32(blk + "ffn_up_exps.weight")
            fused = np.concatenate([gate, up], axis=1).reshape(n_rows, inv.HIDDEN)
            del gate, up
            return w8_from_float(fused, device)
        if leaf == "routed_down":
            return w8_object(source, blk + "ffn_down_exps.weight", None, device)  # (512*2560, 640)
        if leaf == "shared_gate_up":
            g_t, u_t = source.tensor(blk + "ffn_gate_shexp.weight"), source.tensor(blk + "ffn_up_shexp.weight")
            if g_t.type_name in REPACKABLE_TYPES and u_t.type_name in REPACKABLE_TYPES:
                gc, gs, (_, k) = source.planes_exact(blk + "ffn_gate_shexp.weight")
                uc, us, _ = source.planes_exact(blk + "ffn_up_shexp.weight")
                return w8_from_planes(np.concatenate([gc, uc]), np.concatenate([gs, us]), (2 * inv.SHARED_FFN, k))
            fused = np.concatenate([source.float32(blk + "ffn_gate_shexp.weight"), source.float32(blk + "ffn_up_shexp.weight")])
            return w8_from_float(fused, device)
        if leaf == "shared_down":
            return w8_object(source, blk + "ffn_down_shexp.weight", None, device)

    raise KeyError(f"no recipe covers artifact object {name!r}")


def hyper_connection(source: GgufSource, prefix: str, leaf: str, device: torch.device) -> bytes:
    if leaf == "norm":
        return f32_bytes(source.float32(prefix + "norm.weight"), (inv.HC_WIDTH,))
    if leaf == "down":  # GGUF ne (10240, 320) -> logical (320, 10240): 10240 -> 320
        return bf16_bytes(source.float32(prefix + "down.weight"), (inv.HC_LOW_RANK, inv.HC_WIDTH))
    if leaf == "up":  # logical (10240, 320): 320 -> 10240
        return bf16_bytes(source.float32(prefix + "up.weight"), (inv.HC_WIDTH, inv.HC_LOW_RANK))
    if leaf == "inject":  # logical (4, 10240)
        return bf16_bytes(source.float32(prefix + "inject.weight"), (inv.HC_COUNT, inv.HC_WIDTH))
    raise KeyError(f"unknown hyper-connection leaf {leaf!r}")


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


class _Sized:
    """A stand-in with the right length for the object plan; the payload streams separately."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes

    def __len__(self) -> int:
        return self.nbytes


def _stream(array: np.ndarray, chunk_bytes: int = 256 << 20) -> Iterator[bytes]:
    flat = array.reshape(-1).view(np.uint8)
    for begin in range(0, flat.shape[0], chunk_bytes):
        yield memoryview(np.ascontiguousarray(flat[begin : begin + chunk_bytes]))


def convert(gguf: str | Path, frontend_dir: str | Path, out_path: str | Path, *, device: str = "cuda") -> Path:
    started = time.perf_counter()
    inv.validate_inventory()
    source = GgufSource(Path(gguf))
    resolved = pick_device(device)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    table = source.tensor("per_layer_token_embd.weight")
    table_raw = np.asarray(table.reader_tensor.data)
    if table.shape != (inv.PLE_TABLE_ROWS, inv.PLE_HEAD_DIM) or table.type_name != "IQ4_NL":
        raise ValueError(f"unexpected PLE table {table.shape} {table.type_name}")

    frontend = family_conversion.load_resources(frontend_dir, inv.RESOURCE_SPECS)
    resources: dict[str, object] = {item.name: item.data for item in frontend}
    resources[inv.PLE_TABLE_RESOURCE] = _Sized(int(table_raw.nbytes))
    # The tower travels only when the source has one. The community GGUF exports of
    # this model carry 1,224 tensors and none of them vision, so converting from one
    # produces a text-only artifact; a safetensors checkpoint produces the full model.
    has_vision = any(name.startswith(("v.", "mm.")) or "vision" in name
                     for name in source.tensors)
    _, object_specs = inv.active_specs(vision=has_vision)
    print(f"vision tower in source: {'yes' if has_vision else 'no'}", flush=True)

    plan = family_conversion.build_object_plan(object_specs, resources)  # type: ignore[arg-type]

    specs = list(object_specs)
    total = len(specs)
    print(f"converting {total} objects from {len(source.shards)} shards on {resolved}", flush=True)
    with ArtifactWriter(output, ArtifactIdentity(inv.MODEL_ID, inv.WEIGHTS_ID), plan.specs) as writer:
        for index, spec in enumerate(specs, start=1):
            t0 = time.perf_counter()
            if isinstance(spec, inv.ResourceSpec):
                payload: bytes | Iterable[bytes]
                if spec.name == inv.PLE_TABLE_RESOURCE:
                    payload = _stream(table_raw)
                else:
                    payload = resources[spec.name]  # type: ignore[assignment]
            else:
                payload = materialize(source, spec.name, resolved)
            writer.write(spec.name, payload)
            del payload
            if index % 25 == 0 or index == total or spec.name.endswith("routed_gate_up"):
                print(f"[{index}/{total}] {spec.name} ({time.perf_counter() - t0:.1f}s)", flush=True)
    report = {
        "recipe_id": RECIPE_ID,
        "model_id": inv.MODEL_ID,
        "weights_id": inv.WEIGHTS_ID,
        "gguf": [str(s) for s in source.shards],
        "frontend": str(frontend_dir),
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": output.stat().st_size,
    }
    Path(str(output) + ".conversion.json").write_text(json.dumps(report, indent=2))
    print(f"conversion finished in {report['elapsed_seconds']:.0f} s -> {output} ({report['bytes'] / 1e9:.1f} GB)", flush=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, help="first shard of the split GGUF")
    parser.add_argument("--frontend", required=True, help="directory with tokenizer/chat template/configs")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.gguf, args.frontend, args.out, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
