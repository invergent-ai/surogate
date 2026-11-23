"""M2 check: every object of an Inkling-2 artifact against the checkpoint it was made from.

For each tensor object the stored words are decoded through the GENERIC artifact decoders
(``surogate.serve.artifact.layouts``) and compared with the checkpoint's words, read straight from
the safetensors, through code that shares nothing with the converter's payload builders:

- BF16 / FP32: every word, bit for bit, segment by segment for fused / stacked objects.
- NVFP4: codes and natural E4M3 block scales word for word (the layout's swizzle undone), the
  trailing word == 1.0, AND the dequantised values ``e2m1 * e4m3 * fp32(amax[e] / den)`` from the
  artifact's words == the same from the checkpoint's, compared as fp32 bit patterns.
- MXFP8: codes and E8M0 scales word for word, AND the dequantised values (artifact side via
  ``ldexp``, checkpoint side via the reference's exponent bit trick ``_dequant_e4m3``) bit-equal;
  also that the bf16 rounding of every dequantised value is exact.
- FP8 attention variants (derived): equal to a fresh run of the quantiser on the BF16 sources.
- Resources: byte-equal to the checkpoint files; ``inkling2/config.json`` parses.
Also: the artifact holds exactly the inventory's objects, the root geometry and layer types.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from surogate.serve.artifact.container import Artifact, TensorObject
from surogate.serve.artifact.layouts import (
    decode_direct,
    decode_fp8_grid_words,
    decode_mxfp8_words,
    decode_nvfp4_words,
    dequantize_mxfp8_words,
)

from .checkpoint import Checkpoint
from .convert import CONFIG_RESOURCE, FRONTEND_RESOURCES, layer_types, text_geometry
from .fp8_attention import quantize_slices
from .inventory import MTP, build_inventory

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
_WORD = {torch.bfloat16: torch.int16, torch.float32: torch.int32}


class Mismatch(AssertionError):
    pass


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_WORD[t.dtype])


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise Mismatch(what)


def nvfp4_values(codes: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """fp32 values of NVFP4 rows: codes [R, K/2] (low nibble = even column), natural E4M3 scale
    words [R, K/16], global [R] fp32 (one per row: the row's expert's amax / den)."""
    low = _E2M1[(codes & 0x0F).long()]
    high = _E2M1[(codes >> 4).long()]
    values = torch.stack((low, high), dim=-1).reshape(codes.shape[0], -1)
    block = scales.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=1)
    return (values * block) * global_scale[:, None]


def mxfp8_values_reference(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """The reference's dequantisation (``ops/moe_sm120.py`` ``_dequant_e4m3``): the scale as the
    float with exponent field ``s`` (0 -> 0x00400000 = 2^-127), times the E4M3 value, in fp32."""
    s = scales.to(torch.int32)
    bits = torch.where(s == 0, torch.full_like(s, 0x00400000), s << 23)
    scale = bits.view(torch.float32).repeat_interleave(32, dim=1)
    return codes.view(torch.float8_e4m3fn).to(torch.float32) * scale


def _expert_view(t: torch.Tensor, order: str) -> torch.Tensor:
    """Checkpoint [E, N, X] -> artifact row order [E*N, X] (``split``: per expert gate rows
    2j first, then up rows 2j+1), by reshape and transpose."""
    e, n, x = t.shape
    if order == "split":
        t = t.reshape(e, n // 2, 2, x).transpose(1, 2)
    return t.reshape(e * n, x).contiguous()


class Verifier:
    def __init__(self, artifact: Artifact, ckpt: Checkpoint, chunk_rows: int = 8 * 2304, log=print):
        self.a = artifact
        self.c = ckpt
        self.chunk_rows = chunk_rows
        self.log = log
        self.stats: dict[str, int] = {}

    def source(self, name: str) -> torch.Tensor:
        return self.c.get_mtp(name[len(MTP):]) if name.startswith(MTP) else self.c.get(name)

    def tensor(self, name: str) -> torch.Tensor:
        obj = self.a.find(name)
        return decode_direct(self.a.payload(obj), obj.format, obj.shape)

    def count(self, key: str, n: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + n

    # -- per kind --------------------------------------------------------------------------

    def direct(self, entry) -> None:
        flat = self.tensor(entry.name).reshape(-1)
        offset = 0
        for name in entry.sources:
            src = self.source(name).reshape(-1)
            seg = flat[offset: offset + src.numel()]
            _require(seg.dtype == src.dtype, f"{entry.name}: dtype {seg.dtype} vs source {name} {src.dtype}")
            _require(torch.equal(_bits(seg), _bits(src)), f"{entry.name}: words differ from {name}")
            offset += src.numel()
            self.count("values_" + str(src.dtype).replace("torch.", ""), src.numel())
        _require(offset == flat.numel(), f"{entry.name}: {flat.numel()} values, sources {offset}")

    def nvfp4(self, entry) -> None:
        obj = self.a.find(entry.name)
        codes, scales, divisor = decode_nvfp4_words(self.a.payload(obj), obj.shape)
        _require(float(divisor) == 1.0, f"{entry.name}: trailing word {float(divisor)} != 1.0")
        order = entry.extra["gate_up_order"] if entry.extra["projection"] == "w13" else "interleaved"
        src_codes = _expert_view(self.source(entry.sources[0]), order)
        src_scales = _expert_view(self.source(entry.sources[1]).view(torch.uint8), order)
        _require(torch.equal(codes, src_codes), f"{entry.name}: code words differ")
        _require(torch.equal(scales, src_scales), f"{entry.name}: block scale words differ")
        base = entry.name.rsplit("/", 1)[0] + "/" + entry.extra["projection"]
        amax_a = self.tensor(base + "_amax")
        den_a = self.tensor(base + "_outer_scale_denominator")
        prefix = entry.sources[0].rsplit("_weight", 1)[0]
        amax_s = self.source(prefix + "_weight_amax")
        den_s = self.source(prefix + "_weight_outer_scale_denominator")
        global_a = amax_a / den_a  # one fp32 division per expert (vLLM: amax / denominator)
        global_s = amax_s.to(torch.float32) / den_s.to(torch.float32)
        rows = entry.extra["per_expert_rows"]
        for begin in range(0, codes.shape[0], self.chunk_rows):
            end = min(begin + self.chunk_rows, codes.shape[0])
            experts = torch.arange(begin, end) // rows
            a = nvfp4_values(codes[begin:end], scales[begin:end], global_a[experts])
            s = nvfp4_values(src_codes[begin:end], src_scales[begin:end], global_s[experts])
            _require(torch.equal(_bits(a), _bits(s)), f"{entry.name}: dequantised rows {begin}..{end} differ")
            _require(bool(torch.isfinite(a).all()), f"{entry.name}: non-finite dequantised value")
        self.count("nvfp4_values", codes.numel() * 2)

    def mxfp8(self, entry) -> None:
        obj = self.a.find(entry.name)
        codes, scales = decode_mxfp8_words(self.a.payload(obj), obj.shape)
        order = entry.extra["gate_up_order"] if entry.extra["projection"] == "w13" else "interleaved"
        src_codes = _expert_view(self.source(entry.sources[0]).view(torch.uint8), order)
        src_scales = _expert_view(self.source(entry.sources[1]), order)
        _require(torch.equal(codes, src_codes), f"{entry.name}: code words differ")
        _require(torch.equal(scales, src_scales), f"{entry.name}: E8M0 scale words differ")
        for begin in range(0, codes.shape[0], self.chunk_rows):
            end = min(begin + self.chunk_rows, codes.shape[0])
            a = dequantize_mxfp8_words(codes[begin:end], scales[begin:end])
            s = mxfp8_values_reference(src_codes[begin:end], src_scales[begin:end])
            _require(torch.equal(_bits(a), _bits(s)), f"{entry.name}: dequantised rows {begin}..{end} differ")
            _require(torch.equal(_bits(a.to(torch.bfloat16).to(torch.float32)), _bits(a)),
                     f"{entry.name}: rows {begin}..{end} not exact in bf16")
        self.count("mxfp8_values", codes.numel())

    def fp8(self, entry) -> None:
        obj = self.a.find(entry.name)
        q = quantize_slices([self.source(s) for s in entry.sources], entry.extra["scheme"])
        if obj.format == "FP8_E4M3FN_ROW_F32S":
            payload = bytes(self.a.payload(obj))
            n, k = obj.shape
            codes = torch.frombuffer(bytearray(payload[: n * k]), dtype=torch.uint8).reshape(n, k)
            offset = (n * k + 255) // 256 * 256
            scales = torch.frombuffer(bytearray(payload[offset: offset + 4 * n]), dtype=torch.float32)
            _require(torch.equal(_bits(scales), _bits(q.row_scales)), f"{entry.name}: row scales differ")
        else:
            codes, grid = decode_fp8_grid_words(self.a.payload(obj), obj.shape, obj.format)
            _require(torch.equal(_bits(grid), _bits(q.grid)), f"{entry.name}: grid scales differ")
        _require(torch.equal(codes, q.codes), f"{entry.name}: FP8 codes differ from the quantiser")
        self.count("fp8_values", codes.numel())

    # -- whole artifact --------------------------------------------------------------------

    def run(self, *, fp8_schemes, gate_up_order, mtp) -> dict:
        started = time.time()
        g = self.c.geometry
        entries = build_inventory(g, fp8_schemes=fp8_schemes, gate_up_order=gate_up_order, mtp=mtp)
        tensors = [o for o in self.a.objects if isinstance(o, TensorObject)]
        resources = [o for o in self.a.objects if not isinstance(o, TensorObject)]
        _require([o.name for o in tensors] == [e.name for e in entries],
                 "artifact tensor objects differ from the inventory (names or order)")
        for obj, entry in zip(tensors, entries):
            _require(obj.format == entry.format and tuple(obj.shape) == entry.shape
                     and obj.layout == entry.layout, f"{entry.name}: stored {obj.format} {obj.shape} "
                     f"{obj.layout}, inventory {entry.format} {entry.shape} {entry.layout}")
        _require(self.a.identity.architecture == "inkling2", "identity.architecture is not inkling2")
        expected_geometry = text_geometry(g)
        _require(self.a.geometry == {k: (float(v) if isinstance(v, float) else v)
                                     for k, v in expected_geometry.items()}, "root geometry differs")
        _require(self.a.layer_types == layer_types(g), "layer_types differ")
        names = {o.name for o in resources}
        for name, file, required in FRONTEND_RESOURCES:
            path = self.c.dir / file
            if not path.exists():
                _require(name not in names, f"{name} carried but {file} absent")
                continue
            _require(bytes(self.a.payload(name)) == path.read_bytes(), f"{name} differs from {file}")
            self.count("resources")
        config = json.loads(bytes(self.a.payload(CONFIG_RESOURCE)))
        _require(config["visits"]["count"] == len(g.visits), "config visits")
        self.count("resources")
        _require(len(names) == self.stats["resources"], f"unexpected resources: {sorted(names)}")
        for i, entry in enumerate(entries):
            getattr(self, entry.kind)(entry)
            self.count("objects_" + entry.kind)
            if entry.kind != "direct" or i % 100 == 0:
                self.log(f"[{time.time() - started:7.1f}s] OK {i + 1}/{len(entries)} {entry.name}")
        self.stats["seconds"] = int(time.time() - started)
        return self.stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_dir")
    parser.add_argument("artifact")
    parser.add_argument("--attention", default="bf16")
    parser.add_argument("--gate-up-order", default="interleaved")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    from .inventory import parse_attention
    ckpt = Checkpoint(args.model_dir)
    with Artifact(args.artifact) as artifact:
        stats = Verifier(artifact, ckpt, log=lambda m: print(m, flush=True)).run(
            fp8_schemes=parse_attention(args.attention), gate_up_order=args.gate_up_order,
            mtp=not args.no_mtp)
    print("PASS", json.dumps(stats, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
