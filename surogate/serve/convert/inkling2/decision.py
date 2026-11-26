"""An Inkling-2 decision model (System 1.5, e.g. eltokh7/inkling-2-mini-decision) -> a decision ADAPTER for the
inkling2 target: what the checkpoint adds to the base Inkling-2-Mini it was trained on, so one served base answers
chat requests and decision requests side by side.

A System 1.5 checkpoint is the base text model with (measured on eltokh7/inkling-2-mini-decision @dc334283):
- the routed experts unchanged: the base release's NVFP4 / MXFP8 codes dequantised to BF16 (checked here: every value
  within one BF16 ulp of the exact e2m1 x e4m3 x global product; the MXFP8 layers bit for bit, ~22% of the NVFP4
  values one ulp off -- the exporter's rounding order). The engine serves the base's own codes, i.e. the exact values;
- a LoRA merged into the attention projections wq_du, wk_dv, wv_dv, wo_ud and gate_attn of every layer (rank 128:
  the deltas' singular values drop by an order of magnitude after index 127). The merge's BF16 rounding is ~11% of
  the delta (a full-rank noise floor the SVD cannot keep), and this mixture amplifies rounding-level differences
  through its top-8 routing, so the adapter carries the merged attention weights themselves -- the base artifact's
  fused `attn/qkvr` and `attn/output` objects, BF16, bit for bit the checkpoint's -- and the engine swaps them in for
  decision rounds (~1.3 GB). The rank and the truncation error are still measured and reported;
- full-rank retrains of small tensors, carried as replacements: the routers (mlp.gate.weight, FP32 as the reference
  computes them), the loop controllers (down BF16, gain / forget FP32) and the AttnRes / cross-pass queries (FP32);
- every other tensor identical to the base (checked here: the conversion refuses a checkpoint that differs anywhere
  else);
- new modules: the LoopedSharedExpert per core layer (`scratchpad.pt`, BF16) and the heads (`heads.pt`, FP32) with
  their calibration (exit / gate temperatures, conformal and selective-escalation tables).

Output directory: `decision_adapter.json` (geometry, calibration, provenance) + `decision.safetensors` (objects below).
The pickles are read with torch.load(weights_only=True): tensors, dicts and numbers only.

Objects (names are the base artifact's where a base object is replaced, so the engine binds them the same way):
  text/layers/<L>/attn/qkvr                               BF16  [8600, 2304] (wq_du, wr_du, gate_attn, attn_sink_proj,
                                                                 wk_dv, wv_dv: the base artifact's fused rows)
  text/layers/<L>/attn/output                             BF16  [2304, 3072] (wo_ud)
  text/layers/<L>/moe/router                              FP32  [128, 2304]
  text/layers/<L>/loop/down                               BF16  [4, 64, 4608]   (core layers; streams attn, k, v, mlp)
  text/layers/<L>/loop/gain_forget                        FP32  [4, 4608, 64]  (gain then forget per stream)
  text/attn_res/attn_query/<L>, mlp_query/<L>, output_query FP32
  text/loop/cross_pass_queries, cross_pass_kv_queries     FP32  [24, 2304]
  scratchpad/<j>/{norm,gate_proj,up_proj,down_proj,loop_emb,ada_proj_weight,ada_proj_bias}  BF16 (j = core layer - 6)
  heads/<module>/<param>                                  FP32  (the heads.pt state dicts, '.' -> '/')
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LORA_TARGETS = ("wq_du", "wk_dv", "wv_dv", "wo_ud", "gate_attn")
#: The fused qkvr object's row slices in the base artifact's order (inventory.py, Geometry.qkvr_slices).
QKVR_SLICES = ("wq_du", "wr_du", "gate_attn", "attn_sink_proj", "wk_dv", "wv_dv")
LOOP_STREAMS = ("attn", "k", "v", "mlp")
BASE_PREFIX = "model.llm."
KIND = "inkling2-decision"
#: The System 1.5 prompt markers (modeling.py SPECIAL): state, question, option open, option close, decide.
SPECIAL = ("<|content_xml|>", "<|content_invoke_tool_text|>", "<|content_invoke_tool_json|>",
           "<|content_tool_error|>", "<|content_thinking|>")
HEAD_MODULES = ("joint_head", "core_gate", "halt_head", "traj_mixer", "escalation_head", "aux_head", "evidence_pooler")


class Tensors:
    """Every tensor of a safetensors checkpoint directory, by name (lazy)."""

    def __init__(self, directory: Path, files: list[str] | None = None):
        self.dir = directory
        index = directory / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            weight_map = {}
            for f in files or sorted(p.name for p in directory.glob("*.safetensors")):
                with safe_open(str(directory / f), "pt") as h:
                    weight_map.update({k: f for k in h.keys()})
        self.files = weight_map
        self._handles = {}

    def __contains__(self, name: str) -> bool:
        return name in self.files

    def names(self) -> list[str]:
        return sorted(self.files)

    def get(self, name: str) -> torch.Tensor:
        f = self.files[name]
        if f not in self._handles:
            self._handles[f] = safe_open(str(self.dir / f), "pt")
        return self._handles[f].get_tensor(name)

    def rows(self, name: str):
        """A context manager yielding `slice(e) -> tensor` over the first dim of `name`, read through a handle of its
        own that is closed on exit: the mapped pages a long walk over a 1.4 GB expert bank touches are not kept
        resident (a cached handle keeps them mapped -- ~2.5 GB per layer, on a unified-memory box)."""
        import contextlib

        @contextlib.contextmanager
        def ctx():
            with safe_open(str(self.dir / self.files[name]), "pt") as h:
                sl = h.get_slice(name)
                yield lambda e: sl[e:e + 1][0].clone()
        return ctx()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def _dequant_base_expert(codes: torch.Tensor, scale: torch.Tensor, glob: torch.Tensor | None,
                         device: str) -> torch.Tensor:
    """One base expert (codes / scales of `…experts.w13_weight` or `w2_weight`) as fp32, NVFP4 or MXFP8."""
    codes = codes.to(device)
    scale = scale.to(device)
    if codes.dtype == torch.uint8:  # NVFP4: e2m1 pairs (low nibble = even column), e4m3 per 16, global amax / den
        lut = _E2M1.to(device)
        vals = torch.stack((lut[(codes & 15).long()], lut[(codes >> 4).long()]), -1).reshape(codes.shape[0], -1)
        return vals * scale.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1) * glob.to(device)
    s = scale.to(torch.int32)  # MXFP8: e4m3 values, e8m0 per 32 (0 -> 2^-127)
    bits = torch.where(s == 0, torch.full_like(s, 0x00400000), s << 23)
    return codes.float() * bits.view(torch.float32).repeat_interleave(32, -1)


def _lora_factors(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """delta [out, in] fp32 -> (A [r, in], B [out, r]) BF16 with B @ A ~ delta; the truncation's measured errors."""
    u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    root = s[:rank].sqrt()
    b = (u[:, :rank] * root).contiguous()
    a = (root[:, None] * vh[:rank]).contiguous()
    norm = float(delta.norm())
    rec = b.bfloat16().float() @ a.bfloat16().float()
    stats = {"rel_residual": float((delta - b @ a).norm()) / norm,
             "rel_residual_bf16": float((delta - rec).norm()) / norm,
             "sv_rank": float(s[rank - 1]), "sv_next": float(s[rank]) if s.numel() > rank else 0.0}
    return a.bfloat16(), b.bfloat16(), stats


def convert(decision_dir: str | Path, base_dir: str | Path, out_dir: str | Path, *, rank: int = 128,
            device: str = "cpu", verify_experts: bool = True, log=print) -> dict:
    t0 = time.time()
    ddir, bdir, out = Path(decision_dir), Path(base_dir), Path(out_dir)
    dcfg = json.loads((ddir / "config.json").read_text())
    bcfg = json.loads((bdir / "config.json").read_text())
    btxt = bcfg.get("text_config", bcfg)
    if dcfg.get("model_type") != "inkling2_loop_text":
        raise ValueError(f"{ddir}: model_type {dcfg.get('model_type')!r}, expected inkling2_loop_text")
    # The HF port's config (the decision model's) names some of the release's fields differently.
    loop = btxt.get("loop_config", {})
    pairs = (("hidden_size", btxt.get("hidden_size")), ("num_hidden_layers", btxt.get("num_hidden_layers")),
             ("n_routed_experts", btxt.get("n_routed_experts")), ("num_experts_per_tok", btxt.get("num_experts_per_tok")),
             ("moe_intermediate_size", btxt.get("intermediate_size")), ("local_layer_ids", btxt.get("local_layer_ids")),
             ("vocab_size", btxt.get("vocab_size")), ("conv_kernel_size", btxt.get("sconv_kernel_size")),
             ("loop_layer_index_start", loop.get("core_start")), ("loop_layer_index_end", loop.get("core_end")),
             ("loop_gate_rank", loop.get("gate_rank")))
    for key, want in pairs:
        if dcfg.get(key) != want or want is None:
            raise ValueError(f"the decision model and the base disagree on {key}: {dcfg.get(key)} vs {want}")
    dec, base = Tensors(ddir), Tensors(bdir)
    layers, core_start, core_end = dcfg["num_hidden_layers"], dcfg["loop_layer_index_start"], dcfg["loop_layer_index_end"]

    objects: dict[str, torch.Tensor] = {}
    report: dict = {"lora": {}, "replaced": [], "identical": 0, "experts_checked": 0}
    lora_names = {f"layers.{L}.attn.{p}.weight" for L in range(layers) for p in LORA_TARGETS}
    replaced = set()
    for L in range(layers):
        replaced.add(f"layers.{L}.mlp.gate.weight")
        if core_start <= L < core_end:
            for s in LOOP_STREAMS:
                replaced |= {f"layers.{L}.controllers.{s}.{w}" for w in ("down_weight", "gain_weight", "forget_weight")}
    replaced |= {n for n in dec.names() if n.startswith("attn_res.") or n.startswith("middle_cycle_cross_pass_")}

    for name in dec.names():
        if BASE_PREFIX + name not in base:
            raise ValueError(f"{name}: not in the base checkpoint")
        if ".mlp.experts." in name:
            continue  # checked below
        if name in lora_names or name in replaced:
            continue
        x, y = dec.get(name), base.get(BASE_PREFIX + name)
        if x.shape != y.shape or not torch.equal(x.float(), y.float()):
            raise ValueError(f"{name}: differs from the base, and is neither a LoRA target nor a known replacement")
        report["identical"] += 1

    # attention: the merged weights as replacements (the LoRA's rank and truncation error measured for the record)
    for L in range(layers):
        for p in LORA_TARGETS:
            name = f"layers.{L}.attn.{p}.weight"
            delta = dec.get(name).to(device).float() - base.get(BASE_PREFIX + name).to(device).float()
            report["lora"][f"{L}/{p}"] = _lora_factors(delta, rank)[2] if rank > 0 else {}
        objects[f"text/layers/{L}/attn/qkvr"] = torch.cat(
            [dec.get(f"layers.{L}.attn.{p}.weight").bfloat16() for p in QKVR_SLICES]).contiguous()
        objects[f"text/layers/{L}/attn/output"] = dec.get(f"layers.{L}.attn.wo_ud.weight").bfloat16().contiguous()
        if rank > 0:
            log(f"[{time.time() - t0:6.1f}s] layer {L}: attention carried merged; a rank-{rank} LoRA would leave "
                f"{max(report['lora'][f'{L}/{p}']['rel_residual_bf16'] for p in LORA_TARGETS):.4f} of the delta")

    # replacements, in the base artifact's object names and packing
    for L in range(layers):
        objects[f"text/layers/{L}/moe/router"] = dec.get(f"layers.{L}.mlp.gate.weight").float().contiguous()
        if core_start <= L < core_end:
            objects[f"text/layers/{L}/loop/down"] = torch.stack(
                [dec.get(f"layers.{L}.controllers.{s}.down_weight").bfloat16() for s in LOOP_STREAMS]).contiguous()
            objects[f"text/layers/{L}/loop/gain_forget"] = torch.stack(
                [torch.cat([dec.get(f"layers.{L}.controllers.{s}.gain_weight").float(),
                            dec.get(f"layers.{L}.controllers.{s}.forget_weight").float()]) for s in LOOP_STREAMS]
            ).contiguous()
    for L in range(1, layers):
        objects[f"text/attn_res/attn_query/{L}"] = dec.get(f"attn_res.attn_queries.{L - 1}").float().reshape(
            -1, dcfg["hidden_size"]).contiguous()
    for L in range(layers):
        objects[f"text/attn_res/mlp_query/{L}"] = dec.get(f"attn_res.mlp_queries.{L}").float().reshape(1, -1).contiguous()
    objects["text/attn_res/output_query"] = dec.get("attn_res.output_query").float().reshape(1, -1).contiguous()
    objects["text/loop/cross_pass_queries"] = dec.get("middle_cycle_cross_pass_queries").float().contiguous()
    objects["text/loop/cross_pass_kv_queries"] = dec.get("middle_cycle_cross_pass_kv_queries").float().contiguous()
    report["replaced"] = sorted(replaced)

    # experts: the base release's, bit for bit
    if verify_experts:
        for L in range(layers):
            for proj in ("w13_weight", "w2_weight"):
                name = f"layers.{L}.mlp.experts.{proj}"
                bname = BASE_PREFIX + name
                glob = None
                if bname + "_amax" in base:  # NVFP4: per-expert global amax / den
                    glob = base.get(bname + "_amax").float() / base.get(bname + "_outer_scale_denominator").float()[0]
                off, numel, experts = 0, 0, int(dcfg["n_routed_experts"])
                with dec.rows(name) as drow, base.rows(bname) as crow, base.rows(bname + "_scale") as srow:
                  for e in range(experts):
                    ref = _dequant_base_expert(crow(e), srow(e), None if glob is None else glob[e],
                                               device).bfloat16().view(torch.int16).int()
                    got = drow(e).to(device).view(torch.int16).int()
                    numel += got.numel()
                    ulps = (ref - got).abs()
                    if int(ulps.max()) > 1:
                        raise ValueError(f"{name} expert {e}: more than one BF16 ulp from the base release's expert")
                    off += int((ulps != 0).sum())
                report["experts_checked"] += experts
                report.setdefault("experts_one_ulp_fraction", {})[name] = off / numel
            log(f"[{time.time() - t0:6.1f}s] layer {L}: 256 experts are the base's (within one BF16 ulp)")

    # the added modules
    sp = torch.load(ddir / "scratchpad.pt", map_location="cpu", weights_only=True)
    core = core_end - core_start
    for j in range(core):
        for src, dst in (("norm.weight", "norm"), ("gate_proj.weight", "gate_proj"), ("up_proj.weight", "up_proj"),
                         ("down_proj.weight", "down_proj"), ("loop_emb.weight", "loop_emb"),
                         ("ada_proj.weight", "ada_proj_weight"), ("ada_proj.bias", "ada_proj_bias")):
            objects[f"scratchpad/{j}/{dst}"] = sp[f"{j}.{src}"].bfloat16().contiguous()
    heads = torch.load(ddir / "heads.pt", map_location="cpu", weights_only=True)
    for module in HEAD_MODULES:
        if heads.get(module) is None:
            continue
        for k, v in heads[module].items():
            objects[f"heads/{module}/{k.replace('.', '/')}"] = v.float().contiguous()

    out.mkdir(parents=True, exist_ok=False)
    save_file(objects, str(out / "decision.safetensors"))
    tok = json.loads((ddir / "tokenizer_config.json").read_text())
    meta = {
        "kind": KIND,
        "version": 1,
        "protocol": "system15-v5",
        "base": {"model_type": bcfg.get("model_type"), "hidden": dcfg["hidden_size"], "layers": layers,
                 "core": [core_start, core_end]},
        "max_loops": int(heads["halt_head"]["loop_emb.weight"].shape[0]),
        "attention": {"carried": "merged", "objects": ["attn/qkvr", "attn/output"], "qkvr_slices": list(QKVR_SLICES),
                      "lora_rank_measured": rank, "lora_targets": list(LORA_TARGETS)},
        "scratchpad": {"d_ff": int(sp["0.gate_proj.weight"].shape[0]), "d_ada": int(sp["0.loop_emb.weight"].shape[1]),
                       "loops": int(sp["0.loop_emb.weight"].shape[0])},
        "heads": {"joint_d_head": int(heads["joint_head"]["proj_q.weight"].shape[0]), "joint_heads": 8,
                  "evidence_pooler_max_chunks": int(heads.get("evidence_pooler_max_chunks", 64)),
                  "present": [m for m in HEAD_MODULES if heads.get(m) is not None]},
        "exit_temps": {str(k): float(v) for k, v in heads.get("exit_temps", {}).items()},
        "gate_temps": {str(k): float(v) for k, v in heads.get("gate_temps", {}).items()},
        "conformal": heads.get("conformal_cfg"),
        "selective": heads.get("selective_cfg"),
        "special_tokens": list(SPECIAL),
        "max_state_tokens": 8192,
        "pad_token": tok.get("pad_token"),
        "source": {"dir": str(ddir), "files": {p.name: _sha256(p) for p in sorted(ddir.iterdir())
                                                if p.suffix in (".safetensors", ".pt", ".json")}},
        "conversion": {k: v for k, v in report.items() if k != "replaced"} | {"replaced": len(report["replaced"]),
                                                                            "seconds": time.time() - t0},
    }
    (out / "decision_adapter.json").write_text(json.dumps(meta, indent=1, default=float))
    log(f"wrote {out} ({len(objects)} objects) in {time.time() - t0:.0f}s")
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("decision_dir")
    p.add_argument("base_dir", help="the base Inkling-2-Mini checkpoint the decision model was trained on")
    p.add_argument("out_dir")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-verify-experts", action="store_true")
    a = p.parse_args()
    convert(a.decision_dir, a.base_dir, a.out_dir, rank=a.rank, device=a.device, verify_experts=not a.no_verify_experts)


if __name__ == "__main__":
    main()
