"""Compare the engine's per-layer probes against the transformers reference.

The engine dumps a tensor per tag per layer under SUROGATE_SERVE_DUMP_RESIDUAL; this runs the
same checkpoint through `transformers` on the same token ids and reports cosine similarity at
each point. The reference is the arbiter: it is the implementation the checkpoint was published
against, and every quantity below is read out of it rather than recomputed here.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

MODEL = Path(sys.argv[1])
DUMPS = Path(sys.argv[2])
PROMPT = sys.argv[3] if len(sys.argv) > 3 else (
    "The capital of France is Paris and the capital of Italy is")
#: A probe below this is a defect rather than quantisation. Measured on the fixture, the worst
#: is ~0.9973 at the final layer, accumulated over six layers of int8 weights.
THRESHOLD = 0.995


def read_dump(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    magic, n0, n1, occurrence = struct.unpack("<4i", raw[:16])
    body = np.frombuffer(raw[16:], dtype=np.uint16)[: n0 * n1]
    # BF16 -> float32 by placing the bits in the high half.
    f32 = (body.astype(np.uint32) << 16).view(np.float32)
    return f32.reshape(n1, n0)  # [tokens, rows]: ne[0] is the fastest dimension


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float("nan")


def report_routing_ties(got: np.ndarray, want: np.ndarray, probs) -> None:
    """Why a routed branch's cosine is low, per token, rather than as one number.

    A top-k mixture has a failure mode no other layer has: the 8th and 9th experts can be so
    close that the *stored* router cannot tell them apart, and the token then takes a different
    set from the reference. That is one token being entirely different rather than every token
    being slightly off, and the two have completely different causes -- a selection that flips
    on a hair's-breadth margin is the checkpoint's conditioning, where an evenly spread error is
    the engine's arithmetic.

    One number cannot distinguish them, so this prints both halves: each token's own cosine, and
    the reference's margin between the last expert it chose and the first it did not. A token
    whose cosine collapses while its neighbours sit at the quantisation floor, on a margin below
    the router's storage precision, is the former.
    """
    print("      per-token, because a routed branch fails one token at a time:")
    print(f"      {'tok':>3} {'cosine':>9} {'kth prob':>10} {'next prob':>10} {'margin':>10}"
          f" {'relative':>10}")
    for token in range(got.shape[0]):
        a = got[token].astype(np.float64)
        b = want[token].astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        per_token = float(a @ b / denom) if denom else float("nan")
        if probs is None:
            print(f"      {token:3d} {per_token:9.6f}")
            continue
        ordered = np.sort(probs[token].astype(np.float64))[::-1]
        top_k = int(getattr(report_routing_ties, "top_k", 8))
        kth, nxt = ordered[top_k - 1], ordered[top_k]
        margin = kth - nxt
        print(f"      {token:3d} {per_token:9.6f} {kth:10.3e} {nxt:10.3e} {margin:10.3e}"
              f" {margin / kth:10.3e}")
    print("      A relative margin below ~4e-3 is under BF16's resolution: the artifact's "
          "router cannot rank those two experts.")


def main() -> int:
    raw = json.loads((MODEL / "config.json").read_text())
    # The dense sizes are `gemma4_unified` and the E-series `gemma4`; both are Gemma 4 and
    # both are read the same way, so the checkpoint picks its own class rather than this
    # naming one and being wrong for half the family.
    architecture = (raw.get("architectures") or ["Gemma4UnifiedForConditionalGeneration"])[0]
    import transformers
    model_class = getattr(transformers, architecture)
    cfg = AutoConfig.from_pretrained(MODEL)
    torch.set_grad_enabled(False)
    # fp32 on the CPU is the sharpest reference and is what the fixture uses. The published
    # sizes do not fit that way -- the 12B alone would be 48 GB of parameters -- so the two
    # knobs below run the reference in bf16 on a device instead. The comparison then carries
    # the reference's own bf16 rounding as well as the engine's, which shows up as a slightly
    # lower floor rather than as a disagreement.
    dtype = getattr(torch, os.environ.get("GEMMA4_PARITY_DTYPE", "float32"))
    device = os.environ.get("GEMMA4_PARITY_DEVICE", "cpu")
    model = model_class.from_pretrained(MODEL, dtype=dtype).eval()
    model.to(device)
    text = model.model.language_model

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = torch.tensor([tok(PROMPT, add_special_tokens=True)["input_ids"]]).to(device)

    captured: dict[tuple[str, int], torch.Tensor] = {}

    def attn_hook(layer: int):
        def hook(module, args, kwargs, output):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            # The engine dumps `q_proj_raw` straight out of the projection leaf, *before* the
            # family applies the query norm -- so the reference must not normalise it either.
            # The value is the other way round: its norm happens inside the leaf, because on a
            # k_eq_v layer the operand is the key projection's output before the key norm.
            q_raw = module.q_proj(hidden).view(shape)
            captured[("q_proj_raw", layer)] = q_raw
            captured[("q_post_headnorm", layer)] = module.q_norm(q_raw)
            # A layer that shares an earlier layer's keys and values builds none of these.
            if getattr(module, "is_kv_shared_layer", False):
                return
            k_raw = module.k_proj(hidden).view(shape)
            captured[("k_proj_raw", layer)] = k_raw
            v_raw = module.v_proj(hidden).view(shape) if module.v_proj is not None else k_raw
            captured[("v_proj_raw", layer)] = module.v_norm(v_raw)
            captured[("k_post_headnorm", layer)] = module.k_norm(k_raw)
        return hook

    def out_hook(tag: str, layer: int):
        def hook(module, args, output):
            captured[(tag, layer)] = output
        return hook

    def in_hook(tag: str, layer: int):
        def hook(module, args):
            captured[(tag, layer)] = args[0]
        return hook

    def mixture_probs_hook(layer: int):
        """The router's softmax over *every* expert, which is what a margin is measured on."""
        def hook(module, args, output):
            captured[("router_probs", layer)] = output[0].float().cpu().numpy()
        return hook

    def mixture_router_hook(router, layer: int):
        """What the router projects, which is not what the experts read.

        `Gemma4TextRouter.forward` normalises with no weight, then multiplies by a learned
        per-channel vector and by `hidden ** -0.5`. The engine folds the first two into one
        weighted RMSNorm -- Gemma 4 applies `normed * w` with no unit offset, so a weightless
        norm followed by a per-channel multiply *is* a weighted norm -- and applies the scalar
        separately. Both arrive at the same tensor, which is what this captures.
        """
        def hook(module, args, output):
            captured[("ffn_router_in", layer)] = output * router.scale * router.scalar_root_size
        return hook

    handles = []
    for index, layer in enumerate(text.layers):
        handles.append(layer.self_attn.register_forward_hook(attn_hook(index), with_kwargs=True))
        handles.append(layer.input_layernorm.register_forward_hook(out_hook("post_input_norm", index)))
        # Before o_proj: the attention core's output, which is what the engine dumps.
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(in_hook("attn_core", index)))
        # The layer's own input and output, and the point between its two halves.
        handles.append(layer.register_forward_pre_hook(in_hook("residual_in", index)))
        handles.append(layer.register_forward_hook(out_hook("post_mlp_residual", index)))
        handles.append(layer.pre_feedforward_layernorm.register_forward_pre_hook(
            in_hook("post_attention_residual", index)))
        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(
            out_hook("post_attention_norm", index)))
        # A mixture layer's two feed-forward branches, where the layer has them. The family
        # brackets the whole feed-forward already; these four say which branch a mismatch is
        # in, which on a layer that runs a dense MLP *and* 128 routed experts is the first
        # question worth asking.
        if getattr(layer, "enable_moe_block", False):
            handles.append(layer.mlp.gate_proj.register_forward_hook(out_hook("ffn_gate", index)))
            handles.append(layer.mlp.up_proj.register_forward_hook(out_hook("ffn_up", index)))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(in_hook("ffn_act", index)))
            report_routing_ties.top_k = int(cfg.get_text_config().top_k_experts)
            handles.append(layer.router.register_forward_hook(mixture_probs_hook(index)))
            handles.append(layer.mlp.register_forward_hook(out_hook("ffn_dense_out", index)))
            handles.append(layer.experts.register_forward_hook(out_hook("ffn_routed_out", index)))
            handles.append(layer.router.norm.register_forward_hook(
                mixture_router_hook(layer.router, index)))
            handles.append(layer.post_feedforward_layernorm.register_forward_pre_hook(
                in_hook("ffn_combined", index)))
    text(input_ids=ids)
    for handle in handles:
        handle.remove()

    print(f"{'probe':26s} {'layer':>5s} {'shape':>16s}  cosine")
    worst = 1.0
    failures = 0
    for tag in ("residual_in", "post_input_norm", "q_proj_raw", "k_proj_raw", "v_proj_raw",
                "q_post_headnorm", "k_post_headnorm", "attn_core", "post_attention_residual",
                "post_attention_norm", "ffn_router_in", "ffn_dense_out", "ffn_routed_out",
                "ffn_gate", "ffn_up", "ffn_act", "ffn_combined", "post_mlp_residual"):
        for layer in range(cfg.get_text_config().num_hidden_layers):
            path = DUMPS / f"{tag}_{layer}.bin"
            if not path.exists() or (tag, layer) not in captured:
                continue
            got = read_dump(path)
            reference = captured[(tag, layer)]
            if isinstance(reference, tuple):
                reference = reference[0]
            want = reference.reshape(got.shape[0], -1).float().cpu().numpy()
            score = cosine(got, want)
            worst = min(worst, score)
            flag = "" if score > THRESHOLD else "   <-- MISMATCH"
            failures += 0 if score > THRESHOLD else 1
            print(f"{tag:26s} {layer:5d} {str(got.shape):>16s}  {score:.6f}{flag}")
            if tag == "ffn_routed_out" and score <= THRESHOLD:
                report_routing_ties(got, want, captured.get(("router_probs", layer)))
    print(f"\nworst cosine {worst:.6f}; {failures} probe(s) below {THRESHOLD}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
