"""ENGINE-INKLING lane T: the trainer's Inkling forward through the engine's own ops (ift_engine.InklingEngine).

forward(eng, ids) runs embed -> 48 visits -> final mix on ONE prompt segment (positions 0..T-1, B = 1) exactly as
the engine's prefill does, and returns what the LEVEL-0 backward needs (lanes/T/NOTES.md PLAN 3):
  x0       bf16 [T, 2304]   the normed embedding (stack slot 0)
  outs[v]  bf16 [T, 2304]   visit v's output (the stack slot it wrote: kVisits[v].next_partial)
  routing  int32 [48, T, 8] the routed experts of every visit (the engine's own routing; R3 by construction)
  hidden   bf16 [T, 2304]   the final-norm output the LM head reads
Prompt log-probs: logprobs(eng, hidden, ids) = the engine's returned values (lane e1's inkling_head_logprob).
"""

from __future__ import annotations

import torch

from ift_engine import InklingEngine, Segment, VisitTaps

D = 2304


def slot(x: torch.Tensor, s: int) -> torch.Tensor:
    return x[:, s * D:(s + 1) * D]


def make_taps(eng: InklingEngine, v: int, T: int, names=("attn_in", "qkvr", "gated", "attn_out", "mlp_in", "moe_out")):
    """Fresh device buffers for the named VisitTaps fields of visit v; returns (VisitTaps, {name: tensor})."""
    vi = eng.visits[v]
    Q = 3 if (vi.global_ or vi.looped) else 1
    shapes = {"attn_in": (Q, T, D), "qkvr": (T, 8600), "k_dyn": (T, 32), "v_dyn": (T, 32), "gated": (T, 3072),
              "attn_partial": (T, D), "attn_out": (T, D), "mlp_in": (T, D), "moe_out": (T, D),
              "att_q": (T, 24, 128), "att_r": (T, 24, 16), "att_tau": (T,), "att_o": (T, 24, 128), "att_lse": (T, 24),
              "moe_gammas": (T, 8), "moe_gate_up": (T, 8, D), "moe_y": (T, 8, 1152), "moe_down": (T, 8, D),
              "moe_logits": (T, 128)}
    f32 = {"att_tau", "att_o", "att_lse", "moe_gammas", "moe_logits"}
    taps = VisitTaps()
    bufs = {}
    for n in names:
        if n in ("k_dyn", "v_dyn", "att_tau") and not vi.global_:
            continue
        t = torch.empty(shapes[n], dtype=torch.float32 if n in f32 else torch.bfloat16, device=eng.dev)
        bufs[n] = t
        setattr(taps, n, t.data_ptr())
    return taps, bufs


@torch.no_grad()
def forward(eng: InklingEngine, ids: torch.Tensor, save: bool = True, tap_names=None, on_visit=None,
            adapters_per_layer=None):
    """ids int32 [T] on the engine's device. adapters_per_layer: None (base) or 24 MoeAdapters (by physical layer;
    every pass of a core layer reads its layer's table, as the engine). on_visit(v, x, bufs) after every visit."""
    assert ids.dtype == torch.int32 and ids.is_contiguous()
    T = ids.numel()
    x = torch.zeros(T, eng.c["residual"], dtype=torch.bfloat16, device=eng.dev)
    ws = eng.workspace(T, adapters_per_layer is not None)
    seg = Segment(T, eng.dev)
    eng.embed(ids, x, ws)
    x0 = slot(x, 0).clone() if save else None
    routing = torch.empty(eng.c["layers"], T, 8, dtype=torch.int32, device=eng.dev)
    outs = []
    for v in range(eng.c["layers"]):
        seg.conv.zero_()
        taps, bufs = (make_taps(eng, v, T, tap_names) if tap_names else (None, {}))
        ad = adapters_per_layer[eng.visits[v].layer] if adapters_per_layer is not None else None
        eng.run_visit(v, x, T, seg, ws, routing[v], taps, ad)
        if save:
            outs.append(slot(x, eng.visits[v].next_partial).clone())
        if on_visit is not None:
            on_visit(v, x, bufs)
    hidden = torch.empty(T, D, dtype=torch.bfloat16, device=eng.dev)
    eng.final_hidden(x, T, hidden, ws)
    return {"x0": x0, "outs": outs, "routing": routing, "hidden": hidden, "x": x}


@torch.no_grad()
def logprobs(eng: InklingEngine, hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids[t + 1] | prefix up to t) for t = 0..T-2, fp32 [T - 1], bit for bit the engine's prompt log-probs."""
    return eng.head_logprob(hidden[:-1].contiguous(), ids[1:].contiguous())
