"""Lane T8: the lean per-row backward for W5 (lanes/T8/PLAN.md U1, STEP-API.md).

The math is lane T's (ift_long.visit_torch_long, one recompute per visit from the engine's visit outputs) and the reverse loop is
ift_long.LongModel / RX's rx_coda2.CodaModel. What is new here is only where things live:
- only the visit outputs the last K visits read are copied to the host (K = 6 / 18 / 30 / 48 -> 10 / 22 / 34 / 49 of 49 buffers);
- those buffers and the gradient accumulators come from ONE exact-size pinned arena per worker (torch's pinned allocator keeps freed blocks);
- nothing is accumulated on the card: gradients leave per visit through `on_grad`;
- the attention query block follows RX's rule (a smaller block at long rows), with a retry on out-of-memory in the caller.
"""
from __future__ import annotations

import ctypes

import torch

import ift_backward as B
import ift_forward as F
import ift_long as LG
import ift_ref as RA
from ift_engine import Segment

D, EPS = B.D, B.EPS
BF16, F32 = torch.bfloat16, torch.float32
FINAL_SLOTS = (0, 1, 2, 3, 6)
# RX's rule (census_grad.py QB_STEPS, measured on a 32 GB card that also holds the engine): (longest row, query block); floor 8
QB_STEPS = ((32768, 128), (50000, 64), (100000, 32), (200000, 16))
SEGMENTS = (("A", 42), ("B", 30), ("C", 18), ("D1", 6), ("D0", 0))


def qb_for(T: int) -> int:
    for t_max, qb in QB_STEPS:
        if T <= t_max:
            return qb
    return 8


def segment_of(v: int) -> str:
    for name, first in SEGMENTS:
        if v >= first:
            return name
    raise ValueError(v)


def _slots(vi) -> list:
    return list(vi.stack) + [vi.partial] + (list(vi.bank) if vi.looped else [])


def producers(visits):
    """prod[v] = the visit that wrote each input slot of visit v (-1 = x0, as LongModel); fin = the same for the final mix."""
    last = {0: -1}
    prod = []
    for v, vi in enumerate(visits):
        prod.append([last[s] for s in _slots(vi)])
        last[vi.next_partial] = v
    return prod, [last[s] for s in FINAL_SLOTS]


def keep_outputs(visits, v_first: int) -> list:
    """The visit outputs a backward over visits >= v_first reads (x0 is always kept apart)."""
    prod, fin = producers(visits)
    keep = set(fin)
    for v in range(v_first, len(visits)):
        keep |= set(prod[v])
    keep.discard(-1)
    return sorted(keep)


def acc_peak(visits, v_first: int) -> int:
    """Largest number of host gradient accumulators alive at once in backward_row's order."""
    prod, fin = producers(visits)
    n = len(visits)
    live = {p for p in fin if p >= v_first and p != n - 1}
    peak = len(live)
    for v in range(n - 1, v_first - 1, -1):
        live.discard(v)
        live |= {p for p in prod[v] if p >= v_first and p != v - 1}
        peak = max(peak, len(live))
    return peak


def trained_layers(visits, v_first: int) -> list:
    return sorted({visits[v].layer for v in range(v_first, len(visits))})


def _cudart():
    path = next(ln.split()[-1] for ln in open("/proc/self/maps") if "libcudart" in ln)
    return ctypes.CDLL(path)


class HostArena:
    """One pinned host allocation of exactly `size` bytes (cudaHostAlloc; plain memory without CUDA, for the CPU tests)."""

    def __init__(self, size: int):
        self.size, self.ptr, self.rt = size, None, None
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            self.rt = _cudart()
            p = ctypes.c_void_p()
            rc = self.rt.cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(size), ctypes.c_uint(0))
            if rc != 0:
                raise MemoryError(f"cudaHostAlloc({size}) failed with CUDA error {rc}")
            self.ptr = p.value
            self.t = torch.frombuffer((ctypes.c_char * size).from_address(self.ptr), dtype=torch.uint8)
        else:
            self.t = torch.empty(size, dtype=torch.uint8)

    def view(self, offset: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        width = {BF16: 2, F32: 4}[dtype]
        n = width
        for s in shape:
            n *= s
        if offset + n > self.size:
            raise ValueError("arena view out of range")
        return self.t[offset:offset + n].view(dtype).view(shape)

    def close(self) -> None:
        if self.ptr is not None:
            self.t = None
            rc = self.rt.cudaFreeHost(ctypes.c_void_p(self.ptr))
            self.ptr = None
            if rc != 0:
                raise RuntimeError(f"cudaFreeHost failed with CUDA error {rc}")


class RowBuffers:
    """The worker's host buffers for rows up to T_max: x0 + the kept visit outputs (bf16 [T, D]) + a free list of fp32 [T, D] accumulators."""

    def __init__(self, visits, v_first: int, T_max: int):
        self.keep = keep_outputs(visits, v_first)
        self.n_acc = acc_peak(visits, v_first)
        self.T_max = T_max
        self.out_bytes = T_max * D * 2
        self.acc_bytes = T_max * D * 4
        self.bytes = (1 + len(self.keep)) * self.out_bytes + self.n_acc * self.acc_bytes
        self.arena = HostArena(max(self.bytes, 4096))
        self.free = list(range(self.n_acc))

    def outputs(self, T: int) -> dict:
        if T > self.T_max:
            raise ValueError(f"row of {T} tokens > this worker's buffers ({self.T_max})")
        out = {-1: self.arena.view(0, (T, D), BF16)}
        for i, v in enumerate(self.keep):
            out[v] = self.arena.view((i + 1) * self.out_bytes, (T, D), BF16)
        return out

    def acc(self, T: int):
        k = self.free.pop()
        off = (1 + len(self.keep)) * self.out_bytes + k * self.acc_bytes
        t = self.arena.view(off, (T, D), F32)
        t.zero_()
        return k, t

    def release(self, k: int) -> None:
        self.free.append(k)

    def reset(self) -> None:
        self.free = list(range(self.n_acc))

    def close(self) -> None:
        self.arena.close()


@torch.no_grad()
def forward_keep(eng, ids: torch.Tensor, adapters_per_layer, buffers: RowBuffers) -> dict:
    """ift_long.forward_host with only the kept outputs copied to the host. The 48 visits, the final mix and the log-probs are the engine's."""
    T = ids.numel()
    x = torch.zeros(T, eng.c["residual"], dtype=BF16, device=eng.dev)
    ws = eng.workspace(T, adapters_per_layer is not None)
    seg = Segment(T, eng.dev)
    eng.embed(ids, x, ws)
    host = buffers.outputs(T)
    host[-1].copy_(F.slot(x, 0), non_blocking=True)
    routing = torch.empty(eng.c["layers"], T, 8, dtype=torch.int32, device=eng.dev)
    for v in range(eng.c["layers"]):
        seg.conv.zero_()
        ad = adapters_per_layer[eng.visits[v].layer] if adapters_per_layer is not None else None
        eng.run_visit(v, x, T, seg, ws, routing[v], None, ad)
        if v in host:
            host[v].copy_(F.slot(x, eng.visits[v].next_partial), non_blocking=True)
    hidden = torch.empty(T, D, dtype=BF16, device=eng.dev)
    eng.final_hidden(x, T, hidden, ws)
    lp = F.logprobs(eng, hidden, ids)
    torch.cuda.current_stream(eng.device).synchronize()
    del x, seg, ws
    eng._ws = None
    return {"host": host, "routing": routing, "hidden": hidden, "lp": lp}


def backward_row(eng, W, ids, vals: dict, lora_host: list, scale: float, v_first: int, g_lp: torch.Tensor, buffers: RowBuffers,
                 on_grad, head_grad=None, q_block: int | None = None, train_idx=(0, 1, 2, 3)) -> None:
    """The reverse loop of ift_long.LongModel over visits >= v_first. g_lp [T - 1] = the seed on the per-target log-probs.
    lora_host: per physical layer {A13, B13, A2, B2} (fp32 values on the host). on_grad(v, layer, key index, gradient on the card)."""
    dev = g_lp.device
    T = ids.numel()
    visits = eng.visits
    n_v = len(visits)
    prod, fin_prod = producers(visits)
    host_vals = vals["host"]
    acc_dev, acc_host = {}, {}
    buffers.reset()
    qb = q_block if q_block is not None else qb_for(T)
    saved_q_block = LG._q_block
    LG._q_block = lambda T_, window, hq=24: qb  # one block for local and global visits, as RX's worker

    def sink_for(p, next_v):
        if p < v_first:  # produced before the trained visits: nothing there receives a gradient
            return None
        if p == next_v:
            if p not in acc_dev:
                acc_dev[p] = torch.zeros(T, D, dtype=F32, device=dev)
            return acc_dev[p]
        if p not in acc_host:
            acc_host[p] = buffers.acc(T)
        return acc_host[p][1]

    def f_mix(a, b, c, d, e):
        return RA.inkling_attn_res([a.float(), b.float(), c.float(), d.float()], e.float(), None, W.q_out,
                                   W.final_norm.reshape(1, D), EPS)[0]

    try:
        with LG._gemm_mode():
            with B._tick("bwd.head"):
                hid = vals["hidden"][:-1].float()
                g_hid = torch.zeros(T, D, dtype=F32, device=dev)
                g_hid[:-1] = (head_grad or LG._head_grad)(W, hid, ids[1:].contiguous(), g_lp)
                del hid
            with torch.enable_grad(), torch.device(dev):
                ph = [torch.zeros(1, dtype=F32, device=dev).expand(T, D) for _ in fin_prod]
                LG.SINK_DUMMIES.clear()
                hm = LG.tc(f_mix, ph, [0] * 5, [0], store_bf16=[True] * 5, vals=[host_vals[p] for p in fin_prod],
                           sinks=[sink_for(p, n_v - 1) for p in fin_prod])
                torch.autograd.grad(hm, list(LG.SINK_DUMMIES), g_hid, allow_unused=True)
                LG.SINK_DUMMIES.clear()
            del g_hid, hm, ph
            for v in range(n_v - 1, v_first - 1, -1):
                vi = visits[v]
                g_out = acc_dev.pop(v, None)
                if v in acc_host:
                    k, t = acc_host.pop(v)
                    h = t.to(dev)  # blocking: every copy queued into this slot has landed before the slot is reused
                    buffers.release(k)
                    g_out = h if g_out is None else g_out.add_(h)
                    del h
                if g_out is None:
                    continue
                ps = prod[v]
                lo = lora_host[vi.layer]
                lt = [lo[k].to(dev) for k in B.LORA_KEYS]
                n = len(vi.stack)
                sinks = [sink_for(p, v - 1) for p in ps]
                with torch.enable_grad(), torch.device(dev):
                    ins = [torch.zeros(1, dtype=F32, device=dev).expand(T, D) for _ in ps]
                    leaves = [t.detach().requires_grad_(j in train_idx) for j, t in enumerate(lt)]  # keys not trained: no vjp
                    want = [t for t in leaves if t.requires_grad]
                    lora = dict(zip(B.LORA_KEYS, leaves)) | {"scale": scale}
                    LG.SINK_DUMMIES.clear()
                    out = LG.visit_torch_long(W, vi, ins[:n], ins[n], (ins[n + 1], ins[n + 2]) if vi.looped else None,
                                              vals["routing"][v], lora, act_vals=[host_vals[p] for p in ps], act_sinks=sinks)
                    with B._tick("bwd.autograd"):
                        got = torch.autograd.grad(out, want + list(LG.SINK_DUMMIES), g_out, allow_unused=True)
                    it = iter(got[:len(want)])
                    gr = [next(it) if t.requires_grad else None for t in leaves]
                    del got, want
                    LG.SINK_DUMMIES.clear()
                if not vi.first and sinks[n] is not None:  # the MLP-output conv's residual: d partial += d out (STE)
                    LG.sink_add(sinks[n], g_out)
                del out, g_out
                for j, gj in enumerate(gr):
                    if gj is not None:
                        on_grad(v, vi.layer, j, gj.detach())
                del gr, lt, leaves, lora
    finally:
        LG._q_block = saved_q_block
        LG.SINK_DUMMIES.clear()
    torch.cuda.current_stream(dev).synchronize()  # the last host-accumulator copies, before the arena's slots are reused
