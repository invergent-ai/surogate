"""Ctypes binding of libsinfer_inkling_train.so, the fast-route shim over the
engine's libsinfer.so (csrc/src/serve/inkling_fastpath/inkling_train_shim.cpp).

The trainer's Inkling forward IS the engine's: every visit runs the engine's own ops with the engine's arguments
(the shim's run_visit is a marked copy of targets/inkling2/impl/forward.cpp). Torch owns every activation buffer;
the engine owns the weights (loaded by its own loader). Layout convention: an engine [rows, T] buffer (rows
contiguous per column) is a row-major torch tensor [T, rows].

Rule: the trainer consumes the engine's token ids only and never renders text.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
from dataclasses import dataclass

import torch

ABI_VERSION = 2
NAMES = ("hidden", "residual", "layers", "physical_layers", "output_rows", "token_domain", "qkvr_rows",
         "attn_width", "conv_channels", "experts", "top_k", "intermediate", "fp8_attention")

_VP = ctypes.c_void_p
_I32 = ctypes.c_int32
_SZ = ctypes.c_size_t
_I32P = ctypes.POINTER(ctypes.c_int32)
_I64P = ctypes.POINTER(ctypes.c_int64)


class VisitTaps(ctypes.Structure):
    """Mirror of the shim's VisitTaps (field order and count checked against ift_sizeof(0))."""

    _fields_ = [(n, _VP) for n in (
        "attn_in", "qkvr", "k_dyn", "v_dyn", "gated", "attn_partial", "attn_out", "mlp_in", "moe_out",
        "att_q", "att_r", "att_tau", "att_o", "att_lse",
        "moe_gammas", "moe_gate_up", "moe_y", "moe_down", "moe_logits")]


class LoraBank(ctypes.Structure):
    """Mirror of sinfer::ops::LoraBank (api/ops/lora.h)."""

    _fields_ = [("a", _VP), ("b", _VP), ("a_stride", ctypes.c_int64), ("b_stride", ctypes.c_int64),
                ("rank", _I32), ("n", _I32), ("k", _I32), ("gain", _VP), ("bias", _VP)]


class MoeAdapters(ctypes.Structure):
    """Mirror of sinfer::ops::InklingMoeAdapters (api/ops/inkling_moe.h)."""

    _fields_ = [("table", _VP), ("token_slot", _VP), ("uniform_slot", _VP), ("max_rank", _I32)]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class _DevView:
    """__cuda_array_interface__ over engine-owned device memory (no copy)."""

    def __init__(self, ptr: int, shape: tuple, typestr: str):
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": shape, "typestr": typestr, "version": 3,
                                         "strides": None}


def device_view(ptr: int, shape: tuple, dtype: torch.dtype, device: int) -> torch.Tensor:
    """A torch tensor over engine memory. bf16 has no array-interface typestr: viewed as int16 then reinterpreted."""
    if dtype == torch.bfloat16:
        t = torch.as_tensor(_DevView(ptr, shape, "<i2"), device=f"cuda:{device}")
        return t.view(torch.bfloat16)
    typestr = {torch.float32: "<f4", torch.int32: "<i4", torch.uint8: "|u1", torch.int16: "<i2"}[dtype]
    return torch.as_tensor(_DevView(ptr, shape, typestr), device=f"cuda:{device}")


@dataclass
class Lib:
    sinfer: ctypes.CDLL
    shim: ctypes.CDLL
    bin_dir: str


def load_lib(bin_dir: str) -> Lib:
    """libsinfer.so (RTLD_GLOBAL, the same file the rollout engine runs) + the shim from `bin_dir`
    (an inkbuild product dir: <bin>/libsinfer.so, <bin>/serve_tests/libsinfer_inkling_train.so)."""
    sinfer = ctypes.CDLL(os.path.join(bin_dir, "libsinfer.so"), mode=ctypes.RTLD_GLOBAL)
    shim_path = os.path.join(bin_dir, "serve_tests", "libsinfer_inkling_train.so")
    if not os.path.exists(shim_path):
        shim_path = os.path.join(bin_dir, "libsinfer_inkling_train.so")
    shim = ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)
    s = shim
    s.ift_abi_version.restype = ctypes.c_int
    s.ift_last_error.restype = ctypes.c_char_p
    s.ift_load.restype = _VP
    s.ift_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
    s.ift_free.argtypes = [_VP]
    s.ift_constants.restype = ctypes.c_int
    s.ift_constants.argtypes = [_VP, _I32P, ctypes.c_int]
    s.ift_visit_info.argtypes = [ctypes.c_int, _I32P]
    s.ift_workspace_bytes.restype = _SZ
    s.ift_workspace_bytes.argtypes = [_I32, ctypes.c_int]
    s.ift_head_workspace_bytes.restype = _SZ
    s.ift_head_workspace_bytes.argtypes = [_I32]
    s.ift_state_bytes.restype = _SZ
    s.ift_embed.argtypes = [_VP, _VP, _I32, _VP, _VP, _SZ, _VP]
    s.ift_run_visit.argtypes = [_VP, ctypes.c_int, _VP, _I32, _VP, _VP, _VP, _VP, _VP, _I32, _VP, _VP, _VP, _VP, _VP,
                                _VP, _SZ, _VP]
    s.ift_final_hidden.argtypes = [_VP, _VP, _I32, _VP, _VP, _SZ, _VP]
    s.ift_head_logprob.argtypes = [_VP, _VP, _I32, _VP, _VP, _VP, _SZ, _VP]
    s.ift_tensor.argtypes = [_VP, ctypes.c_char_p, _I64P]
    s.ift_sizeof.restype = _SZ
    s.ift_sizeof.argtypes = [ctypes.c_int]
    if hasattr(s, "ift_decode_experts"):
        s.ift_decode_experts.argtypes = [_VP, ctypes.c_int, ctypes.c_int, _I32, _I32, _VP, _VP, _VP]
    if s.ift_abi_version() != ABI_VERSION:
        raise RuntimeError(f"shim ABI {s.ift_abi_version()} != binding {ABI_VERSION}")
    for which, struct in ((0, VisitTaps), (1, MoeAdapters), (2, LoraBank)):
        if s.ift_sizeof(which) != ctypes.sizeof(struct):
            raise RuntimeError(f"struct {struct.__name__}: shim {s.ift_sizeof(which)} B != binding "
                               f"{ctypes.sizeof(struct)} B")
    return Lib(sinfer, shim, bin_dir)


@dataclass
class Visit:
    v: int
    layer: int
    pass_: int
    stack: tuple
    partial: int
    next_partial: int
    bank: tuple
    feature: int
    first: bool
    global_: bool
    core: bool
    looped: bool


class InklingEngine:
    """One loaded model on one GPU (one per process / device)."""

    def __init__(self, bin_dir: str, artifact: str, device: int = 0):
        self.lib = load_lib(bin_dir)
        self.device = device
        torch.cuda.set_device(device)
        torch.cuda.init()
        h = self.lib.shim.ift_load(artifact.encode(), device)
        if not h:
            raise RuntimeError("ift_load: " + self.lib.shim.ift_last_error().decode())
        self.h = h
        buf = (ctypes.c_int32 * 16)()
        n = self.lib.shim.ift_constants(h, buf, 16)
        self.c = {NAMES[i]: buf[i] for i in range(min(n, len(NAMES)))}
        self.visits = []
        row = (ctypes.c_int32 * 16)()
        for v in range(self.c["layers"]):
            self._check(self.lib.shim.ift_visit_info(v, row), "ift_visit_info")
            r = list(row)
            self.visits.append(Visit(v, r[0], r[1], tuple(r[3:3 + r[2]]), r[7], r[8], (r[9], r[10]), r[11],
                                     bool(r[12]), bool(r[13]), bool(r[14]), bool(r[15])))
        # the persistent per-stream state (the attention / MoE ops' counters), zeroed once, never touched again
        self.state = torch.zeros(int(self.lib.shim.ift_state_bytes()) + 256, dtype=torch.uint8, device=self.dev)
        self._ws = None

    @property
    def dev(self) -> str:
        return f"cuda:{self.device}"

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what}: {self.lib.shim.ift_last_error().decode()}")

    def stream(self) -> int:
        return torch.cuda.current_stream(self.device).cuda_stream

    def workspace(self, T: int, adapters: bool = False) -> torch.Tensor:
        need = int(self.lib.shim.ift_workspace_bytes(T, int(adapters)))
        if need == 0:
            raise RuntimeError("ift_workspace_bytes: " + self.lib.shim.ift_last_error().decode())
        if self._ws is None or self._ws.numel() < need:
            self._ws = None
            self._ws = torch.empty(need, dtype=torch.uint8, device=self.dev)
        return self._ws

    # ------------------------------------------------------------------ weights (views, no copy)
    def tensor(self, name: str) -> dict:
        d = (ctypes.c_int64 * 9)()
        self._check(self.lib.shim.ift_tensor(self.h, name.encode(), d), f"ift_tensor({name})")
        return {"ptr": d[0], "dtype": d[1], "numel": d[2], "ne": tuple(d[3:7]), "extra": d[7], "den_bits": d[8]}

    def view(self, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        return device_view(self.tensor(name)["ptr"], shape, dtype, self.device)

    # ------------------------------------------------------------------ the forward pieces
    def embed(self, ids: torch.Tensor, x: torch.Tensor, ws: torch.Tensor) -> None:
        T = ids.numel()
        self._check(self.lib.shim.ift_embed(self.h, ids.data_ptr(), T, x.data_ptr(), ws.data_ptr(), ws.numel(),
                                            self.stream()), "ift_embed")

    def run_visit(self, v: int, x: torch.Tensor, T: int, seg: "Segment", ws: torch.Tensor, routing=None,
                  taps: VisitTaps | None = None, adapters: MoeAdapters | None = None) -> None:
        self._check(self.lib.shim.ift_run_visit(
            self.h, v, x.data_ptr(), T, seg.positions.data_ptr(), seg.meta.data_ptr(), seg.k_pages.data_ptr(),
            seg.v_pages.data_ptr(), seg.block_table.data_ptr(), seg.pages, seg.conv.data_ptr(), self.state.data_ptr(),
            routing.data_ptr() if routing is not None else None,
            ctypes.byref(taps) if taps is not None else None,
            ctypes.byref(adapters) if adapters is not None else None,
            ws.data_ptr(), ws.numel(), self.stream()), f"ift_run_visit({v})")

    def final_hidden(self, x: torch.Tensor, T: int, hidden: torch.Tensor, ws: torch.Tensor) -> None:
        self._check(self.lib.shim.ift_final_hidden(self.h, x.data_ptr(), T, hidden.data_ptr(), ws.data_ptr(),
                                                   ws.numel(), self.stream()), "ift_final_hidden")

    HEAD_CHUNK = 1024

    def head_logprob(self, hidden: torch.Tensor, ids: torch.Tensor, ws: torch.Tensor | None = None) -> torch.Tensor:
        """hidden [N, 2304] bf16 contiguous, ids [N] int32 contiguous -> fp32 [N] (the engine's returned values), in
        column chunks of HEAD_CHUNK (each column's value depends on that column alone; the fp32 logits of a chunk
        are the call's workspace: ~0.8 MB per column)."""
        N = ids.numel()
        out = torch.empty(N, dtype=torch.float32, device=self.dev)
        c = min(N, self.HEAD_CHUNK)
        need = int(self.lib.shim.ift_head_workspace_bytes(c))
        if ws is None or ws.numel() < need:
            ws = torch.empty(need, dtype=torch.uint8, device=self.dev)
        for a in range(0, N, c):
            n = min(c, N - a)
            h = hidden[a:a + n].contiguous()
            t = ids[a:a + n].contiguous()
            o = out[a:a + n]
            self._check(self.lib.shim.ift_head_logprob(self.h, h.data_ptr(), n, t.data_ptr(), o.data_ptr(),
                                                       ws.data_ptr(), ws.numel(), self.stream()), "ift_head_logprob")
        return out

    def decode_experts(self, layer: int, which: str, first: int, count: int, out: torch.Tensor | None = None):
        """The engine's exact unit decode: (unit bf16 [count, 2304, k], factor fp32 [count]); which 'w13' (k 2304) or
        'w2' (k 1152). dX = sum over a token's slots of (dY . unit[e]) * factor[e]."""
        k = 2304 if which == "w13" else 1152
        unit = out if out is not None else torch.empty(count, 2304, k, dtype=torch.bfloat16, device=self.dev)
        factor = torch.empty(count, dtype=torch.float32, device=self.dev)
        self._check(self.lib.shim.ift_decode_experts(self.h, layer, 0 if which == "w13" else 1, first, count,
                                                     unit.data_ptr(), factor.data_ptr(), self.stream()),
                    f"ift_decode_experts({layer}, {which})")
        return unit, factor

    def close(self) -> None:
        if getattr(self, "h", None):
            self.lib.shim.ift_free(self.h)
            self.h = None


class Segment:
    """One prompt segment's per-forward buffers: positions 0..T-1, the segment ints, one KV plane (identity block
    table, 64-token pages, e4m3) and one conv-state slot -- shared by the visits (each visit's attention reads only
    its own keys of this segment, written by its own prologue; the conv state is zeroed before every visit)."""

    def __init__(self, T: int, device: str):
        self.T = T
        self.pages = math.ceil(T / 64)
        self.positions = torch.arange(T, dtype=torch.int32, device=device)
        self.meta = torch.zeros(8, dtype=torch.int32, device=device)  # [0] KV table row 0, [4] conv slot 0
        self.k_pages = torch.zeros(self.pages * 64 * 8 * 128, dtype=torch.uint8, device=device)
        self.v_pages = torch.zeros(self.pages * 64 * 8 * 128, dtype=torch.uint8, device=device)
        self.block_table = torch.arange(self.pages, dtype=torch.int32, device=device)
        self.conv = torch.zeros(6656 * 3, dtype=torch.bfloat16, device=device)
