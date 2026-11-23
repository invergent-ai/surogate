"""S3: the engine binding for libsinfer_inkling_train_t8.so (ABI 3: ift_run_visit takes the visit's explicit S10 attention adapters).
The engine-op trainer's ift_engine.InklingEngine is reused as it is; only the library and the visit call differ. `attn_layers`: None (no attention adapter) or a
list of 24 AttnAdapters (by physical layer; every pass of a core layer reads its layer's tables, as the engine's store does)."""
from __future__ import annotations

import ctypes
import os

import torch

import ift_engine as E

ABI_VERSION = 3
_VP, _I32 = ctypes.c_void_p, ctypes.c_int32


class AttnAdapters(ctypes.Structure):
    """Mirror of the T8 shim's AttnAdapters (size checked against ift_sizeof(3))."""

    _fields_ = [("qkvr_table", _VP), ("wo_table", _VP), ("uniform_slot", _VP), ("rank", _I32)]


def load_lib_t8(bin_dir: str) -> E.Lib:
    sinfer = ctypes.CDLL(os.path.join(bin_dir, "libsinfer.so"), mode=ctypes.RTLD_GLOBAL)
    path = os.path.join(bin_dir, "serve_tests", "libsinfer_inkling_train_t8.so")
    if not os.path.exists(path):
        path = os.path.join(bin_dir, "libsinfer_inkling_train_t8.so")
    s = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    s.ift_abi_version.restype = ctypes.c_int
    s.ift_last_error.restype = ctypes.c_char_p
    s.ift_load.restype = _VP
    s.ift_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
    s.ift_free.argtypes = [_VP]
    s.ift_constants.restype = ctypes.c_int
    s.ift_constants.argtypes = [_VP, ctypes.POINTER(ctypes.c_int32), ctypes.c_int]
    s.ift_visit_info.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
    s.ift_workspace_bytes.restype = ctypes.c_size_t
    s.ift_workspace_bytes.argtypes = [_I32, ctypes.c_int]
    s.ift_head_workspace_bytes.restype = ctypes.c_size_t
    s.ift_head_workspace_bytes.argtypes = [_I32]
    s.ift_state_bytes.restype = ctypes.c_size_t
    s.ift_embed.argtypes = [_VP, _VP, _I32, _VP, _VP, ctypes.c_size_t, _VP]
    s.ift_run_visit.argtypes = [_VP, ctypes.c_int, _VP, _I32, _VP, _VP, _VP, _VP, _VP, _I32, _VP, _VP, _VP, _VP, _VP, _VP,
                                _VP, ctypes.c_size_t, _VP]
    s.ift_final_hidden.argtypes = [_VP, _VP, _I32, _VP, _VP, ctypes.c_size_t, _VP]
    s.ift_head_logprob.argtypes = [_VP, _VP, _I32, _VP, _VP, _VP, ctypes.c_size_t, _VP]
    s.ift_tensor.argtypes = [_VP, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64)]
    s.ift_sizeof.restype = ctypes.c_size_t
    s.ift_sizeof.argtypes = [ctypes.c_int]
    s.ift_decode_experts.argtypes = [_VP, ctypes.c_int, ctypes.c_int, _I32, _I32, _VP, _VP, _VP]
    if s.ift_abi_version() != ABI_VERSION:
        raise RuntimeError(f"T8 shim ABI {s.ift_abi_version()} != binding {ABI_VERSION}")
    for which, struct in ((0, E.VisitTaps), (1, E.MoeAdapters), (2, E.LoraBank), (3, AttnAdapters)):
        if s.ift_sizeof(which) != ctypes.sizeof(struct):
            raise RuntimeError(f"struct {struct.__name__}: shim {s.ift_sizeof(which)} B != binding {ctypes.sizeof(struct)} B")
    return E.Lib(sinfer, s, bin_dir)


class T8Engine(E.InklingEngine):
    def __init__(self, bin_dir: str, artifact: str, device: int = 0):
        real = E.load_lib
        E.load_lib = load_lib_t8                           # InklingEngine.__init__ calls ift_engine.load_lib by its module name
        try:
            super().__init__(bin_dir, artifact, device)
        finally:
            E.load_lib = real
        self.attn_layers = None

    def run_visit(self, v, x, T, seg, ws, routing=None, taps=None, adapters=None):
        at = None if self.attn_layers is None else self.attn_layers[self.visits[v].layer]
        self._check(self.lib.shim.ift_run_visit(
            self.h, v, x.data_ptr(), T, seg.positions.data_ptr(), seg.meta.data_ptr(), seg.k_pages.data_ptr(),
            seg.v_pages.data_ptr(), seg.block_table.data_ptr(), seg.pages, seg.conv.data_ptr(), self.state.data_ptr(),
            routing.data_ptr() if routing is not None else None,
            ctypes.byref(taps) if taps is not None else None,
            ctypes.byref(adapters) if adapters is not None else None,
            ctypes.byref(at) if at is not None else None,
            ws.data_ptr(), ws.numel(), self.stream()), f"ift_run_visit({v})")

    def workspace(self, T: int, adapters: bool = False) -> torch.Tensor:
        # the attention-LoRA low-rank buffers are booked with the expert adapters' flag (Variant::visit_workspace_capacity_bytes, lora_enabled)
        return super().workspace(T, adapters or self.attn_layers is not None)
