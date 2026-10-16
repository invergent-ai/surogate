"""Rollout Routing Replay (R3): carry the rollout engine's MoE expert choices to the trainer.

The engine returns, per request, the experts it selected for every token it ran through the model, in the
vLLM / verifiers payload `choice["routed_experts"] = {"data": <base64>, "shape": [n, layers, top_k],
"start": s, "dtype": ...}`: rows cover the request's prompt+completion positions [s, s + n) (positions before
`s` came from the prefix cache; the final sampled token never runs through the model and may be missing).
verifiers keeps the payload on each trajectory step (`step["tokens"]["routed_experts"]`).

This module turns those payloads into one int16 array per training sample, [sample_len, layers, top_k], aligned
with `prompt_ids + completion_ids` (-1 rows = no replay: the trainer keeps its own selection there), and turns a
micro-batch's rows into the trainer's replay layout [layer][token][k] (`SurogateTrainer.set_routing_replay`).
"""

from __future__ import annotations

import binascii
from typing import Any

import numpy as np

NO_ROUTE = -1


def decode_payload(payload: Any) -> tuple[np.ndarray, int] | None:
    """(rows int16 [n, layers, top_k], start) from a routed-experts payload, or None if absent/malformed."""
    if not payload:
        return None
    try:
        data = payload["data"]
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, str):
            data = data.encode("ascii")
        raw = binascii.a2b_base64(data)
        shape = [int(x) for x in payload["shape"]]
        if len(shape) != 3:
            return None
        arr = np.frombuffer(raw, dtype=np.dtype(payload.get("dtype") or "uint8")).reshape(shape)
        start = int(payload.get("start", 0) or 0)
    except (KeyError, TypeError, ValueError, binascii.Error):
        return None
    return arr.astype(np.int16, copy=False), start


class SampleRouting:
    """Routing rows for one training sample, grown step by step as the sample is stitched.

    Positions are the sample's token positions, which equal each merged step's own prompt+completion positions
    (the extension property). Completion tokens keep the rows of the request that generated them; every other
    position takes the rows of the latest request whose prefill covered it; uncovered positions stay -1.
    """

    def __init__(self) -> None:
        self.rows: np.ndarray | None = None  # int16 [len, layers, top_k]
        self.generated = np.zeros(0, dtype=bool)
        self.dims: tuple[int, int] | None = None
        self.valid = True
        self.length = 0

    def _grow(self, length: int) -> None:
        if length <= self.length:
            return
        self.generated = np.concatenate([self.generated, np.zeros(length - self.length, dtype=bool)])
        if self.rows is not None:
            pad = np.full((length - self.length, *self.rows.shape[1:]), NO_ROUTE, dtype=np.int16)
            self.rows = np.concatenate([self.rows, pad], axis=0)
        self.length = length

    def add_step(self, prompt_len: int, completion_len: int, payload: Any) -> None:
        """Account for one trajectory step whose prompt+completion are positions [0, prompt_len + completion_len)."""
        total = prompt_len + completion_len
        self._grow(total)
        decoded = decode_payload(payload) if self.valid else None
        if decoded is not None:
            arr, start = decoded
            dims = (arr.shape[1], arr.shape[2])
            if self.dims is None:
                self.dims = dims
                self.rows = np.full((self.length, *dims), NO_ROUTE, dtype=np.int16)
            if dims != self.dims:
                self.valid = False  # layer count / top_k changed mid-trajectory: replay nothing for this sample
            else:
                lo, hi = max(start, 0), min(start + arr.shape[0], total)
                if hi > lo:
                    take = ~self.generated[lo:hi]  # completion rows already owned by their generating request
                    self.rows[lo:hi][take] = arr[lo - start : hi - start][take]
        self.generated[prompt_len:total] = True

    def result(self) -> tuple[bytes, list[int]] | None:
        """(int16 bytes [len, layers, top_k], [layers, top_k]) or None when there is nothing to replay."""
        if not self.valid or self.rows is None or not (self.rows[:, 0, 0] >= 0).any():
            return None
        return np.ascontiguousarray(self.rows, dtype="<i2").tobytes(), [int(self.dims[0]), int(self.dims[1])]


def rows_from_bytes(data: bytes | None, dims: list[int] | None, length: int) -> np.ndarray | None:
    """int16 [length, layers, top_k] view of packed routing bytes (None when absent or inconsistent)."""
    if data is None or dims is None:
        return None
    layers, top_k = int(dims[0]), int(dims[1])
    arr = np.frombuffer(data, dtype="<i2")
    if arr.size != length * layers * top_k:
        return None
    return arr.reshape(length, layers, top_k)


def no_route_bytes(length: int, dims: list[int]) -> bytes:
    return np.full((length, int(dims[0]), int(dims[1])), NO_ROUTE, dtype="<i2").tobytes()


def moe_layer_ids(hf_config: Any, get_attr) -> list[int] | None:
    """Block indices of the model's MoE layers in order (the engine's routing rows follow this order).

    Qwen-style configs: a layer is sparse unless listed in mlp_only_layers, and only every decoder_sparse_step-th
    layer is sparse; DeepSeek/GLM-style: the first first_k_dense_replace (or num_dense_layers) layers are dense.
    """
    try:
        n = int(get_attr(hf_config, "num_hidden_layers"))
    except Exception:
        return None
    def opt(name, default):
        try:
            v = get_attr(hf_config, name)
            return default if v is None else v
        except Exception:
            return default
    dense = set(int(x) for x in (opt("mlp_only_layers", []) or []))
    step = int(opt("decoder_sparse_step", 1) or 1)
    first_dense = int(opt("first_k_dense_replace", opt("num_dense_layers", 0)) or 0)
    return [i for i in range(n) if i not in dense and (i + 1) % step == 0 and i >= first_dense]


def trainer_rows(routing: np.ndarray, seq_len: int, num_layers: int, moe_layers: list[int] | None) -> np.ndarray | None:
    """[num_layers, seq_len, top_k] int16 rows for set_routing_replay from a micro-batch's [T, R, top_k] rows.

    R (the engine's MoE layer count) maps onto block indices `moe_layers`; R == num_layers maps one to one.
    Returns None when the layer counts cannot be matched (replay is then skipped for the micro-batch).
    """
    T, R, top_k = routing.shape
    if R == num_layers:
        targets = list(range(num_layers))
    elif moe_layers is not None and len(moe_layers) == R:
        targets = moe_layers
    else:
        return None
    out = np.full((num_layers, seq_len, top_k), NO_ROUTE, dtype=np.int16)
    n = min(T, seq_len)
    out[targets, :n, :] = np.transpose(routing[:n], (1, 0, 2))
    return out
