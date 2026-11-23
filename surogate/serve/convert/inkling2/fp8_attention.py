"""FP8 (E4M3FN) quantisation of Inkling-2-Mini's attention projections: the candidate schemes of
the FP8-attention gate (design notes section 10.1, artifact contract 4.4).

One rule for every scheme, applied to the BF16 weight ``W`` of ONE checkpoint tensor (the fused
``qkvr`` is quantised per source slice, so no scale ever mixes wq / wr / gate / sink / wk / wv):

    for each scale cell B:  amax = max |W_B|                     (exact)
                            scale = amax / 448                    (one fp32 division; 1.0 if amax == 0)
                            code  = e4m3_rne(clamp(W / scale, -448, 448))   (fp32 division, RNE)
    value = e4m3(code) * scale                                    (fp32)

Cells: ``tensor`` the whole tensor, ``channel`` one output row, ``g1x128`` one row x 128 columns,
``b128`` 128 x 128 (the last row block may be partial: ``attn_sink_proj`` has 24 rows).

Every scheme is returned in two equivalent forms: ``row_scales`` [n] (tensor / channel only) and
``grid`` [n, k/128] (every scheme; a cell larger than 1x128 repeats its scale), so the storage
format is a separate choice from the scheme.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

E4M3_MAX = 448.0
SCHEMES = ("tensor", "channel", "g1x128", "b128")
#: How each scheme is stored in the artifact (ARTIFACT.md 4.4).
STORAGE_FORMAT = {
    "tensor": "FP8_E4M3FN_ROW_F32S",
    "channel": "FP8_E4M3FN_ROW_F32S",
    "g1x128": "FP8_E4M3FN_1X128_F32S",
    "b128": "FP8_E4M3FN_1X128_F32S",
}
RULE = ("amax = max|W_cell|; scale = amax / 448 (fp32, 1.0 if amax == 0); "
        "code = e4m3_rne(clamp(W / scale, -448, 448)); value = e4m3(code) * scale; "
        "cells per checkpoint tensor (never across qkvr slices)")


@dataclass(frozen=True)
class Fp8Quantised:
    scheme: str
    codes: torch.Tensor  # uint8 [n, k], E4M3FN words
    grid: torch.Tensor  # fp32 [n, k/128], the scale of each element's 1x128 cell
    row_scales: torch.Tensor | None  # fp32 [n] for tensor / channel, else None

    def dequantize(self) -> torch.Tensor:
        values = self.codes.view(torch.float8_e4m3fn).to(torch.float32)
        return values * self.grid.repeat_interleave(128, dim=1)


def _scales(amax: torch.Tensor) -> torch.Tensor:
    scale = amax.to(torch.float32) / torch.tensor(E4M3_MAX, dtype=torch.float32)
    return torch.where(amax > 0, scale, torch.ones_like(scale))


def quantize(weight: torch.Tensor, scheme: str) -> Fp8Quantised:
    """Quantise one BF16 checkpoint tensor [n, k] (k a multiple of 128) by ``scheme``."""

    if scheme not in SCHEMES:
        raise ValueError(f"unknown FP8 scheme {scheme!r}; expected one of {SCHEMES}")
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise TypeError(f"expected a rank-two bfloat16 weight, got {weight.dtype} {tuple(weight.shape)}")
    n, k = weight.shape
    if k % 128:
        raise ValueError(f"k = {k} is not a multiple of 128")
    w = weight.to(torch.float32)
    magnitude = w.abs()
    groups = k // 128
    row_scales = None
    if scheme == "tensor":
        row_scales = _scales(magnitude.max()).expand(n).contiguous()
        grid = row_scales[:, None].expand(n, groups).contiguous()
    elif scheme == "channel":
        row_scales = _scales(magnitude.amax(dim=1))
        grid = row_scales[:, None].expand(n, groups).contiguous()
    elif scheme == "g1x128":
        grid = _scales(magnitude.reshape(n, groups, 128).amax(dim=2))
    else:  # b128
        blocks = -(-n // 128)
        padded = torch.zeros((blocks * 128, k), dtype=torch.float32)
        padded[:n] = magnitude
        block_amax = padded.reshape(blocks, 128, groups, 128).amax(dim=(1, 3))
        grid = _scales(block_amax).repeat_interleave(128, dim=0)[:n].contiguous()
    full = grid.repeat_interleave(128, dim=1)
    scaled = torch.clamp(w / full, -E4M3_MAX, E4M3_MAX)
    codes = scaled.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return Fp8Quantised(scheme, codes, grid, row_scales)


def quantize_slices(slices: list[torch.Tensor], scheme: str) -> Fp8Quantised:
    """Quantise each checkpoint tensor on its own and stack the rows (the fused ``qkvr``)."""

    parts = [quantize(part, scheme) for part in slices]
    rows = None
    if all(p.row_scales is not None for p in parts):
        rows = torch.cat([p.row_scales for p in parts])
    return Fp8Quantised(scheme, torch.cat([p.codes for p in parts]).contiguous(),
                        torch.cat([p.grid for p in parts]).contiguous(), rows)


def reconstruction_error(weight: torch.Tensor, scheme: str) -> dict[str, float]:
    """Weight reconstruction error of ``scheme`` on one BF16 tensor (DESIGN 10.1 stage A)."""

    q = quantize(weight, scheme)
    w = weight.to(torch.float64)
    error = q.dequantize().to(torch.float64) - w
    norm = torch.linalg.vector_norm(w)
    return {
        "rel_fro": float(torch.linalg.vector_norm(error) / norm) if norm > 0 else 0.0,
        "max_abs": float(error.abs().max()),
        "mean_abs": float(error.abs().mean()),
        "rms_w": float(w.square().mean().sqrt()),
        "scales": int(q.grid.numel() if q.row_scales is None or scheme in ("g1x128", "b128")
                      else (1 if scheme == "tensor" else q.row_scales.numel())),
    }


__all__ = ["E4M3_MAX", "RULE", "SCHEMES", "STORAGE_FORMAT", "Fp8Quantised", "quantize",
           "quantize_slices", "reconstruction_error"]
