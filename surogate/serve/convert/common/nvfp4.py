"""NVFP4 words in torch: decode them, and produce them from real values.

The artifact stores NVFP4 exactly as the checkpoints do -- E2M1 codes packed two to a byte
(the even column in the low nibble), one E4M3FN block scale per 16 values in natural
``[N, K/16]`` order before the engine's swizzle, and a per-tensor global scale. What differs
between exporters is the global scale's direction, so these helpers take it in one convention
only, the engine's: the **divisor**. A value is ``e2m1(code) * e4m3(block) / divisor``.
ModelOpt states the reciprocal (``weight_scale_2`` multiplies); compressed-tensors states the
divisor itself (``weight_global_scale``).

`quantize` is the plain recipe every NVFP4 exporter here uses by default: the block scale is
the block's absolute maximum over six, times the divisor, rounded to E4M3FN; each value is
rounded to the nearest E2M1 point of its block (ties to the even code). Nothing searches for
a better scale -- a caller that wants one passes `scale_candidates`.
"""

from __future__ import annotations

import torch

#: E2M1 magnitudes by code (sign in bit 3).
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: The largest E2M1 magnitude and the largest finite E4M3FN value: their product is the
#: largest magnitude a block can represent at a divisor of one.
E2M1_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16


def _e2m1_table(device: torch.device) -> torch.Tensor:
    magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    return torch.cat((magnitudes, -magnitudes))


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """``[N, K/2]`` uint8 -> ``[N, K]`` int64 E2M1 codes (even column from the low nibble)."""
    if packed.dtype != torch.uint8 or packed.dim() != 2:
        raise TypeError("NVFP4 packed codes must be a uint8 matrix")
    n, half = packed.shape
    codes = torch.empty((n, half * 2), dtype=torch.int64, device=packed.device)
    codes[:, 0::2] = (packed & 0x0F).long()
    codes[:, 1::2] = (packed >> 4).long()
    return codes


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` integer E2M1 codes -> ``[N, K/2]`` uint8, the even column in the low nibble."""
    if codes.dim() != 2 or codes.shape[1] % 2:
        raise ValueError("NVFP4 codes must be a matrix with an even column count")
    low = codes[:, 0::2].to(torch.int16) & 0x0F
    high = codes[:, 1::2].to(torch.int16) & 0x0F
    return (low | (high << 4)).to(torch.uint8)


def dequantize(packed: torch.Tensor, scales: torch.Tensor, divisor: float) -> torch.Tensor:
    """The FP32 values ``[N, K]`` NVFP4 words represent.

    ``scales`` is the natural ``[N, K/16]`` plane, as ``float8_e4m3fn`` or its uint8 words.
    """
    if scales.dtype == torch.uint8:
        scales = scales.view(torch.float8_e4m3fn)
    if scales.dtype != torch.float8_e4m3fn:
        raise TypeError("NVFP4 block scales must be E4M3FN")
    values = _e2m1_table(packed.device)[unpack_codes(packed)]
    block = scales.to(torch.float32).repeat_interleave(BLOCK, dim=1)
    return values * block / float(divisor)


def global_divisor(values: torch.Tensor) -> float:
    """The divisor that maps a tensor's absolute maximum onto the top of the format:
    ``6 * 448 / amax``. A zero tensor takes one, which represents it exactly."""
    amax = float(values.abs().max()) if values.numel() else 0.0
    return E2M1_MAX * E4M3_MAX / amax if amax > 0.0 else 1.0


def _round_e2m1(scaled: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code of each value (already divided by its block scale); ties go to the
    even code, which is the even magnitude index -- the rounding the hardware quantiser uses."""
    table = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=scaled.device)
    magnitude = scaled.abs().clamp(max=E2M1_MAX)
    # Midpoints between consecutive magnitudes; `searchsorted` with right=False puts a value
    # sitting exactly on a midpoint in the lower bucket, and the tie fix-up below moves it to
    # whichever neighbour has the even index.
    midpoints = (table[1:] + table[:-1]) / 2
    index = torch.searchsorted(midpoints, magnitude.contiguous(), right=False)
    tie = (index < len(E2M1_MAGNITUDES) - 1) & (magnitude == midpoints[index.clamp(max=6)])
    index = torch.where(tie & (index % 2 == 1), index + 1, index)
    negative = (scaled < 0) & (index > 0)
    return index + negative.long() * 8


def quantize(values: torch.Tensor, divisor: float,
             scale_candidates: tuple[float, ...] = (1.0,)) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 words for ``values`` ``[N, K]`` at the given global divisor.

    Returns ``(packed [N, K/2] uint8, scales [N, K/16] uint8 E4M3FN words)``. Each block's
    scale is ``amax / 6 * divisor`` rounded to E4M3FN; with ``scale_candidates`` other than
    ``(1.0,)`` each block also tries that scale times each candidate and keeps the one with the
    smallest squared error, which is a cheap search that never does worse than the plain rule.
    """
    if values.dim() != 2 or values.shape[1] % BLOCK:
        raise ValueError("NVFP4 quantisation needs a matrix whose width is a multiple of 16")
    n, k = values.shape
    x = values.to(torch.float32).reshape(n, k // BLOCK, BLOCK) * float(divisor)
    amax = x.abs().amax(dim=2, keepdim=True)
    best_codes = None
    best_scales = None
    best_error = None
    for candidate in scale_candidates:
        raw = (amax / E2M1_MAX * candidate).clamp(max=E4M3_MAX)
        scale = raw.to(torch.float8_e4m3fn)
        decoded = scale.to(torch.float32)
        safe = torch.where(decoded > 0, decoded, torch.ones_like(decoded))
        codes = _round_e2m1(x / safe)
        reconstructed = _e2m1_table(x.device)[codes] * decoded
        error = (reconstructed - x).square().sum(dim=2, keepdim=True)
        if best_error is None:
            best_codes, best_scales, best_error = codes, scale, error
        else:
            better = error < best_error
            best_codes = torch.where(better, codes, best_codes)
            best_scales = torch.where(better, scale.to(torch.float32), best_scales.to(torch.float32)).to(
                torch.float8_e4m3fn)
            best_error = torch.minimum(error, best_error)
    packed = pack_codes(best_codes.reshape(n, k))
    scales = best_scales.reshape(n, k // BLOCK).view(torch.uint8)
    return packed, scales


__all__ = [
    "BLOCK",
    "E2M1_MAGNITUDES",
    "E2M1_MAX",
    "E4M3_MAX",
    "dequantize",
    "global_divisor",
    "pack_codes",
    "quantize",
    "unpack_codes",
]
