"""Inkling-2 (``inkling2_loop``, thinkingmachines/Inkling-2-Mini) -> SInfer v2 artifact.

The artifact contract (interface I-1): text model +
DSpark drafter + frontend resources; NVFP4 experts with raw per-expert globals, MXFP8 experts in
the MXFP8 format, BF16 attention always and FP8 attention variants on request.
"""

from .convert import ARCHITECTURE, convert

__all__ = ["ARCHITECTURE", "convert"]
