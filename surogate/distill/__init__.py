"""Offline knowledge-distillation support: `.kd` sidecar format and teacher capture.

Only the sidecar API is re-exported here; the capture driver
(`surogate.distill.capture`) pulls in torch/transformers and must be imported
explicitly.
"""

from surogate.distill.sidecar import (
    KD_MAGIC,
    KD_MAX_TOP_K,
    SidecarHeader,
    SidecarWriter,
    TokenShardHeader,
    read_sidecar,
    read_sidecar_header,
    read_token_shard_header,
    sidecar_path_for,
    validate_sidecar,
    write_sidecar_header,
)

__all__ = [
    "KD_MAGIC",
    "KD_MAX_TOP_K",
    "SidecarHeader",
    "SidecarWriter",
    "TokenShardHeader",
    "read_sidecar",
    "read_sidecar_header",
    "read_token_shard_header",
    "sidecar_path_for",
    "validate_sidecar",
    "write_sidecar_header",
]
