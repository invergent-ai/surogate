"""Conversion target for the interleaved gated-delta architecture.

One target for Qwen3.5, Qwen3.6 and Qwen3.8: the same decoder at every published size,
and the checkpoint states which. The names re-exported here are the registered size's,
for callers with no checkpoint in hand; a conversion asks `inventory.export_inventory`
for the contract of the checkpoint and export it actually has.
"""

from .inventory import (
    ALIAS_SPECS,
    PROFILES,
    FORMAT_COUNTS,
    LAYOUT_COUNTS,
    LOGICAL_ROW_VIEW_SPECS,
    MODEL_ID,
    OBJECT_SPECS,
    RESOURCE_SPECS,
    TARGET_KEY,
    TENSOR_SPECS,
)

__all__ = [
    "ALIAS_SPECS",
    "PROFILES",
    "FORMAT_COUNTS",
    "LAYOUT_COUNTS",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "OBJECT_SPECS",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "TENSOR_SPECS",
]
