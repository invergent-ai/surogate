"""Selected rows of an inline, unquantized matrix without decoding its full bank."""

from __future__ import annotations

import warnings
import torch


def direct_rows(payload, shape, format_name, rows, *, device, dtype=torch.bfloat16):
    element_type = {"BF16": torch.bfloat16, "FP32": torch.float32}[format_name]
    # The artifact mapping is read-only. index_select copies the selected rows before
    # any caller receives a tensor, so no writable view escapes this function.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given buffer is not writable")
        source = torch.frombuffer(payload, dtype=element_type).reshape(shape)
    indices = torch.as_tensor(rows, dtype=torch.long, device="cpu")
    return source.index_select(0, indices).to(device=device, dtype=dtype)
