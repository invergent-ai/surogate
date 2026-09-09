# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-License-Identifier: MIT
# Vendored from flash-linear-attention v0.5.2; see README.md and LICENSE.

import triton
import triton.language as tl


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


gather = tl.gather
