"""Singular-value fine-tuning (SVF), following Transformer-squared (Sun et al., 2025).

A selected frozen Linear ``W`` is decomposed once at init as ``W = U diag(s) Vᵀ``.
Only a per-singular-value ``scale`` vector is trainable; ``U``, ``s`` and ``Vᵀ`` stay
frozen. The effective weight is ``U diag(s ⊙ scale) Vᵀ``, so ``scale = 1`` reproduces
the original layer exactly. Trainable params per patched matrix = ``min(out, in)``,
which lets the whole router stay under the ~20K-parameter budget.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn


class SVDLinear(nn.Module):
    """Drop-in replacement for a frozen ``nn.Linear`` exposing only SV scales."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.detach()
        out_features, in_features = weight.shape
        self.out_features = out_features
        self.in_features = in_features
        # full_matrices=False => U:(out,k) s:(k,) Vh:(k,in), k=min(out,in)
        U, s, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        self.register_buffer("U", U.to(weight.dtype))
        self.register_buffer("s", s.to(weight.dtype))
        self.register_buffer("Vh", Vh.to(weight.dtype))
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None
        # The one trainable tensor.
        self.scale = nn.Parameter(torch.ones_like(self.s))

    @property
    def rank(self) -> int:
        return self.s.numel()

    def effective_weight(self) -> torch.Tensor:
        return (self.U * (self.s * self.scale)) @ self.Vh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


def _set_submodule(root: nn.Module, dotted: str, value: nn.Module) -> None:
    parent = root
    *path, last = dotted.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, last, value)


def apply_svf(model: nn.Module, patterns: list[str]) -> list[SVDLinear]:
    """Replace every ``nn.Linear`` whose qualified name matches any regex in
    ``patterns`` with an :class:`SVDLinear`. All other parameters are frozen.

    Returns the list of installed SVDLinear modules (their ``scale`` params are the
    only trainable tensors introduced).
    """
    for p in model.parameters():
        p.requires_grad_(False)

    compiled = [re.compile(p) for p in patterns]
    targets = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and any(c.search(name) for c in compiled)
    ]
    installed: list[SVDLinear] = []
    for name in targets:
        linear = model.get_submodule(name)
        svd = SVDLinear(linear).to(linear.weight.device)
        _set_submodule(model, name, svd)
        installed.append(svd)
    return installed


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
