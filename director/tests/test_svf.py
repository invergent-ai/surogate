"""Tests for singular-value fine-tuning."""

from __future__ import annotations

import torch
from torch import nn

from director.fugu.svf import SVDLinear, apply_svf, count_trainable


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 6)
        self.b = nn.Linear(6, 3)

    def forward(self, x):
        return self.b(self.a(x))


def test_svdlinear_identity_at_scale_one():
    lin = nn.Linear(4, 6)
    svd = SVDLinear(lin)
    x = torch.randn(5, 4)
    assert torch.allclose(lin(x), svd(x), atol=1e-4)
    assert svd.rank == 4  # min(out=6, in=4)


def test_apply_svf_freezes_all_but_scales():
    model = Tiny()
    installed = apply_svf(model, [r"^a$"])
    assert len(installed) == 1
    assert isinstance(model.a, SVDLinear)
    assert not isinstance(model.b, SVDLinear)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable == ["a.scale"]
    assert count_trainable(model) == 4  # rank of a

    # buffers U/s/Vh are not trainable
    assert not model.a.U.requires_grad
    assert not model.a.s.requires_grad


def test_scale_gradient_flows():
    model = Tiny()
    apply_svf(model, [r"^a$"])
    x = torch.randn(3, 4)
    out = model(x).sum()
    out.backward()
    assert model.a.scale.grad is not None
    assert model.a.scale.grad.abs().sum() > 0
