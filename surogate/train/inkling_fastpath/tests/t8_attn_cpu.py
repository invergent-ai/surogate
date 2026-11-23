"""S3 CPU checks of the attention-LoRA pieces (no engine): (1) lora_add's forward == the S10 arithmetic written out in fp64
(bf16(bf16(base) + (x . bf16(bf16(A) s)^T) . bf16(B)^T)) up to fp32 summation; (2) its A / B / x gradients through the engine-op trainer's TokenChunked in chunks ==
plain autograd over all tokens (rel <= 1e-5); (3) the key layout: param_shapes / work_views / ParamLayout agree, expert keys first; (4) config refusals."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("IFT_CHUNK", "16")
import ift_long as LG  # noqa: E402
import ift_ref as RA  # noqa: E402
import t8_step as TS  # noqa: E402
import t8_visit as TV  # noqa: E402

BF = torch.bfloat16


def rb(x):
    return x.to(BF).float()


def main():
    g = torch.Generator().manual_seed(3)
    T, k, n, R, s = 40, 96, 72, 8, 0.5
    x = rb(torch.randn(T, k, generator=g))
    base = rb(torch.randn(T, n, generator=g))
    A = rb(torch.randn(R, k, generator=g) / k ** 0.5)
    Bm = rb(torch.randn(n, R, generator=g) * 0.1)
    y = TV.lora_add(base, x, A, Bm, s)
    a64 = (A.to(BF).double() * s).to(BF).double()
    ref = (base.double() + (x.double() @ a64.T) @ Bm.to(BF).double().T).to(BF).float()
    diff = (y != ref).float().mean().item()
    worst = ((y - ref).abs() / ref.abs().clamp(min=1e-3)).max().item()
    print(f"forward: {diff:.4f} of elements differ by one bf16 rounding at most (worst rel {worst:.2e})")
    assert worst <= 2 ** -7, worst
    # gradients: TokenChunked in chunks of 16 vs plain autograd over all tokens
    gy = torch.randn(T, n, generator=g)
    xs = x.clone().requires_grad_(True)
    Al, Bl = A.clone().requires_grad_(True), Bm.clone().requires_grad_(True)
    out = TV.lora_add(base, xs, Al, Bl, s)
    out.backward(gy)
    want = (xs.grad.clone(), Al.grad.clone(), Bl.grad.clone())
    xs2 = x.clone().requires_grad_(True)
    A2, B2 = A.clone().requires_grad_(True), Bm.clone().requires_grad_(True)
    o2 = LG.tc(lambda xx, bb, aa, bm: TV.lora_add(bb, xx, aa, bm, s), [xs2, base], [0, 0], [0], params=[A2, B2], chunk=16)
    o2.backward(gy)
    got = (xs2.grad, A2.grad, B2.grad)
    for name, a, b in zip(("x", "A", "B"), got, want):
        r = float((a - b).norm() / b.norm())
        print(f"grad {name}: rel {r:.2e}")
        assert r <= 1e-5, name
    # key layout (no engine needed for the shapes)
    import t8_worker as TW
    sh = TW.param_shapes(64)
    keys = tuple(sh[0])
    assert keys[:4] == TS.LORA_KEYS and keys[4:] == TS.ATTN_KEYS and keys == TS.param_keys(64)
    assert sh[0]["qA"] == (64, 2304) and sh[0]["qB"] == (3072, 64) and sh[0]["oA"] == (64, 3072) and sh[0]["oB"] == (2304, 64) and sh[0]["sB"] == (24, 64)
    flat = torch.arange(sum(int(np.prod(v)) for d in sh for v in d.values()), dtype=torch.float32)
    wv = TW.work_views(flat, sh)
    assert [k for k in wv[3] if k != "scale"] == list(keys) and float(wv[0]["A13"].reshape(-1)[0]) == 0.0
    lay = TS.ParamLayout(sh, [0, 23], ("B13", "B2", "qA", "qB", "oB"))
    assert lay.keys == keys and set(j for (_, j) in lay.index) == {keys.index(k) for k in ("B13", "B2", "qA", "qB", "oB")}
    n_attn = sum(int(np.prod(sh[0][k])) for k in TS.ATTN_KEYS)
    print(f"attention LoRA r64: {n_attn:,} parameters per layer, {24 * n_attn:,} in all")
    for bad in (dict(attn_rank=128), dict(attn_rank=64, attn_alpha=48.0), dict(attn_rank=0, train_keys=("qA",))):
        kw = {"head": "plain", "segment_keys": ()} | bad
        kw.setdefault("train_keys", ("A13", "B13", "A2", "B2"))
        try:
            TS.validate_config(TS.T8Config(**kw))
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    TS.validate_config(TS.T8Config(head="plain", attn_rank=64, train_keys=TS.param_keys(64), segment_keys=("B13", "B2")))
    # the combined adapter file (S5): round trip, names, config
    import json
    import tempfile

    import t8_adapter as TA
    gg = np.random.default_rng(0)
    layers = []
    for _ in range(24):
        d = {k: gg.standard_normal(v).astype(np.float32) * 0.01 for k, v in TW.param_shapes(64)[0].items()}
        layers.append(d)
    out = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    TA.export_combined(layers, out, 64, 32.0, 16)
    cfg = json.load(open(os.path.join(out, "adapter_config.json")))
    assert cfg["r"] == 64 and cfg["lora_alpha"] == 32.0 and list(cfg["rank_pattern"].values()) == [16] and "alpha_pattern" not in cfg
    back, _ = TA.import_combined(out)
    from surogate.inkling_rl import adapter as AD
    for L in (0, 11, 23):
        for k in layers[L]:
            assert np.array_equal(back[L][k], AD.round_bf16(layers[L][k])), (L, k)
    names = AD.read_safetensors(os.path.join(out, "adapter_model.safetensors"))
    assert f"{AD.PREFIX}.5.attn.attn_sink_proj.lora_B.weight" in names and names[f"{AD.PREFIX}.5.attn.wo_ud.lora_A.weight"][1].shape == (64, 3072)
    assert len(names) == 24 * (14 + 128 * 4)
    print("combined adapter round trip: PASS")
    # the gate's one-ulp helper
    sys.path.insert(0, HERE)
    import t8_gate_s3 as G3
    w = torch.tensor([1.0, -2.0, 0.3, -0.75], dtype=BF).float()
    up1 = torch.nextafter(w.to(BF), torch.full_like(w.to(BF), float("inf"))).float()
    eq, within = G3.ulp_cmp(up1, w)
    assert eq == 0.0 and within
    eq, within = G3.ulp_cmp(w + (up1 - w) * 2.5, w)
    assert not within
    print("ulp helper: PASS")
    print("T8 ATTN CPU TEST: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
